__copyright__ = "Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved."
__license__ = """
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

import re
import socket
import subprocess as sp
import sys
from shutil import which
from typing import Callable, Dict, Iterable, List, Tuple, Union

import carb  # carb logging

# reload(sys.modules["omni.replicator.core"])
import omni.replicator.core as rep
import omni.timeline
import pynvml as pynvml
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry
from omni.replicator.core.scripts.utils.utils import ReplicatorItem
from omni.replicator.core.scripts.utils.viewport_manager import HydraTexture

__version__ = "0.0.1"

# [Change Log]
# 09/15/2023    Verified on omni.replicator.core-1.10.4+kit105.1.

# print(rep.__file__)
# # /isaac-sim/extscache/omni.replicator.core-1.10.4+105.1.lx64.r.cp310/omni/replicator/core/__init__.py
# NvEnc only has a 48bit per pixel (unpacked) format. 64bit, 128bit per pixel formats are not supported by NVENC.
# https://forums.developer.nvidia.com/t/dxgi-nvenc-yuv444-10bit-format-compatability/189782
# NVENC can perform end-to-end encoding for H.264, HEVC 8-bit, HEVC 10-bit, AV1 8-bit and AV1 10-bit. Pixel channel
# datatypes of np.float16 or np.float32 cannot be supported. Fallback to libx264 (software) codec on grayf32le/be,
# rgba64le/be, gbrapf32le/be pixel formats.
# https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/nvenc-application-note/index.html#nvenc-capabilities
# _annotator_type_uint8_channel_1 = [
#     'pointcloud_pointSemantic',
# ]
#
# https://gitlab-master.nvidia.com/omniverse/synthetic-data/omni.replicator/-/blob/develop/source/extensions/omni.replicator.core/python/scripts/annotators_default.py#L9
_annotator_type_uint8_channel_4 = [
    "LdrColor",
    "rgb",
    "semantic_segmentation",  # colorize = True
    "instance_id_segmentation",  # colorize = True
    "instance_segmentation",  # colorize = True
    "DiffuseAlbedo",  # ??? pixel format is wrong ???
    "Roughness",
    # 'pointcloud_pointRgb',
]

_annotator_type_float16_channel_1 = ["EmissionAndForegroundMask"]

_annotator_type_float32_channel_1 = [
    "distance_to_camera",
    "distance_to_image_plane",
    "DepthLinearized",  # ??? pixel format is wrong ???
]

_annotator_type_float16_channel_4 = [
    "HdrColor",
    # 'AmbientOcclusion', # ??? data['AmbientOcclusion'] is empty ???
    # 'SpecularAlbedo', # ??? data['AmbientOcclusion'] is empty ???
    # 'DirectDiffuse', # ??? data['AmbientOcclusion'] is empty ???
    # 'DirectSpecular', # ??? data['AmbientOcclusion'] is empty ???
    # 'IndirectDiffuse', # ??? data['AmbientOcclusion'] is empty ???
    # # Path Tracing AOVs
    # 'PtDirectIllumation',
    # 'PtGlobalIllumination',
    # 'PtReflections',
    # 'PtRefractions',
    # 'PtSelfIllumination',
    # 'PtBackground',
    # 'PtWorldNormal',
    # 'PtWorldPos',
    # 'PtZDepth',
    # 'PtVolumes',
    # 'PtDiffuseFilter',
    # 'PtReflectionFilter',
    # 'PtRefractionFilter',
    # 'PtMultiMatte0',
    # 'PtMultiMatte1',
    # 'PtMultiMatte2',
    # 'PtMultiMatte3',
    # 'PtMultiMatte4',
    # 'PtMultiMatte5',
    # 'PtMultiMatte6',
    # 'PtMultiMatte7',
]

# _annotator_type_float32_channel_3 = [
#     'pointcloud', # data['pointcloud']['data']
# ]

# !!!Not able to play grbapf32le/be - Need help on ffmpeg encoding.!!!
# https://stackoverflow.com/questions/71725213/ffmpeg-cant-recognize-3-channels-with-each-32-bit
# _annotator_type_float32_channel_4 = [
#     'normals',
#     'motion_vectors',
#     'cross_correspondence',
#     'SmoothNormal', # ??? data['AmbientOcclusion'] is empty ???
#     'BumpNormal', # ??? data['AmbientOcclusion'] is empty ???
#     'Motion2d', # ??? data['AmbientOcclusion'] is empty ???
#     'Reflections', # ??? data['AmbientOcclusion'] is empty ???
#     # 'pointcloud_pointNormals',
# ]

_nvenc_annotators = _annotator_type_uint8_channel_4
# 'pointcloud', # Empty from camera. Is this a bug???

_software_annotators = (
    _annotator_type_float16_channel_1 + _annotator_type_float32_channel_1 + _annotator_type_float16_channel_4
)

_supported_annotators = _nvenc_annotators + _software_annotators

# Default GPU device where NVENC operates on
_default_device = 0
# Default annotator of each render product to capture and stream
_default_annotator = "LdrColor"


class RTSPCamera:
    """The class records a render products (HydraTexture) by its prim path.
    The class also records the ffmpeg subprocess command which is customized
    by the render product's camera parameters, e.g. fps, width, height, annotator.
    The published RTSP URL of each RTSPCamera instance is constructed by appending
    the render product's camera prim path and the annotator name to the base output directory.

    Notes:
        The supported annotators are:
            'LdrColor' / 'rgb',
            'semantic_segmentation',
            'instance_id_segmentation',
            'instance_segmentation',
            'DiffuseAlbedo',
            'Roughness',
            'EmissionAndForegroundMask'
            'distance_to_camera',
            'distance_to_image_plane',
            'DepthLinearized',
            'HdrColor'
        Please refer to https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html
        for more details on the supported annotators. Annotators: 'LdrColor' / 'rgb', 'semantic_segmentation',
        'instance_id_segmentation', 'instance_segmentation', are accelerated by NVENC while the rest annotators
        are software encoded by CPU. The default video stream format is HEVC.
    """

    _default_device = _default_device
    _default_fps = 30  # RTSP stream frame rate. Should be consistent with render product frame rate.
    _default_bitrate = 5000  # encoding bitrate in kb.
    _default_width = 512  # encoded image width in pixels
    _default_height = 512  # encoded image height in pixels
    _default_annotator = _default_annotator

    _name = "RTSPCamera"
    _version = __version__

    def __init__(
        self,
        device: int = _default_device,  # GPU device for encoding acceleration
        fps: int = _default_fps,  # Streaming FPS
        width: int = _default_width,  # RTSP stream window width
        height: int = _default_height,  # RTSP stream window height
        bitrate: int = _default_bitrate,  # Encoding bit rate
        annotator: str = _default_annotator,  # Only support ONE annotator
        # https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html#annotators-information
        output_dir: str = None,  # Base RTSP URL
        prim_path: str = None,  # Camera prim path (how to do "from .utils.viewport_manager import HydraTexture")
    ):
        """Create an RTSP camera instance

        Args:
            device:     GPU device id to be used for hardware accelerated encoding.
            fps:        The encoded video frame rate.
            width:      The encoded video frame width in pixels.
            height:     The encoded video frame height in pixels.
            bitrate:    The maximal value of the encoded video bitrate in kb/s.
            annotator: The annotator data to be streamed. The supported annotators are:
                'LdrColor' / 'rgb',
                'semantic_segmentation',
                'instance_id_segmentation',
                'instance_segmentation',
                'DiffuseAlbedo',
                'Roughness',
                'EmissionAndForegroundMask'
                'distance_to_camera',
                'distance_to_image_plane',
                'DepthLinearized',
                'HdrColor'
                Please refer to
                https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html#annotators-information
            output_dir: The base RTSP URL defined by server, port, and topic. Each
                        render product has the unique RTSP URL which is constructed by
                        appending the render product name by the end of output_dir.
            prim_path:  The full camera prim path
        """

        self.initialize(
            device=device,
            fps=fps,
            width=width,
            height=height,
            bitrate=bitrate,
            annotator=annotator,
            output_dir=output_dir,
            prim_path=prim_path,
        )

    def initialize(
        self,
        device: int = _default_device,  # GPU device for encoding acceleration
        fps: int = _default_fps,  # Streaming FPS
        width: int = _default_width,  # RTSP stream window width
        height: int = _default_height,  # RTSP stream window height
        bitrate: int = _default_bitrate,  # x264enc bit rate
        annotator: str = _default_annotator,  # Only support ONE annotator
        output_dir: str = None,  # Base RTSP URL
        prim_path: str = None,  # RTSP URL
    ):
        """Initialize the writer.

        Args:
            device:     GPU device id to be used for hardware accelerated encoding.
            fps:        The encoded video frame rate.
            width:      The encoded video frame width in pixels.
            height:     The encoded video frame height in pixels.
            bitrate:    The maximal value of the encoded video bitrate in kb/s.
            annotator: The annotator data to be streamed. The supported annotators are:
                'LdrColor' / 'rgb',
                'semantic_segmentation',
                'instance_id_segmentation',
                'instance_segmentation',
                'DiffuseAlbedo',
                'Roughness',
                'EmissionAndForegroundMask'
                'distance_to_camera',
                'distance_to_image_plane',
                'DepthLinearized',
                'HdrColor'
                Please refer to
                https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html#annotators-information
            output_dir: The base RTSP URL defined by server, port, and topic. Each
                        render product has the unique RTSP URL which is constructed by
                        appending the render product name by the end of output_dir.
            prim_path:  The full camera prim path
        Returns:
            None
        """

        self.device = device
        self.fps = fps
        self.width = width
        self.height = height
        self.bitrate = bitrate
        self.annotator = annotator
        self.output_dir = output_dir
        self.prim_path = prim_path
        self._verify_init_arguments()

        # TODO: Support more annotators described in https://gitlab-master.nvidia.com/omniverse/synthetic-data/omni.replicator/-/blob/develop/source/extensions/omni.replicator.core/python/scripts/annotators_default.py#L9
        # https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html?highlight=hdrcolor#annotator-output
        # https://gitlab-master.nvidia.com/omniverse/synthetic-data/omni.replicator/-/blob/develop/source/extensions/omni.replicator.core/python/scripts/annotators_default.py
        # Annotators encoded by NVENC.
        if self.annotator in _annotator_type_uint8_channel_4:
            self.pixel_fmt = "rgba"
        elif self.annotator in _annotator_type_float16_channel_1:
            if sys.byteorder == "little":
                self.pixel_fmt = "gray16le"
            else:
                self.pixel_fmt = "gray16be"
        elif self.annotator in _annotator_type_float32_channel_1:
            if sys.byteorder == "little":
                self.pixel_fmt = "grayf32le"
            else:
                self.pixel_fmt = "grayf32be"
        elif self.annotator in _annotator_type_float16_channel_4:
            if sys.byteorder == "little":
                self.pixel_fmt = "rgba64le"
            else:
                self.pixel_fmt = "rgba64be"
        # elif self.annotator in ['pointcloud']: # (N,3) - np.float32
        #     # pc_data = pointcloud_anno.get_data()
        #     # print(pc_data)
        #     # # {
        #     # #     'data': array([...], shape=(<num_points>, 3), dtype=float32),
        #     # #     'info': {
        #     # #         'pointNormals': [ 0.000e+00 1.00e+00 -1.5259022e-05 ... 0.00e+00 -1.5259022e-05 1.00e+00], shape=(<num_points> * 4), dtype=float32),
        #     # #         'pointRgb': [241 240 241 ... 11  12 255], shape=(<num_points> * 4), dtype=uint8),
        #     # #         'pointSemantic': [2 2 2 ... 2 2 2], shape=(<num_points>), dtype=uint8),
        #     # #
        #     # #     }
        #     # # }
        #     # https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html?highlight=hdrcolor#point-cloud
        #     # Due to the 48bit per pixel limitation in NVENC, 'data' and 'info'.'pointNormals' are not supported.
        #     # By default, render 'info'.'pointRgb' - (N, 4) - np.uint8
        #     self.pixel_fmt = 'rgba'
        # Annotators encoded by software due to NVENC 48bit/pixel limit
        # elif self.annotator in _annotator_type_float32_channel_4:
        #     # output_data_type=np.float32, output_channels=4
        #     # https://stackoverflow.com/questions/1346034/whats-the-most-pythonic-way-of-determining-endianness
        #     if sys.byteorder == 'little':
        #         self.pixel_fmt = 'gbrapf32le'
        #     else:
        #         self.pixel_fmt = 'gbrapf32be'
        else:
            raise ValueError(f"Publishing {self.annotator} annotator is not supported.")

        # Replace special character/separator with '_'
        prim_path = self.prim_path.replace("/", "_")
        prim_path = prim_path.replace(" ", "_")
        output_dir = self.output_dir + prim_path + "_" + self.annotator

        # --- Verify ffmpeg installed ---
        if not which("ffmpeg"):
            raise ValueError(f"ffmpeg cannot be found in the system.")
        # --- Verify ffmpeg installed ---

        # Video writer
        # https://video.stackexchange.com/questions/12905/repeat-loop-input-video-with-ffmpeg
        # To enable GPU acceleration, please uncommented these two lines. One for the input. One for the output.
        # '-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda', '-hwaccel_device', str(device),
        # '-c:v', 'h264_nvenc', '-c:a', 'copy', # encodes all video streams with h264_nvenc.
        self.command = ["ffmpeg"]

        if self.annotator in _nvenc_annotators:
            self.command += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-hwaccel_device", str(self.device)]
            vcodec = "hevc_nvenc"  # 'h264_nvenc'
        elif self.annotator in _software_annotators:  # Software encoding
            vcodec = "libx265"  # 'libx264'
        else:
            raise ValueError(f"Publishing {self.annotator} annotator is not supported.")

        # https://stackoverflow.com/questions/71725213/ffmpeg-cant-recognize-3-channels-with-each-32-bit
        self.command += [
            "-re",
            "-use_wallclock_as_timestamps",
            "1",
            "-f",
            "rawvideo",
            # '-thread_queue_size', '4096',  # May help https://stackoverflow.com/questions/61723571/correct-usage-of-thread-queue-size-in-ffmpeg
            # '-vcodec', 'rawvideo',
            "-pix_fmt",
            self.pixel_fmt,  # 'bgr24',
            # '-src_range', '1',
            "-s",
            f"{self.width}x{self.height}",
            # '-r', str(fps),
            # '-stream_loop', '-1', # Loop infinite times.
            "-i",
            "-",
            "-c:a",
            "copy",
            "-c:v",
            vcodec,
            "-preset",
            "ll",
            # '-pix_fmt', 'yuv420p',
            # '-preset', 'ultrafast',
            # '-vbr', '5', # Variable Bit Rate (VBR). Valid values are 1 to 5. 5 has the best quality.
            # '-b:v', f'{bitrate}k',
            "-maxrate:v",
            f"{self.bitrate}k",
            "-bufsize:v",
            "64M",  # Buffering is probably required
            # passthrough (0) - Each frame is passed with its timestamp from the demuxer to the muxer.
            # -vsync 0 cannot be applied together with -r/-fpsmax.
            # cfr (1) - Frames will be duplicated and dropped to achieve exactly the requested constant frame rate.
            # vfr (2) - Frames are passed through with their timestamp or dropped so as to prevent 2 frames from having the same timestamp.
            # '-vsync', 'passthrough',
            "-vsync",
            "cfr",
            "-r",
            str(self.fps),
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",  # udp is the most performant. But udp does not support NAT/firewall nor encryption
            # '-vf', 'scale=in_range=full:out_range=full',
            # '-dst_range', '1', '-color_range', '2',
            output_dir,
        ]
        # carb.log_info(f'[{self._name}] command = {self.command}')

        carb.log_info(
            f'"{self.annotator}" of "{self.prim_path}" will be published to "{output_dir}" encoded by "{vcodec}".'
        )

        self.pipe = None

    def _verify_init_arguments(self):
        if self.device <= 0:
            self.device = self._default_device

        if self.fps <= 0:
            self.fps = self._default_fps

        if self.width <= 0:
            self.width = self._default_width

        if self.height <= 0:
            self.height = self._default_height

        if self.bitrate <= 0:
            self.bitrate = self._default_bitrate

        if self.annotator not in _supported_annotators:
            raise ValueError(f"{self.annotator} is not part of the supported annotators, {_supported_annotators}.")

        if not self.output_dir:
            raise ValueError(f'{self._name} member "output_dir" cannot be empty.')

        if not self.prim_path:
            raise ValueError(f'{self._name} member "prim_path" cannot be empty.')


class RTSPWriter(Writer):
    """Publish annotations of attached render products to an RTSP server.

    The Writer tracks a dictionary of render products (HydraTexture) by the combo of the
    annotator name and the render product's prim path. Each render product is recorded as
    an instance of RTSPCamera. The published RTSP URL of each RTSPCamera instance is
    constructed by appending the render product's camera prim path and the annotator name
    to the base output directory.

    The supported annotators are:
        'LdrColor' / 'rgb',
        'semantic_segmentation',
        'instance_id_segmentation',
        'instance_segmentation',
        'DiffuseAlbedo',
        'Roughness',
        'EmissionAndForegroundMask'
        'distance_to_camera',
        'distance_to_image_plane',
        'DepthLinearized',
        'HdrColor'
    Please refer to https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html
    for more details on the supported annotators. Annotators: 'LdrColor' / 'rgb', 'semantic_segmentation',
    'instance_id_segmentation', 'instance_segmentation', are accelerated by NVENC while the rest annotators
    are software encoded by CPU. The video stream format is HEVC.
    """

    # Default RTSP server, port, and topic.
    _default_device = _default_device
    _default_annotator = _default_annotator
    _default_server = "localhost"
    _default_port = 8554
    _default_topic = "RTSPWriter"
    _default_output_dir = f"rtsp://{_default_server}:{_default_port}/{_default_topic}"

    _name = "RTSPWriter"
    _version = __version__

    def __init__(
        self,
        device: int = _default_device,  # GPU device where NVENC operates on.
        annotator: str = _default_annotator,  # Only support ONE annotator
        output_dir: str = _default_output_dir,  # RTSP URL
    ):
        """Initialize the writer.

        Args:
            output_dir: The base RTSP URL defined by server, port, and topic. Each
                render product has the unique RTSP URL which is constructed by
                appending the render product name by the end of output_dir.
        """

        super().__init__()
        self.initialize(device=device, annotator=annotator, output_dir=output_dir)

    # def initialize(self, **kwargs):
    def initialize(
        self,
        device: int = _default_device,  # GPU device where NVENC operates on.
        annotator: str = _default_annotator,  # Only support ONE annotator
        output_dir: str = _default_output_dir,  # RTSP URL
    ):
        """Initialize the writer.

        Args:
            output_dir: The base RTSP URL defined by server, port, and topic. Each
                render product has the unique RTSP URL which is constructed by
                appending the render product name by the end of output_dir.

        Returns:
            None
        """
        self.device = device
        if self.device < 0:
            self.device = _default_device

        if annotator == "rgb":
            annotator = "LdrColor"
        if annotator not in _supported_annotators:
            raise ValueError(f"{annotator} is not part of the supported annotators, {_supported_annotators}.")

        self.annotators = []
        if annotator in ["semantic_segmentation", "instance_id_segmentation", "instance_segmentation"]:
            # https://gitlab-master.nvidia.com/omniverse/synthetic-data/omni.replicator/-/blob/develop/source/extensions/omni.replicator.core/python/scripts/writers_default/basicwriter.py
            # If ``True``, semantic segmentation is converted to an image where semantic IDs are mapped to colors
            # and saved as a uint8 4 channel PNG image. If ``False``, the output is saved as a uint32 PNG image.
            # Defaults to ``True``.
            self.annotators.append(AnnotatorRegistry.get_annotator(annotator, init_params={"colorize": True}))
        else:
            self.annotators.append(AnnotatorRegistry.get_annotator(annotator))
        # print(f'DEBUG: dir(self.annotators[0]) = {dir(self.annotators[0])}')
        # print(f'DEBUG: self.annotators[0].get_name() = {self.annotators[0].get_name()}')

        self.output_dir = output_dir
        # --- Verify RTSP URL ---
        match = re.search(r"^rtsp\://(.+)\:([0-9]+)/(.+)$", self.output_dir)
        if not match:
            raise ValueError(
                f'{self.output_dir} is not a valid RTSP stream URL. The format is "rtsp://<hostname>:<port>/<topic>".'
            )

        hostname = match.group(1)
        port = match.group(2)
        # topic = match.group(3)

        # Verify RTSP server is live
        sock = socket.socket()
        try:
            sock.connect((hostname, int(port)))
            # originally, it was
            # except Exception, e:
            # but this syntax is not supported anymore.
        except Exception as e:
            raise ValueError(f"RTSP server at {hostname}:{port} is not accessible with exception {e}.")
        finally:
            sock.close()
        # carb.log_info(f'INFO: output_dir = {self.output_dir}')
        # --- Verify RTSP URL ---

        self.backend = None  # BackendDispatch({"paths": {"out_dir": "/tmp"}}) # None
        self._backend = self.backend  # Kept for backwards compatibility

        self.cameras = {}  # map render_product name to a subprocess
        self._frame_id = 0
        self._tcp_retries = 0

        pynvml.nvmlInit()
        try:
            self.nDevices = pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError as error:
            carb.log_warn(error)
            self.nDevices = 1
            pass
        pynvml.nvmlShutdown()

    @staticmethod
    def default_device():
        return _default_device

    @staticmethod
    def default_annotator():
        return _default_annotator

    @staticmethod
    def supported_annotators():
        return _supported_annotators

    @staticmethod
    def default_output_dir():
        return RTSPWriter._default_output_dir

    @staticmethod
    def params_values():
        # params refer to the initialize function
        return {
            "annotator": RTSPWriter.default_annotator(),
            "output_dir": RTSPWriter.default_output_dir(),
        }

    @staticmethod
    def params_labels():
        # params refer to the initialize function
        return {
            "annotator": "Annotator",
            "output_dir": "Output Path",
        }

    @staticmethod
    def tooltip():
        return f"""
            RTSPWriter
            - It will use the output path under 'Parameters'.
            - It supports one annotator at a time. All available annotators:
                {str(RTSPWriter.supported_annotators())}.
        """

    def on_final_frame(self):
        """Run after final frame is written.

        Notes:
            When "stop" button is clicked in Isaac Sim UI, this function is called.
            The ffmpeg subprocesses are killed (SIGKILL).
        """
        super().on_final_frame()

        for key, camera in self.cameras.items():
            if camera.pipe:
                camera.pipe.stdin.close()
                camera.pipe.wait()
                camera.pipe.kill()
                camera.pipe = None
                carb.log_info(f'Subprocess on "{camera.prim_path}" has been terminated.')

        self.cameras.clear()
        self._frame_id = 0

    # Failed in gitlab-master.nvidia.com:5005/isaac/omni_isaac_sim/isaac-sim:latest-2023.1 with
    # 2023-09-14 21:23:55 [74,802ms] [Error] [omni.kit.app.plugin] [py stderr]: IndexError: list index out of range
    #
    # At:
    #   /isaac-sim/extscache/omni.replicator.core-1.10.4+105.1.lx64.r.cp310/omni/replicator/core/scripts/writers.py(750): _attach
    #   /isaac-sim/extscache/omni.replicator.core-1.10.4+105.1.lx64.r.cp310/omni/replicator/core/scripts/writers.py(551): attach
    #   /isaac-sim/extscache/omni.replicator.core-1.10.4+105.1.lx64.r.cp310/omni/replicator/core/scripts/writers.py(388): attach
    #   /tmp/carb.h59qSD/script_1694726635.py(374): attach
    def _config_cameras(self, data):
        """Attach one or a list of render products to the writer.

        This is the base function called by either attach() or attach_async().
        The function constructs the ffmpeg command for each render product.
        Each render product associates with an unique RTSP URL which is built
        by appending render product's camera prim path to self.output_dir.

        Args:
            data: A dictionary containing the annotator data for the current frame.

        Returns:
            None
        """
        # Clear existing camera pipes
        if self.cameras:
            self.on_final_frame()

        annotator_name = self.annotators[0].get_name()

        render_products = [k for k in data.keys() if k.startswith("rp_")]
        if not isinstance(render_products, list):
            render_products = [render_products]

        # ex: render_products = ['rp_RenderProduct_Replicator', 'rp_RenderProduct_Replicator_01']
        rp_path_idx = 0
        for rp_path in render_products:
            camera_prim_path = data[rp_path]["camera"]
            resolution = data[rp_path]["resolution"]

            # Make keys in "self.cameras" match keys in "data".
            # When there is a single render product, key = <annotator>
            # When there are multiple render products, key = <annotator>-<rp_path[3:]>
            # Please refer to RTSPWriter.write() for details.
            if len(render_products) == 1:
                key = annotator_name
            else:
                key = annotator_name + "-" + rp_path[3:]
            # carb.log_info(f'key = {key}')

            # Distribute ffmpeg HW encoding among multiple GPUs
            if self.nDevices > 1:
                device = rp_path_idx % self.nDevices
            else:
                device = 0

            camera = RTSPCamera(
                device=device,
                width=resolution[0],  # rp.hydra_texture.width,
                height=resolution[1],  # rp.hydra_texture.height,
                annotator=annotator_name,
                output_dir=self.output_dir,
                prim_path=camera_prim_path,  # Camera prim path
            )

            camera.pipe = sp.Popen(camera.command, stdin=sp.PIPE)
            if not camera.pipe:
                raise Exception(f"Can't start ffmpeg RTSP client writer on {camera.prim_path}.")

            self.cameras[key] = camera
            rp_path_idx += 1

    def write(self, data):
        """Write function called from the OgnWriter node on every frame to process annotator output.

        The function publishes one of _supported_annotators of each attached render product
        to the RTSP server. If the render product name does not exist in self.pipes (dict),
        the function creates a new ffmpeg subprocess.

        Args:
            data: A dictionary containing the annotator data for the current frame.

        Returns:
            None

        Raises:
            Exception: Can't open RTSP client writer.
        """

        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing():
            return

        # Create RTSPCamera's on the first frame.
        if self._frame_id == 0:
            self._config_cameras(data)

        # DEBUG
        # print(f'cameras = {self.cameras}')
        # for k in data.keys():
        #    print(f'DEBUG: data[{k}] = {data[k]}')
        #
        # !!!Please pay attention to the dictionary format!!!
        # ---When there is a single render product---
        # data[pointcloud] = {'data': array([], shape=(0, 3), dtype=float32), 'info': {'pointNormals': array([], dtype=float32), 'pointRgb': array([], dtype=uint8), 'pointSemantic': array([], dtype=uint8)}}
        # data[swhFrameNumber] = 0
        # data[reference_time] = (306607, 30)
        # data[distribution_outputs] = {}
        # data[trigger_outputs] = {}
        # data[named_outputs] = {}
        # data[rp_RenderProduct_Replicator] = {'camera': '/World/Camera', 'resolution': array([512, 512], dtype=int32)}
        # data[LdrColor] = [[[  0   0   0 255]
        #
        # ---When there are multiple render products---
        # data[pointcloud-RenderProduct_Replicator] = {'data': array([], shape=(0, 3), dtype=float32), 'info': {'pointNormals': array([], dtype=float32), 'pointRgb': array([], dtype=uint8), 'pointSemantic': array([], dtype=uint8)}}
        # data[pointcloud-RenderProduct_Replicator_01] = {'data': array([], shape=(0, 3), dtype=float32), 'info': {'pointNormals': array([], dtype=float32), 'pointRgb': array([], dtype=uint8), 'pointSemantic': array([], dtype=uint8)}}
        # data[swhFrameNumber] = 0
        # data[reference_time] = (81317, 30)
        # data[distribution_outputs] = {}
        # data[trigger_outputs] = {}
        # data[named_outputs] = {}
        # data[rp_RenderProduct_Replicator] = {'camera': '/World/Camera', 'resolution': array([512, 512], dtype=int32)}
        # data[rp_RenderProduct_Replicator_01] = {'camera': '/World/Camera_01', 'resolution': array([1280,  720], dtype=int32)}
        # data[LdrColor-RenderProduct_Replicator] = [[[  0   0   0 255] ...
        # data[LdrColor-RenderProduct_Replicator_01] = [[[  0   0   0 255] ...
        # data[semantic_segmentation-RenderProduct_Replicator] = {'data': array([[0, 0, 0, ..., 0, 0, 0], ...
        #        [0, 0, 0, ..., 0, 0, 0]], dtype=uint32), 'info': {'_uniqueInstanceIDs': array([1], dtype=uint8), 'idToLabels': {'(0, 0, 0, 0)': {'class': 'BACKGROUND'}, '(0, 0, 0, 255)': {'class': 'UNLABELLED'}}}}
        # data[semantic_segmentation-RenderProduct_Replicator_01] = {'data': array([[0, 0, 0, ..., 0, 0, 0], ...
        #        [0, 0, 0, ..., 0, 0, 0]], dtype=uint32), 'info': {'_uniqueInstanceIDs': array([1], dtype=uint8), 'idToLabels': {'(0, 0, 0, 0)': {'class': 'BACKGROUND'}, '(0, 0, 0, 255)': {'class': 'UNLABELLED'}}}}

        # When there is a single render product, key = <annotator>
        # When there are multiple render products, key = <annotator>-<rp.path[8:]>
        for key, camera in self.cameras.items():
            # print(f'DEBUG: key = {key}')
            if (
                key.startswith("semantic_segmentation")
                or key.startswith("instance_id_segmentation")
                or key.startswith("instance_segmentation")
            ):
                camera.pipe.stdin.write(data[key]["data"].tobytes())
            # elif key.startswith('pointcloud'):
            #     camera.pipe.stdin.write(data[key]['info']['pointRgb'].tobytes())
            else:
                camera.pipe.stdin.write(data[key].tobytes())
            # self._backend.write_blob(bbox_filepath, buf.getvalue())

        self._frame_id += 1
        # carb.log_info(f'frame id : {self._frame_id}')

    def write_metadata(self):
        # pass
        self._is_warning_backend_posted = True
        self._is_metadata_written = True


rep.WriterRegistry.register(RTSPWriter)
