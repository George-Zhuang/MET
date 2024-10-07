__copyright__ = "Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved."
__license__ = """
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
import asyncio
import doctest
import inspect
import os

import carb
import numpy as np
import omni.kit

# NOTE:
#   omni.kit.test - std python's unittest module with additional wrapping to add suport for async/await tests
#   For most things refer to unittest docs: https://docs.python.org/3/library/unittest.html
# Import extension python module we are testing with absolute import path, as if we are external user (other extension)
import omni.kit.test
import omni.replicator.core as rep
from metspace.core.writers.rtsp import RTSPCamera, RTSPWriter
from omni.replicator.core import Writer, WriterRegistry
from omni.syntheticdata import SyntheticData


class TestRTSPCamera(omni.kit.test.AsyncTestCase):
    _class_name = "TestRTSPCamera"

    # Before running each test
    async def setUp(self):
        pass

    # After running each test
    async def tearDown(self):
        pass

    async def test_initialize(self):
        # ---Test---
        # output_dir and prim_path have to be specified.
        with self.assertRaises(ValueError):
            camera = RTSPCamera()
        # ---Test---

        # ---Test---
        output_dir = "output_dir"
        prim_path = "/World/Camera"
        camera = RTSPCamera(output_dir=output_dir, prim_path=prim_path)
        self.assertTrue(camera.device == camera._default_device)
        self.assertTrue(camera.fps == camera._default_fps)
        self.assertTrue(camera.width == camera._default_width)
        self.assertTrue(camera.height == camera._default_height)
        self.assertTrue(camera.bitrate == camera._default_bitrate)
        self.assertTrue(camera.annotator == camera._default_annotator)
        self.assertTrue(camera.output_dir == output_dir)
        self.assertTrue(camera.prim_path == prim_path)

        # Default command
        output_dir = camera.output_dir + "_World_Camera" + "_" + camera.annotator
        vcodec = "hevc_nvenc"  # 'h264_nvenc'
        command = [
            "ffmpeg",
            "-hwaccel",
            "cuda",
            "-hwaccel_output_format",
            "cuda",
            "-hwaccel_device",
            str(camera.device),
            "-re",
            "-use_wallclock_as_timestamps",
            "1",
            "-f",
            "rawvideo",
            # '-thread_queue_size', '4096',  # May help https://stackoverflow.com/questions/61723571/correct-usage-of-thread-queue-size-in-ffmpeg
            # '-vcodec', 'rawvideo',
            "-pix_fmt",
            camera.pixel_fmt,  # 'bgr24',
            # '-src_range', '1',
            "-s",
            f"{camera.width}x{camera.height}",
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
            f"{camera.bitrate}k",
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
            str(camera.fps),
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",  # udp is the most performant. But udp does not support NAT/firewall nor encryption
            # '-vf', 'scale=in_range=full:out_range=full',
            # '-dst_range', '1', '-color_range', '2',
            output_dir,
        ]
        self.assertTrue(camera.command == command)
        # ---Test---

        # ---Test---
        # Fully speced parameters
        device = 1
        fps = 45
        width = 1024
        height = 512
        bitrate = 1234567
        annotator = "DiffuseAlbedo"
        output_dir = "output_dir"
        prim_path = "/World/Camera"

        camera = RTSPCamera(
            device=device,
            fps=fps,
            width=width,
            height=height,
            bitrate=bitrate,
            annotator=annotator,
            output_dir=output_dir,
            prim_path=prim_path,
        )
        self.assertTrue(camera.device == device)
        self.assertTrue(camera.fps == fps)
        self.assertTrue(camera.width == width)
        self.assertTrue(camera.height == height)
        self.assertTrue(camera.bitrate == bitrate)
        self.assertTrue(camera.annotator == annotator)
        self.assertTrue(camera.output_dir == output_dir)
        self.assertTrue(camera.prim_path == prim_path)

        # Default command
        output_dir = camera.output_dir + "_World_Camera" + "_" + camera.annotator
        vcodec = "hevc_nvenc"  # 'h264_nvenc'
        command = [
            "ffmpeg",
            "-hwaccel",
            "cuda",
            "-hwaccel_output_format",
            "cuda",
            "-hwaccel_device",
            str(camera.device),
            "-re",
            "-use_wallclock_as_timestamps",
            "1",
            "-f",
            "rawvideo",
            # '-thread_queue_size', '4096',  # May help https://stackoverflow.com/questions/61723571/correct-usage-of-thread-queue-size-in-ffmpeg
            # '-vcodec', 'rawvideo',
            "-pix_fmt",
            camera.pixel_fmt,  # 'bgr24',
            # '-src_range', '1',
            "-s",
            f"{camera.width}x{camera.height}",
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
            f"{camera.bitrate}k",
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
            str(camera.fps),
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",  # udp is the most performant. But udp does not support NAT/firewall nor encryption
            # '-vf', 'scale=in_range=full:out_range=full',
            # '-dst_range', '1', '-color_range', '2',
            output_dir,
        ]
        self.assertTrue(camera.command == command)
        # ---Test---

        # ---Test---
        # Test parameter autocorrection
        output_dir = "output_dir"
        prim_path = "/World/Camera"

        camera = RTSPCamera(device=-1, output_dir=output_dir, prim_path=prim_path)
        self.assertTrue(camera.device == camera._default_device)

        camera = RTSPCamera(fps=-30, output_dir=output_dir, prim_path=prim_path)
        self.assertTrue(camera.fps == camera._default_fps)

        camera = RTSPCamera(width=-30, output_dir=output_dir, prim_path=prim_path)
        self.assertTrue(camera.width == camera._default_width)

        camera = RTSPCamera(height=-30, output_dir=output_dir, prim_path=prim_path)
        self.assertTrue(camera.height == camera._default_height)

        camera = RTSPCamera(bitrate=-30, output_dir=output_dir, prim_path=prim_path)
        self.assertTrue(camera.bitrate == camera._default_bitrate)
        # ---Test---

        # ---Test---
        # Invalid annotator
        output_dir = "output_dir"
        prim_path = "/World/Camera"
        annotator = "wrong_annotator"
        with self.assertRaises(ValueError):
            camera = RTSPCamera(annotator=annotator, output_dir=output_dir, prim_path=prim_path)
        # ---Test---

        function_name = inspect.currentframe().f_code.co_name
        print(f"{self._class_name}.{function_name}() completes successfully.")


# # Before running the tests, an RTSP server should be started first. The RTSP server hostname
# # is defined in macro ${rtsp_server}.
# # Having a test class dervived from omni.kit.test.AsyncTestCase declared on the root of module will make it auto-discoverable by omni.kit.test
# class TestRTSPWriter(omni.kit.test.AsyncTestCase):
#     _class_name = "TestRTSPWriter"

#     # Before running each test
#     async def setUp(self):
#         await omni.usd.get_context().new_stage_async()
#         await omni.kit.app.get_app().next_update_async()

#         stage = omni.usd.get_context().get_stage()
#         stage.DefinePrim("/World", "Xform")
#         self.camera = stage.DefinePrim("/World/Camera", "Camera")
#         self.camera_01 = stage.DefinePrim("/World/Camera_01", "Camera")

#         self.out_dir = carb.tokens.get_tokens_interface().resolve("${temp}/test_RTSPWriter")
#         pass

#     # After running each test
#     async def tearDown(self):
#         await omni.usd.get_context().new_stage_async()
#         pass

#     async def test_initialize(self):
#         cam_path = "/World/Camera"
#         render_products = [rep.create.render_product(cam_path, (640, 480))]

#         rtsp_server = "localhost"
#         rtsp_port = 8554
#         rtsp_topic = "RTSPWriter"
#         writer = rep.WriterRegistry.get(rtsp_topic)

#         # ---Test---
#         # Test default values
#         writer.initialize()

#         # Default device ID is 0.
#         self.assertTrue(writer.device == writer._default_device)

#         # Default annotator is LdrColor. Only 1 annotator.
#         self.assertTrue(len(writer.annotators) == 1)

#         # rgb should be changed to LdrColor
#         annotator_name = writer.annotators[0].get_name()
#         self.assertTrue(annotator_name == writer._default_annotator)

#         # Default RTSP server
#         self.assertTrue(writer.output_dir == writer._default_output_dir)

#         self.assertTrue(writer.backend == None)
#         self.assertTrue(writer._backend == None)
#         self.assertTrue(writer.cameras == {})
#         self.assertTrue(writer._frame_id == 0)
#         self.assertTrue(writer._tcp_retries == 0)
#         # ---Test---

#         # ---Test---
#         writer.initialize(device=-1, annotator="rgb")

#         # Input device ID < 0, it is reset to 0.
#         self.assertTrue(writer.device == 0)

#         # Only 1 annotator.
#         self.assertTrue(len(writer.annotators) == 1)

#         # rgb should be changed to LdrColor
#         annotator_name = writer.annotators[0].get_name()
#         self.assertTrue(annotator_name == "LdrColor")
#         # ---Test---

#         # ---Test---
#         # Valid URL
#         url = f"rtsp://{rtsp_server}:{rtsp_port}/{rtsp_topic}"
#         writer.initialize(output_dir=url)

#         self.assertTrue(writer.output_dir == url)
#         # ---Test---

#         # ---Test---
#         # Valid URL
#         url = f"rtsp://{rtsp_server}:{rtsp_port}/{rtsp_topic}/"
#         writer.initialize(output_dir=url)

#         self.assertTrue(writer.output_dir == url)
#         # ---Test---

#         # ---Test---
#         # URL miss topic
#         with self.assertRaises(ValueError):
#             url = f"rtsp://{rtsp_server}:{rtsp_port}"
#             writer.initialize(output_dir=url)
#         # ---Test---

#         # ---Test---
#         # URL ends with with wrong protocol '$rtsp://'
#         with self.assertRaises(ValueError):
#             url = f"$rtsp://{rtsp_server}:{rtsp_port}/{rtsp_topic}"
#             writer.initialize(output_dir=url)
#         # ---Test---

#         # ---Test---
#         # URL hostname:port pair is invalid
#         with self.assertRaises(ValueError):
#             url = f"rtsp://{rtsp_server}:12345/{rtsp_topic}"
#             writer.initialize(output_dir=url)
#         # ---Test---

#         # ---Test---
#         # URL hostname:port pair is invalid
#         with self.assertRaises(ValueError):
#             url = f"rtsp://invalid_hostname:{rtsp_port}/{rtsp_topic}"
#             writer.initialize(output_dir=url)
#         # ---Test---

#         # ---Test---
#         # Test invalid annotator
#         with self.assertRaises(ValueError):
#             writer.initialize(annotator="arbitrary")

#         # String value has to be exact.
#         with self.assertRaises(ValueError):
#             writer.initialize(annotator="ldrColor")
#         # ---Test---

#         function_name = inspect.currentframe().f_code.co_name
#         print(f"{self._class_name}.{function_name}() completes successfully.")
