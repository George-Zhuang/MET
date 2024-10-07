__copyright__ = "Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved."
__license__ = """
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

import csv
import ctypes
import io
import json
import math
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import List

import carb
import numpy as np
import omni.usd
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry
from omni.syntheticdata.scripts.SyntheticData import SyntheticData
from pxr import Semantics, Usd, UsdSkel

EPS = 1e-5
# Procuring standard KITTI Labels for objects annotated in the KITTI-format
# The dictionary is ordered where label idx corresponds to semantic ID
# See https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py


__version__ = "0.0.2"


class LidarWriter(Writer):
    """
    Lidar data output
    """

    def __init__(
        self,
        output_dir: str,
        s3_bucket: str = None,
        s3_region: str = None,
        s3_endpoint: str = None,
        s3: bool = False,
        frame_num: int = 50,
    ):
        """Create a KITTI Writer

        Args:
            output_dir: Output directory to which KITTI annotations will be saved.
            semantic_types: List of semantic types to consider. If ``None``, only consider semantic types ``"class"``.
            omit_semantic_type: If ``True``, only record the semantic data (ie. ``class: car`` becomes ``car``).
            bbox_height_threshold: The minimum valid bounding box height, in pixels. Value must be positive integers.
            partly_occluded_threshold: Minimum occlusion factor for bounding boxes to be considered partly occluded.
            fully_visible_threshold: Minimum occlusion factor for bounding boxes to be considered fully visible.
            mapping_path: json path to the label to color mapping for KITTI
            colorize_instance_segmentation: bool value representing whether image is to be colorized or not.
            use_kitti_dir_names: If True, use standard KITTI directory names: `rgb`->`image_02`,
                `semantic_segmentation`->`semantic`, `instance_segmentation`->`instance`, `object_detection`->`label_02`

        """
        self._date = datetime.now(datetime.now().astimezone().tzinfo)
        self.year = self._date.year
        self.version = __version__
        self._frame_id = 0
        self._frame_num = frame_num
        if s3:
            self.backend = BackendDispatch(
                {
                    "use_s3": True,
                    "paths": {
                        "out_dir": output_dir,
                        "s3_bucket": s3_bucket,
                        "s3_region": s3_region,
                        "s3_endpoint_url": s3_endpoint,
                    },
                }
            )
        else:
            self.backend = BackendDispatch({"paths": {"out_dir": output_dir}})
        self._backend = self.backend  # Kept for backwards compatibility

        self.annotators = ["RtxSensorCpuIsaacCreateRTXLidarScanBufferLocal"]

    @staticmethod
    def params_values():
        # params refer to the initialize function
        return {
            "output_dir": os.path.abspath(
                carb.settings.get_settings().get("/exts/metspace/default_replicator_output_path")
            )
        }

    @staticmethod
    def params_labels():
        return {"output_dir": "output_dir"}

    def write_buffered_lidar(self, data, sub_dir: str, annotator: str):
        point_cloud = data[annotator]["data"]
        pc_and_intensity_matrix = np.append(point_cloud, data[annotator]["intensity"].reshape(-1, 1), axis=1)
        buf = io.BytesIO()
        np.save(buf, pc_and_intensity_matrix)
        self._backend.write_blob(f"{sub_dir}/{self._frame_id}.npy", buf.getvalue())

    def write(self, data):

        render_products = [k for k in data.keys() if k.startswith("rp_")]

        if len(render_products) == 1:
            self.write_buffered_lidar(
                data,
                data[render_products[0]]["camera"].replace("/", "_"),
                "RtxSensorCpuIsaacCreateRTXLidarScanBufferLocal",
            )
        else:
            for render_product in render_products:
                self.write_buffered_lidar(
                    data,
                    data[render_product]["camera"].replace("/", "_"),
                    "RtxSensorCpuIsaacCreateRTXLidarScanBufferLocal-{}".format(render_product.split("_", 1)[-1]),
                )

        self._frame_id += 1


annotator_name = "RtxSensorCpu" + "IsaacCreateRTXLidarScanBufferLocal"
AnnotatorRegistry.register_annotator_from_node(
    name=annotator_name,
    input_rendervars=["RtxSensorCpuPtr"],
    node_type_id="omni.isaac.sensor.IsaacCreateRTXLidarScanBuffer",
    output_data_type=np.float32,
    output_channels=3,
    init_params={"transformPoints": False},
)
WriterRegistry.register(LidarWriter)
