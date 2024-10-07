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
import random
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List

import carb
import numpy as np
import omni.anim.graph.core as ag
import omni.usd
from omni.anim.people import PeopleSettings
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry
from omni.syntheticdata.scripts.SyntheticData import SyntheticData
from pxr import Semantics, Usd, UsdSkel

from .character_data_collector import CharacterDataCollector
from .utils import Utils

EPS = 1e-5
# Procuring standard KITTI Labels for objects annotated in the KITTI-format
# The dictionary is ordered where label idx corresponds to semantic ID
# See https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
KITTI_LABELS = {
    "unlabelled": (0, 0, 0, 0),
    "ego vehicle": (0, 0, 0, 0),
    "rectification border": (0, 0, 0, 0),
    "out of roi": (0, 0, 0, 0),
    "static": (0, 0, 0, 0),
    "dynamic": (111, 74, 0, 255),
    "ground": (81, 0, 81, 255),
    "road": (128, 64, 128, 255),
    "sidewalk": (244, 35, 232, 255),
    "parking": (250, 170, 160, 255),
    "rail track": (230, 150, 140, 255),
    "building": (70, 70, 70, 255),
    "wall": (102, 102, 156, 255),
    "fence": (190, 153, 153, 255),
    "guard rail": (180, 165, 180, 255),
    "bridge": (150, 100, 100, 255),
    "tunnel": (150, 120, 90, 255),
    "pole": (153, 153, 153, 255),
    "polegroup": (153, 153, 153, 255),
    "traffic light": (250, 170, 30, 255),
    "traffic sign": (220, 220, 0, 255),
    "vegetation": (107, 142, 35, 255),
    "terrain": (152, 251, 152, 255),
    "background": (70, 130, 180, 255),  # Sky is always labelled as BACKGROUND
    "person": (220, 20, 60, 255),
    "rider": (255, 0, 0, 255),
    "car": (0, 0, 142, 255),
    "truck": (0, 0, 70, 255),
    "bus": (0, 60, 100, 255),
    "caravan": (0, 0, 90, 255),
    "trailer": (0, 0, 110, 255),
    "train": (0, 80, 100, 255),
    "motorcycle": (0, 0, 230, 255),
    "bicycle": (119, 11, 32, 255),
    "license plate": (0, 0, 142, 255),
}


__version__ = "0.0.2"


class LidarFusionWriter(Writer):
    """Writer outputting data in the KITTI annotation format http://www.cvlibs.net/datasets/kitti/
        Development work to provide full support is ongoing.

    Supported Annotations:
        RGB
        Object Detection (partial 2D support, see notes)
        Depth
        Semantic Segmentation
        Instance Segmentation

    Notes:
        Object Detection
        Bounding boxes with a height smaller than 25 pixels are discarded

        Supported: bounding box extents, semantic labels
        Partial Support: occluded (occlusion is estimated from the area ratio of tight / loose bounding boxes)
        Unsupported: alpha, dimensions, location, rotation_y, truncated (all set to default values of 0.0)
    """

    def __init__(
        self,
        output_dir: str,
        s3_bucket: str = None,
        s3_region: str = None,
        s3_endpoint: str = None,
        semantic_types: List[str] = None,
        omit_semantic_type: bool = False,
        renderproduct_idxs: List[tuple] = None,
        mapping_path: str = None,
        semantic_filter_predicate: str = None,
        valid_unoccluded_threshold: float = 0.6,
        s3: bool = False,
        frame_num: int = 50,
        rgb: bool = True,
        bbox: bool = True,
        lidar: bool = True,
    ):
        """Create a KITTI Writer

        Args:
            output_dir: Output directory to which KITTI annotations will be saved.
            semantic_types: List of semantic types to consider. If ``None``, only consider semantic types ``"class"``.
            omit_semantic_type: If ``True``, only record the semantic data (ie. ``class: car`` becomes ``car``).
            mapping_path: json path to the label to color mapping for KITTI
        """
        self._date = datetime.now(datetime.now().astimezone().tzinfo)
        self.year = self._date.year
        self.version = __version__
        self._frame_id = 0

        # Track numbering of the next frame to be written.
        self._write_id = 0
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
        self._sqlite_lock = threading.Lock()
        self._omit_semantic_type = omit_semantic_type
        self._render_product_idxs = renderproduct_idxs
        self.frame_delay = 0
        self.valid_unoccluded_threshold = valid_unoccluded_threshold
        ##container used to store skeleton information
        self.character_data_collector = CharacterDataCollector(self.valid_unoccluded_threshold, None, self.frame_delay)
        self.character_data_collector._create_skeleton_dicts()
        self.camera_frame_dict = {}
        self.output_dir = output_dir
        self.format = "KITTI"
        # Lidar parameters
        self.lidar_to_world_transform = {}
        self.test_points = []
        self.lidar_buffer = []
        self.lidar_collection_count = 0

        if mapping_path:
            self.mapping_dict = self._procure_labels_from_json(mapping_path)
        else:
            # update the color dictionary to sign character with distinguish color
            color_dict = {}
            color_dict = self.character_data_collector.create_random_color_dict(KITTI_LABELS)
            color_dict.update(KITTI_LABELS)
            self.mapping_dict = color_dict

        # Specify the semantic types that will be included in output
        if semantic_types is not None:
            if semantic_filter_predicate is None:
                semantic_filter_predicate = ":*; ".join(semantic_types) + ":*"
            else:
                raise ValueError(
                    "`semantic_types` and `semantic_filter_predicate` are mutually exclusive. Please choose only one."
                )
        elif semantic_filter_predicate is None:
            semantic_filter_predicate = "class:*"

        if semantic_filter_predicate is not None:
            SyntheticData.Get().set_instance_mapping_semantic_filter(semantic_filter_predicate)

        # annotator list
        self.annotators = []
        self.write_rgb = rgb
        self.write_bbox = bbox
        self.write_lidar = lidar

        if self.write_rgb:
            self.annotators.append("rgb")

        if self.write_bbox:
            self.annotators.append("bounding_box_3d_fast")
            self.annotators.append("bounding_box_2d_tight_fast")
            self.annotators.append("bounding_box_2d_loose_fast")
            self.annotators.append("camera_params")

        if self.write_lidar:
            self.annotators.append("RtxSensorCpuIsaacCreateRTXLidarScanBufferLocal")

        self.skip_frames = carb.settings.get_settings().get(
            "/persistent/exts/metspace/skip_starting_frames"
        )
        self.writer_interval = carb.settings.get_settings().get(
            "/persistent/exts/metspace/frame_write_interval"
        )

    @staticmethod
    def params_values():
        # Params to be recoginized in config file format and their default values
        # Refer to the initialize function
        return {
            "output_dir": os.path.abspath(
                carb.settings.get_settings().get("/exts/metspace/default_replicator_output_path")
            ),
            "rgb": True,
            "bbox": True,
            "lidar": True,
        }

    @staticmethod
    def params_labels():
        # Params to be display in the UI and their display labels
        return {
            "output_dir": "output_dir",
            "rgb": "rgb",
            "bbox": "bbox",
            "lidar": "lidar",
        }

    @staticmethod
    def tooltip():
        return f"""
            LidarFusionWriter
            - Generates point cloud data from lidar sensors.
            - Generates 3d bbox, 2d bbox and rgb from cameras.
            - Camera and lidars are paired together. Camera_x and Lidar_x are expected to be fused together.
            - The number of cameras and lidars passed to the writer must be the same.
        """

    def _write_rgb(self, data, sub_dir: str, annotator: str):
        # Save the rgb data under the correct path
        rgb_dir_name = "rgb"
        rgb_file_path = os.path.join(sub_dir, rgb_dir_name, f"{self._write_id}.png")
        self._backend.write_image(rgb_file_path, data[annotator])

    @staticmethod
    def tooltip():
        return f"""
            LidarFusionWriter
            - Generates point cloud data from lidar sensors.
            - Generates 3d bbox, 2d bbox and rgb from cameras.
            - Camera and lidars are paired together. Camera_x and Lidar_x are expected to be fused together.
            - The number of cameras and lidars passed to the writer must be the same.
        """

    def _write_object_detection(
        self,
        data,
        sub_dir: str,
        render_product_annotator: str,
        bbox_2d_tight_annotator: str,
        bbox_2d_loose_annotator: str,
        bbox_3d_annotator: str,
        camera_params_annotator: str,
    ):
        """
        Saves the labels for the object detection data in Kitti format.

        Unsupported fields: alpha, rotation_y, truncated (all set to default values of 0.0)

        Notes on occlusion:
        # This estimation relies on the ratio between loose (unoccluded) and tight bounding boxes
        # and may produce unexpected results in certain cases:
        #
        #        //           XXXX                 //  XXXX
        #  _____//____/_______XXXX          ______//___XXXX______
        # )   __          __  XXXX         )   __      XXXX_     \
        # |__/  \________/  \_XXXX         |__/  \_____XXXX \____|
        # ___\__/________\__/_XXXX__      ____\_ /_____XXXX_/______
        # PARTLY OCCLUDED (OK!)           FULLY VISIBLE (INCORRECT)
        """
        label_set = []
        rp_width = data[render_product_annotator]["resolution"][0]
        rp_height = data[render_product_annotator]["resolution"][1]
        camera_info = data[camera_params_annotator]
        camera_view = np.reshape(camera_info["cameraViewTransform"], (4, 4))
        camera_projection = np.reshape(camera_info["cameraProjection"], (4, 4))
        view_proj_mat = np.dot(camera_view, camera_projection)

        # width and height of the image
        rp_width = data[render_product_annotator]["resolution"][0]
        rp_height = data[render_product_annotator]["resolution"][1]

        bbox_tight = data[bbox_2d_tight_annotator]["data"]
        bbox_loose = data[bbox_2d_loose_annotator]["data"]
        bbox_tight_bbox_ids = data[bbox_2d_tight_annotator]["info"]["bboxIds"]
        bbox_loose_bbox_ids = data[bbox_2d_loose_annotator]["info"]["bboxIds"]

        # For box in tight, find the corresponding index of box in loose
        bbox_loose_indices = np.where(np.isin(bbox_loose_bbox_ids, bbox_tight_bbox_ids))[0]
        selected_bbox_loose = bbox_loose[bbox_loose_indices]

        box_3d_annotator = data[bbox_3d_annotator]

        label_to_bbox_3d_info = {}
        bbox_id_to_label = box_3d_annotator["info"]["idToLabels"]
        bbox_3d_data = box_3d_annotator["data"]

        for bbox_3d in bbox_3d_data:
            semantic_id = bbox_3d["semanticId"]
            bbox_3d_label = bbox_id_to_label[semantic_id]["class"]
            if self.character_data_collector.is_character(str(bbox_3d_label)):
                label_to_bbox_3d_info[str(bbox_3d_label)] = bbox_3d

        for box_tight, box_loose in zip(bbox_tight, selected_bbox_loose):
            # return if the object is not a character, or not recorded in our character dict
            if self._frame_id < self.frame_delay:
                return
            semantic_label = data[bbox_2d_tight_annotator]["info"]["idToLabels"][box_tight["semanticId"]]["class"]
            # check whether the label belong to certain character in the scene
            if not self.character_data_collector.is_character(semantic_label):
                continue
            if semantic_label not in label_to_bbox_3d_info.keys():
                continue

            label = []
            # occlusionRate = box_tight["occlusionRatio"]
            tight_box = [box_tight["x_max"], box_tight["x_min"], box_tight["y_max"], box_tight["y_min"]]
            loose_box = [box_loose["x_max"], box_loose["x_min"], box_loose["y_max"], box_loose["y_min"]]
            viewport_box = [rp_width, 0, rp_height, 0]
            valid_character, true_body_box = self.character_data_collector.valid_character(
                semantic_label, tight_box, loose_box, viewport_box, view_proj_mat
            )
            if not valid_character:
                continue

            character_position, character_rotation = self.character_data_collector.get_character_transform(
                semantic_label
            )

            scale = Utils.get_scale(label_to_bbox_3d_info[semantic_label])
            box_3d_info = {
                "scale": scale,
                "rot_deg_z": Utils.convert_to_angle(character_rotation),
                "translate": [character_position[0], character_position[1], character_position[2]],
            }
            # calculate character's bounding box info
            bbox_3d_data = Utils.generate_bounding_box_information(
                camera_view, camera_projection, box_3d_info, rp_width, rp_height
            )

            if self.format == "KITTI":
                ## if kitts format output
                label.append(semantic_label)  # semantic
                label.append(f"{0.00:.2f}")  # truncated (not supported)
                label.append(f"{0.00:.2f}")  # truncation (not supported)
                label.append(f"{0.00:.2f}")  # alpha (not supported)
                label.extend(true_body_box)
                label.extend([bbox_3d_data["scale"][0], bbox_3d_data["scale"][1], bbox_3d_data["scale"][2]])
                label.extend([bbox_3d_data["location"][0], bbox_3d_data["location"][1], bbox_3d_data["location"][2]])
                label.extend(
                    [
                        bbox_3d_data["rotate_in_degree"][0],
                        bbox_3d_data["rotate_in_degree"][1],
                        bbox_3d_data["rotate_in_degree"][2],
                    ]
                )
                label.append(f"{0.00:.2f}")  ## score (not supported)
                label_set.append(label)

        if self.format == "KITTI":
            kitti_filepath = os.path.join(sub_dir, "object_detection", f"{self._write_id}.txt")
            buf = io.StringIO()
            writer = csv.writer(buf, delimiter=" ")
            writer.writerows(label_set)
            self.backend.write_blob(kitti_filepath, bytes(buf.getvalue(), "utf-8"))

    def _procure_labels_from_json(self, json_path):
        with open(json_path, "r") as f:
            labels_dict = json.load(f)
        return labels_dict

    def write_camera_info(self, data, sub_dir, render_product_annotator, camera_params_annotator):
        rp_width = data[render_product_annotator]["resolution"][0]
        rp_height = data[render_product_annotator]["resolution"][1]
        camera_info = data[camera_params_annotator]
        camera_view = np.reshape(camera_info["cameraViewTransform"], (4, 4))
        camera_intrinsics = Utils.get_camera_info(
            rp_width, rp_height, camera_info["cameraFocalLength"], camera_info["cameraAperture"][0]
        )

        label_set = []
        split_list = sub_dir.split("_")
        if split_list[-1] == "Camera":
            camera_id = "_"
        else:
            camera_id = split_list[-1]
        if camera_id in self.lidar_to_world_transform:
            lidar_to_camera_transform = np.reshape(self.lidar_to_world_transform[camera_id], (4, 4)) @ camera_view
            label = ["lidar_to_camera_transform:"]
            label.extend(lidar_to_camera_transform.flatten().tolist())
            label_set.append(label)
        label = ["camera_projection"]
        label.extend(np.reshape(camera_info["cameraProjection"], (4, 4)).flatten().tolist())
        label_set.append(label)
        label_set.append(["rp_height:", rp_height])
        label_set.append(["rp_width:", rp_width])

        intrinsic = [
            "fx_fy_cx_cy",
            camera_intrinsics["fx"],
            camera_intrinsics["fy"],
            camera_intrinsics["cx"],
            camera_intrinsics["cy"],
        ]
        label_set.append(intrinsic)
        calibration_filepath = os.path.join(sub_dir, "calibration", f"{self._write_id}.txt")
        buf = io.StringIO()
        writer = csv.writer(buf, delimiter=" ")
        writer.writerows(label_set)
        self.backend.write_blob(calibration_filepath, bytes(buf.getvalue(), "utf-8"))

    def write_buffered_lidar(self, data, sub_dir: str, annotator: str):
        # Store lidar transform based on lidar_id
        split_list = sub_dir.split("_")
        if split_list[-1] == "Lidar":
            lidar_id = "_"
        else:
            lidar_id = split_list[-1]
        self.lidar_to_world_transform[lidar_id] = data[annotator]["info"]["transform"]

        point_cloud = data[annotator]["data"]
        pc_and_intensity_matrix = np.append(point_cloud, data[annotator]["intensity"].reshape(-1, 1), axis=1)
        buf = io.BytesIO()
        np.save(buf, pc_and_intensity_matrix)
        self._backend.write_blob(f"{sub_dir}/{self._write_id}.npy", buf.getvalue())

    def write(self, data):

        # Skip starting frames and wait for timeline to start.
        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing() or self.skip_frames > 0:
            self.skip_frames -= 1
            return

        render_products = [k for k in data.keys() if k.startswith("rp_")]

        # check whether we need to update character pos
        if self.write_bbox:
            # update character's position and rotation information, store those information in to buffer
            self.character_data_collector.update_character_pos()

        if self._frame_id % self.writer_interval != 0:
            self._frame_id += 1
            return

        if len(render_products) == 1:
            if (
                data[render_products[0]]["camera"].startswith("/World/Lidars")
                or "LIDAR" in data[render_products[0]]["camera"]
            ):
                cam_path = data[render_products[0]]["camera"]
                sub_dir = str(cam_path).replace("/", "_")
                if sub_dir[0] == "_":
                    sub_dir = sub_dir[1:]
                if self.write_lidar:
                    self.write_buffered_lidar(data, sub_dir, "RtxSensorCpuIsaacCreateRTXLidarScanBufferLocal")
            else:
                cam_path = data[render_products[0]]["camera"]
                sub_dir = str(cam_path).replace("/", "_")
                if sub_dir[0] == "_":
                    sub_dir = sub_dir[1:]
                if self.write_rgb:
                    self._write_rgb(data, sub_dir, "rgb")
                if self.write_bbox:
                    self._write_object_detection(
                        data,
                        sub_dir,
                        render_products[0],
                        "bounding_box_2d_tight_fast",
                        "bounding_box_2d_loose_fast",
                        "bounding_box_3d_fast",
                        "camera_params",
                    )
                    self.write_camera_info(data, sub_dir, render_products[0], "camera_params")
        else:
            for render_product in render_products:
                if (
                    data[render_product]["camera"].startswith("/World/Lidars")
                    or "LIDAR" in data[render_product]["camera"]
                ):
                    cam_path = data[render_product]["camera"]
                    sub_dir = str(cam_path).replace("/", "_")
                    if sub_dir[0] == "_":
                        sub_dir = sub_dir[1:]
                    if self.write_lidar:
                        self.write_buffered_lidar(
                            data,
                            sub_dir,
                            "RtxSensorCpuIsaacCreateRTXLidarScanBufferLocal-{}".format(
                                render_product.split("_", 1)[-1]
                            ),
                        )
                else:
                    render_product_name = render_product[3:]
                    cam_path = data[render_product]["camera"]
                    sub_dir = str(cam_path).replace("/", "_")
                    if sub_dir[0] == "_":
                        sub_dir = sub_dir[1:]
                    if self.write_rgb:
                        self._write_rgb(data, sub_dir, f"rgb-{render_product_name}")
                    if self.write_bbox:
                        self._write_object_detection(
                            data,
                            sub_dir,
                            render_product,
                            f"bounding_box_2d_tight_fast-{render_product_name}",
                            f"bounding_box_2d_loose_fast-{render_product_name}",
                            f"bounding_box_3d_fast-{render_product_name}",
                            f"camera_params-{render_product_name}",
                        )
                        self.write_camera_info(data, sub_dir, render_product, f"camera_params-{render_product_name}")

        self._write_id += 1
        self._frame_id += 1


annotator_name = "RtxSensorCpu" + "IsaacCreateRTXLidarScanBufferLocal"
if annotator_name not in AnnotatorRegistry.get_registered_annotators():
    AnnotatorRegistry.register_annotator_from_node(
        name=annotator_name,
        input_rendervars=["RtxSensorCpuPtr"],
        node_type_id="omni.isaac.sensor.IsaacCreateRTXLidarScanBuffer",
        output_data_type=np.float32,
        output_channels=3,
        init_params={"transformPoints": False},
    )
WriterRegistry.register(LidarFusionWriter)
