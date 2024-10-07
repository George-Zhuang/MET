__copyright__ = "Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved."
__license__ = """
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

import csv
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
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry
from omni.replicator.core.scripts.writers_default.tools import *
from omni.syntheticdata.scripts.SyntheticData import SyntheticData
from pxr import Semantics, Usd, UsdSkel
from scipy.spatial.transform import Rotation as R
from omni.isaac.core.utils.rotations import quat_to_euler_angles

from .character_data_collector import CharacterDataCollector
from .utils import Objectron_Utils, Utils
from metspace.core.scripts.track_manager import TrackManager
from omni.isaac.core.utils import prims
from omni.isaac.core import World


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


class METWriter(Writer):
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
        renderproduct_idxs: List[tuple] = None,
        mapping_path: str = None,
        semantic_filter_predicate: str = None,
        valid_unoccluded_threshold: float = 0.6,
        s3: bool = False,
        frame_num: int = 50,
        rgb: bool = True,
        traj: bool = True,
        bbox: bool = True,
        semantic_segmentation=True,
        distance_to_camera=True,
    ):
        """Create a KITTI Writer

        Args:
            output_dir: Output directory to which KITTI annotations will be saved.
            semantic_types: List of semantic types to consider. If ``None``, only consider semantic types ``"class"``.
            mapping_path: json path to the label to color mapping for KITTI

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
        self._sqlite_lock = threading.Lock()
        self._render_product_idxs = renderproduct_idxs
        self.frame_delay = 0
        self.valid_unoccluded_threshold = valid_unoccluded_threshold

        self.segmentation_seed = 12345
        # container used to store skeleton information
        self.character_data_collector = CharacterDataCollector(
            self.valid_unoccluded_threshold, self.segmentation_seed, self.frame_delay
        )
        # initialize the skeleton dict
        self.character_data_collector._create_skeleton_dicts()
        self.camera_frame_dict = {}
        self.output_dir = output_dir
        self.format = "COCO"

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

        self.annotators = []

        self.write_rgb = rgb
        self.write_bbox = bbox
        self.write_position_and_yaw = traj
        self.write_semantic_segmentation = semantic_segmentation
        self.write_distance_to_camera = distance_to_camera

        if self.write_bbox:
            self.annotators.append("bounding_box_3d_fast")
            self.annotators.append("bounding_box_2d_tight_fast")
            self.annotators.append("bounding_box_2d_loose_fast")
            self.annotators.append("camera_params")

        if self.write_semantic_segmentation:
            self.annotators.append("semantic_segmentation")

        if self.write_rgb:
            self.annotators.append("rgb")

        if self.write_position_and_yaw:
            self.annotators.append("camera_params")

        if self.write_distance_to_camera:
            self.annotators.append("distance_to_camera")

        self.skip_frames = carb.settings.get_settings().get(
            "/persistent/exts/metspace/skip_starting_frames"
        )
        self.writer_interval = carb.settings.get_settings().get(
            "/persistent/exts/metspace/frame_write_interval"
        )

        # Track numbering of the next frame to be written.
        self._write_id = 0

    @staticmethod
    def params_values():
        # Params to be recoginized in config file format and their default values
        # Refer to the initialize function
        return {
            "output_dir": os.path.abspath(
                carb.settings.get_settings().get("/exts/default_replicator_output_path")
            ),
            "rgb": True,
            "bbox": True,
            "semantic_segmentation": True,
            "distance_to_camera": True,
        }

    @staticmethod
    def params_labels():
        # Params to be display in the UI and their display labels
        return {
            "output_dir": "output_dir",
            "rgb": "rgb",
            "bbox": "bbox",
            "semantic_segmentation": "semantic_segmentation",
            "distance_to_camera": "distance_to_camera",
        }

    def _write_rgb(self, data, sub_dir: str, annotator: str):
        # Save the rgb data under the correct path
        rgb_dir_name = "rgb"
        rgb_file_path = os.path.join(sub_dir, rgb_dir_name, f"{self._write_id}.jpeg")
        self._backend.write_image(rgb_file_path, data[annotator])

        # create random color for instance segmentation

    def _write_position_yaw_distance(self, data, sub_dir: str, annotator: str):
        # get file path
        traj_dir_name = "traj"
        traj_file_path = os.path.join(self.output_dir, sub_dir, traj_dir_name, f"{self._write_id}.json")
        os.makedirs(os.path.dirname(traj_file_path), exist_ok=True)
        # get camera transform
        camera_prim_path = str(data[annotator]["camera"])
        # camera_transform = np.reshape(Objectron_Utils.get_camera_transform(camera_prim_path), (4, 4))
        # # get camera intrisic data
        # camera_translate = [camera_transform[3][0], camera_transform[3][1], camera_transform[3][2]]
        # camera_rotation = camera_transform[:3, :3]
        # compute distance of the camera to the target
        distance = np.array([0, 0])
        agent_pos = np.array([0, 0])
        agent_yaw = 0
        all_target_positions = TrackManager.get_instance().get_target_position()
        for agent_prim_path in all_target_positions.keys():
            if agent_prim_path in camera_prim_path:
                agent = World.instance().scene.get_object(agent_prim_path)
                position, orientation = agent.get_world_pose()
                agent_pos = position[:2]
                agent_yaw = quat_to_euler_angles(orientation)[-1]
                target_pos = all_target_positions[agent_prim_path][:2]
                distance = np.array(target_pos) - np.array(agent_pos)
                break
        # # compute yaw
        # eular_angles = matrix_to_euler_angles(camera_rotation)
        # yaw = eular_angles[-1]
        traj = {
            'position': agent_pos.tolist(), 
            'yaw': agent_yaw,
            'distance': distance.tolist()}
        # carb.log_warn(f"traj: {traj}")
        with open(traj_file_path, 'w') as f:
            json.dump(traj, f)

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
        objs = []
        camera_prim_path = str(data[render_product_annotator]["camera"])
        rp_width = data[render_product_annotator]["resolution"][0]
        rp_height = data[render_product_annotator]["resolution"][1]
        camera_info = data[camera_params_annotator]

        # get camera transform
        camera_transform = np.reshape(Objectron_Utils.get_camera_transform(camera_prim_path), (4, 4))
        camera_view = np.linalg.inv(camera_transform)

        # get camera intrisic data
        camera_translate = [camera_transform[3][0], camera_transform[3][1], camera_transform[3][2]]
        camera_rotation = camera_transform[:3, :3]

        camera_intrinsics = Objectron_Utils.get_camera_info(
            camera_translate,
            camera_rotation,
            rp_width,
            rp_height,
            camera_info["cameraFocalLength"],
            camera_info["cameraAperture"][0],
            camera_info["cameraNearFar"][0],
            camera_info["cameraNearFar"][1],
        )

        # get camera projection matrix
        camera_projection = Objectron_Utils.get_projection_matrix(camera_intrinsics)
        rotation_quatd = list(R.from_matrix(np.transpose(camera_transform[:3, :3])).as_quat())

        view_proj_mat = np.dot(camera_view, camera_projection)
        bbox_tight = data[bbox_2d_tight_annotator]["data"]
        bbox_loose = data[bbox_2d_loose_annotator]["data"]
        bbox_tight_bbox_ids = data[bbox_2d_tight_annotator]["info"]["bboxIds"]
        bbox_loose_bbox_ids = data[bbox_2d_loose_annotator]["info"]["bboxIds"]

        # For box in tight, find the corresponding index of box in loose
        bbox_loose_indices = np.where(np.isin(bbox_loose_bbox_ids, bbox_tight_bbox_ids))[0]
        selected_bbox_loose = bbox_loose[bbox_loose_indices]

        # get 3d bounding box annotator data
        box_3d_annotator = data[bbox_3d_annotator]

        label_to_bbox_3d_info = {}
        bbox_id_to_label = box_3d_annotator["info"]["idToLabels"]
        bbox_3d_data = box_3d_annotator["data"]

        # record current 3d bbox info to a dictionary
        for bbox_3d in bbox_3d_data:
            semantic_id = bbox_3d["semanticId"]
            bbox_3d_label = bbox_id_to_label[semantic_id]["class"]
            if self.character_data_collector.is_character(str(bbox_3d_label)):
                label_to_bbox_3d_info[str(bbox_3d_label)] = bbox_3d

        for box_tight, box_loose in zip(bbox_tight, selected_bbox_loose):

            if self._frame_id < self.frame_delay:
                return

            semantic_label = data[bbox_2d_tight_annotator]["info"]["idToLabels"][box_tight["semanticId"]]["class"]

            if not self.character_data_collector.is_character(str(semantic_label)):
                continue

            if semantic_label not in label_to_bbox_3d_info.keys():
                continue

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
            # calculate character's scale from bounding box information
            scale = Utils.get_scale(label_to_bbox_3d_info[semantic_label])

            # collect bounding box 's scale, rotation and translation
            box_3d_info = {
                "scale": scale,
                "rot_deg_z": Utils.convert_to_angle(character_rotation),
                "translate": [character_position[0], character_position[1], character_position[2]],
            }

            # calcualte the bounding box information follow the objectron format
            full_meta_data = Objectron_Utils.generate_bounding_box_information(
                camera_intrinsics, camera_projection, box_3d_info, rp_width, rp_height
            )
            if self.format == "COCO":
                full_meta_data["label"] = semantic_label
                objs.append(full_meta_data)

        if self.format == "COCO":
            # compose data to objectron json format
            proj_mat_output = Objectron_Utils.projection_matrix_format(camera_projection)
            json_dict = {
                "camera_data": {
                    "camera_projection_matrix": proj_mat_output,
                    "camera_view": Objectron_Utils.to_array(camera_view),
                    "intrinsics": {
                        "cx": float(rp_width / 2),
                        "cy": float(rp_height / 2),
                        "fx": float(rp_width * camera_info["cameraFocalLength"] / camera_info["cameraAperture"][0]),
                        "fy": float(
                            rp_height
                            * camera_info["cameraFocalLength"]
                            / (rp_height / rp_width * camera_info["cameraAperture"][0])
                        ),
                    },
                    "width": int(rp_width),
                    "height": int(rp_height),
                    "quaternion_world_xyzw": [float(i) for i in rotation_quatd],
                    "location_world": [float(i) for i in camera_translate],
                },
                "objects": objs,
            }

            buf = io.BytesIO()
            buf.write(json.dumps(json_dict, indent=4).encode())
            coco_filepath = os.path.join(sub_dir, "coco_annotator", f"{self._write_id}.json")
            self._backend.write_blob(coco_filepath, buf.getvalue())

    def _procure_labels_from_json(self, json_path):
        with open(json_path, "r") as f:
            labels_dict = json.load(f)
        return labels_dict

    def write(self, data):

        timeline = omni.timeline.get_timeline_interface()
        # if timeline is not playing return
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
            cam_path = data[render_products[0]]["camera"]
            sub_dir = str(cam_path).replace("/", "_")
            if sub_dir[0] == "_":
                sub_dir = sub_dir[1:]
            if self.write_rgb:
                self._write_rgb(data, sub_dir, "rgb")
            if self.write_position_and_yaw:
                self._write_position_yaw_distance(data, sub_dir, render_products[0])
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
            if self.write_semantic_segmentation:
                self._write_segmentation(data, sub_dir, "semantic_segmentation")
            if self.write_distance_to_camera:
                self._write_distance_to_camera(data, sub_dir, "distance_to_camera")
        else:

            for render_product in render_products:
                render_product_name = render_product[3:]
                ## current camera path
                cam_path = data[render_product]["camera"]
                sub_dir = str(cam_path).replace("/", "_")
                if sub_dir[0] == "_":
                    sub_dir = sub_dir[1:]
                if self.write_rgb:
                    self._write_rgb(data, sub_dir, f"rgb-{render_product_name}")
                if self.write_position_and_yaw:
                    self._write_position_yaw_distance(data, sub_dir, render_product)
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
                if self.write_semantic_segmentation:
                    self._write_segmentation(data, sub_dir, f"semantic_segmentation-{render_product_name}")
                if self.write_distance_to_camera:
                    self._write_distance_to_camera(data, sub_dir, f"distance_to_camera-{render_product_name}")

        self._frame_id += 1
        self._write_id += 1

    # output objects' distance to camera to a rgb image
    def _write_distance_to_camera(self, data, sub_dir: str, annotator: str):
        distance_to_camera_metres = data[annotator]
        distance_to_camera_metres = np.nan_to_num(distance_to_camera_metres, posinf=0.0)
        distance_to_camera_uint16 = (distance_to_camera_metres * 256).astype(np.uint16)
        file_path = os.path.join(sub_dir, "depth", f"{self._write_id}.png")
        self._backend.write_image(file_path, distance_to_camera_uint16)

    # output segmentation data according to the current color mapping dict
    def _write_segmentation(self, data, sub_dir: str, sem_annotator: str):
        """
        Instance segmentation follows the format specified by here: https://www.vision.rwth-aachen.de/page/mots
        """
        sem_rgb_dir_name = "semantic_segmentation"
        seg_col_filepath = os.path.join(sub_dir, sem_rgb_dir_name, f"{self._write_id}.png")
        semantic_seg_data_colorized = colorize_segmentation(
            data[sem_annotator]["data"], data[sem_annotator]["info"]["idToLabels"], mapping=self.mapping_dict
        )
        self._backend.write_image(seg_col_filepath, semantic_seg_data_colorized)


WriterRegistry.register(METWriter)
