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
from subprocess import PIPE, STDOUT, Popen
from typing import List

import carb
import numpy as np
import omni.anim.graph.core as ag
import omni.usd
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry
from omni.replicator.core.scripts.writers_default.tools import *
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
    "character": (0, 0, 100, 255),
}


__version__ = "0.0.2"


class TaoWriter(Writer):
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
        bbox: bool = True,
        semantic_segmentation=False,
        video=False,
    ):
        """Create a KITTI Writer

        Args:
            output_dir: Output directory to which KITTI annotations will be saved.
            semantic_types: List of semantic types to consider. If ``None``, only consider semantic types ``"class"``.
            partly_occluded_threshold: Minimum occlusion factor for bounding boxes to be considered partly occluded.
            mapping_path: json path to the label to color mapping for KITTI
            colorize_instance_segmentation: bool value representing whether image is to be colorized or not.

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
        ##container used to store skeleton information
        self.segmentation_seed = 12345
        self.character_data_collector = CharacterDataCollector(
            self.valid_unoccluded_threshold, self.segmentation_seed, self.frame_delay
        )
        self.character_data_collector._create_skeleton_dicts()
        self.camera_frame_dict = {}
        self.output_dir = output_dir
        self.format = "KITTI"

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
        self.write_semantic_segmentation = semantic_segmentation
        self.video = video

        if self.write_bbox:
            self.annotators.append("bounding_box_3d_fast")
            self.annotators.append("bounding_box_2d_tight_fast")
            self.annotators.append("bounding_box_2d_loose_fast")
            self.annotators.append("camera_params")

        if self.write_semantic_segmentation:
            if mapping_path:
                self.mapping_dict = self._procure_labels_from_json(mapping_path)
            else:
                color_dict = {}
                color_dict = self.character_data_collector.create_random_color_dict(KITTI_LABELS)
                color_dict.update(KITTI_LABELS)
                self.mapping_dict = color_dict
            self.annotators.append("semantic_segmentation")

        if self.write_rgb:
            self.annotators.append("rgb")

        self.rgb_dir_list = {}
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
                carb.settings.get_settings().get("/exts/metspace/default_replicator_output_path")
            ),
            "rgb": True,
            "bbox": True,
            "semantic_segmentation": False,
            "video": False,
        }

    @staticmethod
    def params_labels():
        # Params to be display in the UI and their display labels
        return {
            "output_dir": "output_dir",
            "rgb": "rgb",
            "bbox": "bbox",
            "semantic_segmentation": "semantic_segmentation",
            "video": "video",
        }

    @staticmethod
    def tooltip():
        return f"""
            TaoWriter
            - Generates 3d bbox, 2d bbox, sematic segmentation and rgb from cameras.
            - Follows TAO labeeling standards when generating 2d and 3d bboxes.
        """

    def _write_rgb(self, data, sub_dir: str, annotator: str):
        # Save the rgb data under the correct path

        if not sub_dir in self.rgb_dir_list:
            dir_path = os.path.join(self.output_dir, sub_dir, "rgb")
            self.rgb_dir_list[sub_dir] = dir_path
        rgb_dir_name = "rgb"
        rgb_file_path = os.path.join(sub_dir, rgb_dir_name, f"{self._write_id}.jpeg")
        self._backend.write_image(rgb_file_path, data[annotator])

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

        # the wdith and height of the image
        rp_width = data[render_product_annotator]["resolution"][0]
        rp_height = data[render_product_annotator]["resolution"][1]

        bbox_tight = data[bbox_2d_tight_annotator]["data"]
        bbox_loose = data[bbox_2d_loose_annotator]["data"]

        bbox_tight_bbox_ids = data[bbox_2d_tight_annotator]["info"]["bboxIds"]
        bbox_loose_bbox_ids = data[bbox_2d_loose_annotator]["info"]["bboxIds"]

        bbox_loose_indices = np.where(np.isin(bbox_loose_bbox_ids, bbox_tight_bbox_ids))[0]
        selected_bbox_loose = bbox_loose[bbox_loose_indices]

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

            # get current prim's semantic label
            semantic_label = data[bbox_2d_tight_annotator]["info"]["idToLabels"][box_tight["semanticId"]]["class"]

            # if check whether the semantic label belong to one of characters in the scene
            if not self.character_data_collector.is_character(str(semantic_label)):
                continue

            if semantic_label not in label_to_bbox_3d_info.keys():
                continue

            label = []
            # tight and loose bbox information of the character
            tight_box = [box_tight["x_max"], box_tight["x_min"], box_tight["y_max"], box_tight["y_min"]]
            loose_box = [box_loose["x_max"], box_loose["x_min"], box_loose["y_max"], box_loose["y_min"]]
            viewport_box = [rp_width, 0, rp_height, 0]

            # check whether character is trancated, out of image, or within image
            valid_character, true_body_box = self.character_data_collector.valid_character(
                semantic_label, tight_box, loose_box, viewport_box, view_proj_mat
            )
            if not valid_character:
                continue

            # get character's position and rotation
            character_position, character_rotation = self.character_data_collector.get_character_transform(
                semantic_label
            )

            # calculate character's scale
            scale = Utils.get_scale(label_to_bbox_3d_info[semantic_label])

            # calculate character's 3d bounding box data
            box_3d_info = {
                "scale": scale,
                "rot_deg_z": Utils.convert_to_angle(character_rotation),
                "translate": [character_position[0], character_position[1], character_position[2]],
            }
            bbox_3d_data = Utils.generate_bounding_box_information(
                camera_view, camera_projection, box_3d_info, rp_width, rp_height
            )
            bbox_2d_info = bbox_3d_data["projected_cuboid"]

            # get joint position data from the anim.graph
            joint_3d, joint_2d = self.character_data_collector.get_joint_information(
                semantic_label, view_proj_mat, rp_width, rp_height
            )
            converted_joint_3d = [[joint_pos[0], joint_pos[1], joint_pos[2]] for joint_pos in joint_3d]

            if self.format == "KITTI":
                ## if kitts format output
                label.append(semantic_label)
                label.append(f"{0.00:.2f}")
                label.append(f"{0.00:.2f}")
                label.append("2d bounding box")
                label.append("tight bounding box")
                label.append(true_body_box)
                label.append("loose bounding box")
                label.append(loose_box)
                label.append("character position")
                label.append(character_position)
                label.append("3d_joint_position")
                label.append(converted_joint_3d)
                label.append("2d_joint_position")
                label.append(joint_2d)
                label.append("3d_bounding_box")
                label.append("scale")
                label.append([bbox_3d_data["scale"][0], bbox_3d_data["scale"][1], bbox_3d_data["scale"][2]])
                label.append("location")
                label.append([bbox_3d_data["location"][0], bbox_3d_data["location"][1], bbox_3d_data["location"][2]])
                label.append("rotate_in_degree")
                label.append(
                    [
                        bbox_3d_data["rotate_in_degree"][0],
                        bbox_3d_data["rotate_in_degree"][1],
                        bbox_3d_data["rotate_in_degree"][2],
                    ]
                )
                label.append("2d_vertexs")
                label.append(bbox_2d_info)
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

    # convert generated image to video
    def on_final_frame(self):
        if self.video:
            rgb_folders = self.rgb_dir_list.values()
            try:
                for folder in rgb_folders:
                    # Get parent folder name
                    camer_folder_name = os.path.basename(os.path.dirname(folder))

                    Popen(
                        "ffmpeg -r 30 -f image2 -s 1920x1080 -start_number 0 -y -i {}/%d.jpeg -vcodec libx264 -crf 23 -pix_fmt yuv420p {}/{}.mp4".format(
                            folder, folder, camer_folder_name
                        ),
                        shell=True,
                    )
            except Exception as e:
                carb.log_error(
                    "Could not run ffmpeg command due to error - {}. \n Note, this operation requires ffmpeg to be installed".format(
                        e
                    )
                )
                return False

    def write(self, data):

        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing() or self.skip_frames > 0:
            self.skip_frames -= 1
            return

        render_products = [k for k in data.keys() if k.startswith("rp_")]
        # check whether we need to update character pos
        if self.write_bbox:
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

        else:
            for render_product in render_products:
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

                if self.write_semantic_segmentation:
                    self._write_segmentation(data, sub_dir, f"semantic_segmentation-{render_product_name}")

        self._write_id += 1
        self._frame_id += 1

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


WriterRegistry.register(TaoWriter)
