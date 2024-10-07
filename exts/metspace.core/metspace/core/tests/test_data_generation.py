__copyright__ = "Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved."
__license__ = """
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

import sys
import unittest
from pathlib import Path

import carb.settings
import numpy as np
import omni.kit
import omni.kit.test
import omni.replicator.core as rep
import omni.usd
from internal.unit_test import *
from omni.isaac.core.utils.prims import get_all_matching_child_prims
from metspace.core.data_generation import DataGeneration
from metspace.core.stage_util import CameraUtil, LidarCamUtil
from omni.replicator.core import utils
from omni.replicator.core.scripts.create import render_product
from pxr import Gf, Sdf, Semantics, UsdGeom, UsdShade

from ..settings import PrimPaths

CHARACTER_ASSETS_PATH = "/exts/metspace/asset_settings/character_assets_path"
BEHAVIOR_SCRIPT_PATH = "/exts/metspace/behavior_script_settings/behavior_script_path"
CHARACTERS_PARENT_PRIM_PATH = "/exts/metspace/characters_parent_prim_path"
CAMERAS_PARENT_PRIM_PATH = "/exts/metspace/cameras_parent_prim_path"
LIDAR_CAMERAS_PARENT_PRIM_PATH = "/exts/metspace/lidar_cameras_parent_prim_path"


class TestDataGeneration(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        async with TestStage():
            rep.set_global_seed(1)

    async def tearDown(self):
        carb.settings.get_settings().set("/omni/replicator/RTSubframes", 1)

    async def test_get_camera_list(self):
        async with TestStage():
            data_generation = DataGeneration()
            stage = omni.usd.get_context().get_stage()
            for i in range(5):
                CameraUtil.spawn_camera(spawn_location=carb.Float3(0, 0, 0))

            cam_list = data_generation._get_camera_list(2, 3)
            cam_list_paths = [str(cam.GetPath()) for cam in cam_list]

            cam_list_1 = data_generation._get_camera_list(5, 0)
            cam_list_paths_1 = [str(cam.GetPath()) for cam in cam_list_1]

            # invalid start index will produce empty list
            cam_list_2 = data_generation._get_camera_list(6, 10)
            cam_list_paths_2 = [str(cam.GetPath()) for cam in cam_list_2]

            self.assertListEqual(cam_list_paths, ["/World/Cameras/Camera_03", "/World/Cameras/Camera_04"])
            self.assertListEqual(
                cam_list_paths_1,
                [
                    "/World/Cameras/Camera",
                    "/World/Cameras/Camera_01",
                    "/World/Cameras/Camera_02",
                    "/World/Cameras/Camera_03",
                    "/World/Cameras/Camera_04",
                ],
            )
            self.assertListEqual(cam_list_paths_2, [])

    async def test_get_lidar_list(self):
        async with TestStage():
            data_generation = DataGeneration()

            for i in range(5):
                CameraUtil.spawn_camera(spawn_location=carb.Float3(0, 0, 0))
                LidarCamUtil.spawn_lidar_camera(spawn_location=carb.Float3(0, 0, 0))

            lidar_list = data_generation._get_lidar_list(2, 3)
            lidar_list_paths = [str(lidar.GetPath()) for lidar in lidar_list]

            lidar_list_1 = data_generation._get_lidar_list(5, 0)
            lidar_list_paths_1 = [str(lidar.GetPath()) for lidar in lidar_list_1]

            # invalid start index will produce empty list
            lidar_list_2 = data_generation._get_lidar_list(6, 10)
            lidar_list_paths_2 = [str(lidar.GetPath()) for lidar in lidar_list_2]

            self.assertListEqual(lidar_list_paths, ["/World/Lidars/Lidar_03", "/World/Lidars/Lidar_04"])
            self.assertListEqual(
                lidar_list_paths_1,
                [
                    "/World/Lidars/Lidar",
                    "/World/Lidars/Lidar_01",
                    "/World/Lidars/Lidar_02",
                    "/World/Lidars/Lidar_03",
                    "/World/Lidars/Lidar_04",
                ],
            )
            self.assertListEqual(lidar_list_paths_2, [])
