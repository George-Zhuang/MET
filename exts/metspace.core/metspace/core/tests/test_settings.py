__copyright__ = "Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved."
__license__ = """
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""


import unittest

import omni.kit
import omni.kit.test
import omni.usd
from metspace.core.settings import *

CHARACTER_ASSETS_PATH = "/exts/metspace/asset_settings/character_assets_path"
BEHAVIOR_SCRIPT_PATH = "/exts/metspace/behavior_script_settings/behavior_script_path"
CHARACTERS_PARENT_PRIM_PATH = "/exts/metspace/characters_parent_prim_path"
CAMERAS_PARENT_PRIM_PATH = "/exts/metspace/cameras_parent_prim_path"
LIDAR_CAMERAS_PARENT_PRIM_PATH = "/exts/metspace/lidar_cameras_parent_prim_path"


class TestSettings(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        pass

    async def tearDown(self):
        pass

    async def test_asset_paths_cache(self):
        ext_instance = metspace.core.get_instance()
        if not ext_instance.check_startup_async_done():

            def done_callback(context):
                assert AssetPaths.cache_biped_asset_path
                assert AssetPaths.cache_default_character_path
                assert AssetPaths.cache_scene_asset_path

            ext_instance.add_startup_async_done_callback(done_callback)
        else:
            assert AssetPaths.cache_biped_asset_path
            assert AssetPaths.cache_default_character_path
            assert AssetPaths.cache_scene_asset_path
