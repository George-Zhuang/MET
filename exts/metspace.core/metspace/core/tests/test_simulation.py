__copyright__ = "Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved."
__license__ = """
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

import asyncio
import os
import sys
import unittest
from pathlib import Path

import carb.settings
import numpy as np
import omni.kit
import omni.kit.test
import omni.usd
from internal.unit_test import *
from metspace.core.file_util import TextFileUtil
from metspace.core.randomization.randomizer_util import RandomizerUtil
from metspace.core.settings import AssetPaths, PrimPaths
from metspace.core.simulation import SimulationManager
from metspace.core.stage_util import CameraUtil, CharacterUtil, LidarCamUtil
from pxr import AnimGraphSchema, Usd, UsdGeom

CHARACTERS_PARENT_PRIM_PATH = "/exts/metspace/characters_parent_prim_path"
CAMERAS_PARENT_PRIM_PATH = "/exts/metspace/cameras_parent_prim_path"
LIDAR_CAMERAS_PARENT_PRIM_PATH = "/exts/metspace/lidar_cameras_parent_prim_path"


class TestSimulation(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        pass

    async def tearDown(self):
        pass

    async def test_load_save_command_file(self):
        """
        Test character and robot command files can be loaded and saved through SimulationManager
        """
        sim = create_example_sim_manager()
        # Change contentes in command files
        test_commands_list = ["Character Idle 5"]
        test_robot_commands_list = ["Carter Idle 5"]
        sim.save_commands(test_commands_list)
        sim.save_robot_commands(test_robot_commands_list)
        # Check if the changes are saved
        commands_list = sim.load_commands()
        robot_commands_list = sim.load_robot_commands()
        self.assertTrue(check_list_equal(test_commands_list, commands_list))
        self.assertTrue(check_list_equal(test_robot_commands_list, robot_commands_list))

    async def test_command_file_relative_path(self):
        """
        Test if config file can handle relative character and robot command file paths
        """
        sim = create_example_sim_manager()
        # Create new command files in config file location
        location = Path(sim.config_file_path).parent.resolve()
        command_relative_path = "test_command.txt"
        robot_command_relative_path = "test_robot_command.txt"
        commands_list = ["Character Idle 5"]
        robot_commands_list = ["Carter Idle 5"]
        self.assertTrue(TextFileUtil.create_text_file(str(location / command_relative_path), commands_list[0]))
        self.assertTrue(
            TextFileUtil.create_text_file(str(location / robot_command_relative_path), robot_commands_list[0])
        )
        # Point SimulationManager to relative paths and check if that can be loaded
        sim.yaml_data["character"]["command_file"] = command_relative_path
        sim.yaml_data["robot"]["command_file"] = robot_command_relative_path
        loaded_commands_list = sim.load_commands()
        loaded_robot_commands_list = sim.load_robot_commands()
        self.assertTrue(check_list_equal(loaded_commands_list, commands_list))
        self.assertTrue(check_list_equal(loaded_robot_commands_list, robot_commands_list))

    # ======= Test Scene ========

    async def test_load_scene(self):
        """
        Test if scenes can be loaded correctly through SimulationManager
        """
        # On an empty scene
        async with TestStage():
            # Simulation Manager loads the default scene
            sim = create_example_sim_manager()
            sim.yaml_data["scene"]["asset_path"] = AssetPaths.default_scene_path()
            sim.yaml_data["character"]["num"] = 0
            sim.yaml_data["global"]["camera_num"] = 0
            # Wait for simulation set up done
            sim.set_up_simulation_from_config_file()
            await wait_for_simulation_set_up_done(sim)
            # Check if the default scene is loaded
            current_url = omni.usd.get_context().get_stage_url()
            self.assertEqual(current_url, sim.yaml_data["scene"]["asset_path"])

    # ======= Test Characters ========

    async def test_load_default_skeleton_and_animations(self):
        """
        Test if default skeletons and animations can be loaded through SimulationManager
        """
        async with TestStage():
            sim = SimulationManager()
            sim.load_default_skeleton_and_animations()
            stage = omni.usd.get_context().get_stage()
            setting_dict = carb.settings.get_settings()
            self.characters_parent_prim_path = setting_dict.get(CHARACTERS_PARENT_PRIM_PATH)
            self.assertTrue(stage.GetPrimAtPath("{}".format(self.characters_parent_prim_path)).IsValid())
            self.assertTrue(stage.GetPrimAtPath("{}/Biped_Setup".format(self.characters_parent_prim_path)).IsValid())
            self.assertTrue(
                stage.GetPrimAtPath(
                    "{}/Biped_Setup/CharacterAnimation".format(self.characters_parent_prim_path)
                ).IsValid()
            )
            self.assertTrue(
                stage.GetPrimAtPath(
                    "{}/Biped_Setup/biped_demo_meters".format(self.characters_parent_prim_path)
                ).IsValid()
            )

    async def test_load_characters(self):
        """
        Test if correct amounts of characters can be loaded
        """
        # On an empty scene
        async with TestStage():
            # Load 5 default characters from default characters folder into default scene
            sim = create_example_sim_manager()
            sim.yaml_data["character"]["num"] = 5
            # Wait for simulation set up done
            sim.set_up_simulation_from_config_file()
            await wait_for_simulation_set_up_done(sim)
            # Check if 5 characters are loaded
            stage = omni.usd.get_context().get_stage()
            char_list = CharacterUtil.get_characters_root_in_stage()
            self.assertTrue(len(char_list) == 5)
            # Reduce number to 3, check if the last 2 are not deleted
            sim.yaml_data["character"]["num"] = 3
            sim.load_setup_characters_from_config_file()
            char_list = CharacterUtil.get_characters_root_in_stage()
            self.assertTrue(len(char_list) == 5)
            # Increase character numbers to 7, check if the additional 2 are spawned
            sim.yaml_data["character"]["num"] = 7
            sim.load_setup_characters_from_config_file()
            char_list = CharacterUtil.get_characters_root_in_stage()
            self.assertTrue(len(char_list) == 7)

    async def test_setup_characters(self):
        """
        Test if behavior scripts and animation graphs have been setup to each character
        """
        # On an empty scene
        async with TestStage():
            # Load 5 characters to default scene
            sim = create_example_sim_manager()
            sim.yaml_data["character"]["num"] = 5
            # Wait for simulation set up done
            sim.set_up_simulation_from_config_file()
            await wait_for_simulation_set_up_done(sim)
            # Go through each character
            skelroot_list = CharacterUtil.get_characters_in_stage()
            self.assertTrue(len(skelroot_list) == 5)
            # Python scripts
            script_path = AssetPaths.behavior_script_path()
            # Animation graph
            default_biped = CharacterUtil.get_default_biped_character()
            anim_graph_path = CharacterUtil.get_anim_graph_from_character(default_biped).GetPrimPath()
            for skelroot in skelroot_list:
                attr = skelroot.GetAttribute("omni:scripting:scripts").Get()
                self.assertTrue(attr[0].path == script_path)
                anim_graph_ref = AnimGraphSchema.AnimationGraphAPI(skelroot).GetAnimationGraphRel()
                self.assertTrue(anim_graph_ref.GetTargets()[0] == anim_graph_path)

    # ======= Test Camera Randomization ========
    # The following code aiming at testing whether Camera Randomization Works
    async def test_random_camera_height(self):
        """
        Test if all the camera's height are in range defined by user
        """
        # get the original camera_randomization setting:
        # (we need to reset this value at the end of test)
        original_aim_character_setting = RandomizerUtil.do_aim_camera_to_character()
        RandomizerUtil.set_aim_camera_to_character(True)
        max_camera_height = RandomizerUtil.get_max_camera_height()
        min_camera_height = RandomizerUtil.get_min_camera_height()
        xformoptype_setting_path = "/persistent/app/primCreation/DefaultXformOpType"
        original_xform_order_setting = carb.settings.get_settings().get(xformoptype_setting_path)
        carb.settings.get_settings().set(xformoptype_setting_path, "Scale, Orient, Translate")
        carb.settings.get_settings().set(RandomizerUtil.MAX_CAMERA_FOCALLENGTH, 23)
        carb.settings.get_settings().set(RandomizerUtil.MIN_CAMERA_FOCALLENGTH, 13)

        # On an empty scene
        async with TestStage():
            sim = create_example_sim_manager()
            sim.yaml_data["scene"]["asset_path"] = AssetPaths.default_scene_path()
            sim.yaml_data["character"]["num"] = 20
            sim.yaml_data["global"]["camera_num"] = 5
            sim.set_up_simulation_from_config_file()
            await wait_for_simulation_set_up_done(sim)
            stage = omni.usd.get_context().get_stage()
            camera_root_prim = stage.GetPrimAtPath(PrimPaths.cameras_parent_path())
            if camera_root_prim is None:
                carb.log_error("no valid camera in the scene")
                return False
            for prim in Usd.PrimRange(camera_root_prim):
                if prim.IsA(UsdGeom.Camera):
                    translate_value = prim.GetAttribute("xformOp:translate").Get()
                    self.assertTrue(min_camera_height <= translate_value[2] <= max_camera_height)

            RandomizerUtil.set_aim_camera_to_character(original_aim_character_setting)
            carb.settings.get_settings().set(xformoptype_setting_path, original_xform_order_setting)

    async def test_random_camera_focallength(self):
        """
        Test if all the camera's focal length are in range defined by user
        """
        # get the original camera_randomization setting:
        # (we need to reset this value at the end of test)

        original_aiming_setting = RandomizerUtil.do_aim_camera_to_character()
        original_info_setting = RandomizerUtil.do_randomize_camera_info()
        xformoptype_setting_path = "/persistent/app/primCreation/DefaultXformOpType"
        original_xform_order_setting = carb.settings.get_settings().get(xformoptype_setting_path)
        carb.settings.get_settings().set(xformoptype_setting_path, "Scale, Orient, Translate")

        RandomizerUtil.set_aim_camera_to_character(True)
        RandomizerUtil.set_randomize_camera_info(True)

        carb.settings.get_settings().set(RandomizerUtil.MAX_CAMERA_FOCALLENGTH, 23)
        carb.settings.get_settings().set(RandomizerUtil.MIN_CAMERA_FOCALLENGTH, 13)

        max_focal_length = RandomizerUtil.get_max_camera_focallength()
        min_focal_length = RandomizerUtil.get_min_camera_focallength()

        carb.log_warn("Test message: This is current max focallength " + str(max_focal_length))
        carb.log_warn("Test message: This is current min focallength " + str(min_focal_length))

        # On an empty scene
        async with TestStage():
            sim = create_example_sim_manager()
            sim.yaml_data["scene"]["asset_path"] = AssetPaths.default_scene_path()
            sim.yaml_data["character"]["num"] = 20
            sim.yaml_data["global"]["camera_num"] = 5
            sim.set_up_simulation_from_config_file()
            await wait_for_simulation_set_up_done(sim)
            stage = omni.usd.get_context().get_stage()
            camera_root_prim = stage.GetPrimAtPath(PrimPaths.cameras_parent_path())
            if camera_root_prim is None:
                carb.log_error("no valid camera in the scene")
                return False
            for prim in Usd.PrimRange(camera_root_prim):
                if prim.IsA(UsdGeom.Camera):
                    focal_length_value = prim.GetAttribute("focalLength").Get()
                    self.assertTrue(min_focal_length <= focal_length_value <= max_focal_length)

            RandomizerUtil.set_aim_camera_to_character(original_aiming_setting)
            RandomizerUtil.set_randomize_camera_info(original_info_setting)
            carb.settings.get_settings().set(xformoptype_setting_path, original_xform_order_setting)

    # do raycast from each character to camera
    # make sure that there are at least one character in the camera viewport
    def check_character_aiming(self, spawn_location):
        stage = omni.usd.get_context().get_stage()
        character_root_prim = stage.GetPrimAtPath(PrimPaths.characters_parent_path())
        for character_prim in character_root_prim.GetChildren():
            character_prim_path = character_prim.GetPath()
            in_camera, pos = RandomizerUtil.check_character_visible_in_pos(character_prim_path, spawn_location)
            if in_camera:
                return True
        return False
        # _ , center = RandomizerUtil.get_character_radius_and_center(character_path)

    async def test_random_camera_lidar_matching(self):
        """
        Test whether every lidar camera have same transform as the camera with the same index
        """
        # get the original camera_randomization setting:
        # (we need to reset this value at the end of test)
        original_aiming_setting = RandomizerUtil.do_aim_camera_to_character()
        original_info_setting = RandomizerUtil.do_randomize_camera_info()
        carb.settings.get_settings().set(RandomizerUtil.MAX_CAMERA_FOCALLENGTH, 23)
        carb.settings.get_settings().set(RandomizerUtil.MIN_CAMERA_FOCALLENGTH, 13)
        RandomizerUtil.set_aim_camera_to_character(True)
        xformoptype_setting_path = "/persistent/app/primCreation/DefaultXformOpType"
        original_xform_order_setting = carb.settings.get_settings().get(xformoptype_setting_path)
        carb.settings.get_settings().set(xformoptype_setting_path, "Scale, Orient, Translate")
        # On an empty scene
        async with TestStage():
            sim = create_example_sim_manager()
            sim.yaml_data["scene"]["asset_path"] = AssetPaths.default_scene_path()
            sim.yaml_data["character"]["num"] = 20
            sim.yaml_data["global"]["camera_num"] = 5
            sim.yaml_data["global"]["lidar_num"] = 5
            sim.set_up_simulation_from_config_file()
            await wait_for_simulation_set_up_done(sim)
            stage = omni.usd.get_context().get_stage()
            camera_parent_prim_path = PrimPaths.cameras_parent_path()
            lidar_parent_prim_path = PrimPaths.lidar_cameras_parent_path()

            for i in range(0, 5):
                lidar_name = LidarCamUtil.get_lidar_name_by_index(i)
                lidar_path = lidar_parent_prim_path + "/" + lidar_name
                lidar_prim = stage.GetPrimAtPath(lidar_path)
                # When prim is missing
                if not lidar_prim.IsValid():
                    if RandomizerUtil.do_aim_camera_to_character():
                        camera_name = CameraUtil.get_camera_name_by_index(i)
                        camera_path = camera_parent_prim_path + "/" + camera_name
                        camera_prim = stage.GetPrimAtPath(camera_path)
                        if camera_prim.IsValid():
                            camera_rot = camera_prim.GetAttribute("xformOp:orient").Get()
                            camera_pos = camera_prim.GetAttribute("xformOp:translate").Get()
                            camera_focallength = camera_prim.GetAttribute("focalLength").Get()
                            lidar_rot = lidar_prim.GetAttribute("xformOp:orient").Get()
                            lidar_pos = camera_prim.GetAttribute("xformOp:translate").Get()
                            lidar_focallength = lidar_prim.GetAttribute("focalLength").Get()
                            self.assertTrue(camera_rot, lidar_rot)
                            self.assertTrue(camera_pos, lidar_pos)
                            self.assertTrue(camera_focallength, lidar_focallength)

            RandomizerUtil.set_aim_camera_to_character(original_aiming_setting)
            RandomizerUtil.set_randomize_camera_info(original_info_setting)
            carb.settings.get_settings().set(xformoptype_setting_path, original_xform_order_setting)

        # do raycast from each camera to make sure that there are character in the camera viewport

    async def test_random_camera_aim(self):
        """
        Test if all the camera's aim at at least on character
        """
        # get the original camera_randomization setting:
        # (we need to reset this value at the end of test)
        original_aim_character_setting = RandomizerUtil.do_aim_camera_to_character()
        RandomizerUtil.set_aim_camera_to_character(True)
        carb.settings.get_settings().set(RandomizerUtil.MAX_CAMERA_FOCALLENGTH, 23)
        carb.settings.get_settings().set(RandomizerUtil.MIN_CAMERA_FOCALLENGTH, 13)

        xformoptype_setting_path = "/persistent/app/primCreation/DefaultXformOpType"
        original_xform_order_setting = carb.settings.get_settings().get(xformoptype_setting_path)
        carb.settings.get_settings().set(xformoptype_setting_path, "Scale, Orient, Translate")

        # On an empty scene
        async with TestStage():
            sim = create_example_sim_manager()
            sim.yaml_data["scene"]["asset_path"] = AssetPaths.default_scene_path()
            sim.yaml_data["character"]["num"] = 20
            sim.yaml_data["global"]["camera_num"] = 5
            sim.set_up_simulation_from_config_file()
            await wait_for_simulation_set_up_done(sim)
            stage = omni.usd.get_context().get_stage()
            camera_root_prim = stage.GetPrimAtPath(PrimPaths.cameras_parent_path())
            if camera_root_prim is None:
                carb.log_error("no valid camera in the scene")
                return False
            all_aiming_at_character = True
            for prim in Usd.PrimRange(camera_root_prim):
                if prim.IsA(UsdGeom.Camera):
                    translate_value = prim.GetAttribute("xformOp:translate").Get()
                    aiming_at_character = self.check_character_aiming(translate_value)
                    if not aiming_at_character:
                        carb.log_warn("Camera do not have character in scope: camera path" + str(prim.GetPath()))
                        all_aiming_at_character = False

            self.assertTrue(all_aiming_at_character)

            RandomizerUtil.set_aim_camera_to_character(original_aim_character_setting)
            carb.settings.get_settings().set(xformoptype_setting_path, original_xform_order_setting)
