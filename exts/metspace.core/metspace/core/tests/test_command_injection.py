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
from internal.unit_test.simulation import create_example_sim_manager, wait_for_simulation_set_up_done
from internal.unit_test.stage import TestStage
from omni.anim.people.scripts.global_agent_manager import GlobalAgentManager
from omni.kit.scripting.scripts.script_manager import ScriptManager
from metspace.core.file_util import TextFileUtil
from metspace.core.randomization.randomizer_util import RandomizerUtil
from metspace.core.settings import AssetPaths, PrimPaths
from metspace.core.simulation import SimulationManager
from metspace.core.stage_util import CameraUtil, CharacterUtil, LidarCamUtil, RobotUtil
from pxr import AnimGraphSchema, Usd, UsdGeom

CHARACTERS_PARENT_PRIM_PATH = "/exts/metspace/characters_parent_prim_path"
ROBOTS_PARENT_PRIM_PATH = "/exts/metspace/robots_parent_prim_path"


class TestCommandInjection(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        pass

    async def tearDown(self):
        pass

    # ======= Test Command Injection========

    async def test_command_injection(self):
        """
        Test if the injected command is indeed injected into the character's behavior script
        """
        # On an empty scene
        async with TestStage():
            # Load 5 default characters from default characters folder into default scene
            sim = create_example_sim_manager()
            sim.yaml_data["character"]["num"] = 2
            sim.yaml_data["robot"]["nova_carter_num"] = 2
            # Wait for simulation set up done
            sim.set_up_simulation_from_config_file()
            await wait_for_simulation_set_up_done(sim)

            # Go through each character
            skelroot_list = CharacterUtil.get_characters_in_stage()
            self.assertTrue(len(skelroot_list) == 2)
            # Python scripts
            character_script_path = AssetPaths.behavior_script_path()
            # Animation graph
            default_biped = CharacterUtil.get_default_biped_character()
            anim_graph_path = CharacterUtil.get_anim_graph_from_character(default_biped).GetPrimPath()

            # Check if the behavior scripts are attached and if they are using the right anim graph
            for skelroot in skelroot_list:
                attr = skelroot.GetAttribute("omni:scripting:scripts").Get()
                self.assertTrue(attr[0].path == character_script_path)
                anim_graph_ref = AnimGraphSchema.AnimationGraphAPI(skelroot).GetAnimationGraphRel()
                self.assertTrue(anim_graph_ref.GetTargets()[0] == anim_graph_path)

            # Go throgh each robot
            robot_list = RobotUtil.get_robots_in_stage()
            self.assertTrue(len(robot_list) == 2)
            # Python scripts
            robot_script_path = AssetPaths.robot_behavior_script_path().replace("robot", "nova_carter")

            for robot in robot_list:
                attr = robot.GetAttribute("omni:scripting:scripts").Get()
                self.assertTrue(attr[0].path == robot_script_path)

            # Set up the agents
            script_manager = ScriptManager.get_instance()
            for scripts in script_manager._prim_to_scripts.values():
                for _, inst in scripts.items():
                    if inst:
                        # setup robots
                        if hasattr(inst, "setup_for_play"):
                            inst.setup_for_play()
                        # setup characters
                        else:
                            inst.init_character()

            # Insert commands to the agents
            global_agent_manager = GlobalAgentManager.get_instance()
            global_agent_manager.inject_command_for_all_agents(
                ["Nova_Carter Idle 5", "Nova_Carter GoTo 0 0 0", "Character Idle 5", "Character GoTo 0 0 0 0"], True
            )

            for scripts in script_manager._prim_to_scripts.values():
                for _, inst in scripts.items():
                    if inst:
                        self.assertTrue(inst.commands[0][0] == "Idle")
                        self.assertTrue(inst.commands[1][0] == "GoTo")
