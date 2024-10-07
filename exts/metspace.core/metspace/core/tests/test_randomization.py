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

import carb
import carb.settings
import numpy as np
import omni.anim.navigation.core as nav
import omni.kit
import omni.kit.test
import omni.replicator.core as rep
import omni.usd
from internal.unit_test.stage import TestStage
from metspace.core.randomization.character_randomizer import CharacterRandomizer
from metspace.core.randomization.randomizer import Randomizer
from metspace.core.randomization.randomizer_util import RandomizerUtil
from metspace.core.settings import AssetPaths
from metspace.core.simulation import SimulationManager
from metspace.core.stage_util import CharacterUtil, StageUtil
from omni.replicator.core import utils
from omni.replicator.core.scripts.create import render_product
from pxr import Sdf, Semantics, UsdGeom, UsdShade

# This determines number of agent
TEST_ITERATION = 50


class TestRandomization(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        async with TestStage():
            rep.set_global_seed(1)

    async def tearDown(self):
        pass

    async def test_randomization(self):
        self.randomizer = CharacterRandomizer(0)
        try:
            async with TestStage(AssetPaths.default_scene_path()):
                # Tests can only start after the navmesh is loaded
                def nav_mesh_ready_callback(event):
                    if event.type == nav.EVENT_TYPE_NAVMESH_READY:
                        self._nav_mesh_event_handle = None
                        _agent_pos_list = self._test_spawn()
                        self._test_command(_agent_pos_list)

                _nav = nav.nav.acquire_interface()
                _nav.start_navmesh_baking()
                self._nav_mesh_event_handle = _nav.get_navmesh_event_stream().create_subscription_to_pop(
                    nav_mesh_ready_callback
                )
        except Exception as e:
            # Release event handle and do not procceed loading assets
            carb.log_error("Load scene ({0}) fails. No assets will be loaded.".format())
            self._load_stage_handle = None

    def _test_spawn(self):
        pos_list = [self.randomizer.get_random_position(i) for i in range(TEST_ITERATION)]

        # Test if the distance of any two agents are always greater than 1
        # Test if seed overflow is handled
        for seed in range(1 + sys.maxsize, TEST_ITERATION):
            randomizer = Randomizer(seed)
            pos_list_test = [randomizer.get_random_position(i) for i in range(TEST_ITERATION)]
            for i in range(len(pos_list_test)):
                for j in range(i + 1, len(pos_list_test)):
                    self.assertEqual(RandomizerUtil.dist3(pos_list_test[i], pos_list_test[j]) > 1, True)

            # Test if a different seed results in a different result
            list_different = False
            for i in range(len(pos_list)):
                if not RandomizerUtil.equal3(pos_list[i], pos_list_test[i]):
                    list_different = True
            self.assertEqual(list_different, True)

            # Test if randomizers with the same seed produce the same result
            randomizer.reset()
            randomizer.update_seed(0)
            pos_list_test = [randomizer.get_random_position(i) for i in range(TEST_ITERATION)]
            list_different = False
            for i in range(len(pos_list)):
                if not RandomizerUtil.equal3(pos_list[i], pos_list_test[i]):
                    list_different = True
            self.assertEqual(list_different, False)

            # Test if two seperate random position calls produce the same end result as one
            pos_list_test = [randomizer.get_random_position(i) for i in range(TEST_ITERATION // 2)] + [
                randomizer.get_random_position(i) for i in range(TEST_ITERATION // 2, TEST_ITERATION)
            ]
            list_different = False
            for i in range(len(pos_list)):
                if not RandomizerUtil.equal3(pos_list[i], pos_list_test[i]):
                    list_different = True
            self.assertEqual(list_different, False)
        return pos_list

    def _test_command(self, pos_list):
        agent_list = {}
        command_duration = 600
        for idx in range(TEST_ITERATION):
            agent_list["a" + str(idx)] = pos_list[idx]
        commands = self.randomizer.generate_commands(0, command_duration, agent_list)

        agent_idx = 0
        goto_count = 0
        idle_count = 0
        lookaround_count = 0

        # Seperate the commands for each agent
        agent_command_list = {}
        for command in commands:
            command = command.split()
            agent_name = command[0]
            if agent_name not in agent_command_list:
                agent_command_list[agent_name] = []
            agent_command_list[agent_name].append(command[1:])

        # Test each agent has enough commands to last for the given duration
        for agent in agent_command_list.keys():
            commands = agent_command_list[agent]
            duration = 0
            for command in commands:
                # Get the count for each command for testing the random distribution
                if command[0] == "GoTo":
                    goto_count += 1
                elif command[0] == "Idle":
                    idle_count += 1
                elif command[0] == "LookAround":
                    lookaround_count += 1

                # Get the command duration
                if command[0] != "GoTo":
                    duration += float(command[1])
                else:
                    duration += (
                        RandomizerUtil.dist3(
                            pos_list[agent_idx], carb.Float3(float(command[1]), float(command[2]), float(command[3]))
                        )
                        / 1.1
                    )
                    pos_list[agent_idx] = [float(command[1]), float(command[2]), float(command[3])]

            self.assertEqual(duration > command_duration, True)
            agent_idx += 1

        # Test whether the commands are generated according to the Markov Chain
        total_commands = goto_count + lookaround_count + idle_count
        # Test GoTo convergence ~ 0.3946
        self.assertEqual(abs(goto_count / total_commands - 0.3946) < 0.01, True)
        # Test Idle convergence ~ 0.2915
        self.assertEqual(abs(idle_count / total_commands - 0.2915) < 0.01, True)
        # Test LookAround convergence ~ 0.3139
        self.assertEqual(abs(lookaround_count / total_commands - 0.3139) < 0.01, True)
