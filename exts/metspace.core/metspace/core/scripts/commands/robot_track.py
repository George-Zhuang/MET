# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import carb
import numpy as np

from .robot_command import Robot_Command
from metspace.core.scripts.track_manager import TrackManager
from metspace.core.stage_util import CharacterUtil

class Robot_Track(Robot_Command):
    """
    Command class to go to a location/locations.
    """

    def __init__(self, robot, controller, command, navigation_manager, speed=None, min_dist=0.5):
        super().__init__(robot, controller, command, navigation_manager, speed)
        # update params
        self.update_target_position_frequency = 5
        self.time_elapsed_update = 0
        self.min_dist = min_dist
        # get tracking target
        assert len(command) > 1, "Track command must have a target"
        self.target = command[1]
        self.target_prim = None

        target_list = TrackManager.get_instance()._merged_object_character_prim_list
        carb.log_warn(f'target_list: {target_list}')
        for target in target_list:
            if CharacterUtil.get_character_name(target) == self.target:
                self.target_prim = target
                break
        if self.target_prim is None:
            carb.log_error(f"Target {self.target} not found")
        carb.log_warn(f'robot: {self.robot}')
        carb.log_warn(f"target_prim: {self.target_prim}")
    def setup(self):
        super().setup()
        target_pos = CharacterUtil.get_character_current_pos(self.target_prim)
        self.navigation_manager.generate_goto_path(target_pos)

    def update(self, dt):
        self.time_elapsed += dt
        self.time_elapsed_update += dt
        # update target position
        if self.time_elapsed_update > 1 / self.update_target_position_frequency:
            target_pos = CharacterUtil.get_character_current_pos(self.target_prim)
            self.navigation_manager.generate_goto_path(target_pos)
            self.time_elapsed_update = 0
        # agent_pos = self.robot.get_world_pose()[0]
        # distance = np.linalg.norm(agent_pos[:2] - target_pos[:2])
        # if distance < self.min_dist:
        #     return self.exit_command()
        # if self.move(dt):
        #     return self.exit_command()
        self.move(dt)
