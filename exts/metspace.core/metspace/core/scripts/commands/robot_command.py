# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import carb
import numpy as np
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import quat_to_euler_angles

class Robot_Command:
    """
    Base class for command, provides default implementation for setup, update, execute and exit_command.
    This follows the same structure as found in the character Command base class
    Also implements the move function which moves a robot to a location based on the target location set in NavigationManager.
    """

    def __init__(self, robot, controller, command, navigation_manager, speed=None, distance_threshold=1.0):
        self.robot = robot
        self.controller = controller
        self.command = command
        self.navigation_manager = navigation_manager
        self.time_elapsed = 0
        self.is_setup = False
        self.duration = 5
        self.finished = False

        self.speed = speed
        self.distance_threshold = distance_threshold

    def setup(self):
        self.time_elapsed = 0
        self.is_setup = True

    def exit_command(self):
        self.is_setup = False
        return True

    def update(self, dt):
        self.time_elapsed += dt
        if self.time_elapsed > self.duration:
            return self.exit_command()

    def execute(self, dt):
        if self.finished:
            return True

        if not self.is_setup:
            self.setup()
        return self.update(dt)

    def force_quit_command(self):
        position, orientation = self.robot.get_world_pose()
        self.robot.apply_wheel_actions(
            self.controller.forward(start_position=position, start_orientation=orientation, goal_position=position)
        )
        self.is_setup = False
        self.finished = True
        return

    def move(self, dt):
        self.navigation_manager.update_current_path_point()

        path_points = self.navigation_manager.get_path_points()
        position, orientation = self.robot.get_world_pose()
        # yaw = quat_to_euler_angles(orientation)[-1]
        # carb.log_warn(f'position: {position}, yaw: {yaw}')

        if len(path_points) and np.linalg.norm(path_points[-1][:2] - position[:2]) > self.distance_threshold:

            target_position = np.array([path_points[0][0], path_points[0][1]])

            self.robot.apply_wheel_actions(
                self.controller.forward(
                    start_position=position,
                    start_orientation=orientation,
                    goal_position=target_position,
                    # lateral_velocity=0.5,
                    lateral_velocity=self.speed,
                )
            )
            # After reaching the target position feed the robot with its current position to make it stop
            return False
        else:
            self.robot.apply_wheel_actions(
                self.controller.forward(start_position=position, start_orientation=orientation, goal_position=position)
            )
            return True

    def lift_up(self, dt):
        current_lift_pos = self.robot.get_joint_positions(4)[0]
        if current_lift_pos >= 0.039:
            return True
        else:
            lift_pos = 0.04 * dt + current_lift_pos
            self.robot.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=[None, None, None, None, lift_pos, None, None])
            )

    def lift_down(self, dt):
        current_lift_pos = self.robot.get_joint_positions(4)[0]
        if current_lift_pos <= 0.001:
            return True
        else:
            lift_pos = current_lift_pos - 0.04 * dt
            self.robot.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=[None, None, None, None, lift_pos, None, None])
            )
