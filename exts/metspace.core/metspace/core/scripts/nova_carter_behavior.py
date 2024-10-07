# ------------------------------------------------------------------------------
# Embodied Visual Tracking
#   1. Add random speed within an appropriate range
#   2. Max speed 3.3 from https://wiki.ros.org/Robots/NovaCarter
# ------------------------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# ------------------------------------------------------------------------------
import carb
import random

# from omni.anim.people.scripts.commands.robot_goto import *
# from omni.anim.people.scripts.commands.robot_idle import *

from .robot_behavior import RobotBehavior
from .commands.robot_goto import *
from .commands.robot_idle import *
from .commands.robot_track import *

"""
    Control the Isaac Nova Carter Robot
    Given a command file, the robot will execute the commands sequentially
"""


class TNovaCarterBehavior(RobotBehavior):
    def on_init(self):
        super().on_init()
        self._base_speed = 0.5
        self._min_speed = 1.2
        self._max_speed = 2.0 # 3.3
        # self._speed = random.uniform(self._min_speed, self._max_speed)
        self._speed = 0.5
        # self._speed = self._max_speed
        self.wheel_joints = ["joint_wheel_left", "joint_wheel_right"]
        self.wheel_radius = 0.152
        self.wheel_base = 0.413
        self.set_up_robot(self.wheel_joints, self.wheel_radius, self.wheel_base)

    def get_command(self, command):
        """
        Returns an instance of a command object based on the command.

        :param list[str] command: list of strings describing the command.
        :return: instance of a command object.
        :rtype: python object
        """
        if command[0] == "GoTo":
            return Robot_GoTo(self._wheeled_robot, self._controller, command, self.navigation_manager, self._speed)
        elif command[0] == "Idle":
            return Robot_Idle(self._wheeled_robot, self._controller, command, self.navigation_manager, self._speed)
        elif command[0] == "Track":
            return Robot_Track(self._wheeled_robot, self._controller, command, self.navigation_manager, self._speed)
        else:
            return None
