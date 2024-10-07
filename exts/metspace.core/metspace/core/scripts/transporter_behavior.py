# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from omni.anim.people.scripts.commands.robot_goto import *
from omni.anim.people.scripts.commands.robot_idle import *
from omni.anim.people.scripts.commands.robot_liftdown import *
from omni.anim.people.scripts.commands.robot_liftup import *
# from omni.anim.people.scripts.robot_behavior import RobotBehavior

from .robot_behavior import RobotBehavior
from .commands.robot_track import *

"""
    Control the Isaac Transporter Robot
    Given a command file, the robot will execute the commands sequentially
"""


class TransporterBehavior(RobotBehavior):
    def on_init(self):
        super().on_init()
        self.wheel_joints = ["left_wheel_joint", "right_wheel_joint"]
        self.wheel_radius = 0.08
        self.wheel_base = 0.58
        self._speed = 0.5
        self.set_up_robot(self.wheel_joints, self.wheel_radius, self.wheel_base)

    def get_command(self, command):
        """
        Returns an instance of a command object based on the command.

        :param list[str] command: list of strings describing the command.
        :return: instance of a command object.
        :rtype: python object
        """
        if command[0] == "GoTo":
            return Robot_GoTo(self._wheeled_robot, self._controller, command, self.navigation_manager)
        elif command[0] == "Idle":
            return Robot_Idle(self._wheeled_robot, self._controller, command, self.navigation_manager)
        elif command[0] == "LiftUp":
            return Robot_LiftUp(self._wheeled_robot, self._controller, command, self.navigation_manager)
        elif command[0] == "LiftDown":
            return Robot_LiftDown(self._wheeled_robot, self._controller, command, self.navigation_manager)
        elif command[0] == "Track":
            return Robot_Track(self._wheeled_robot, self._controller, command, self.navigation_manager, self._speed)
        else:
            return None
