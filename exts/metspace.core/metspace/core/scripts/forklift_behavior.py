# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import asyncio

import carb
import omni
import omni.anim.navigation.core as nav
from omni.anim.people import PeopleSettings
from omni.anim.people.scripts.global_agent_manager import GlobalAgentManager
from omni.isaac.core import World
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.isaac.wheeled_robots.controllers.wheel_base_pose_controller import WheelBasePoseController
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.kit.scripting import BehaviorScript
from pxr import Gf

from .robot_navigation_manager import RobotNavigationManager
from .commands.robot_goto import *
from .commands.robot_idle import *
from .commands.robot_liftdown import *
from .commands.robot_liftup import *
from .commands.robot_track import *
from .track_manager import TrackManager

"""
    Control the Isaac Nova _robot Robot
    Given a command file, the robot will execute the commands sequentially
"""


class ForkliftBehavior(BehaviorScript):
    def on_init(self):
        # Loading Isaac.Core.World, which is created and managed by the Simulation Manager
        if World.instance() is None:
            ForkliftBehavior.set_robot_world()
        self.world = World.instance()  # Path: prim path, Name: the name to be used in the command file

        self._robot_path = str(self.prim_path)
        self._robot_name = self._robot_path.split("/")[-1]
        self._action_name = self._robot_name + "_actions"
        self.commands = []

        # Read settings from anim.people
        self.setting = carb.settings.get_settings()
        self.command_path = self.setting.get(PeopleSettings.ROBOT_COMMAND_FILE_PATH)
        self.navmeshEnabled = self.setting.get(PeopleSettings.NAVMESH_ENABLED)
        self.avoidanceOn = self.setting.get(PeopleSettings.DYNAMIC_AVOIDANCE_ENABLED)
        self.navigation_manager = None
        self.current_command = None

        self._wheeled_robot = None
        self._controller = None

        self.wheel_joints = ["back_wheel_drive", "back_wheel_swivel"]
        self.wheel_radius = 0.03
        self.wheel_base = 0.1125
        self._speed = 0.5

        self.paused = False  # Used to handle when the user pauses the play

    def set_up_robot(self, wheel_joints, wheel_radius, wheel_base):
        # Register the robot and its controller
        if self.world.scene.get_object(self._robot_path) is None:
            self._wheeled_robot = WheeledRobot(
                prim_path=self._robot_path, name=self._robot_path, wheel_dof_names=wheel_joints, create_robot=False
            )
            self.world.scene.add(self._wheeled_robot)
        else:
            self._wheeled_robot = self.world.scene.get_object(self._robot_path)
        self._controller = WheelBasePoseController(
            name="cool_controller_" + self._robot_name,
            open_loop_wheel_controller=DifferentialController(
                name="refine_control_" + self._robot_name, wheel_radius=wheel_radius, wheel_base=wheel_base
            ),
            is_holonomic=False,
        )

    # Clean the navigation manager, robot registry, and physics callback when removing the robot from the scene
    def on_destroy(self):
        if self.navigation_manager is not None:
            self.navigation_manager.destroy()
            self.navigation_manager = None

        if self.global_agent_manager is not None:
            self.global_agent_manager.destroy()
            self.global_agent_manager = None

        if self.world.scene.get_object(self._robot_path) is not None:
            self.world.scene.remove_object(self._robot_path, True)

        if self.world.physics_callback_exists(self._action_name):
            self.world.remove_physics_callback(self._action_name)

    # Must clean the robot physics callback and current command when stop playing
    def on_stop(self):
        if self.navigation_manager is not None:
            self.navigation_manager = None

        if self.global_agent_manager is not None:
            self.global_agent_manager.destroy()
            self.global_agent_manager = None

        self.current_command = None
        if self.world.physics_callback_exists(self._action_name):
            self.world.remove_physics_callback(self._action_name)

    def on_pause(self):
        self.paused = True

    def on_play(self):
        # If paused rather than stopped, no need to reset the robot
        if self.paused:
            self.paused = False

        self.set_up_robot(self.wheel_joints, self.wheel_radius, self.wheel_base)
        self.setup_for_play()

        # Add physics callback and initialize the robot, must be called in async and everytime before it plays
        async def init_robot():
            self.world.add_physics_callback(self._action_name, callback_fn=self.send_robot_actions)
            self._wheeled_robot.initialize()
            return

        asyncio.ensure_future(init_robot())

        # Equivalent to on_update(), step_size is fixed and is determined by world: physics_dt

    def send_robot_actions(self, step_size):
        carb.log_info("This is current agent manager " + str(self.global_agent_manager))
        if self.navigation_manager is None:
            err_msg = self.name + "has no navigation manager. No action will be executed."
            carb.log_error(err_msg)
            return
        
        # if TrackManager.get_instance()._is_tracking:
        #     TrackManager.get_instance().inject_tracking_command(self._robot_path, self._robot_name, step_size)

        if self.commands:
            self.execute_command(self.commands, step_size)
        # Publish the position of the robot after it's been updated
        if self.avoidanceOn:
            # Set the radius of the robot to be avoided by the characters
            self.navigation_manager.publish_robot_position(step_size, 1)

    # Commands and Navigation settings may be changed every time before playing, hence need to be set before play
    def setup_for_play(self):
        self.command_path = self.setting.get(PeopleSettings.ROBOT_COMMAND_FILE_PATH)
        self.commands = self.get_simulation_commands()
        self.navmeshEnabled = self.setting.get(PeopleSettings.NAVMESH_ENABLED)
        self.avoidanceOn = self.setting.get(PeopleSettings.DYNAMIC_AVOIDANCE_ENABLED)
        self.navigation_manager = RobotNavigationManager(
            self._robot_path, self._wheeled_robot, self.navmeshEnabled, self.avoidanceOn
        )
        self.global_agent_manager = GlobalAgentManager.get_instance()
        self.global_agent_manager.add_agent(str(self._robot_path), self)
        return

    def execute_command(self, commands, delta_time):
        """
        Executes commands in commands list in sequence. Removes a command once completed.

        :param list[list] commands: list of commands.
        :param float delta_time: time elapsed since last execution.
        """
        while not self.current_command:
            if not commands:
                return
            else:
                next_cmd = self.get_command(commands[0])

                if next_cmd:
                    self.current_command = self.get_command(commands[0])
                # skip invalid command
                else:
                    carb.log_error(f"{commands[0]} is not a valid command")
                    commands.pop(0)
        try:
            if self.current_command.execute(delta_time):
                commands.pop(0)
                self.current_command = None
        except:
            carb.log_error(
                "{}: invalid command. Abort this execution.".format(
                    self.get_origin_command_string(self.current_command.command)
                )
            )
            self.current_command.exit_command()
            commands.pop(0)
            self.current_command = None

    """
    The following three methods are related to command loading
    Recommended Action: make a base class for all agents and put those functions in the base class
    """

    def read_commands_from_file(self):
        """
        Reads commands from file pointed by self.command_path. Creates a Queue using queue manager if a queue is specified.
        :return: List of commands.
        :rtype: python list
        """
        result, version, context = omni.client.read_file(self.command_path)
        if result != omni.client.Result.OK:
            return []

        cmd_lines = memoryview(context).tobytes().decode("utf-8").splitlines()
        return cmd_lines

    def get_command(self, command):
        """
        Returns an instance of a command object based on the command.

        :param list[str] command: list of strings describing the command.
        :return: instance of a command object.
        :rtype: python object
        """
        # carb.log_error("get_command() not implemented for " + self._robot_path)
        if command[0] == "Track":
            return Robot_Track(self._wheeled_robot, self._controller, command, self.navigation_manager, self._speed)

    def convert_str_to_command(self, cmd_line):
        if not cmd_line:
            return None
        words = cmd_line.strip().split(" ")
        if words[0] == self._robot_name:
            command = []
            command = [str(word) for word in words[1:] if word != ""]
            return command
        if words[0][0] == "#":
            return None

    def get_simulation_commands(self):
        cmd_lines = []
        # Get commands from cmd_file
        cmd_lines.extend(self.read_commands_from_file())
        commands = []
        for cmd_line in cmd_lines:
            command = self.convert_str_to_command(cmd_line)
            if command is not None:
                commands.append(command)
        return commands

    # Add a physics ground if there isn't one yet
    def set_robot_world():
        world = World.instance()

        async def _init_sim_context():
            if World.instance() is None:
                carb.log_error("Attempt to initialize the simulation context before World is created")
            else:
                world = World.instance()
                await world.initialize_simulation_context_async()
            return

        if world is None:
            world = World()
            asyncio.ensure_future(_init_sim_context())

        stage = omni.usd.get_context().get_stage()
        ground_prim = stage.GetPrimAtPath("/World/defaultGroundPlane")
        # Create the physics ground plane if it hasn't been created
        if not ground_prim.IsValid() or not ground_prim.IsActive():
            world.scene.add_default_ground_plane()
            inav = nav.acquire_interface()
            origin_point = carb.Float3(0, 0, 0)
            inav.closest_navmesh_point(target=(0, 0, 0), point=origin_point)
            ground_plane_pos = Gf.Vec3d(origin_point[0], origin_point[1], origin_point[2])
            ground_prim = stage.GetPrimAtPath("/World/defaultGroundPlane")
            ground_prim.GetAttribute("xformOp:translate").Set(ground_plane_pos)
            ground_prim.GetAttribute("visibility").Set("invisible")
        # Needs to wait for one frame to make sure that it added the world and simulation context
        # await omni.kit.app.get_app().next_update_async()

    # inject commands to robot's command list
    def inject_command(self, command_list):
        cmd_array = []
        for command_line in command_list:
            listed_cmd = self.convert_str_to_command(command_line)
            if listed_cmd is not None:
                cmd_array.append(listed_cmd)
        if self.commands and cmd_array:
            self.commands[1:1] = cmd_array
        else:
            self.commands[0:0] = cmd_array

    def get_agent_name(self):
        return self._robot_name

    # force the character to end current command
    def end_current_command(self):
        # if current command is not None, force the character to quit.
        if self.current_command is not None:
            self.current_command.force_quit_command()
