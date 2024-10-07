# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import importlib
import math

import carb
import omni.anim.graph.core as ag
import omni.usd
from omni.anim.people import PeopleSettings
from omni.anim.people.python_ext import get_instance
from omni.anim.people.scripts.global_agent_manager import GlobalAgentManager
from omni.anim.people.scripts.global_character_position_manager import GlobalCharacterPositionManager
from omni.anim.people.scripts.global_queue_manager import GlobalQueueManager
from omni.anim.people.scripts.navigation_manager import NavigationManager
from omni.anim.people.ui_components import CommandTextWidget
from omni.kit.scripting import BehaviorScript

from omni.anim.people.scripts.commands.dequeue import *
from omni.anim.people.scripts.commands.goto import *
from omni.anim.people.scripts.commands.idle import *
from omni.anim.people.scripts.commands.look_around import *
from omni.anim.people.scripts.commands.queue import *
from omni.anim.people.scripts.commands.sit import *
from omni.anim.people.scripts.custom_command.command_manager import *
from omni.anim.people.scripts.custom_command.command_templates import *
from omni.anim.people.scripts.utils import Utils


class CharacterBehavior(BehaviorScript):
    """
    Character controller class that reads commands from a command file and drives character actions.
    """

    def on_init(self):
        """
        Called when a script is attached to characters and when a stage is loaded. Uses renew_character_state() to initialize character state.
        """
        self.renew_character_state()

    def on_play(self):
        """
        Called when entering runtime (when clicking play button). Uses renew_character_state() to initialize character state.
        """
        self.renew_character_state()

    def on_stop(self):
        """
        Called when exiting runtime (when clicking stop button). Uses on_destroy() to clear state.
        """
        self.on_destroy()

    def on_destroy(self):
        """
        Clears character state by deleting global variable instances.
        """
        self.character_name = None
        if self.character_manager is not None:
            self.character_manager.destroy()
            self.character_manager = None

        if self.navigation_manager is not None:
            self.navigation_manager.destroy()
            self.navigation_manager = None

        if self.queue_manager is not None:
            self.queue_manager.destroy()
            self.queue_manager = None

        if self.global_character_manager is not None:
            self.global_character_manager.destroy()
            self.global_character_manager = None

    def renew_character_state(self):
        """
        Defines character variables and loads settings.
        """
        self.setting = carb.settings.get_settings()
        self.command_path = self.setting.get(PeopleSettings.COMMAND_FILE_PATH)
        self.number_of_loop = self.setting.get_as_string(PeopleSettings.NUMBER_OF_LOOP)
        if self.number_of_loop == "inf":
            self.number_of_loop = math.inf
        else:
            self.number_of_loop = int(self.number_of_loop)
        self.navmeshEnabled = self.setting.get(PeopleSettings.NAVMESH_ENABLED)
        self.avoidanceOn = self.setting.get(PeopleSettings.DYNAMIC_AVOIDANCE_ENABLED)
        self.character_name = self.get_agent_name()
        carb.log_info("Character name is {}".format(self.character_name))
        self.character = None
        self.current_command = None
        self.loop_commands = None
        self.loop_commands_count = 1
        self.navigation_manager = None
        self.character_manager = None
        self.queue_manager = None
        self.global_character_manager = None
        self.in_queue = False
        self.commands = []

    # force the character to end current command
    def end_current_command(self):

        # if current command is not None, force the character to quit.
        if self.current_command is not None:
            self.current_command.force_quit_command()

            # if character is conducting "Queue" command, remove next several commands related to this behavior.
            if self.current_command.get_command_name() == "QueueCmd" or self.in_queue:
                self.clean_unclosed_dequeue()

        # if character is currently inside the queue, remove character from the queue.
        if self.in_queue:
            # Free the queue spot occupied by this character.
            self.queue_manager.remove_character_from_queue(str(self.character_name))
            self.in_queue = None

    # Remove the following queue behavior and "Dequeue" command once the "Queue" command is interrupted by command injection
    def clean_unclosed_dequeue(self):
        if len(self.commands) > 1:
            list_length = len(self.commands)
            for i in range(1, list_length):
                # if we hit the "Queue" command
                if str(self.commands[i][0]) == "Queue":
                    break
                # if we hit uncompleted Dequeue command.
                if str(self.commands[i][0]) == "Dequeue":
                    # remove all the command between this command and the second command.
                    if i == 1:
                        self.commands.pop(i)
                    else:
                        self.commands[1:i] = []
                    break

    def get_agent_name(self):
        """
        For this character asset find its name used in the command file.
        """
        character_path = str(self.prim_path)
        split_path = character_path.split("/")
        prim_name = split_path[-1]
        root_path = self.setting.get(PeopleSettings.CHARACTER_PRIM_PATH)
        # If a character is loaded through the spawn command, the commands for the character can be given by using the encompassing parent name.
        if character_path.startswith(str(root_path)):
            parent_len = len(root_path.split("/"))
            parent_name = split_path[parent_len]
            return parent_name
        return prim_name

    def init_character(self):
        """
        Initializes global variables and fetches animation graph attached to the character. Called after entering runtime as ag.get_character() can only be used in runtime.
        """
        self.character = ag.get_character(str(self.prim_path))
        if self.character is None:
            return False

        self.custom_command_manager = get_instance().get_custom_command_manager()

        self.global_character_manager = GlobalAgentManager.get_instance()
        self.global_character_manager.add_agent(str(self.prim_path), self)
        self.navigation_manager = NavigationManager(str(self.prim_path), self.navmeshEnabled, self.avoidanceOn)
        self.character_manager = GlobalCharacterPositionManager.get_instance()
        self.queue_manager = GlobalQueueManager.get_instance()
        if not self.navigation_manager or not self.character_manager or not self.queue_manager:
            return False

        self.commands = self.get_simulation_commands()

        # Store all registered custom commands beforehand
        self.custom_command_names = self.custom_command_manager.get_all_custom_command_names()

        # Prepare loop command
        if self.number_of_loop > 0:
            # Character go to original spot to form the loop
            originPos, originRot = Utils.get_character_transform(self.character)
            originAngle = Utils.convert_to_angle(originRot)
            self.commands.append(["GoTo", str(originPos[0]), str(originPos[1]), str(originPos[2]), str(originAngle)])
            self.loop_commands = self.commands.copy()

        self.character.set_variable("Action", "None")
        carb.log_info("Initialize the character")
        return True

    def read_commands_from_file(self):
        """
        Reads commands from file pointed by self.command_path. Creates a Queue using queue manager if a queue is specified.
        :return: List of commands.
        :rtype: python list
        """
        if not self.command_path:
            carb.log_warn("Command file field is empty.")
            return []
        result, version, context = omni.client.read_file(self.command_path)
        if result != omni.client.Result.OK:
            carb.log_error("Unable to read command file at {}.".format(self.command_path))
            return []

        cmd_lines = memoryview(context).tobytes().decode("utf-8").splitlines()
        return cmd_lines

    def read_commands_from_UI(self):
        ui_commands = CommandTextWidget.textbox_commands
        if ui_commands:
            cmd_lines = ui_commands.splitlines()
            return cmd_lines
        return []

    def get_combined_user_commands(self):
        cmd_lines = []

        # Get commands from cmd_file
        cmd_lines.extend(self.read_commands_from_file())

        # Get commands from UI
        cmd_lines.extend(self.read_commands_from_UI())

        return cmd_lines

    # convert command string to command list. split character name, command name, and command parameters
    def convert_str_to_command(self, cmd_line):
        if not cmd_line:
            return None
        words = str(cmd_line).strip().split(" ")
        if words[0] == self.character_name:
            command = []
            command = [str(word) for word in words[1:] if word != ""]
            return command
        if words[0] == "Queue":
            self.queue_manager.create_queue(words[1])
            return None
        if words[0] == "Queue_Spot":
            queue = self.queue_manager.get_queue(words[1])
            queue.create_spot(
                int(words[2]),
                carb.Float3(float(words[3]), float(words[4]), float(words[5])),
                Utils.convert_angle_to_quatd(float(words[6])),
            )
            return None
        if words[0][0] == "#":
            return None

        return None

    # get simulation commands from both UI and command file
    def get_simulation_commands(self):
        cmd_lines = self.get_combined_user_commands()
        commands = []
        for cmd_line in cmd_lines:
            command = self.convert_str_to_command(cmd_line)
            if command is not None:
                commands.append(command)

        return commands

    # get character's position
    def get_current_position(self):
        return Utils.get_character_pos(self.character)

    # inject commands to character's command list
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

        # Debug: output character's current command list
        # carb.log_warn("{} current self.commands: {}".format(str(self.character_name), str(self.commands)))

    def get_command(self, command):
        """
        Returns an instance of a command object based on the command.

        :param list[str] command: list of strings describing the command.
        :return: instance of a command object.
        :rtype: python object
        """
        if command[0] == "GoTo":
            return GoTo(self.character, command, self.navigation_manager)
        elif command[0] == "Idle":
            return Idle(self.character, command, self.navigation_manager)
        elif command[0] == "Queue":
            return QueueCmd(self.character, command, self.navigation_manager, self.queue_manager, self.character_name)
        elif command[0] == "Dequeue":
            return Dequeue(self.character, command, self.navigation_manager, self.queue_manager, self.character_name)
        elif command[0] == "LookAround":
            return LookAround(self.character, command, self.navigation_manager)
        elif command[0] == "Sit":
            return Sit(self.character, command, self.navigation_manager)
        elif command[0] in self.custom_command_names:
            custom_command_item = self.custom_command_manager.get_custom_command_by_name(command[0])
            if custom_command_item.template == CustomCommandTemplate.TIMING:
                return TimingTemplate(self.character, command, self.navigation_manager, custom_command_item.name)
            elif custom_command_item.template == CustomCommandTemplate.TIMING_TO_OBJECT:
                return TimingToObjectTemplate(
                    self.character, command, self.navigation_manager, custom_command_item.name
                )
            elif custom_command_item.template == CustomCommandTemplate.GOTO_BLEND:
                return GoToBlendTemplate(self.character, command, self.navigation_manager, custom_command_item.name)
            return None
        else:
            module_str = ".commands.{}".format(command[0].lower(), package=None)
            try:
                custom_class = getattr(importlib.import_module(module_str, package=__package__), command[0])
            except (ImportError, AttributeError) as error:
                carb.log_error("Module or Class for the command do not exist. Check the command again.")
            return custom_class(self.character, command, self.navigation_manager)

    def get_origin_command_string(self, command):
        line = self.character_name
        for str in command:
            if str != self.character_name:
                line = line + " " + str
        return line

    def execute_command(self, commands, delta_time):
        """
        Executes commands in commands list in sequence. Removes a command once completed.

        :param list[list] commands: list of commands.
        :param float delta_time: time elapsed since last execution.
        """
        while not self.current_command:
            if not commands:
                return
            next_cmd = self.get_command(commands[0])
            if next_cmd:
                self.current_command = next_cmd
            else:
                commands.pop(0)  # Skip the command that cannot be executed
        try:
            if self.current_command.execute(delta_time):

                if self.current_command.get_command_name() == "QueueCmd":
                    # check whether character has occupied a spot in the queue
                    self.in_queue = self.current_command.current_spot is not None

                if self.current_command.get_command_name() == "Dequeue":
                    # set character's status to "not in queue"
                    self.in_queue = False

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

    def on_update(self, current_time: float, delta_time: float):
        """
        Called on every update. Initializes character at start, publishes character positions and executes character commands.
        :param float current_time: current time in seconds.
        :param float delta_time: time elapsed since last update.
        """
        if self.character is None:
            if not self.init_character():
                return

        if self.navigation_manager and self.avoidanceOn:
            self.navigation_manager.publish_character_positions(delta_time, 0.5)

        if self.commands:
            self.execute_command(self.commands, delta_time)

        elif self.number_of_loop > self.loop_commands_count and self.loop_commands:
            self.commands = self.loop_commands.copy()
            self.loop_commands_count += 1
