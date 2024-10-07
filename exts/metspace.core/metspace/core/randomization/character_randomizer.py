import os

import carb
import carb.events
import numpy as np
import omni.kit.app
from omni.anim.people.scripts.custom_command.command_manager import CustomCommandManager
from omni.anim.people.scripts.custom_command.defines import CustomCommand, CustomCommandTemplate
from metspace.core.extension import get_ext_path
from metspace.core.file_util import FileUtil, JSONFileUtil

from .randomizer import Command, Randomizer
from .randomizer_util import RandomizerUtil

"""
    This list determines what commands are available for the character randomizer and their probability of being picked from the last command
    Each time a new command is added, add it in this list and create a class that inherits the Command class for it
"""
# transition_matrix = {"GoTo": [0, 0.5, 0.5], "Idle": [0.6, 0, 0.4], "LookAround": [0.7, 0.3, 0]}
# Convergence rate: [0.3946, 0.2915, 0.3139]
"""
Class for the Character Randomizer
    Initialize special attributes for the characters
"""


class CharacterRandomizer(Randomizer):
    def __init__(self, global_seed):
        super().__init__(global_seed)
        self.default_command = Idle()  # Will be used whenever an invalid command is randomly generated
        self.command_manager = CustomCommandManager.get_instance()
        self.register_command_manager()
        ext_path = get_ext_path()
        self.transition_matrix_path = os.path.join(ext_path, "data/character_command_transition_matrix.json")
        self.load_transition_matrix()
        self.event_update_transition_matrix_file = carb.events.type_from_string(
            "metspace.update_transition_matrix_file_event"
        )
        self.update_transition_matrix_from_command_manager()

    def update_transition_matrix_from_command_manager(self):
        # Make sure all registered commands are added to transition matrix
        registered_commands = self.command_manager.get_all_custom_commands()
        for cmd in registered_commands:
            if cmd.name not in self.transition_matrix:
                self.add_to_transition_matrix(cmd)

    def update_transition_matrix(self, data):
        self.transition_matrix = data
        self.command_list = list(self.transition_matrix.keys())
        self.update_command_list()
        JSONFileUtil.write_to_file(self.transition_matrix_path, self.transition_matrix)
        self.notify_updated_transition_matrix_file()

    # Load the transition matrix from a file
    # The transition matrix determines the possiblity for transiting from one command to the next
    def load_transition_matrix(self):
        self.transition_matrix = JSONFileUtil.load_from_file(self.transition_matrix_path)
        self.command_list = list(self.transition_matrix.keys())
        self.update_command_list()

    def notify_updated_transition_matrix_file(self):
        bus = omni.kit.app.get_app().get_message_bus_event_stream()
        bus.push(self.event_update_transition_matrix_file, payload={})

    def update_command_list(self):
        # create a mapping of command name to its class
        for command in self.command_list:
            if command in globals():
                self.command_map[command] = globals()[command](
                    agent_speed=1.1
                )  # character's speed is about 1m/s, 1.1 is used to underestimate the duration
            else:
                if self.command_manager.get_command_template_by_name(command) == CustomCommandTemplate.TIMING:
                    new_command = type(command, (Idle,), {})
                    self.command_map[command] = new_command(agent_speed=1.1)
                elif self.command_manager.get_command_template_by_name(command) == CustomCommandTemplate.GOTO_BLEND:
                    new_command = type(command, (GoTo,), {})
                    self.command_map[command] = new_command(agent_speed=1.1)
                else:
                    carb.log_info(
                        command
                        + " has a invalid command template. It must be either TIMING or GOTO_BLEND to be randomized."
                    )

    def add_to_transition_matrix(self, new_command: CustomCommand):
        # TIMING_TO_OBJECT template commands will not support randomization
        if new_command.template == CustomCommandTemplate.TIMING_TO_OBJECT:
            carb.log_info(new_command.name + " will not be randomized.")
            return
        for command in self.transition_matrix.keys():
            self.transition_matrix[command].append(0.0)

        self.transition_matrix[new_command.name] = [0.0 for _ in range(len(list(self.transition_matrix.items())) + 1)]
        JSONFileUtil.write_to_file(self.transition_matrix_path, self.transition_matrix)
        self.notify_updated_transition_matrix_file()
        self.update_command_list()

    def remove_from_transition_matrix(self, command: CustomCommand):
        # If this command is not in the randomization list
        if command.name not in self.transition_matrix.keys():
            return
        cmd_idx = 0
        for command_name in self.transition_matrix.keys():
            if command_name == command.name:
                del self.transition_matrix[command.name]
                break
            cmd_idx += 1
        for command in self.transition_matrix.keys():
            self.transition_matrix[command].pop(cmd_idx)
        JSONFileUtil.write_to_file(self.transition_matrix_path, self.transition_matrix)
        self.notify_updated_transition_matrix_file()
        self.update_command_list()

    def register_command_manager(self):
        self.command_manager.register_listener("ADD_COMMAND", self.add_to_transition_matrix)
        self.command_manager.register_listener("REMOVE_COMMAND", self.remove_from_transition_matrix)


"""
Command supported by the Character Randomizer
    Need to implement the get_parameters() method for each command
"""


class GoTo(Command):
    def get_parameters(self):
        distance = np.random.uniform(5, 20)  # a reasonable distance for each GoTo command is about 5 to 20 meters

        # Find a valid and reachable point on the navmesh
        agent_pos = self.agent_list[self.agent]
        closest_point = carb.Float3(0, 0, 0)
        valid = False  # Whether the target point is a point on the navmesh
        reachable = None  # Whether there exists a path from the starting point to the target point
        num_attempts = 0  # Number of attempts to find a valid path

        while not valid or reachable == None:
            if num_attempts > 1000:
                # If failed to find a good GoTo destination after 1000 attempts, terminate and it will be reset to the default command
                carb.log_info("Can't find a valid random point for " + self.agent + ". GoTo will be reset to Idle")
                return
            loc_offset = RandomizerUtil.decompose_distance(distance)
            target_point = RandomizerUtil.add3(agent_pos, loc_offset)
            valid = self.inav.closest_navmesh_point(
                target_point, closest_point, carb.Float3(0.1, 0.1, 0.1)
            )  # snap a invalid point back to the navmesh
            reachable = self.inav.query_navmesh_path(agent_pos, closest_point)
            num_attempts += 1

        # Round the numbers based on given precision
        closest_point = np.round(np.array([closest_point.x, closest_point.y, closest_point.z]), 2)

        self.parameter = str(closest_point[0]) + " " + str(closest_point[1]) + " " + str(closest_point[2]) + " _"
        self.duration = np.linalg.norm(closest_point[:2] - np.array([agent_pos[0], agent_pos[1]])) / self.agent_speed
        # Update the starting position of this agent for the next GoTo command to work
        self.agent_list[self.agent] = [closest_point[0], closest_point[1], closest_point[2]]


class Idle(Command):
    def get_parameters(self):
        duration = np.random.uniform(2, 6)  # a reasonable idle duration is about 2 to 6 seconds
        self.duration = duration
        self.parameter = str(round(duration, self.num_precision))


class LookAround(Command):
    def get_parameters(self):
        duration = np.random.uniform(2, 4)  # a reasonble look-around duration is about 2 to 4 seconds
        self.duration = duration
        self.parameter = str(round(duration, self.num_precision))
