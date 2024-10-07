import carb
import numpy as np

from .randomizer import Command, Randomizer
from .randomizer_util import RandomizerUtil

"""
    This list determines what commands are available for the character randomizer and their probability of being picked from the last command
    Each time a new command is added, add it in this list and create a class that inherits the Command class for it
"""
transition_matrix = {"GoTo": [0.8, 0.2], "Idle": [0.8, 0.2]}

"""
Class for the Robot Randomizer
    Initialize special attributes for the robots
"""


class RobotRandomizer(Randomizer):
    def __init__(self, global_seed):
        super().__init__(global_seed)
        self.command_list = list(transition_matrix.keys())
        # create a mapping of command name to its class
        for command in self.command_list:
            self.command_map[command] = globals()[command](
                agent_speed=0.6
            )  # robot's speed is about 0.5m/s, 0.6 is used to underestimate the duration
        self.transition_matrix = transition_matrix
        self.default_command = Idle()  # Will be used whenever an invalid command is randomly generated


"""
Command supported by the Robot Randomizer
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

        self.duration = RandomizerUtil.dist3(closest_point, agent_pos) / self.agent_speed  # Duration = Dist / Speed
        self.parameter = (
            str(round(closest_point.x, self.num_precision))
            + " "
            + str(round(closest_point.y, self.num_precision))
            + " "
            + str(round(closest_point.z, self.num_precision))
        )
        # Update the starting position of this agent for the next GoTo command to work
        self.agent_list[self.agent] = [closest_point.x, closest_point.y, closest_point.z]


class Idle(Command):
    def get_parameters(self):
        duration = np.random.uniform(5, 10)  # a reasonable idle duration for the robot is about 5 to 10 seconds
        self.duration = duration
        self.parameter = str(round(duration, self.num_precision))
