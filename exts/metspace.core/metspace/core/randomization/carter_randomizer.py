import carb
import numpy as np

from .randomizer import Command, Randomizer
from .randomizer_util import RandomizerUtil
from .robot_randomizer import Idle, RobotRandomizer

"""
Class for the Robot Randomizer
    Initialize special attributes for the robots
"""


class CarterRandomizer(RobotRandomizer):
    def __init__(self, global_seed):
        super().__init__(global_seed)
        # Carter's AABB min and max
        self.extent = [(-1.101184962241601, -0.45593118604327454), (0.1407164487669442, 0.45593120044146757)]
        self.default_command = Idle()  # Will be used whenever an invalid command is randomly generated
