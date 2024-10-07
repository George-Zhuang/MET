import carb
import numpy as np

from .randomizer import Command, Randomizer
from .randomizer_util import RandomizerUtil
from .robot_randomizer import Idle, RobotRandomizer

"""
Class for the Robot Randomizer
    Initialize special attributes for the robots
"""


class TransporterRandomizer(RobotRandomizer):
    def __init__(self, global_seed):
        super().__init__(global_seed)
        # Transporter's AABB min and max
        self.extent = [(-1.0332839734863342, -0.3290388065636699), (0.3975681979007675, 0.3300932763404468)]
        self.default_command = Idle()  # Will be used whenever an invalid command is randomly generated
