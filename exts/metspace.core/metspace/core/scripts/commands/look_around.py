# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from .base_command import Command


class LookAround(Command):
    """
    Command class to look around (moving head from left to right).
    """

    def __init__(self, character, command, navigation_manager=None):
        super().__init__(character, command, navigation_manager)
        if len(command) > 1:
            self.duration = float(command[1])
        self.command_name = "LookAround"

    def setup(self):
        super().setup()
        self.character.set_variable("Action", "None")
        self.character.set_variable("lookaround", 1.0)

    def exit_command(self):
        self.character.set_variable("lookaround", 0.0)
        return super().exit_command()

    def update(self, dt):
        return super().update(dt)

    def force_quit_command(self):
        self.character.set_variable("lookaround", 0.0)
        return super().force_quit_command()
