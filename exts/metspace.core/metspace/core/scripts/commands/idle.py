# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from .base_command import Command


class Idle(Command):
    """
    Command class to stay idle
    """

    def __init__(self, character, command, navigation_manager):
        super().__init__(character, command, navigation_manager)
        if len(command) > 1:
            self.duration = float(command[1])

        self.command_name = "Idle"

    def setup(self):
        super().setup()
        self.character.set_variable("Action", "None")

    def update(self, dt):
        return super().update(dt)

    def force_quit_command(self):
        return super().force_quit_command()
