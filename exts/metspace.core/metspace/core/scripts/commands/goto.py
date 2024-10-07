# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from .base_command import Command


class GoTo(Command):
    """
    Command class to go to a location/locations.
    """

    def __init__(self, character, command, navigation_manager):
        super().__init__(character, command, navigation_manager)
        self.command_name = "GoTo"

    def setup(self):
        super().setup()
        self.character.set_variable("Action", "Walk")
        self.navigation_manager.generate_goto_path(self.command[1:])

    def execute(self, dt):
        if self.finished:
            return True

        if not self.is_setup:
            self.setup()
        return self.update(dt)

    def update(self, dt):
        self.time_elapsed += dt
        if self.walk(dt):
            return self.exit_command()

    def force_quit_command(self):
        return super().force_quit_command()
