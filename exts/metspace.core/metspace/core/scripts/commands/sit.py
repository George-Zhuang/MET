# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import carb
import omni.usd
from pxr import Gf, UsdGeom

from ..utils import Utils
from .base_command import Command


class Sit(Command):
    """
    Command class that implements the sit command.
    """

    def __init__(self, character, command, navigation_manager):
        super().__init__(character, command, navigation_manager)
        self.stage = omni.usd.get_context().get_stage()
        self.seat_prim = self.stage.GetPrimAtPath(command[1])
        if len(command) > 2:
            self.duration = float(command[2])
        self.sit_time = 0
        self.current_action = None
        self.command_name = "Sit"

    def setup(self):
        super().setup()
        self.walk_to_pos, self.walk_to_rot, self.interact_pos, self.interact_rot = Utils.get_interact_prim_offsets(
            self.stage, self.seat_prim
        )
        character_pos = Utils.get_character_pos(self.character)
        self.navigation_manager.generate_path([character_pos, self.walk_to_pos], self.walk_to_rot)
        self.current_action = "walk"
        self._char_lerp_t = 0
        self.stand_animation_time = 0

    def force_quit_command(self):
        if self.current_action == "walk" or self.current_action == None:
            return super().force_quit_command()

        if self.current_action == "sit":
            self.character.set_variable("Action", "Sit")
            self.character.set_variable("Action", "None")
            self._char_lerp_t = 0.0
            self.current_action = "stand"
            return

    def update(self, dt):
        if self.current_action == "walk" or self.current_action == None:
            if self.walk(dt):
                self.current_action = "sit"
                self._char_start_pos, self._char_start_rot = Utils.get_character_transform(self.character)

        elif self.current_action == "sit":
            # Start to play sit animation
            # At the same time adjust players's tranlatation to fit the seat
            self._char_lerp_t = min(self._char_lerp_t + dt, 1.0)
            lerp_pos = Utils.lerp3(self._char_start_pos, self.interact_pos, self._char_lerp_t)
            self.character.set_world_transform(lerp_pos, self._char_start_rot)
            self.character.set_variable("Action", "Sit")
            self.sit_time += dt
            if self.sit_time > self.duration:
                self.character.set_variable("Action", "None")
                self._char_lerp_t = 0.0
                self.current_action = "stand"

        elif self.current_action == "stand":
            if self.stand_animation_time < 1.5:
                # adjust character's position while play "stand" animation
                self._char_lerp_t = min(self._char_lerp_t + dt, 1.0)
                lerp_pos = Utils.lerp3(self.interact_pos, self._char_start_pos, self._char_lerp_t)
                current_pos, current_rot = Utils.get_character_transform(self.character)
                self.character.set_world_transform(lerp_pos, current_rot)
                self.stand_animation_time += dt

            if self.stand_animation_time > 1.5:
                # set character's position to position before the sit animation, enter the idle stage
                current_pos, current_rot = Utils.get_character_transform(self.character)
                self.character.set_world_transform(self._char_start_pos, current_rot)
                return self.exit_command()
