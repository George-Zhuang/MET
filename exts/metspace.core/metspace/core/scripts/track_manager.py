from __future__ import annotations

import os
import carb
import json
import random
import omni.usd
from omni.anim.people.scripts.global_agent_manager import GlobalAgentManager
from metspace.core.stage_util import CharacterUtil, RobotUtil

class TrackManager:
    """Global class which stores current and predicted positions of all characters and moving objects."""

    __instance: TrackManager = None

    def __init__(self):
        if self.__instance is not None:
            raise RuntimeError("Only one instance of GlobalCharacterPositionManager is allowed")
        self._is_tracking = True
        self._config_loaded = False
        
        # character dict that match character object with prim path
        self._random_agent_target_pair_indexes = []
        self._merged_object_character_prim_list = []
        self._merged_object_character_prim_path_list = []
        self._robot_prim_list = []
        self._robot_prim_path_list = []
        self._tracking_goal_dict = {}
        
        self._num_objects = 0
        self._num_characters = 0
        self._num_robots = 0
        self._num_type_robots = []

        # self._robot_types = ["Nova_Carter", "Transporter"]
        self._agent_target_pairs = []
        self._tracking_goal_dict = {}

        # 
        self._passed_time = 0

        TrackManager.__instance = self

    def destroy(self):
        TrackManager.__instance = None

    def __del__(self):
        self.destroy()

    @classmethod
    def get_instance(cls) -> TrackManager:
        if cls.__instance is None:
            TrackManager()
        return cls.__instance

    def load_yaml_config(self, yaml_data):
        if "object" in yaml_data:
            if "num" in yaml_data["object"]:
                self._num_objects = yaml_data["object"]["num"]
        if "character" in yaml_data:
            if "num" in yaml_data["character"]:
                self._num_characters = yaml_data["character"]["num"]
        if "robot" in yaml_data:
            for key in yaml_data["robot"].keys():
                if key.endswith("_num"):
                    self._num_type_robots.append(yaml_data["robot"][key])
                    self._num_robots += yaml_data["robot"][key]

        # Select agents and targets for tracking
        if self._num_characters + self._num_objects > 0 and self._num_robots > 0:
            # num_pairs = random.randint(1, min(self._num_characters + self._num_objects, self._num_robots))
            num_pairs = min(self._num_characters + self._num_objects, self._num_robots)
            targets = random.sample(range(self._num_characters + self._num_objects), num_pairs)
            agents = random.sample(range(self._num_robots), num_pairs)
            self._random_agent_target_pair_indexes = list(zip(agents, targets))

        self._config_loaded = True
        
    def setup_tracking_elements(self):
        if self._num_characters + self._num_objects > 0:
            object_prim_list = []
            object_prim_path_list = []
            character_prim_list = CharacterUtil.get_characters_in_stage()
            character_prim_path_list = [c.GetPath().pathString for c in character_prim_list]
            self._merged_object_character_prim_list = object_prim_list + character_prim_list
            self._merged_object_character_prim_path_list = object_prim_path_list + character_prim_path_list
        carb.log_warn(f'character_prim_list: {character_prim_list}')
        carb.log_warn(f'merged_object_character_prim_list: {self._merged_object_character_prim_list}')

        # carb.log_warn(f'num robots: {self._num_robots}')
        if self._num_robots > 0:
            self._robot_prim_list = []
            # carb.log_warn(f'setting up robot list')

            # for robot_id, robot_type in enumerate(self._robot_types):
            #     _robot_prim_list = RobotUtil.get_robots_in_stage(robot_type)
            #     _robot_prim_path_list = [r.GetPath().pathString for r in _robot_prim_list]
            #     self._robot_prim_list += _robot_prim_list
            #     self._robot_prim_path_list += _robot_prim_path_list

            _robot_prim_list = RobotUtil.get_robots_in_stage()
            _robot_prim_path_list = [r.GetPath().pathString for r in _robot_prim_list]
            self._robot_prim_list += _robot_prim_list
            self._robot_prim_path_list += _robot_prim_path_list

    def generate_tracking_command(self):
        tracking_command_list = []
        for agent_index, target_index in self._random_agent_target_pair_indexes:
            agent_prim = self._robot_prim_list[agent_index]
            agent_name = RobotUtil.get_robot_name(agent_prim)
            target_prim = self._merged_object_character_prim_list[target_index]
            # carb.log_warn(f'target_prim: {target_prim}')
            target_name = CharacterUtil.get_character_name(target_prim)
            self._tracking_goal_dict[agent_prim] = target_prim
            tracking_command_list.append(
                agent_name + " Track " + target_name
            )
        return tracking_command_list

    def inject_tracking_command(self, agent_prim_path, agent_name, step_size, frequency=5, force=True,):
        # inject tracking command to the agent following the frequency
        self._passed_time += step_size
        inject_interval = 1 / frequency
        if self._passed_time < inject_interval:
            return
        self._passed_time = 0

        agent_index = self._robot_prim_path_list.index(agent_prim_path)
        # if the agent is not selected in the agent-target pairs, return
        if agent_index not in [pair[0] for pair in self._random_agent_target_pair_indexes]:
            return
        agent_prim = self._robot_prim_list[agent_index]
        target_index = self._random_agent_target_pair_indexes[agent_index][1]
        target_prim = self._merged_object_character_prim_list[target_index]

        target_pos = CharacterUtil.get_character_current_pos(target_prim)
        agent_pos = RobotUtil.get_robot_pos(agent_prim)
        tracking_command_list = [
            agent_name + " GoTo " + str(target_pos[0]) + " " + str(target_pos[1]) + " " + str(agent_pos[2])
            ]
        if not GlobalAgentManager.has_instance():
            carb.log_warn("Global Character Manager is None. Please check out whether simulation has already start")
        GlobalAgentManager.get_instance().inject_command(
            agent_prim_path, tracking_command_list, force=force)
    
    def get_agent_cameras(self):
        # carb.log_warn(f'random_agent_target_pair_indexes: {self._random_agent_target_pair_indexes}')
        cameras = []
        for (agent_index, target_index) in self._random_agent_target_pair_indexes:
            agent_prim = self._robot_prim_list[agent_index]
            cameras += RobotUtil.get_specific_cameras_on_robot(agent_prim)
        
        return list(set(cameras))
    
    # def get_agent_target_distance(self):
    #     distances = {}
    #     for (agent_index, target_index) in self._random_agent_target_pair_indexes:
    #         agent_prim = self._robot_prim_list[agent_index]
    #         agent_prim_path = self._robot_prim_path_list[agent_index]
    #         target_prim = self._merged_object_character_prim_list[target_index]
    #         agent_pos = RobotUtil.get_robot_pos(agent_prim)
    #         target_pos = CharacterUtil.get_character_current_pos(target_prim)
    #         distances[agent_prim_path] = ((agent_pos[0] - target_pos[0]) ** 2 + (agent_pos[1] - target_pos[1]) ** 2) ** 0.5
    #     return distances

    def get_target_position(self):
        target_positions = {}
        for (agent_index, target_index) in self._random_agent_target_pair_indexes:
            agent_prim_path = self._robot_prim_path_list[agent_index]
            target_prim = self._merged_object_character_prim_list[target_index]
            target_pos = CharacterUtil.get_character_current_pos(target_prim)
            target_positions[agent_prim_path] = target_pos
        return target_positions
    
    def save_agent_target_pairs_prim_path(self, save_path):
        data = {}
        for (agent_index, target_index) in self._random_agent_target_pair_indexes:
            agent_prim_path = self._robot_prim_path_list[agent_index]
            target_prim_path = self._merged_object_character_prim_path_list[target_index]
            data[agent_prim_path] = target_prim_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(data, f, indent=4, separators=(",", ": "))
        return True
    