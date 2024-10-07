import asyncio
import json
import os
import re

import carb
import numpy as np
import omni.anim.navigation.core as nav
import omni.client
import omni.kit
import omni.usd
from omni.anim.people.scripts.custom_command.populate_anim_graph import populate_anim_graph
from omni.anim.people.scripts.robot_behavior import RobotBehavior
from omni.isaac.core.utils import prims
from pxr import Sdf, Usd

from .config_file_util import ConfigFileUtil
from .data_generation import DataGeneration
from .file_util import FileUtil, TextFileUtil, YamlFileUtil
from .randomization.camera_randomizer import CameraRandomizer, LidarCameraRandomizer
from .randomization.carter_randomizer import CarterRandomizer
from .randomization.character_randomizer import CharacterRandomizer
from .randomization.randomizer_util import RandomizerUtil
from .randomization.robot_randomizer import RobotRandomizer
from .randomization.transporter_randomizer import TransporterRandomizer
from .settings import AssetPaths, PrimPaths, Settings
from .stage_util import CameraUtil, CharacterUtil, LidarCamUtil, RobotUtil, StageUtil
from .verification import SimulationVerification
from .scripts.track_manager import TrackManager

EXCLUSIVE_CHARACTER_FOLDERS = "/exts/metspace/asset_settings/exclusive_character_assets_folders"

OMNI_ANIM_PEOPLE_COMMAND_PATH = "/exts/omni.anim.people/command_settings/command_file_path"
OMNI_ANIM_PEOPLE_ROBOT_COMMAND_PATH = "/exts/omni.anim.people/command_settings/robot_command_file_path"


class SimulationManager:
    """
    Simulation Manager class that takes in config file to set up simulation accordingly.
    """

    SET_UP_SIMULATION_DONE_EVENT = carb.events.type_from_string("metspace.SET_UP_SIMULATION_DONE")

    def __init__(self):
        self.characters_parent_prim_path = PrimPaths.characters_parent_path()
        self.default_biped_prim_path = PrimPaths.default_biped_prim_path()
        self.default_biped_asset_path = AssetPaths.default_biped_asset_path()
        self.default_biped_name = AssetPaths.default_biped_asset_name()
        self.exclusive_character_folders = []
        self.lidar_cameras_parent_prim_path = PrimPaths.lidar_cameras_parent_path()
        self.cameras_parent_prim_path = PrimPaths.cameras_parent_path()
        self.robots_parent_prim_path = PrimPaths.robots_parent_path()
        self.character_assets_list = (
            []
        )  # List of all characters inside the character asset folders, provided by config file
        self.available_character_list = []  # Character list after filtering and shuffling
        # Config file variables
        self.clear_config_file()
        # Randomizers
        self._character_randomizer = CharacterRandomizer(0)
        self._nova_carter_randomizer = CarterRandomizer(0)
        self._transporter_randomizer = TransporterRandomizer(0)
        self._carter_v1_randomizer = CarterRandomizer(0)
        self._dingo_randomizer = CarterRandomizer(0)
        self._jetbot_randomizer = CarterRandomizer(0)
        self._robot_randomizers = {
            "nova_carter": self._nova_carter_randomizer,
            "transporter": self._transporter_randomizer,
            "carter_v1": self._carter_v1_randomizer,
            "dingo": self._dingo_randomizer,
            "jetbot": self._jetbot_randomizer,
        }
        self._camera_randomizer = CameraRandomizer(0)
        self._lidar_camera_randomizer = LidarCameraRandomizer(0)
        self._agent_positions = []
        # State variables for assets loading
        self._bus = omni.kit.app.get_app().get_message_bus_event_stream()
        self._seed_is_updated = False
        self._filter_is_updated = False
        self._character_folder_is_updated = False
        self._command_file_is_updated = False
        self._robot_command_file_is_updated = False
        self.will_run_data_generation = False
        self._load_stage_handle = None
        self._nav_mesh_event_handle = None
        self._dg = None
        self._dg_task = None
        self._simulation_verification = SimulationVerification()
        # command settings
        self._command_valid = True
        self._robot_command_valid = True
        self.output_path = ''
        
        self.initialize_settings()

    def initialize_settings(self):
        """
        Loads extension settings. Apply default values when needed.
        """
        setting_dict = carb.settings.get_settings()
        self.exclusive_character_folders = setting_dict.get(EXCLUSIVE_CHARACTER_FOLDERS)

        # Handle default values
        if not self.exclusive_character_folders:
            self.exclusive_character_folders = ["biped_demo"]

    # ========= Set Up Characters/Robots =========

    def load_filters(self):
        """
        Load the filters from the asset folder
        The filter must be a json file named "filter" and located in the asset root directory
        """
        if self.yaml_data == None:
            return None
        if "character" not in self.yaml_data:
            return None
        file_path = self.yaml_data["character"]["asset_path"] + "filter.json"
        result, _, content = omni.client.read_file(file_path)
        data = {}
        if result == omni.client.Result.OK:
            data = json.loads(memoryview(content).tobytes().decode("utf-8"))
        # Handling the case if the file does not exist
        else:
            carb.log_warn("Filter file does not exist. Asset filtering will not function.")
            return None
        return data

    def load_default_skeleton_and_animations(self):
        """
        Loads the default biped skeleton that Animation Graph requires. Also loads character animations for idle,
        walking, lookaround and sitting.
        """
        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath(self.characters_parent_prim_path):
            prims.create_prim(self.characters_parent_prim_path, "Xform")

        if not stage.GetPrimAtPath(self.default_biped_prim_path):
            prim = prims.create_prim(
                self.default_biped_prim_path,
                "Xform",
                usd_path=self.default_biped_asset_path,
            )
            prim.GetAttribute("visibility").Set("invisible")

        populate_anim_graph()

    def spawn_character_by_idx(self, spawn_location, spawn_rotation, idx):
        """
        Spawns character according to index in the character folder list at provided spawn_location and spawn_rotation.
        Ensures duplicate characters are not spawned, until all character assets have been utilized.
        If all character assets have been utilized, duplicates will be spawned.
        """
        # Character name
        char_name = CharacterUtil.get_character_name_by_index(idx)
        # Characters will be spawned in the same order again if all unique assets are used
        list_len = len(self.available_character_list)
        # Loop the list if there are multiple characters
        idx = idx % list_len
        # The character assets are randomly sorted depending on the global seed when the assets is selected
        # This draws the character based on the index, producing a deterministic result
        char_asset_name = self.available_character_list[idx]
        character_folder = "{}/{}".format(self.yaml_data["character"]["asset_path"], char_asset_name)
        # Get the usd present in the character folder
        character_usd_name = self._get_character_usd_in_folder(character_folder)
        if not character_usd_name:
            carb.log_error("Unable to spawn character due to finding a character folder with no usd present")
            return
        character_usd_path = "{}/{}".format(character_folder, character_usd_name)
        if not character_usd_path:
            carb.log_error("Unable to spawn character due to finding a character folder with no usd present")
            return
        # Spawn character
        return CharacterUtil.load_character_usd_to_stage(character_usd_path, spawn_location, spawn_rotation, char_name)

    def _get_character_usd_in_folder(self, character_folder_path):
        result, folder_list = omni.client.list(character_folder_path)
        if result != omni.client.Result.OK:
            carb.log_error("Unable to read character folder path at {}".format(character_folder_path))
            return None
        for item in folder_list:
            if item.relative_path.endswith(".usd"):
                return item.relative_path
        carb.log_error("Unable to file a .usd file in {} character folder".format(character_folder_path))
        return None

    def read_character_asset_list(self):
        """
        Read character assets into list according to the character asset path in config file
        """
        assets_root_path = self.yaml_data["character"]["asset_path"]
        # List all files in characters directory
        result, folder_list = omni.client.list("{}/".format(assets_root_path))
        if result != omni.client.Result.OK:
            carb.log_error("Unable to get character assets from provided asset path.")
            self.character_assets_list = []
            return
        # Prune items from folder list that are not directories.
        pruned_folder_list = [
            folder.relative_path
            for folder in folder_list
            if (folder.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN) and not folder.relative_path.startswith(".")
        ]
        if pruned_folder_list == None or len(pruned_folder_list) == 0:
            self.character_assets_list = []
            return
        # Prune folders that do not have usd inside
        pruned_usd_folder_list = []
        for folder in pruned_folder_list:
            result, file_list = omni.client.list("{}/{}/".format(assets_root_path, folder))
            for file in file_list:
                post_fix = file.relative_path[file.relative_path.rfind(".") + 1 :].lower()
                if post_fix == "usd" or post_fix == "usda":
                    pruned_usd_folder_list.append(folder)
                    break
        # Prune the default biped character
        if self.default_biped_name in pruned_usd_folder_list:
            pruned_usd_folder_list.remove(self.default_biped_name)
        # Prune exclusive folders
        for folder in self.exclusive_character_folders:
            if folder in pruned_usd_folder_list:
                pruned_usd_folder_list.remove(folder)
        self.character_assets_list = pruned_usd_folder_list

    def refresh_available_character_asset_list(self):
        """
        Set avaliable character asset list by filtering and shuffling the character assets list.
        """
        if len(self.character_assets_list) == 0:
            self.available_character_list = []
            return
        labels_str = self.yaml_data["character"]["filters"]
        if not labels_str:
            labels_str = ""
        labels = re.split(r"[,\s]+", labels_str)  # Split the string by comma, space, or both
        self.available_character_list = self.character_assets_list.copy()
        filters = self.load_filters()
        self.filter_character_asset_list(filters, labels)
        self.shuffle_character_asset_list()

    def shuffle_character_asset_list(self):
        """
        Shuffles the order of characters in the avaliable asset list
        """
        if not self.yaml_data_seed_valid:
            carb.log_warn("Seed invalid. Shuffle character asset list fails.")
            return
        np.random.seed(RandomizerUtil.handle_overflow(self.yaml_data["global"]["seed"]))
        self.available_character_list = np.random.choice(
            self.available_character_list, size=len(self.available_character_list), replace=False
        )

    def filter_character_asset_list(self, filters, labels):
        """
        Given labels, return character assets with these labels
        """
        # Filter file does not exist, skip the filtering
        if filters != None:
            filtered = self.available_character_list
            for label in labels:
                if label in filters:
                    filtered = [char for char in filtered if char in filters[label]]
                # Handle non-existent labels
                else:
                    if label != "" and label != " ":
                        warn_msg = (
                            'Invalid character filter label: "'
                            + str(label)
                            + '". Available labels: '
                            + str(list(filters.keys()))[1:-1]
                        )
                        carb.log_warn(warn_msg)
                        labels.remove(label)
            self.available_character_list = filtered

    def setup_animation_graph_to_character(self, character_list: []):
        """
        Add animation graph for all characters in stage
        """
        # Search for the animation graph in default biped
        anim_graph_prim = None
        default_biped_prim = PrimPaths.default_biped_prim_path()
        stage = omni.usd.get_context().get_stage()
        anim_graph_prim = CharacterUtil.get_anim_graph_from_character(stage.GetPrimAtPath(default_biped_prim))
        if anim_graph_prim is None:
            carb.log_error("Unable to find an animation graph on stage.")
            return
        # Apply animation graph to each character
        for prim in character_list:
            # remove animation graph attribute if it exists
            omni.kit.commands.execute("RemoveAnimationGraphAPICommand", paths=[Sdf.Path(prim.GetPrimPath())])
            omni.kit.commands.execute(
                "ApplyAnimationGraphAPICommand",
                paths=[Sdf.Path(prim.GetPrimPath())],
                animation_graph_path=Sdf.Path(anim_graph_prim.GetPrimPath()),
            )

    def setup_python_scripts_to_character(self, character_list):
        """
        Add behavior script to all characters in stage
        """
        for prim in character_list:
            omni.kit.commands.execute("ApplyScriptingAPICommand", paths=[Sdf.Path(prim.GetPrimPath())])
            attr = prim.GetAttribute("omni:scripting:scripts")
            script_path = AssetPaths.behavior_script_path()
            attr.Set([r"{}".format(script_path)])

    def setup_python_scripts_to_robot(self, robot_list, robot_type):
        """
        Add behavior script to all characters in stage
        """
        for prim in robot_list:
            omni.kit.commands.execute("ApplyScriptingAPICommand", paths=[Sdf.Path(prim.GetPrimPath())])
            attr = prim.GetAttribute("omni:scripting:scripts")
            script_path = AssetPaths.robot_behavior_script_path()
            # Get the corresponding robot script
            script_path = script_path.replace("robot", robot_type.lower())
            attr.Set([r"{}".format(script_path)])

    # ========= Config File =========

    def check_config_file_values_valid(self, new_yaml, error_list=None):
        """
        Check if each field in config file is valid. Mark the invalid fields for later use.
        """

        def add_error(err_str):
            if error_list is not None and isinstance(error_list, list):
                error_list.append(err_str)
            carb.log_error(err_str)

        if new_yaml["global"]["seed"] < 0:
            add_error("global seed ({0}) can not be negative".format(str(new_yaml["global"]["seed"])))
            self.yaml_data_seed_valid = False
        else:
            self.yaml_data_seed_valid = True
        if (
            "camera_num" in new_yaml["global"]
            and new_yaml["global"]["camera_num"] < 0
            and new_yaml["global"]["camera_num"] != -1
        ):
            add_error("camera number ({0}) can not be negative".format(str(new_yaml["global"]["camera_num"])))
            self.yaml_data_cam_num_valid = False
        else:
            self.yaml_data_cam_num_valid = True
        if (
            "lidar_num" in new_yaml["global"]
            and new_yaml["global"]["lidar_num"] < 0
            and new_yaml["global"]["lidar_num"] != -1
        ):
            add_error("lidar camera number ({0}) can not be negative".format(str(new_yaml["global"]["lidar_num"])))
            self.yaml_data_lidar_num_valid = False
        else:
            self.yaml_data_lidar_num_valid = True
        if new_yaml["global"]["simulation_length"] < 0:
            add_error(
                "simulation length ({0}) can not be negative".format(str(new_yaml["global"]["simulation_length"]))
            )
            self.yaml_data_sim_length_valid = False
        else:
            self.yaml_data_sim_length_valid = True
        if "character" in new_yaml and new_yaml["character"]["num"] < 0:
            add_error("character number ({0}) can not be negative".format(str(new_yaml["character"]["num"])))
            self.yaml_data_character_num_valid = False
        else:
            self.yaml_data_character_num_valid = True

        self.yaml_data_robot_num_valid = True
        if "robot" in new_yaml:
            for robot in Settings.ROBOT_CATEGORY:
             if new_yaml["robot"][f"{robot}_num"] < 0:
                carb.log_error(
                        f"{robot} robot number ({new_yaml['robot'][f'{robot}_num']}) can not be negative"
                )
                self.yaml_data_robot_num_valid = False

        self.yaml_data_is_valid = (
            self.yaml_data_seed_valid
            and self.yaml_data_cam_num_valid
            and self.yaml_data_lidar_num_valid
            and self.yaml_data_sim_length_valid
            and self.yaml_data_character_num_valid
            and self.yaml_data_robot_num_valid
        )

        if self.yaml_data_is_valid == False:
            carb.log_error("Config file contains invalid values.")
            return False
        else:
            return True

    def check_config_file_data_need_update(self, new_yaml):
        """
        Check if each field in this config file is different from last config file. Mark the updated fields for later use.
        """
        # Check if seed needs udpate
        if self.yaml_data == None or self.yaml_data["global"]["seed"] != new_yaml["global"]["seed"]:
            self._seed_is_updated = True
        else:
            self._seed_is_updated = False
        # Check if character section needs update
        if "character" in new_yaml:
            if self.yaml_data == None or "character" not in self.yaml_data:
                self._character_folder_is_updated = True
                self._filter_is_updated = True
                self._command_file_is_updated = True
            else:
                self._character_folder_is_updated = bool(
                    self.yaml_data["character"]["asset_path"] != new_yaml["character"]["asset_path"]
                )
                self._filter_is_updated = bool(
                    self.yaml_data["character"]["filters"] != new_yaml["character"]["filters"]
                )
                self._command_file_is_updated = bool(
                    self.yaml_data["character"]["command_file"] != new_yaml["character"]["command_file"]
                )
        else:
            self._character_folder_is_updated = False
            self._filter_is_updated = False
            self._command_file_is_updated = False
        # Check if robot section needs update
        if "robot" in new_yaml:
            if self.yaml_data == None or "robot" not in self.yaml_data:
                self._robot_command_file_is_updated = True
            else:
                self._robot_command_file_is_updated = bool(
                    self.yaml_data["robot"]["command_file"] != new_yaml["robot"]["command_file"]
                )
        else:
            self._robot_command_file_is_updated = False

    def load_config_file(self, file_path):
        """
        Load config file and set it up.
        Params:
            - file_path: the config file path
        Return:
            - (A, B)
            - A: the config file object being loaded
            - B: if the config file is modified during setup
        """
        # Remove the last load
        self.clear_config_file()
        # Load file check
        raw_yaml = YamlFileUtil.load_yaml(file_path)
        # Store config file path
        self.config_file_path = file_path
        if not raw_yaml:
            self.config_file_path = None
            return (None, False)
        # Header check
        yaml_data = ConfigFileUtil.remove_config_file_header(raw_yaml)
        if not yaml_data:
            self.config_file_path = None
            return (None, False)
        # Version check
        if not ConfigFileUtil.check_config_file_version(yaml_data):
            self.config_file_path = None
            return (None, False)
        # Property valid check
        if not ConfigFileUtil.check_config_file_property(yaml_data):
            self.config_file_path = None
            return (None, False)
        # Set up config file (sections and fields)
        is_modified = ConfigFileUtil.setup_config_file(yaml_data)
        
        # Set up output path
        if yaml_data["replicator"]["parameters"]["output_dir"] == '':
            # self.output_path = SaveFileUtil.get_save_path(yaml_data)
            self.output_path = yaml_data["replicator"]["parameters"]["output_dir"]
        else:
            self.output_path = yaml_data["replicator"]["parameters"]["output_dir"]
        # Make sure command files are created
        if "character" in yaml_data:
            file_path = self.parse_command_file_path(yaml_data["character"]["command_file"])
            if not TextFileUtil.is_text_file_exist(file_path):
                carb.log_warn(f"No character command file found in path: {file_path}. A new command file is created.")
                TextFileUtil.create_text_file(file_path)
        if "robot" in yaml_data:
            file_path = self.parse_command_file_path(yaml_data["robot"]["command_file"])
            if not TextFileUtil.is_text_file_exist(file_path):
                carb.log_warn(f"No robot command file found in path: {file_path}. A new command file is created.")
                TextFileUtil.create_text_file(file_path)

        # Update the loaded config file to tracking manager
        TrackManager.get_instance().load_yaml_config(yaml_data)
        # Update the loaded config file to self
        self.update_config_file(yaml_data)
        return (self.yaml_data, is_modified)

    def save_config_file(self):
        if not self.yaml_data:
            return False
        yaml_raw = ConfigFileUtil.add_config_file_header(self.yaml_data)
        return YamlFileUtil.save_yaml(self.config_file_path, yaml_raw)

    def save_as_config_file(self, new_path):
        if not self.yaml_data:
            return False
        # Save command files if needed
        if "character" in self.yaml_data:
            command_file_path = self.parse_command_file_path(self.yaml_data["character"]["command_file"])
            new_command_file_path = os.path.join(new_path, "command.txt")
            self.yaml_data["character"]["command_file"] = new_command_file_path
            TextFileUtil.copy_text_file(command_file_path, new_command_file_path)
        if "robot" in self.yaml_data:
            robot_command_file_path = self.parse_command_file_path(self.yaml_data["robot"]["command_file"])
            new_robot_command_file_path = os.path.join(new_path, "robot_command.txt")
            self.yaml_data["robot"]["command_file"] = new_robot_command_file_path
            TextFileUtil.copy_text_file(robot_command_file_path, new_robot_command_file_path)
        # Update config file path and save
        self.config_file_path = os.path.join(new_path, "config.yaml")
        return self.save_config_file()

    def get_config_file(self):
        return self.yaml_data

    def clear_config_file(self):
        """
        Remove the last loaded config file and reset associated varibales
        """
        self.config_file_path = None
        self.yaml_data = None
        self.yaml_data_is_valid = False
        self.yaml_data_seed_valid = False
        self.yaml_data_cam_num_valid = False
        self.yaml_data_lidar_num_valid = False
        self.yaml_data_sim_length_valid = False
        self.yaml_data_character_num_valid = False
        self.yaml_data_robot_num_valid = False

    def update_config_file(self, new_yaml, error_list=None):
        """
        Update the loaded config file with new_yaml
        """
        self.check_config_file_values_valid(new_yaml, error_list)
        # Mark the updated fields. The actual changes will be applied in assets loading
        self.check_config_file_data_need_update(new_yaml)
        # Update the config file
        self.yaml_data = new_yaml
        # Udpate randomizer
        # Create or Udpate randomizer
        # The order for creating the randomizers matter now since they determin the ID
        seed = self.yaml_data["global"]["seed"]
        if self.yaml_data_seed_valid and self._seed_is_updated:
            if self._character_randomizer == None:
                self._character_randomizer = CharacterRandomizer(seed)
            else:
                self._character_randomizer.reset()
                self._character_randomizer.update_seed(seed)
            if self._camera_randomizer == None:
                self._camera_randomizer = CameraRandomizer(seed)
            else:
                self._camera_randomizer.reset()
                self._camera_randomizer.update_seed(seed)
            if self._lidar_camera_randomizer == None:
                self._lidar_camera_randomizer = LidarCameraRandomizer(seed)
            else:
                self._lidar_camera_randomizer.reset()
                self._lidar_camera_randomizer.update_seed(seed)

            for robot_type in Settings.ROBOT_CATEGORY:
                if robot_type not in self._robot_randomizers:
                    self._robot_randomizers[robot_type] = RobotRandomizer(seed)
                else:
                    self._robot_randomizers[robot_type].reset()
                    self._robot_randomizers[robot_type].update_seed(seed)

        # Refresh character list to make sure loading characters succeed
        if "character" in new_yaml:
            self.read_character_asset_list()
            self.refresh_available_character_asset_list()  # Character folder change triggers available character list refresh
        # Special case: when command file is changed, the change should be applied to anim.people immediately
        if self._command_file_is_updated:
            self.setup_anim_people_command_from_config_file()
        if self._robot_command_file_is_updated:
            self.setup_anim_people_robot_command_from_config_file()

    # ========= Characters/Robots Commands =========

    def parse_command_file_path(self, command_file_path):
        return FileUtil.get_absolute_path(self.config_file_path, command_file_path)

    def load_commands(self):
        if not self.yaml_data:
            return None
        if "character" not in self.yaml_data:
            return None
        command_path = self.parse_command_file_path(self.yaml_data["character"]["command_file"])
        cmd_str = TextFileUtil.read_text_file(command_path)
        if not cmd_str:
            return None
        else:
            return cmd_str.splitlines()

    def load_robot_commands(self):
        if not self.yaml_data:
            return None
        if "robot" not in self.yaml_data:
            return None
        command_path = self.parse_command_file_path(self.yaml_data["robot"]["command_file"])
        cmd_str = TextFileUtil.read_text_file(command_path)
        if not cmd_str:
            return None
        else:
            return cmd_str.splitlines()

    def save_commands(self, commands_list):
        if not self.yaml_data:
            return False
        if "character" not in self.yaml_data:
            carb.log_warn("character section is not present. Save commands fails.")
            return False
        command_path = self.parse_command_file_path(self.yaml_data["character"]["command_file"])
        command_str = ""
        for cmd in commands_list:
            command_str += cmd
            command_str += "\n"
        return TextFileUtil.write_text_file(command_path, command_str)

    def save_robot_commands(self, commands_list):
        if not self.yaml_data:
            return False
        if "robot" not in self.yaml_data:
            carb.log_warn("robot section is not present. Save commands fails.")
            return False
        command_path = self.parse_command_file_path(self.yaml_data["robot"]["command_file"])
        command_str = ""
        for cmd in commands_list:
            command_str += cmd
            command_str += "\n"
        return TextFileUtil.write_text_file(command_path, command_str)

    def generate_random_commands(self):
        """
        Generate random character commands by the current config file
        """
        if self.yaml_data == None:
            carb.log_error("Config file is invalid. Generate random commands fails.")
            return None
        if self.yaml_data_seed_valid == False:
            carb.log_error("Randomization seed is invalid. Generate random commands fails.")
            return None
        if not self.is_valid_transition_matrix(self._character_randomizer.transition_matrix):
            carb.log_error(
                "The randomization transition matrix is invalid. The sum of probabilities for a command must equal to 1. Please edit the transition matrix and try again."
            )
            return None
        seed = self.yaml_data["global"]["seed"]
        duration = self.yaml_data["global"]["simulation_length"]
        character_list = CharacterUtil.get_characters_in_stage()
        character_dict = {}  # <name, pos>
        count = self.yaml_data["character"]["num"]
        for i, c in enumerate(character_list):
            if i < count:
                name = CharacterUtil.get_character_name(c)
                pos = CharacterUtil.get_character_pos(c)
                character_dict[name] = pos
        commands = self._character_randomizer.generate_commands(seed, duration, character_dict)
        return commands

    def generate_robot_commands_by_type(self, robot_type):
        seed = self.yaml_data["global"]["seed"]
        duration = self.yaml_data["global"]["simulation_length"]
        robot_list = RobotUtil.get_robots_in_stage(robot_type)
        robot_dict = {}
        count = self.yaml_data["robot"][robot_type.lower() + "_num"]
        for i, c in enumerate(robot_list):
            if i < count:
                name = RobotUtil.get_robot_name(c)
                pos = RobotUtil.get_robot_pos(c)
                robot_dict[name] = pos
        commands = self._nova_carter_randomizer.generate_commands(seed, duration, robot_dict)
        return commands

    def generate_random_robot_commands(self):
        """
        Generate random robot commands by the current config file
        """
        if self.yaml_data == None:
            carb.log_error("Config file is invalid. Generate random commands fails.")
            return None
        if self.yaml_data_seed_valid == False:
            carb.log_error("Randomization seed is invalid. Generate random commands fails.")
            return None
        commands = self.generate_robot_commands_by_type("Nova_Carter") + self.generate_robot_commands_by_type(
            "Transporter"
        )
        return commands

    # ========= Data Generation =========

    def data_generation_done_callback(self):
        """
        Release handle when data generation is finished.
        """
        # Clean up reference
        self._dg = None
        self._dg_task = None
        carb.log_info("One data generation completes.")

    def run_data_generation(self):
        """
        Start data generation if config file is valid and last run is finished.
        """
        if self.yaml_data == None:
            carb.log_error("Config file is invalid. Start data generation fails.")
            return

        if self.yaml_data_sim_length_valid == False:
            carb.log_error("Simulation Length is invalid. Start data generation fails.")
            return

        if self.yaml_data["global"]["simulation_length"] == 0:
            self._simulation_verification.verify_simulation(self.yaml_data)
            return

        # Do not allow multiple data generation running
        if self._dg:
            carb.log_warn("Last data generation is running. Please wait for it to complete.")
            return

        # Start data generation
        self._dg = DataGeneration(self.yaml_data)
        self._dg.register_recorder_done_callback(self.data_generation_done_callback)
        run_replicator_async = carb.settings.get_settings().get(
            "/persistent/exts/metspace/run_replicator_async"
        )
        if run_replicator_async:
            carb.log_warn("Running replicator in async mode")
            self._dg_task = asyncio.ensure_future(self._dg.start_recorder_async())
        else:
            self._dg_task = asyncio.ensure_future(self._dg.start_recorder_sync())

    # ========= Set Up Simulation by Config File =========

    def set_up_simulation_from_config_file(self):
        self.load_scene_from_config_file()

    def register_set_up_simulation_done_callback(self, on_event):
        sub = self._bus.create_subscription_to_push_by_type(SimulationManager.SET_UP_SIMULATION_DONE_EVENT, on_event)
        return sub

    def load_scene_from_config_file(self):
        """
        Load scene by config file and triggers load assets when scene is loaded.
        """
        if self.yaml_data["scene"]["asset_path"] != omni.usd.get_context().get_stage_url():
            # Load scene done callback
            def load_scene_from_config_file_callback(event):
                if event.type == int(omni.usd.StageEventType.ASSETS_LOADED):
                    # Release stage handle
                    self._load_stage_handle = None
                    # Load assets other than scene
                    self.load_assets_to_scene()

            # Subscribe stage event
            self._load_stage_handle = (
                omni.usd.get_context()
                .get_stage_event_stream()
                .create_subscription_to_pop(load_scene_from_config_file_callback)
            )
            try:
                StageUtil.open_stage(self.yaml_data["scene"]["asset_path"])
            except Exception as e:
                # Release event handle and do not procceed loading assets
                carb.log_error(
                    "Load scene ({0}) fails. No assets will be loaded.".format(self.yaml_data["scene"]["asset_path"])
                )
                self._load_stage_handle = None
        else:
            # Skip loading stage. Load assets other than scene
            carb.log_info("The current scene matches the scene path in config file. Scene loading is skipped.")
            self.load_assets_to_scene()

    def load_assets_to_scene(self):
        """
        Load assets into scene by config file after navmesh is ready
        """

        # Load assets other than scene
        # It first makes sure nav mesh is ready, then load cameras and characters in navmesh ready callback
        def nav_mesh_callback(event):
            if event.type == nav.EVENT_TYPE_NAVMESH_READY:
                # Release event handle
                self._nav_mesh_event_handle = None
                # When nav mesh is ready, load cameras and characters
                self.load_agents_cameras_from_config_file()
                # For headless mode
                if self.will_run_data_generation:
                    self.run_data_generation()
            elif event.type == nav.EVENT_TYPE_NAVMESH_BAKE_FAILED:
                carb.log_error("Navmesh baking fails. Will not proceed loading assets into scene.")
                self._nav_mesh_event_handle = None

        _nav = nav.nav.acquire_interface()
        # Do not proceed if navmesh volume does not exist
        if _nav.get_navmesh_volume_count() == 0:
            carb.log_error("Scene does not have navigation volume. Will not proceed loading assets into scene.")
            return

        _nav.start_navmesh_baking()
        self._nav_mesh_event_handle = _nav.get_navmesh_event_stream().create_subscription_to_pop(nav_mesh_callback)

    def load_agents_cameras_from_config_file(self):
        """
        Load characters, robots, cameras into scene by config file.
        Cameras are loaded at the end because their positions are affected by characters
        """
        # Clean last load
        self.reset_randomizers()
        RobotUtil.clean_robot_world()
        # Load assets
        self.load_robot_from_config_file()
        self.setup_anim_people_command_from_config_file()
        self.setup_anim_people_robot_command_from_config_file()
        self.load_setup_characters_from_config_file()
        #################### setup tracking elements ####################
        TrackManager.get_instance().setup_tracking_elements()
        TrackManager.get_instance().save_agent_target_pairs_prim_path(os.path.join(self.output_path, "agent_target_pairs.json"))
        #################### generate random command ####################
        self.load_camera_from_config_file()
        self.load_lidar_from_config_file()
        # When loading completes, reset states
        self._seed_is_updated = False
        self._filter_is_updated = False
        self._character_folder_is_updated = False
        self._command_file_is_updated = False
        self._robot_command_file_is_updated = False
        # Mark complete
        self._bus.push(SimulationManager.SET_UP_SIMULATION_DONE_EVENT)

    def load_robot_by_type(self, robot_type, randomizer):
        # Invalid values handle
        if self.yaml_data_robot_num_valid == False:
            carb.log_error(robot_type, " number is invalid. Loading robots fails.")
            return

        # Skipping the rest of the code when there is no loading happening
        # Early out if requested robots already exist
        robot_count = self.yaml_data["robot"][robot_type.lower() + "_num"]
        robot_count_in_stage = len(RobotUtil.get_robots_in_stage(robot_type))
        if robot_count <= robot_count_in_stage:
            return
        if self._seed_is_updated == True:
            randomizer.update_seed(self.yaml_data["global"]["seed"])
        # Make sure all required robots exist
        stage = omni.usd.get_context().get_stage()
        parent_path = self.robots_parent_prim_path
        randomizer.update_agent_positions(self._agent_positions)
        robot_pos_list = [randomizer.get_random_position(i) for i in range(robot_count)]
        self._agent_positions.extend(robot for robot in robot_pos_list if robot not in self._agent_positions)
        new_robot_list = []
        for i in range(robot_count):
            robot_name = RobotUtil.get_robot_name_by_index(robot_type, i)
            robot_path = parent_path + "/" + robot_name
            robot_prim = stage.GetPrimAtPath(robot_path)
            if not robot_prim.IsValid():
                robot_prim = RobotUtil.spawn_robot(robot_type, robot_pos_list[i], 0, robot_path)
                new_robot_list.append(robot_prim)

        # Make sure new spawned robot have been setup
        # Robot world setup is in its own behavior script
        self.setup_python_scripts_to_robot(new_robot_list, robot_type)

    def load_robot_from_config_file(self):
        """
        Load to enough robots by config file. Return if no load is needed.
        """
        # When section is missing
        if "robot" not in self.yaml_data:
            carb.log_info("'robot' section is missing. Skip loading robots.")
            return

        if self.yaml_data_seed_valid == False:
            carb.log_error("Randomization seed is invalid. Loading robots fails.")
            return

        # self.load_robot_by_type("Nova_Carter", self._nova_carter_randomizer)
        # self.load_robot_by_type("Transporter", self._transporter_randomizer)
        for robot_type in Settings.ROBOT_CATEGORY:
            self.load_robot_by_type(robot_type, self._robot_randomizers[robot_type])

    def load_camera_from_config_file(self):
        """
        Load to enough cameras by config file.
        Loaded camera will aim to one of the character if it is present
        Return if no load is needed.
        """
        # Invalid values handle
        if self.yaml_data_cam_num_valid == False:
            carb.log_error("Camera number is invalid. Loading cameras fails.")
            return
        if self.yaml_data_seed_valid == False:
            carb.log_error("Randomization seed is invalid. Loading cameras fails.")
            return
        if "camera_num" not in self.yaml_data["global"]:
            return
        # Update seed first
        if self._seed_is_updated == True:
            # Set up randomizer
            self._camera_randomizer.update_seed(self.yaml_data["global"]["seed"])
        # Make sure camera root prim exist
        stage = omni.usd.get_context().get_stage()
        cam_root_prim = stage.GetPrimAtPath(self.cameras_parent_prim_path)
        if not cam_root_prim.IsValid():
            omni.kit.commands.execute(
                "CreatePrimCommand", prim_type="Xform", prim_path=self.cameras_parent_prim_path, select_new_prim=False
            )
            cam_root_prim = stage.GetPrimAtPath(self.cameras_parent_prim_path)
        # Make sure all cameras indicated by camera_num exist

        cam_count = self.yaml_data["global"]["camera_num"]
        # if no camera need to be generated, return
        if cam_count == 0 or cam_count == -1:
            return

        cam_transform_dict = {}
        # we need to get current character list.
        character_prim_list = CharacterUtil.get_characters_root_in_stage(count_invisible=False)
        # get current characters in the scene
        character_list = [prim.GetPath() for prim in character_prim_list]
        # get randomized focallength for each camera
        focallengths = self._camera_randomizer.get_random_camera_focallength_list(cam_count)
        # calculate rotation and translation for each camera
        cam_transform_dict = self._camera_randomizer.get_random_position_rotation(cam_count, character_list)

        # get current camera in the stage:
        original_cam_list = CameraUtil.get_cameras_in_stage()
        if cam_count <= len(original_cam_list):
            return
        new_cam_list = []

        # spawn the extra cameras
        for i in range(cam_count - len(original_cam_list)):
            camera_prim = CameraUtil.spawn_camera(spawn_location=[0, 0, 0])
            new_cam_list.append(camera_prim)

        cam_list = new_cam_list + original_cam_list

        # sort the new camera list with camera name
        cam_list.sort(key=lambda camera: camera.GetName())

        # find all element in new cam list 's index in sorted cam_list
        new_cam_indices = [cam_list.index(camera) for camera in new_cam_list]

        # fetch the location and rotation for new generated cameras
        for i in new_cam_indices:
            target_camera = cam_list[i]
            focallength = focallengths[i]
            cam_pos, cam_rot = cam_transform_dict[i]
            CameraUtil.set_camera(target_camera, cam_pos, cam_rot, focallength)

    def load_lidar_from_config_file(self):
        """
        Load to enough lidar cameras by config file.
        Loaded camera will aim to one of the character if it is present
        Return if no load is needed.
        """
        # Invalid values handle
        if self.yaml_data_lidar_num_valid == False:
            carb.log_error("Lidar camera number is invalid. Loading lidar cameras fails.")
            return
        if self.yaml_data_seed_valid == False:
            carb.log_error("Randomization seed is invalid. Loading lidar cameras fails.")
            return
        if "lidar_num" not in self.yaml_data["global"]:
            return
        # Update first
        if self._seed_is_updated:
            # Set up randomizer
            self._lidar_camera_randomizer.update_seed(self.yaml_data["global"]["seed"])
        # Make sure all required lidar cameras exist
        stage = omni.usd.get_context().get_stage()
        parent_path = self.lidar_cameras_parent_prim_path
        lidar_count = self.yaml_data["global"]["lidar_num"]

        # if no extra lidar camera need to be generated, return
        if lidar_count == 0 or lidar_count == -1:
            return

        camera_parent_path = self.cameras_parent_prim_path

        for i in range(lidar_count):
            lidar_name = LidarCamUtil.get_lidar_name_by_index(i)
            lidar_path = parent_path + "/" + lidar_name
            lidar_prim = stage.GetPrimAtPath(lidar_path)

            if not lidar_prim.IsValid():
                # When there is no lidar camera prim with the same index:
                camera_name = CameraUtil.get_camera_name_by_index(i)
                camera_path = camera_parent_path + "/" + camera_name
                camera_prim = stage.GetPrimAtPath(camera_path)

                # spawn lidar camera only when there are camera with matching index exist in the stage.
                if camera_prim is not None and camera_prim.IsValid():
                    # if camera with same index exist
                    # generate lidar camera according to camera's translation rotaion and focallength
                    lidar_rot = camera_prim.GetAttribute("xformOp:orient").Get()
                    lidar_pos = camera_prim.GetAttribute("xformOp:translate").Get()
                    lidar_focalLength = camera_prim.GetAttribute("focalLength").Get()
                    lidar_prim = LidarCamUtil.spawn_lidar_camera(lidar_path, lidar_pos, lidar_rot, lidar_focalLength)

                else:
                    # send warning message to user when stage do not have matching camera exist
                    carb.log_error(
                        "Camera Prim : {camera_path} is not a valid prim. Lidar Camera {lidar_camera_path} is not generted. ".format(
                            camera_path=camera_path, lidar_camera_path=lidar_path
                        )
                    )

    def setup_anim_people_command_from_config_file(self):
        """
        Link character command file to omni.anim.people
        """
        if not self.yaml_data:
            return
        if "character" not in self.yaml_data:
            return
        target_path = self.parse_command_file_path(self.yaml_data["character"]["command_file"])
        carb.settings.get_settings().set(OMNI_ANIM_PEOPLE_COMMAND_PATH, target_path)

    def setup_anim_people_robot_command_from_config_file(self):
        """
        Link robot command file to omni.anim.people
        """
        if not self.yaml_data:
            return
        if "robot" not in self.yaml_data:
            return
        target_path = self.parse_command_file_path(self.yaml_data["robot"]["command_file"])
        carb.settings.get_settings().set(OMNI_ANIM_PEOPLE_ROBOT_COMMAND_PATH, target_path)

    def load_setup_characters_from_config_file(self):
        """
        Load to enough characters by config file and set them up.
        Load default biped character when needed.
        """
        # When section is missing
        if "character" not in self.yaml_data:
            carb.log_info("'character' section is missing. Skip loading characters.")
            return
        # Invalid values handle
        if self.yaml_data_character_num_valid == False:
            carb.log_error("Character number is invalid. Loading characters fails.")
            return
        if self.yaml_data_seed_valid == False:
            carb.log_error("Randomization seed is invalid. Loading characters fails.")
            return
        # Skipping the rest of the code when there is no loading happening
        character_count = self.yaml_data["character"]["num"]
        character_count_in_stage = len(CharacterUtil.get_characters_in_stage())
        if character_count <= character_count_in_stage:
            return
        # Set up character asset list
        if self._character_folder_is_updated:
            self.read_character_asset_list()
            self.refresh_available_character_asset_list()  # Character folder change triggers available character list refresh
        if len(self.available_character_list) == 0:
            carb.log_error("No character assets found.")
            return
        # Set up randomizer
        if self._seed_is_updated:
            self._character_randomizer.update_seed(self.yaml_data["global"]["seed"])
            self.refresh_available_character_asset_list()  # Seed change triggers available character list refresh
        # Set up available list
        if self._filter_is_updated and not self._seed_is_updated and not self._character_folder_is_updated:
            self.refresh_available_character_asset_list()
        # Make sure skeleton and animation loaded
        self.load_default_skeleton_and_animations()
        # Make sure all required characters exist
        stage = omni.usd.get_context().get_stage()
        parent_path = self.characters_parent_prim_path
        self._character_randomizer.update_agent_positions(self._agent_positions)
        character_pos_list = [self._character_randomizer.get_random_position(i) for i in range(character_count)]
        self._agent_positions.extend(
            character for character in character_pos_list if character not in self._agent_positions
        )
        new_character_skelroot_list = []
        new_character_path_list = []

        i, cnt, num_spawned = 0, 0, 0
        while cnt < character_count:
            character_name = CharacterUtil.get_character_name_by_index(cnt)
            character_path = parent_path + "/" + character_name
            character_prim = stage.GetPrimAtPath(character_path)
            # When prim is missing
            if not character_prim.IsValid():
                character_prim = self.spawn_character_by_idx(character_pos_list[cnt], 0, i)
                if character_prim:
                    # Successfully spawned
                    new_character_skelroot_list.append(CharacterUtil.get_character_skelroot_by_root(character_prim))
                    new_character_path_list.append(character_path)
                    cnt += 1
                    num_spawned += 1
                else:
                    if num_spawned == 0 and i >= character_count:
                        # the folder does not contain any usd file
                        carb.log_error("no usd file found in folder; no character spawned")
                        break
            else:
                cnt += 1
            i += 1
        # Make all spawned characters have been setup (anim graph, python script, semantic)
        self.setup_animation_graph_to_character(new_character_skelroot_list)
        self.setup_python_scripts_to_character(new_character_skelroot_list)
        DataGeneration.add_ppl_semantics(new_character_path_list)

    def reset_randomizers(self):
        self._agent_positions = []
        self._character_randomizer.reset()
        # self._nova_carter_randomizer.reset()
        # self._transporter_randomizer.reset()
        for robot_type in Settings.ROBOT_CATEGORY:
            self._robot_randomizers[robot_type].reset()
        self._camera_randomizer.reset()
        self._lidar_camera_randomizer.reset()

    def get_character_transition_matrix_as_text(self):
        transition_matrix = self._character_randomizer.transition_matrix
        max_width = max([len(command) for command in transition_matrix.keys()]) + 2
        text = "".rjust(max_width)
        text += "".join([str(command).rjust(max_width) for command in transition_matrix.keys()])
        text += "\n"
        for command in transition_matrix.keys():
            text += (
                str(command).rjust(max_width)
                + "".join([str(num).rjust(max_width) for num in transition_matrix[command]])
                + "\n"
            )
        return text

    def set_character_transition_matrix(self, data):
        self._character_randomizer.update_transition_matrix(data)

    def is_valid_transition_matrix(self, data):
        for command in data.keys():
            if abs(sum(data[command]) - 1.0) > 0.0001:
                return False
        return True
