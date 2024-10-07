import os
import random
from collections import OrderedDict
from datetime import datetime

import carb

from .extension import Main as MainExtension
from .extension import get_ext_path
from .file_util import YamlFileUtil
from .settings import AssetPaths
from .writers import *


class ConfigFileUtil:

    CONFIG_HEADER = "/persistent/exts/metspace/header"

    DEFAULT_CONFIG_VALUES_PATH = "data/config_file/config_file_default_values.yaml"

    def remove_config_file_header(raw_yaml):
        header_name = carb.settings.get_settings().get(ConfigFileUtil.CONFIG_HEADER)
        if header_name not in raw_yaml:
            carb.log_error("Yaml file does not contain '{0}' header.".format(header_name))
            return None
        return raw_yaml[header_name]

    def add_config_file_header(yaml_data):
        header_name = carb.settings.get_settings().get(ConfigFileUtil.CONFIG_HEADER)
        raw_yaml = {}
        raw_yaml[header_name] = yaml_data
        return raw_yaml

    def setup_config_file(yaml_data):
        """
        Set up a yaml data to be a proper config file, including filling up missing sections and empty fields.
        Return True if yaml_data is modified, return False if no change is applied.
        """
        default_yaml = ConfigFileUtil.get_config_file_default_values()
        return ConfigFileUtil._setup_config_file_with_default_value(yaml_data, default_yaml)

    def _setup_config_file_with_default_value(yaml_data, default_yaml):
        # Empty config file handle
        if yaml_data == None:
            return False
        # First check if all required sections are present in the config file
        # Add default sections for the missing.
        # 'character' and 'robot' section are optional
        use_default_value = False
        if "global" not in yaml_data:
            carb.log_info("'global' section is missing. Default global section is added.")
            yaml_data["global"] = {}
            use_default_value = True
        if "scene" not in yaml_data:
            carb.log_info("'scene' section is missing. Default scene section is added.")
            yaml_data["scene"] = {}
            use_default_value = True
        if "replicator" not in yaml_data:
            carb.log_info("'replicator' section is missing. Default replicator section is added.")
            yaml_data["replicator"] = {}
            use_default_value = True
        if "parameters" not in yaml_data["replicator"]:
            carb.log_info("'parameters' in replicator section is missing. Default parameters are added.")
            yaml_data["replicator"]["parameters"] = {}
            use_default_value = True

        def try_apply_default_value(panel, property, default_val):
            if not yaml_data[panel]:
                yaml_data[panel] = {}
            if property not in yaml_data[panel] or yaml_data[panel][property] == None:
                yaml_data[panel][property] = default_val
                carb.log_info(
                    "New value is applied to the config file [{0}][{1}]: {2}.".format(panel, property, default_val)
                )
                return True
            return False

        # Global settings
        use_default_value |= try_apply_default_value("global", "seed", default_yaml["global"]["seed"])
        use_default_value |= try_apply_default_value(
            "global", "simulation_length", default_yaml["global"]["simulation_length"]
        )
        # Sensor settings
        if "camera_list" in yaml_data["global"]:
            use_default_value |= try_apply_default_value("global", "camera_list", default_yaml["global"]["camera_list"])
            use_default_value |= try_apply_default_value("global", "lidar_list", default_yaml["global"]["lidar_list"])
        else:
            use_default_value |= try_apply_default_value("global", "camera_num", default_yaml["global"]["camera_num"])
            use_default_value |= try_apply_default_value("global", "lidar_num", default_yaml["global"]["lidar_num"])

        # Scene settings
        use_default_value |= try_apply_default_value("scene", "asset_path", default_yaml["scene"]["asset_path"])
        # Character settings
        if "character" in yaml_data:
            use_default_value |= try_apply_default_value(
                "character", "asset_path", default_yaml["character"]["asset_path"]
            )
            use_default_value |= try_apply_default_value("character", "num", default_yaml["character"]["num"])
            use_default_value |= try_apply_default_value("character", "filters", default_yaml["character"]["filters"])
            use_default_command_file = try_apply_default_value(
                "character", "command_file", default_yaml["character"]["command_file"]
            )
            if use_default_command_file == True:
                # Create command file at default location
                # TODO:: create command file
                pass
            use_default_value |= use_default_command_file
        # Robot settings
        if "robot" in yaml_data:
            use_default_value |= try_apply_default_value(
                "robot", "nova_carter_num", default_yaml["robot"]["nova_carter_num"]
            )
            use_default_value |= try_apply_default_value(
                "robot", "transporter_num", default_yaml["robot"]["transporter_num"]
            )
            use_default_value |= try_apply_default_value("robot", "write_data", default_yaml["robot"]["write_data"])
            use_default_command_file = try_apply_default_value(
                "robot", "command_file", default_yaml["robot"]["command_file"]
            )
            if use_default_command_file == True:
                # Create command file at current config file location
                # TODO:: create command file
                pass
            use_default_value |= use_default_command_file
        # Replicator settings (writer and parameters)
        use_default_value |= try_apply_default_value("replicator", "writer", default_yaml["replicator"]["writer"])
        writers_parameters = ConfigFileUtil.get_writers_params_values()
        writer = yaml_data["replicator"]["writer"]
        if writer in writers_parameters.keys():
            # Apply default values for empty section or empty fields
            if yaml_data["replicator"]["parameters"] == None:
                yaml_data["replicator"]["parameters"] = writers_parameters[writer]
                use_default_value = True
            else:
                for param, default_value in writers_parameters[writer].items():
                    if (
                        (yaml_data["replicator"]["parameters"] == None)
                        or (param not in yaml_data["replicator"]["parameters"])
                        or (yaml_data["replicator"]["parameters"][param] == None)
                    ):
                        yaml_data["replicator"]["parameters"][param] = default_value
                        carb.log_info("{0} will use default value : {1}".format(str(param), str(default_value)))
                        use_default_value = True
        return use_default_value

    def check_config_file_version(yaml_data):
        # Check if version attribute exists
        if "version" not in yaml_data:
            carb.log_error("version info is missing.")
            return False
        # Check if major version matches
        major = str(MainExtension.ext_version).split(".")[0]
        yaml_major = str(yaml_data["version"]).split(".")[0]
        if major != yaml_major:
            carb.log_error("Invalid config file version. The version must match with the current extension version.")
            return False
        return True

    def check_config_file_property(yaml_data):
        """
        Verify if a config file's properties are valid.
        This include mutual exclusive properties and maybe must-have properties in the future.
        """
        if "global" in yaml_data:
            # Check camera and lidar exclusive properties
            if "camera_num" in yaml_data["global"] and "camera_list" in yaml_data["global"]:
                carb.log_error(" 'camera_num' and ''camera_list' is mutual exclusive. Config file is not valid.")
                return False
            if "lidar_num" in yaml_data["global"] and "lidar_list" in yaml_data["global"]:
                carb.log_error(" 'lidar_num' and ''lidar_list' is mutual exclusive. Config file is not valid.")
                return False
            if "camera_num" in yaml_data["global"] and "lidar_list" in yaml_data["global"]:
                carb.log_error(
                    " 'camera_num' cannot pair with ''lidar_list'. Please use 'lidar_num' instead. Config file is not valid."
                )
            if "camera_list" in yaml_data["global"] and "lidar_num" in yaml_data["global"]:
                carb.log_error(
                    " 'camera_list' cannot pair with ''lidar_num'. Please use 'lidar_list' instead. Config file is not valid."
                )
        return True

    def create_config_file(path, overwrites=None):
        """
        Create a config file at given path with default values.
        Overwrite some values if needed.
            - overwrite_content_list: a list of [section, property, new_content] to replace the default values
        """
        # Get default config file values
        yaml_data = ConfigFileUtil.get_config_file_default_values()
        # Temperary solution to handle exclusive values
        if "camera_list" in yaml_data["global"]:
            yaml_data["global"].pop("camera_list", None)
        if "lidar_list" in yaml_data["global"]:
            yaml_data["global"].pop("lidar_list", None)
        # Overwrite some values
        if overwrites:
            for overwrite in overwrites:
                section = overwrite[0]
                property = overwrite[1]
                new_content = overwrite[2]
                yaml_data[section][property] = new_content
        # Complete a full yaml data and save
        yaml_data["version"] = MainExtension.ext_version
        raw_yaml = ConfigFileUtil.add_config_file_header(yaml_data)
        return YamlFileUtil.save_yaml(path, raw_yaml)

    # Config file default values

    def get_random_seed():
        """
        Generate a random seed based on current timestamp
        Python 3 int has no max limits. Here we use limit in most popular languages (2147483647) to avoid potential problems
        """
        random.seed(datetime.now().timestamp())
        return random.randrange(0, 2147483647)

    def get_config_file_default_values():
        """
        Grab all default values for config file
        """
        # Load default value file
        path = f"{get_ext_path()}/{ConfigFileUtil.DEFAULT_CONFIG_VALUES_PATH}"
        default_yaml = YamlFileUtil.load_yaml(path)
        # Code generated seed
        default_yaml["global"]["seed"] = ConfigFileUtil.get_random_seed()
        # Asset paths
        default_yaml["scene"]["asset_path"] = AssetPaths.default_scene_path()
        default_yaml["character"]["asset_path"] = AssetPaths.default_character_path()

        return default_yaml

    def get_default_config_file_path():
        path = get_ext_path()
        path += "/config/default_config.yaml"
        return path

    # Default values in writer parameters

    def get_writers_params_values():
        # Get all writer parameter names and default values
        writer_dict = OrderedDict()
        # BasicWriter is from Replicator so here we only specify the annotators we are interested in.
        writer_dict["BasicWriter"] = OrderedDict()
        writer_dict["BasicWriter"]["output_dir"] = os.path.abspath(
            carb.settings.get_settings().get("/exts/metspace/default_replicator_output_path")
        )
        writer_dict["BasicWriter"]["rgb"] = True
        writer_dict["BasicWriter"]["bounding_box_2d_tight"] = True
        writer_dict["BasicWriter"]["bounding_box_2d_loose"] = False
        writer_dict["BasicWriter"]["bounding_box_3d"] = False
        writer_dict["BasicWriter"]["semantic_segmentation"] = False

        writer_dict["TaoWriter"] = TaoWriter.params_values()
        writer_dict["LidarWriter"] = LidarWriter.params_values()
        writer_dict["LidarFusionWriter"] = LidarFusionWriter.params_values()
        writer_dict["RTSPWriter"] = RTSPWriter.params_values()
        writer_dict["ObjectronWriter"] = ObjectronWriter.params_values()

        return writer_dict

    def get_writers_params_labels():
        # Get all writers parameter names and labels to display in UI
        writer_dict = OrderedDict()
        writer_dict["BasicWriter"] = {
            "output_dir": "output_dir",
            "rgb": "rgb",
            "bounding_box_2d_tight": "bounding_box_2d_tight",
            "bounding_box_2d_loose": "bounding_box_2d_loose",
            "bounding_box_3d": "bounding_box_3d",
            "semantic_segmentation": "semantic_segmentation",
        }
        writer_dict["TaoWriter"] = TaoWriter.params_labels()
        writer_dict["LidarWriter"] = LidarWriter.params_labels()
        writer_dict["LidarFusionWriter"] = LidarFusionWriter.params_labels()
        writer_dict["RTSPWriter"] = RTSPWriter.params_labels()
        writer_dict["ObjectronWriter"] = ObjectronWriter.params_labels()
        return writer_dict

    def get_writer_tooltips():
        writer_dict = OrderedDict()
        writer_dict["BasicWriter"] = ""
        writer_dict["TaoWriter"] = TaoWriter.tooltip()
        writer_dict["LidarWriter"] = ""
        writer_dict["LidarFusionWriter"] = LidarFusionWriter.tooltip()
        writer_dict["RTSPWriter"] = RTSPWriter.tooltip()
        writer_dict["ObjectronWriter"] = ""
        return writer_dict
