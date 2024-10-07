import os
import carb
import omni.kit
from omni.isaac.nucleus import get_assets_root_path, get_assets_root_path_async


class Settings:
    """
    Manager class to handle general settings
    """

    EXTEND_DATA_GENERATION_LENGTH = "/exts/metspace/extend_data_generation_length"
    SAVE_ROOT_PATH = "/isaac-sim/tmp"
    ASSETS_ROOT_PATH = "/isaac-sim/Assets/Isaac/4.0"
    PROMPT_PATH = "/isaac-sim/Assets/Isaac/4.0/Isaac/People/character.json"
    ROBOT_CATEGORY = ['nova_carter', 'transporter', 'carter_v1',  'dingo', 'jetbot']

    def extend_data_generation_length():
        return Settings.__get_value_by_key(Settings.EXTEND_DATA_GENERATION_LENGTH, 0)

    def __get_value_by_key(key, default_val):
        val = carb.settings.get_settings().get(key)
        if not val:
            val = default_val
        return val


class AssetPaths:
    """
    Manager class to handle all asset paths.
    It first reads path from the extension setting. If not present, it will read a default value instead.
    - For remote path, the default value is a path in the Neuclus Server.
        - All default remote paths are cached during extension start up.
    - For local path, the default value is a path locally.
    """

    cache_asset_default_root_path = None

    # ====== Remote Paths ======

    DEFAULT_BIPED_ASSET_PATH = os.path.join(Settings.ASSETS_ROOT_PATH, "Isaac/People/Characters/Biped_Setup.usd") 
    DEFAULT_SCENE_PATH = os.path.join(Settings.ASSETS_ROOT_PATH, "Isaac/Environments/Simple_Warehouse/warehouse_01.usd")
    DEFAULT_CHARACTER_PATH = os.path.join(Settings.ASSETS_ROOT_PATH, "Isaac/People/Characters/")
    DEFAULT_NOVA_CARTER_PATH = os.path.join(Settings.ASSETS_ROOT_PATH, "Isaac/Robots/Carter/nova_carter_sensors.usd")
    DEFAULT_TRANSPORTER_PATH = os.path.join(Settings.ASSETS_ROOT_PATH, "Isaac/Robots/Transporter/transporter_sensors.usd")
    DEFAULT_CARTER_V1_PATH = os.path.join(Settings.ASSETS_ROOT_PATH, "Isaac/Robots/Carter/carter_v1.usd")
    DEFAULT_DINGO_PATH = os.path.join(Settings.ASSETS_ROOT_PATH, "Isaac/Robots/Clearpath/Dingo/dingo.usd")
    DEFAULT_JETBOT_PATH = os.path.join(Settings.ASSETS_ROOT_PATH, "Isaac/Robots/Jetbot/jetbot.usd")
    cache_biped_asset_path = ""
    cache_scene_asset_path = ""
    cache_default_nova_carter_path = ""
    cache_default_transporter_path = ""
    cache_default_character_path = ""
    cache_default_carter_v1_path = ""
    cache_default_dingo_path = ""
    cache_default_jetbot_path = ""

    # ====== Local Paths ======

    # DEFAULT_BEHAVIOR_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts/character_behavior.py")
    # DEFAULT_ROBOT_BEHAVIOR_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts/robot_behavior.py")

    DEFAULT_BEHAVIOR_SCRIPT_PATH = "/exts/metspace/behavior_script_settings/behavior_script_path"
    DEFAULT_ROBOT_BEHAVIOR_SCRIPT_PATH = (
        "/exts/metspace/behavior_script_settings/robot_behavior_script_path"
    )

    async def cache_paths_async():
        AssetPaths.cache_asset_default_root_path = Settings.ASSETS_ROOT_PATH

        AssetPaths.cache_biped_asset_path = AssetPaths.DEFAULT_BIPED_ASSET_PATH
        AssetPaths.cache_scene_asset_path = AssetPaths.DEFAULT_SCENE_PATH
        AssetPaths.cache_default_character_path = AssetPaths.DEFAULT_CHARACTER_PATH
        AssetPaths.cache_default_nova_carter_path = AssetPaths.DEFAULT_NOVA_CARTER_PATH
        AssetPaths.cache_default_transporter_path = AssetPaths.DEFAULT_TRANSPORTER_PATH
        AssetPaths.cache_default_carter_v1_path = AssetPaths.DEFAULT_CARTER_V1_PATH
        AssetPaths.cache_default_dingo_path = AssetPaths.DEFAULT_DINGO_PATH
        AssetPaths.cache_default_jetbot_path = AssetPaths.DEFAULT_JETBOT_PATH
        carb.log_info("AssetPaths cache remote pahts done.")

    def default_biped_asset_path():
        return AssetPaths.DEFAULT_BIPED_ASSET_PATH

    def default_biped_asset_name():
        return "Biped_Setup"
    
    def default_scene_path():
        return AssetPaths.DEFAULT_SCENE_PATH
    
    def default_character_path():
        return AssetPaths.DEFAULT_CHARACTER_PATH

    def default_nova_carter_path():
        return AssetPaths.DEFAULT_NOVA_CARTER_PATH

    def default_transpoter_path():
        return AssetPaths.DEFAULT_TRANSPORTER_PATH
    
    def default_carter_v1_path():
        return AssetPaths.DEFAULT_CARTER_V1_PATH

    def default_dingo_path():
        return AssetPaths.DEFAULT_DINGO_PATH

    def default_jetbot_path():
        carb.log_warn(f"default_jetbot_path: {AssetPaths.DEFAULT_JETBOT_PATH}")
        return AssetPaths.DEFAULT_JETBOT_PATH

    def behavior_script_path():
        return AssetPaths.DEFAULT_BEHAVIOR_SCRIPT_PATH

    def robot_behavior_script_path():
        return AssetPaths.DEFAULT_ROBOT_BEHAVIOR_SCRIPT_PATH


class PrimPaths:
    """
    Manager class to handle all prim paths
    """

    CHARACTERS_PARENT_PATH = "/exts/metspace/characters_parent_prim_path"
    ROBOTS_PARENT_PATH = "/exts/metspace/robots_parent_prim_path"
    CAMERAS_PARENT_PATH = "/exts/metspace/cameras_parent_prim_path"
    LIDAR_CAMERAS_PARENT_PATH = "/exts/metspace/lidar_cameras_parent_prim_path"

    def default_biped_prim_path():
        biped_name = AssetPaths.default_biped_asset_name()
        return "{}/{}".format(PrimPaths.characters_parent_path(), biped_name)

    def characters_parent_path():
        return PrimPaths.__get_path_by_key(PrimPaths.CHARACTERS_PARENT_PATH, "/World/Characters")

    def robots_parent_path():
        return PrimPaths.__get_path_by_key(PrimPaths.ROBOTS_PARENT_PATH, "/World/Robots")

    def cameras_parent_path():
        return PrimPaths.__get_path_by_key(PrimPaths.CAMERAS_PARENT_PATH, "/World/Cameras")

    def lidar_cameras_parent_path():
        return PrimPaths.__get_path_by_key(PrimPaths.LIDAR_CAMERAS_PARENT_PATH, "/World/Lidars")

    def __get_path_by_key(key, default_path):
        path = carb.settings.get_settings().get(key)
        if not path:
            path = default_path
        return path
