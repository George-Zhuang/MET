import os 
import carb
import omni.kit
from omni.isaac.nucleus import get_assets_root_path, get_assets_root_path_async


class Settings:
    """
    Manager class to handle general settings
    """
    ROBOT_CATEGORY = ['nova_carter', 'transporter', 'carter_v1',  'dingo', 'jetbot']
    EXTEND_DATA_GENERATION_LENGTH = "/exts/metspace/extend_data_generation_length"

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

    DEFAULT_BIPED_ASSET_PATH = "/exts/metspace/asset_settings/default_biped_assets_path"
    DEFAULT_SCENE_PATH = "/exts/metspace/asset_settings/default_scene_path"
    DEFAULT_CHARACTER_PATH = "/exts/metspace/asset_settings/default_character_asset_path"
    DEFAULT_NOVA_CARTER_PATH = "/exts/metspace/asset_settings/default_nova_carter_asset_path"
    DEFAULT_TRANSPORTER_PATH = "/exts/metspace/asset_settings/default_transporter_asset_path"
    DEFAULT_CARTER_V1_PATH = "/exts/metspace/asset_settings/default_carter_v1_asset_path"
    DEFAULT_DINGO_PATH = "/exts/metspace/asset_settings/default_dingo_asset_path"
    DEFAULT_JETBOT_PATH = "/exts/metspace/asset_settings/default_jetbot_asset_path"
    cache_biped_asset_path = ""
    cache_scene_asset_path = ""
    cache_default_nova_carter_path = ""
    cache_default_transporter_path = ""
    cache_default_carter_v1_path = ""
    cache_default_dingo_path = ""
    cache_default_jetbot_path = ""
    cache_default_character_path = ""

    # ====== Local Paths ======

    # DEFAULT_BEHAVIOR_SCRIPT_PATH = "/exts/metspace/behavior_script_settings/behavior_script_path"
    # DEFAULT_ROBOT_BEHAVIOR_SCRIPT_PATH = (
    #     "/exts/metspace/behavior_script_settings/robot_behavior_script_path"
    # )
    DEFAULT_BEHAVIOR_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts/character_behavior.py")
    DEFAULT_ROBOT_BEHAVIOR_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts/robot_behavior.py")

    async def cache_paths_async():
        AssetPaths.cache_asset_default_root_path = await get_assets_root_path_async()

        AssetPaths.cache_biped_asset_path = await AssetPaths.__get_remote_asset_path_by_key_async(
            AssetPaths.DEFAULT_BIPED_ASSET_PATH, "/Isaac/People/Characters/Biped_Setup.usd"
        )
        AssetPaths.cache_scene_asset_path = await AssetPaths.__get_remote_asset_path_by_key_async(
            AssetPaths.DEFAULT_SCENE_PATH, "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
        )
        AssetPaths.cache_default_character_path = await AssetPaths.__get_remote_asset_path_by_key_async(
            AssetPaths.DEFAULT_CHARACTER_PATH, "/Isaac/People/Characters/"
        )
        AssetPaths.cache_default_nova_carter_path = await AssetPaths.__get_remote_asset_path_by_key_async(
            AssetPaths.DEFAULT_NOVA_CARTER_PATH, "/Isaac/Robots/Carter/nova_carter_sensors.usd"
        )
        AssetPaths.cache_default_transporter_path = await AssetPaths.__get_remote_asset_path_by_key_async(
            AssetPaths.DEFAULT_TRANSPORTER_PATH, "/Isaac/Robots/Transporter/transporter_sensors.usd"
        )
        AssetPaths.cache_default_carter_v1_path = await AssetPaths.__get_remote_asset_path_by_key_async(
            AssetPaths.DEFAULT_CARTER_V1_PATH, "/Isaac/Robots/Carter/carter_v1.usd"
        )
        AssetPaths.cache_default_dingo_path = await AssetPaths.__get_remote_asset_path_by_key_async(
            AssetPaths.DEFAULT_DINGO_PATH, "/Isaac/Robots/Clearpath/Dingo/dingo.usd"
        )
        AssetPaths.cache_default_jetbot_path = await AssetPaths.__get_remote_asset_path_by_key_async(
            AssetPaths.DEFAULT_JETBOT_PATH, "/Isaac/Robots/Jetbot/jetbot.usd"
        )
        carb.log_info("AssetPaths cache remote pahts done.")

    def default_biped_asset_path():
        if AssetPaths.cache_biped_asset_path:
            return AssetPaths.cache_biped_asset_path
        else:
            return AssetPaths.__get_remote_asset_path_by_key(
                AssetPaths.DEFAULT_BIPED_ASSET_PATH, "/Isaac/People/Characters/Biped_Setup.usd"
            )

    def default_biped_asset_name():
        asset_path = None
        if AssetPaths.cache_biped_asset_path:
            asset_path = AssetPaths.cache_biped_asset_path
        else:
            asset_path = AssetPaths.default_biped_asset_path()
        return str(asset_path).split("/")[-1].replace(".usd", "").replace(".usda", "")

    def default_scene_path():
        if AssetPaths.cache_scene_asset_path:
            return AssetPaths.cache_scene_asset_path
        else:
            return AssetPaths.__get_remote_asset_path_by_key(
                AssetPaths.DEFAULT_SCENE_PATH, "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
            )

    def default_character_path():
        if AssetPaths.cache_default_character_path:
            return AssetPaths.cache_default_character_path
        else:
            return AssetPaths.__get_remote_asset_path_by_key(
                AssetPaths.DEFAULT_CHARACTER_PATH, "/Isaac/People/Characters/"
            )

    def default_nova_carter_path():
        if AssetPaths.cache_default_nova_carter_path:
            return AssetPaths.cache_default_nova_carter_path
        else:
            return AssetPaths.__get_remote_asset_path_by_key(
                AssetPaths.DEFAULT_NOVA_CARTER_PATH, "/Isaac/Robots/Carter/nova_carter_sensors.usd"
            )

    def default_transpoter_path():
        if AssetPaths.cache_default_transporter_path:
            return AssetPaths.cache_default_transporter_path
        else:
            return AssetPaths.__get_remote_asset_path_by_key(
                AssetPaths.DEFAULT_TRANSPORTER_PATH, "/Isaac/Robots/Transporter/transporter_sensors.usd"
            )

    def default_carter_v1_path():
        if AssetPaths.cache_default_carter_v1_path:
            return AssetPaths.cache_default_carter_v1_path
        else:
            return AssetPaths.__get_remote_asset_path_by_key(
                AssetPaths.DEFAULT_CARTER_V1_PATH, "/Isaac/Robots/Carter/carter_v1.usd"
            )
    
    def default_dingo_path():
        if AssetPaths.cache_default_dingo_path:
            return AssetPaths.cache_default_dingo_path
        else:
            return AssetPaths.__get_remote_asset_path_by_key(
                AssetPaths.DEFAULT_DINGO_PATH, "/Isaac/Robots/Clearpath/Dingo/dingo.usd"
            )

    def default_jetbot_path():
        if AssetPaths.cache_default_jetbot_path:
            return AssetPaths.cache_default_jetbot_path
        else:
            return AssetPaths.__get_remote_asset_path_by_key(
                AssetPaths.DEFAULT_JETBOT_PATH, "/Isaac/Robots/Jetbot/jetbot.usd"
            )

    def behavior_script_path():
        default_path = (
            omni.kit.app.get_app().get_extension_manager().get_extension_path_by_module("omni.anim.people")
            + "/omni/anim/people/scripts/character_behavior.py"
        )
        return AssetPaths.__get_local_asset_path_by_key(AssetPaths.DEFAULT_BEHAVIOR_SCRIPT_PATH, default_path)

    def robot_behavior_script_path():
        # default_path = (
        #     omni.kit.app.get_app().get_extension_manager().get_extension_path_by_module("omni.anim.people")
        #     + "/omni/anim/people/scripts/robot_behavior.py"
        # )
        # return AssetPaths.__get_local_asset_path_by_key(AssetPaths.DEFAULT_ROBOT_BEHAVIOR_SCRIPT_PATH, default_path)
        return AssetPaths.DEFAULT_ROBOT_BEHAVIOR_SCRIPT_PATH

    def __get_local_asset_path_by_key(key, default_path):
        path = carb.settings.get_settings().get(key)
        if not path:
            path = default_path
        return path

    def __get_remote_asset_path_by_key(key, default_path_in_root):
        path = carb.settings.get_settings().get(key)
        if not path:
            cache_root_path = AssetPaths.cache_asset_default_root_path
            if not cache_root_path:
                path = get_assets_root_path()
                AssetPaths.cache_asset_default_root_path = path
            else:
                path = cache_root_path
            if not path:
                carb.log_error("Get asset path fails.")
                return None
            return path + default_path_in_root
        else:
            return path

    async def __get_remote_asset_path_by_key_async(key, default_path_in_root):
        path = carb.settings.get_settings().get(key)
        if not path:
            cache_root_path = AssetPaths.cache_asset_default_root_path
            if not cache_root_path:
                path = await get_assets_root_path_async()
            else:
                path = cache_root_path
                AssetPaths.cache_asset_default_root_path = path
            if not path:
                carb.log_error("Get asset path fails.")
                return None
            return path + default_path_in_root
        else:
            return path


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
