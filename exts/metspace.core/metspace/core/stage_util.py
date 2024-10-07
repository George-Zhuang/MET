import asyncio

import carb
import omni.anim.navigation.core as nav
import omni.usd
import omni.anim.graph.core as ag
from omni.anim.people.scripts.utils import Utils
from omni.isaac.core import World
from omni.isaac.core.utils import prims
from omni.isaac.core.utils.rotations import lookat_to_quatf
from omni.isaac.core.utils.stage import create_new_stage_async, update_stage_async
from pxr import Gf, Sdf, Usd, UsdGeom

from .randomization.randomizer_util import RandomizerUtil
from .settings import AssetPaths, PrimPaths


class StageUtil:
    def open_stage(usd_path: str, ignore_unsave=True):
        if not Usd.Stage.IsSupportedFile(usd_path):
            raise ValueError("Only USD files can be loaded")
        import carb.settings
        import omni.kit.window.file

        IGNORE_UNSAVED_CONFIG_KEY = "/app/file/ignoreUnsavedStage"
        old_val = carb.settings.get_settings().get(IGNORE_UNSAVED_CONFIG_KEY)
        carb.settings.get_settings().set(IGNORE_UNSAVED_CONFIG_KEY, ignore_unsave)
        omni.kit.window.file.open_stage(usd_path, omni.usd.UsdContextInitialLoadSet.LOAD_ALL)
        carb.settings.get_settings().set(IGNORE_UNSAVED_CONFIG_KEY, old_val)

    def get_prim_pos(prim):
        if prim:
            matrix = omni.usd.get_world_transform_matrix(prim)
            return matrix.ExtractTranslation()
        else:
            carb.log_error("Invalid prim")
            return None

    def get_prim_rot_quat(prim):
        if prim:
            gf_rotation = UsdGeom.XformCache().GetLocalToWorldTransform(prim).ExtractRotationQuat()
            return gf_rotation
        else:
            carb.log_error("Invalid prim")
            return None

    # Set the xform transformation type to be Scale, Orient, Trans, and return the original order
    def set_xformOpType_SOT():
        xformoptype_setting_path = "/persistent/app/primCreation/DefaultXformOpType"
        original_xform_order_setting = carb.settings.get_settings().get(xformoptype_setting_path)
        carb.settings.get_settings().set(xformoptype_setting_path, "Scale, Orient, Translate")
        return original_xform_order_setting

    def recover_xformOpType(original_xform_order_setting):
        xformoptype_setting_path = "/persistent/app/primCreation/DefaultXformOpType"
        carb.settings.get_settings().set(xformoptype_setting_path, original_xform_order_setting)


class CameraUtil:
    def get_camera_name_by_index(i):
        if i == 0:
            return "Camera"
        elif i < 10:
            return "Camera_0" + str(i)
        else:
            return "Camera_" + str(i)

    def has_a_valid_name(name):
        if name == "Camera":
            return True
        # if name starts with "Camera"
        if name.startswith("Camera_"):
            return True
        return False

    def get_camera_name(camera_prim):
        return camera_prim.GetName()

    def get_camera_name_without_prefix(camera_prim):
        camera_name = None
        name = CameraUtil.get_camera_name(camera_prim)
        if name != None and CameraUtil.has_a_valid_name(name):
            if name == "Camera":
                return ""
            camera_name = name.split("_")[1]
        # camera_name will be None if invalid
        return camera_name

    def get_cameras_in_stage():
        camera_list = []
        # get camera root prim in the stage:
        camera_root_prim = CameraUtil.get_camera_root_prim()

        # if the camera root prim is not valid: return an emtpy list
        if camera_root_prim is None:
            return camera_list

        # all child camera prim would be added to the list:
        for camera_prim in camera_root_prim.GetChildren():
            if camera_prim.GetTypeName() == "Camera":
                camera_list.append(camera_prim)

        # then we sorted the camera prim base on their prim Name
        camera_list = sorted(camera_list, key=lambda camera: camera.GetName())
        return camera_list

    def set_camera(camera_prim, spawn_location=None, spawn_rotation=None, focallength=None):
        if spawn_location is None:
            spawn_location = Gf.Vec3d(0.0)

        if (not RandomizerUtil.do_aim_camera_to_character()) or (spawn_rotation is None):
            # Camera height is fixed in 5 by default
            camera_pos = Gf.Vec3d(spawn_location[0], spawn_location[1], 5)
        else:
            camera_pos = Gf.Vec3d(spawn_location[0], spawn_location[1], spawn_location[2])

        if spawn_rotation is None:
            # Camera will be always looking at the origin when it spawns
            spawn_rotation = Gf.Quatd(lookat_to_quatf(Gf.Vec3d(0.0), camera_pos, Gf.Vec3d(0, 0, 1)))

        camera_prim.GetAttribute("xformOp:orient").Set(spawn_rotation)
        camera_prim.GetAttribute("xformOp:translate").Set(camera_pos)

        if focallength is not None:
            camera_prim.GetAttribute("focalLength").Set(focallength)

    def spawn_camera(spawn_path=None, spawn_location=None, spawn_rotation=None, focallength=None):
        # set xformOp order to Scale, Orient, Translate, and store the setting
        original_xform_order_setting = StageUtil.set_xformOpType_SOT()
        stage = omni.usd.get_context().get_stage()
        camera_path = ""
        if spawn_path:
            camera_path = spawn_path
        else:
            camera_path = Sdf.Path(
                omni.usd.get_stage_next_free_path(stage, PrimPaths.cameras_parent_path() + "/Camera", False)
            )

        omni.kit.commands.execute("CreatePrimCommand", prim_type="Camera", prim_path=camera_path, select_new_prim=False)
        camera_prim = stage.GetPrimAtPath(camera_path)
        CameraUtil.set_camera(camera_prim, spawn_location, spawn_rotation, focallength)

        # set the xform setting back to original value
        StageUtil.recover_xformOpType(original_xform_order_setting)
        return camera_prim

    def delete_camera_prim(cam_name):
        stage = omni.usd.get_context().get_stage()
        if not Sdf.Path.IsValidPathString(PrimPaths.cameras_parent_path()):
            carb.log_error(str(PrimPaths.cameras_parent_path()) + "is not a valid prim path")
            return
        camera_prim = stage.GetPrimAtPath("{}/{}".format(PrimPaths.cameras_parent_path(), cam_name))
        if camera_prim and camera_prim.IsValid() and camera_prim.IsActive():
            prims.delete_prim(camera_prim.GetPath())

    def delete_camera_prims():
        camera_root_prim = CameraUtil.get_camera_root_prim()
        for camera_prim in camera_root_prim.GetChildren():
            if camera_prim and camera_prim.IsValid() and camera_prim.IsActive():
                prims.delete_prim(camera_prim.GetPath())

    def get_camera_root_prim():
        stage = omni.usd.get_context().get_stage()
        if not Sdf.Path.IsValidPathString(PrimPaths.cameras_parent_path()):
            carb.log_error(str(PrimPaths.cameras_parent_path()) + "is not a valid prim path")
            return None
        camera_root_prim = stage.GetPrimAtPath(PrimPaths.cameras_parent_path())
        if camera_root_prim and camera_root_prim.IsValid() and camera_root_prim.IsActive():
            return camera_root_prim

        carb.log_warn("No valid camera root prim exist.")
        return None


class LidarCamUtil:
    def get_lidar_name_by_index(i):
        if i == 0:
            return "Lidar"
        elif i < 10:
            return "Lidar_0" + str(i)
        else:
            return "Lidar_" + str(i)

    # check lidar name base on format
    def has_a_valid_name(name):
        if name == "Lidar":
            return True
        # if name starts with "Lidar"
        if name.startswith("Lidar_"):
            return True
        return False

    def get_lidar_name(lidar_prim):
        return lidar_prim.GetName()

    # return the actual name of the lidar camera, should be the part after "Lidar_"
    def get_lidar_name_without_prefix(lidar_prim):
        lidar_name = None
        name = LidarCamUtil.get_lidar_name(lidar_prim)
        # Return none if not a valid name
        if name != None and LidarCamUtil.has_a_valid_name(name):
            if name == "Lidar":
                return ""
            lidar_name = name.split("_")[1]
        return lidar_name

    # get lidar camera root prim
    def get_lidar_camera_root_prim():
        stage = omni.usd.get_context().get_stage()
        # if the lidar camera root prim does not exist, return empty list
        if not Sdf.Path.IsValidPathString(PrimPaths.lidar_cameras_parent_path()):
            carb.log_error(str(PrimPaths.lidar_cameras_parent_path()) + "is not a valid prim path")
            return None
        # fetch and return lidar camera root prim
        lidar_root_prim = stage.GetPrimAtPath(PrimPaths.lidar_cameras_parent_path())
        if lidar_root_prim and lidar_root_prim.IsValid() and lidar_root_prim.IsActive():
            return lidar_root_prim

        carb.log_warn("No valid camera root prim exist.")
        return None

    # get all camera prims under lidar camera root prim
    def get_lidar_cameras_in_stage():
        lidar_camera_list = []
        # get lidar camera root prim
        camera_root_prim = LidarCamUtil.get_lidar_camera_root_prim()
        # if the camera root prim is not valid: return an emtpy list
        if camera_root_prim is None:
            return lidar_camera_list

        # all child camera prim would be added to the list:
        for lidar_camera_prim in camera_root_prim.GetChildren():
            if lidar_camera_prim.GetTypeName() == "Camera":
                lidar_camera_list.append(lidar_camera_prim)

        lidar_camera_list = sorted(lidar_camera_list, key=lambda camera: camera.GetName())
        return lidar_camera_list

    # get all the lidar cameras that has a matching camera in stage
    def get_valid_lidar_cameras_in_stage():
        valid_lidar_camera_list = []
        lidar_camera_list = LidarCamUtil.get_lidar_cameras_in_stage()

        # For matching names with Lidar
        camera_list = CameraUtil.get_cameras_in_stage()

        # Check if they have a matching camera
        for lidar_camera in lidar_camera_list:
            lidar_name = LidarCamUtil.get_lidar_name_without_prefix(lidar_camera)
            has_match = False
            for camera in camera_list:
                if lidar_name == CameraUtil.get_camera_name_without_prefix(camera):
                    has_match = True
                    valid_lidar_camera_list.append(lidar_camera)
            if not has_match:
                carb.log_warn(LidarCamUtil.get_lidar_name(lidar_camera) + " has no matching camera")
        return valid_lidar_camera_list

    def spawn_lidar_camera(spawn_path=None, spawn_location=None, spawn_rotation=None, focallength=None):

        # ensure the default orientation system is base on orient system :
        original_xform_order_setting = StageUtil.set_xformOpType_SOT()

        stage = omni.usd.get_context().get_stage()

        camera_path = ""
        if spawn_path:
            camera_path = spawn_path
        else:
            camera_path = Sdf.Path(
                omni.usd.get_stage_next_free_path(stage, PrimPaths.lidar_cameras_parent_path() + "/Lidar", False)
            )

        camera_name = str(camera_path).replace(PrimPaths.lidar_cameras_parent_path(), "")
        camera_prim = stage.GetPrimAtPath(camera_path)

        lidar_config = "Example_Solid_State"
        _, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path=camera_name,
            parent=PrimPaths.lidar_cameras_parent_path(),
            config=lidar_config,
        )

        camera_prim = stage.GetPrimAtPath(camera_path)
        CameraUtil.set_camera(camera_prim, spawn_location, spawn_rotation, focallength)

        # set the xform setting back to original value
        StageUtil.recover_xformOpType(original_xform_order_setting)
        return camera_prim

    # Delete one lidar camera prim by the given name
    def delete_lidar_camera_prim(cam_name):
        stage = omni.usd.get_context().get_stage()
        if not Sdf.Path.IsValidPathString(PrimPaths.lidar_cameras_parent_path()):
            carb.log_error(str(PrimPaths.lidar_cameras_parent_path()) + "is not a valid prim path")
            return
        lidar_camera_prim = stage.GetPrimAtPath("{}/{}".format(PrimPaths.lidar_cameras_parent_path(), cam_name))
        if lidar_camera_prim and lidar_camera_prim.IsValid() and lidar_camera_prim.IsActive():
            prims.delete_prim(lidar_camera_prim.GetPath())

    # Delete all lidar camera prims in the stage
    def delete_lidar_camera_prims():
        stage = omni.usd.get_context().get_stage()
        if not Sdf.Path.IsValidPathString(PrimPaths.lidar_cameras_parent_path()):
            carb.log_error(str(PrimPaths.lidar_cameras_parent_path()) + "is not a valid prim path")
            return
        lidar_camera_root_prim = stage.GetPrimAtPath(PrimPaths.lidar_cameras_parent_path())
        if lidar_camera_root_prim and lidar_camera_root_prim.IsValid() and lidar_camera_root_prim.IsActive():
            for lidar_camera_prim in lidar_camera_root_prim.GetChildren():
                if lidar_camera_prim and lidar_camera_prim.IsValid() and lidar_camera_prim.IsActive():
                    prims.delete_prim(lidar_camera_prim.GetPath())


class CharacterUtil:
    def get_character_skelroot_by_root(character_prim):
        for prim in Usd.PrimRange(character_prim):
            if prim.GetTypeName() == "SkelRoot":
                return prim
        return None

    def get_character_name_by_index(i):
        if i == 0:
            return "Character"
        elif i < 10:
            return "Character_0" + str(i)
        else:
            return "Character_" + str(i)

    def get_character_name(character_prim):
        # For characters under /World/Characters, names are root names
        # For the rest, names are skelroot names
        prim_path = prims.get_prim_path(character_prim)
        if prim_path.startswith(PrimPaths.characters_parent_path()):
            return prim_path.split("/")[3]
        else:
            return prim_path.split("/")[-1]

    def get_character_pos(character_prim):
        matrix = omni.usd.get_world_transform_matrix(character_prim)
        return matrix.ExtractTranslation()
    
    def get_character_current_pos(character_prim):
        # get position of the character in the current frame
        skeleton_root = CharacterUtil.get_character_skelroot_by_root(character_prim)
        skeleton_root_path = skeleton_root.GetPath()
        animator = ag.get_character(str(skeleton_root_path))
        char_pos, char_rot = Utils.get_character_transform(animator)
        return char_pos

    def get_characters_root_in_stage(count_invisible=False):
        stage = omni.usd.get_context().get_stage()
        character_list = []
        character_root_path = PrimPaths.characters_parent_path()
        folder_prim = stage.GetPrimAtPath(character_root_path)
        if not folder_prim.IsValid() or not folder_prim.IsActive():
            return []

        children = folder_prim.GetAllChildren()
        for c in children:
            if count_invisible == True or UsdGeom.Imageable(c).ComputeVisibility() != UsdGeom.Tokens.invisible:
                character_list.append(c)
        return character_list

    def get_characters_in_stage(count_invisible=False):
        # Get a list of SkelRoot prims as characters
        stage = omni.usd.get_context().get_stage()
        character_list = []
        for prim in stage.Traverse():
            if prim.GetTypeName() == "SkelRoot":
                if count_invisible == True or UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible:
                    character_list.append(prim)
        return character_list

    def load_character_usd_to_stage(character_usd_path, spawn_location, spawn_rotation, character_stage_name):
        # ensure the default orientation system is base on orient system :
        original_xform_order_setting = StageUtil.set_xformOpType_SOT()
        stage = omni.usd.get_context().get_stage()
        # This automatically append number to the character name
        character_stage_name = omni.usd.get_stage_next_free_path(
            stage,
            f"{PrimPaths.characters_parent_path()}/{character_stage_name}",
            False,
        )
        # Load usd into stage and set character translation and rotation.
        prim = prims.create_prim(character_stage_name, "Xform", usd_path=character_usd_path)
        prim.GetAttribute("xformOp:translate").Set(
            Gf.Vec3d(float(spawn_location[0]), float(spawn_location[1]), float(spawn_location[2]))
        )
        if type(prim.GetAttribute("xformOp:orient").Get()) == Gf.Quatf:
            prim.GetAttribute("xformOp:orient").Set(
                Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), float(spawn_rotation)).GetQuat())
            )
        else:
            prim.GetAttribute("xformOp:orient").Set(Gf.Rotation(Gf.Vec3d(0, 0, 1), float(spawn_rotation)).GetQuat())

        # set the xform setting back to original value
        StageUtil.recover_xformOpType(original_xform_order_setting)
        return prim

    def get_anim_graph_from_character(character_prim):
        for prim in Usd.PrimRange(character_prim):
            if prim.GetTypeName() == "AnimationGraph":
                return prim
        return None

    def get_default_biped_character():
        stage = omni.usd.get_context().get_stage()
        return stage.GetPrimAtPath(PrimPaths.default_biped_prim_path())

    # Delete one character prim bt the given name
    def delete_character_prim(char_name):
        stage = omni.usd.get_context().get_stage()
        if not Sdf.Path.IsValidPathString(PrimPaths.characters_parent_path()):
            carb.log_error(str(PrimPaths.characters_parent_path()) + " is not a valid prim path")
            return

        character_prim = stage.GetPrimAtPath("{}/{}".format(PrimPaths.characters_parent_path(), char_name))
        if character_prim and character_prim.IsValid() and character_prim.IsActive():
            prims.delete_prim(character_prim.GetPath())

    # Delete all character prims in the stage
    def delete_character_prims():
        """
        Delete previously loaded character prims. Also deletes the default skeleton and character animations if they
        were loaded using load_default_skeleton_and_animations. Also deletes state corresponding to characters
        loaded onto stage.
        """
        stage = omni.usd.get_context().get_stage()
        if not Sdf.Path.IsValidPathString(PrimPaths.characters_parent_path()):
            carb.log_error(str(PrimPaths.characters_parent_path()) + " is not a valid prim path")
            return

        character_root_prim = stage.GetPrimAtPath(PrimPaths.characters_parent_path())
        if character_root_prim and character_root_prim.IsValid() and character_root_prim.IsActive():
            for character_prim in character_root_prim.GetChildren():
                if character_prim and character_prim.IsValid() and character_prim.IsActive():
                    prims.delete_prim(character_prim.GetPath())


class RobotUtil:
    WORLD_SETTINGS = {"physics_dt": 1.0 / 30.0, "stage_units_in_meters": 1.0, "rendering_dt": 1.0 / 30.0}
    TYPE_TO_USD_PATH = {
        "nova_carter": AssetPaths.default_nova_carter_path(),
        "transporter": AssetPaths.default_transpoter_path(),
        "carter_v1": AssetPaths.default_carter_v1_path(),
        "dingo": AssetPaths.default_dingo_path(),
        "jetbot": AssetPaths.default_jetbot_path(),
    }

    def get_robot_name_by_index(robot_type, i):
        if i == 0:
            return robot_type
        elif i < 10:
            return robot_type + "_0" + str(i)
        else:
            return robot_type + "_" + str(i)

    def get_robot_name(robot_prim):
        # For robots under /World/Robots, names are root names
        prim_path = prims.get_prim_path(robot_prim)
        if prim_path.startswith(PrimPaths.robots_parent_path()):
            return prim_path.split("/")[3]

    def get_robot_pos(robot_prim):
        matrix = omni.usd.get_world_transform_matrix(robot_prim)
        return matrix.ExtractTranslation()

    def get_robots_in_stage(robot_type_name=None):
        robot_xform = prims.get_prim_at_path(PrimPaths.robots_parent_path())
        if not robot_xform.IsValid() or not robot_xform.IsActive():
            return []
        prims_under_robots = prims.get_prim_children(robot_xform)
        robot_list = []
        for prim in prims_under_robots:
            path = prims.get_prim_path(prim)
            if robot_type_name == None:
                robot_list.append(prim)
            else:
                if path.startswith(PrimPaths.robots_parent_path() + "/" + robot_type_name):
                    robot_list.append(prim)
        return robot_list

    # Get all the cameras on the given robot
    def get_cameras_on_robot(robot_prim):
        stage = omni.usd.get_context().get_stage()
        robot_path = prims.get_prim_path(robot_prim)
        camera_list = []
        for prim in stage.Traverse():
            path = prims.get_prim_path(prim)
            if (
                prim.GetTypeName() == "Camera"
                and path.startswith(robot_path)
                and path.split("/")[-1].startswith("camera")
            ):
                camera_list.append(prim)
        return camera_list
    
    def get_specific_cameras_on_robot(robot_prim=None, robot_prim_path=None):
        # for nova carter, only need the front_owl/camera
        stage = omni.usd.get_context().get_stage()
        robot_path = robot_prim_path or prims.get_prim_path(robot_prim)
        camera_list = []
        if "nova_carter" in robot_path.lower():
            for prim in stage.Traverse():
                path = prims.get_prim_path(prim)
                if (
                    prim.GetTypeName() == "Camera" 
                    and path.startswith(robot_path) 
                    and "front_owl" == path.split("/")[-2]
                    and path.split("/")[-1].startswith("camera")
                ):
                    camera_list.append(prim)
        elif "transporter" in robot_path.lower():
            for prim in stage.Traverse():
                path = prims.get_prim_path(prim)
                if (
                    prim.GetTypeName() == "Camera" 
                    and path.startswith(robot_path) 
                    and path.split("/")[-1] == "transporter_camera_first_person"
                ):
                    camera_list.append(prim)
        elif "carter_v1" in robot_path.lower():
            for prim in stage.Traverse():
                path = prims.get_prim_path(prim)
                if (
                    prim.GetTypeName() == "Camera" 
                    and path.startswith(robot_path) 
                    and path.split("/")[-1] == "carter_camera_first_person"
                ):
                    camera_list.append(prim)
        elif "dingo" in robot_path.lower():
            for prim in stage.Traverse():
                path = prims.get_prim_path(prim)
                if (
                    prim.GetTypeName() == "Camera" 
                    and path.startswith(robot_path) 
                    and path.split("/")[-1] == "realsense_right_stereo_camera"
                ):
                    camera_list.append(prim)
        elif "jackal" in robot_path.lower():
            for prim in stage.Traverse():
                path = prims.get_prim_path(prim)
                if (
                    prim.GetTypeName() == "Camera" 
                    and path.startswith(robot_path) 
                    and path.split("/")[-1] == "bumblebee_stereo_right_camera"
                ):
                    camera_list.append(prim)
        elif "forklift" in robot_path.lower():
            for prim in stage.Traverse():
                path = prims.get_prim_path(prim)
                if (
                    prim.GetTypeName() == "Camera" 
                    and path.startswith(robot_path) 
                    and path.split("/")[-1] == "camera_right"
                ):
                    camera_list.append(prim)
        elif "jetbot" in robot_path.lower():
            for prim in stage.Traverse():
                path = prims.get_prim_path(prim)
                if (
                    prim.GetTypeName() == "Camera" 
                    and path.startswith(robot_path) 
                    and path.split("/")[-1] == "jetbot_camera"
                ):
                    camera_list.append(prim)

        return camera_list

    # Get all the lidar cameras on the given robot
    def get_lidar_cameras_on_robot(robot_prim):
        stage = omni.usd.get_context().get_stage()
        robot_path = prims.get_prim_path(robot_prim)
        camera_list = []
        for prim in stage.Traverse():
            path = prims.get_prim_path(prim)
            if prim.GetTypeName() == "Camera" and path.startswith(robot_path) and "LIDAR" in path.split("/")[-1]:
                camera_list.append(prim)
        return camera_list

    # Get all the cameras on all the robots in the stage
    def get_robot_cameras():
        cameras = [cam for robot in RobotUtil.get_robots_in_stage() for cam in RobotUtil.get_cameras_on_robot(robot)]
        return cameras

    # Get the fisrt n cameras on all the robots in the stage
    def get_n_robot_cameras(n):
        cameras = [
            cam for robot in RobotUtil.get_robots_in_stage() for cam in RobotUtil.get_cameras_on_robot(robot)[:n]
        ]
        return cameras

    # Get all the lidar cameras on all the robots in the stage
    def get_robot_lidar_cameras():
        lidars = [
            lidar for robot in RobotUtil.get_robots_in_stage() for lidar in RobotUtil.get_lidar_cameras_on_robot(robot)
        ]
        return lidars

    # Get all the lidar cameras on all the robots in the stage
    def get_n_robot_lidar_cameras():
        lidars = [
            lidar
            for robot in RobotUtil.get_robots_in_stage()
            for lidar in RobotUtil.get_lidar_cameras_on_robot(robot)[:n]
        ]
        return lidars

    def spawn_robot(spawn_type, spawn_location, spawn_rotation=0, spawn_path=None):

        # ensure the default orientation system is base on orient system :
        original_xform_order_setting = StageUtil.set_xformOpType_SOT()
        stage = omni.usd.get_context().get_stage()
        if spawn_type in RobotUtil.TYPE_TO_USD_PATH:
            robot_name = spawn_type
            robot_usd_path = RobotUtil.TYPE_TO_USD_PATH[spawn_type]
        else:
            carb.log_error("Invalid robot type: ", spawn_type)

        # This automatically append number to the robot name
        robot_stage_name = omni.usd.get_stage_next_free_path(
            stage, f"{PrimPaths.robots_parent_path()}/{robot_name}", False
        )
        if spawn_path:
            robot_stage_name = spawn_path
        # Load usd into stage and set character translation and rotation.
        prim = prims.create_prim(robot_stage_name, "Xform", usd_path=robot_usd_path)
        prim.GetAttribute("xformOp:translate").Set(
            Gf.Vec3d(float(spawn_location[0]), float(spawn_location[1]), float(spawn_location[2]))
        )
        if type(prim.GetAttribute("xformOp:orient").Get()) == Gf.Quatf:
            prim.GetAttribute("xformOp:orient").Set(
                Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), float(spawn_rotation)).GetQuat())
            )
        else:
            prim.GetAttribute("xformOp:orient").Set(Gf.Rotation(Gf.Vec3d(0, 0, 1), float(spawn_rotation)).GetQuat())

        # set the xform setting back to original value
        StageUtil.recover_xformOpType(original_xform_order_setting)

        return prim

    # Delete one character prim bt the given name
    def delete_robot_prim(robot_name):
        stage = omni.usd.get_context().get_stage()
        if not Sdf.Path.IsValidPathString(PrimPaths.robots_parent_path()):
            carb.log_error(str(PrimPaths.robots_parent_path()) + " is not a valid prim path")
            return

        robot_prim = stage.GetPrimAtPath("{}/{}".format(PrimPaths.robots_parent_path(), robot_name))
        if robot_prim and robot_prim.IsValid() and robot_prim.IsActive():
            prims.delete_prim(robot_prim.GetPath())

    # Delete all character prims in the stage
    def delete_robot_prims():
        """
        Delete previously loaded character prims. Also deletes the default skeleton and character animations if they
        were loaded using load_default_skeleton_and_animations. Also deletes state corresponding to characters
        loaded onto stage.
        """
        stage = omni.usd.get_context().get_stage()
        if not Sdf.Path.IsValidPathString(PrimPaths.robots_parent_path()):
            carb.log_error(str(PrimPaths.robots_parent_path()) + " is not a valid prim path")
            return

        robot_root_prim = stage.GetPrimAtPath(PrimPaths.robots_parent_path())
        if robot_root_prim and robot_root_prim.IsValid() and robot_root_prim.IsActive():
            for robot_prim in robot_root_prim.GetChildren():
                if robot_prim and robot_prim.IsValid() and robot_prim.IsActive():
                    prims.delete_prim(robot_prim.GetPath())

    def clean_robot_world():
        async def destroy_world():
            prev_world = World.instance()
            if prev_world is not None:
                prev_world.clear_all_callbacks()
                prev_world.clear_instance()
                prev_world = None
                await update_stage_async()

        asyncio.ensure_future(destroy_world())
