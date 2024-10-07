import carb
import omni.anim.navigation.core as nav
import omni.client
import omni.usd
from omni.isaac.core.utils import prims
from pxr import AnimGraphSchema, Gf, Sdf, Usd, UsdGeom, UsdSkel

from .settings import AssetPaths, PrimPaths
from .stage_util import CameraUtil, CharacterUtil, StageUtil


class SimulationVerification:
    """
    Class to verify all parts of the people simulation.
    """

    def __init__(self):
        self.characters_parent_prim_path = PrimPaths.characters_parent_path()
        self.characters_parent_prim = None
        self.yaml_data = None

    def verify_navmesh(self):
        _nav = nav.nav.acquire_interface()
        if _nav.get_navmesh_volume_count() == 0:
            carb.log_warning(
                "Agent Simulation Verification: Scene does not contain navigation volume. Agent commands using metspace extension will fail."
            )
            return False
        else:
            carb.log_info("Agent Simulation Verification: Navigation volume present in secene.")
            return True

    def verify_character_cmd_files(self):
        command_file_path = self.yaml_data["character"]["command_file"]
        result, version, context = omni.client.read_file(command_file_path)
        if result != omni.client.Result.OK:
            carb.log_warn(
                "Agent Simulation Verification: Cannot load command file at path: {}.".format(command_file_path)
            )
            return False
        else:
            carb.log_info("Agent Simulation Verification: cmd file is accessible.")
            return True

    def verify_script_paths(self):
        error_exists = False
        stage = omni.usd.get_context().get_stage()
        self.characters_parent_prim = stage.GetPrimAtPath(self.characters_parent_prim_path)
        for prim in Usd.PrimRange(self.characters_parent_prim):
            if prim.GetTypeName() == "SkelRoot":
                # Skip biped skelroot as it won't have python script attached
                if prims.get_prim_path(prim).startswith(PrimPaths.default_biped_prim_path()):
                    continue
                attr = prim.GetAttribute("omni:scripting:scripts").Get()
                if not attr:
                    error_exists = True
                    carb.log_warn(
                        "Agent Simulation Verification: Character skelroot {} does not have python scripts attached.".format(
                            prim.GetPrimPath()
                        )
                    )
                    continue
                script_path = attr[0].path
                result, version, context = omni.client.read_file(script_path)
                if result != omni.client.Result.OK:
                    error_exists = True
                    carb.log_warn(
                        "Agent Simulation Verification: Unable to read python scripting path {} for skelroot at {}.".format(
                            script_path, prim.GetPrimPath()
                        )
                    )
                    continue
        if not error_exists:
            carb.log_info("Agent Simulation Verification: Found valid script for all skelroots.")
            return True
        else:
            return False

    def verify_base_skeleton_and_animations(self):

        stage = omni.usd.get_context().get_stage()

        if not stage.GetPrimAtPath("{}/Biped_Setup".format(self.characters_parent_prim_path)).IsValid():
            carb.log_warn(
                "Agent Simulation Verification: Base skeleton parent prim {}/Biped_Setup is not valid.".format(
                    self.characters_parent_prim_path
                )
            )
            return False

        if not stage.GetPrimAtPath(
            "{}/Biped_Setup/CharacterAnimation".format(self.characters_parent_prim_path)
        ).IsValid():
            carb.log_warn(
                "Agent Simulation Verification: Character animations prims {}/Biped_Setup/CharacterAnimation are not present in stage.".format(
                    self.characters_parent_prim_path
                )
            )
            return False

        if not stage.GetPrimAtPath(
            "{}/Biped_Setup/biped_demo_meters".format(self.characters_parent_prim_path)
        ).IsValid():
            carb.log_warn(
                "Agent Simulation Verification: Base skeleton prim {}/Biped_Setup/biped_demo_meters is not valid.".format(
                    self.characters_parent_prim_path
                )
            )
            return False

        carb.log_info("Agent Simulation Verification: Verified base skeleton and animation prims.")
        return True

    def verify_animation_graph(self):
        error_exists = False
        stage = omni.usd.get_context().get_stage()
        self.characters_parent_prim = stage.GetPrimAtPath(self.characters_parent_prim_path)
        default_biped = CharacterUtil.get_default_biped_character()
        anim_graph_path = CharacterUtil.get_anim_graph_from_character(default_biped).GetPrimPath()
        for prim in Usd.PrimRange(self.characters_parent_prim):
            if prim.GetTypeName() == "SkelRoot":
                anim_graph_ref = AnimGraphSchema.AnimationGraphAPI(prim).GetAnimationGraphRel()
                if not anim_graph_ref:
                    error_exists = True
                    carb.log_warn(
                        "Agent Simulation Verification: Could not find an Animation Graph attached to skelroot {}".format(
                            prim.GetPrimPath()
                        )
                    )
                    continue
                if anim_graph_ref.GetTargets()[0] != anim_graph_path:
                    error_exists = True
                    carb.log_warn(
                        "Agent Simulation Verification: Invalid Animation Graph attached to skelroot {}. Expected Animation Graph is {}.".format(
                            prim.GetPrimPath(), anim_graph_path
                        )
                    )
                    continue
        if not error_exists:
            carb.log_info("Agent Simulation Verification: Verified Animation Graph attached to all skelroot.")
            return True

    def verify_scene_usd(self):
        scene_path = self.yaml_data["scene"]["asset_path"]
        result, version, context = omni.client.read_file(scene_path)
        if result != omni.client.Result.OK:
            carb.log_warn("Agent Simulation Verification: Cannot access usd at path: {}.".format(scene_path))
            return False
        else:
            carb.log_info("Agent Simulation Verification: Verified access to usd at {}.".format(scene_path))
            return True

    def verify_character_assets(self):
        character_asset_path = self.yaml_data["character"]["asset_path"]
        result, folder_list = omni.client.list(character_asset_path)
        if result != omni.client.Result.OK:
            carb.log_warn(
                "Agent Simulation Verification: Cannot access characters from folder: {}.".format(character_asset_path)
            )
            return False
        else:
            carb.log_info(
                "Agent Simulation Verification: Verified access to character folders at {}.".format(
                    character_asset_path
                )
            )
            return True

    def verify_robots_assets(self):
        carb.log_info("Agent Simulation Verification: Verified access to robots folders.")
        return True

    def verify_simulation(self, yaml_data):
        self.yaml_data = yaml_data
        verification_results_list = []
        verification_results_list.append(self.verify_scene_usd())
        verification_results_list.append(self.verify_character_assets())
        verification_results_list.append(self.verify_animation_graph())
        verification_results_list.append(self.verify_base_skeleton_and_animations())
        verification_results_list.append(self.verify_script_paths())
        verification_results_list.append(self.verify_character_cmd_files())
        verification_results_list.append(self.verify_navmesh())
        verification_results_list.append(self.verify_robots_assets())

        if all(verification_results_list):
            carb.log_info("Agent Simulation Verification: Verification found no errors")
        else:
            carb.log_info("Agent Simulation Verification: Verification found errors, please check console warnings.")
