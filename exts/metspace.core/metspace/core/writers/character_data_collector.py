##container used to store skeleton information
from collections import deque

import carb
import omni.anim.graph.core as ag
import omni.timeline
import omni.usd
import usdrt
from pxr import Gf, Semantics, Usd, UsdGeom, UsdSkel

from ..settings import PrimPaths
from .utils import Utils


class CharacterDataCollector:
    def __init__(self, valid_unoccluded_threshold, segmentation_seed, frame_delay):
        # character label to joint position set
        self.label_to_joint_pos = {}
        self.label_to_head_pos = {}
        self.label_to_character_pos = {}
        # label to 3d bounding box
        self.label_to_3d_box = {}
        self.skeleton_path_to_animator = {}
        self.skeleton_path_to_label = {}
        self.skeleton_path_to_skeleton_root_path = {}
        self.skeleton_path_to_tags_dict = {}
        self.skeleton_path_to_tags_dict_prim_path = {}
        self.frame_delay = frame_delay
        # seed value we use to generate random color
        self.segmentation_seed = segmentation_seed
        self.valid_unoccluded_threshold = valid_unoccluded_threshold

    # create skeleotn dict
    # extract labeled character's information and initialize data structure to record dynamic values
    def _create_skeleton_dicts(self):
        self.skeleton_path_to_label = {}
        stage = omni.usd.get_context().get_stage()
        character_root_path = PrimPaths.characters_parent_path()
        character_root_prim = stage.GetPrimAtPath(character_root_path)
        if not character_root_prim.IsValid() or not character_root_prim.IsActive():
            carb.log_warn("Fail to fetch character information, no characters in the scene.")
            return
        # check the child prim of our character root prim
        for character_prim in character_root_prim.GetChildren():
            if not character_prim.HasAPI(Semantics.SemanticsAPI):
                continue
            semantics_data = None
            # check whether the character's semantic label is a "class" label
            for name in character_prim.GetPropertyNames():
                if str(name).startswith("semantic:Semantics") and str(name).endswith("params:semanticType"):
                    semantics_type = character_prim.GetAttribute(str(name))
                    if str(semantics_type.Get()).lower() == "class":
                        semantics_data_name = str(name)[:-4] + "Data"
                        semantics_data = character_prim.GetAttribute(str(semantics_data_name)).Get()

            if not semantics_data:
                continue

            skeleton_root_path = None
            skeleton_path = None
            semantics_data = str(semantics_data).lower()

            # for each character, match character's skeleton and animator with label
            for prim_child in Usd.PrimRange(character_prim):

                if prim_child.GetTypeName() == "Skeleton":
                    skeleton_prim = prim_child
                    # get character's skeleton prim path
                    skeleton_path = skeleton_prim.GetPath()
                if prim_child.GetTypeName() == "SkelRoot":
                    skeleton_root_prim = prim_child
                    # get character's skelroot prim path
                    skeleton_root_path = skeleton_root_prim.GetPath()
            if skeleton_path and skeleton_root_path:

                # initialize data strucutres used to store character information
                self.skeleton_path_to_label[str(skeleton_path)] = str(semantics_data)
                self.label_to_head_pos[str(semantics_data)] = deque()
                self.label_to_character_pos[str(semantics_data)] = deque()
                self.label_to_joint_pos[str(semantics_data)] = deque()

                animator = ag.get_character(str(skeleton_root_path))
                if not animator:
                    carb.log_warn(
                        "No animator found in '{}'. Will gather data from stage instead.".format(
                            str(skeleton_root_path)
                        )
                    )
                # get skeleton name for each character base on retarget tags
                (
                    self.skeleton_path_to_tags_dict[str(skeleton_path)],
                    self.skeleton_path_to_tags_dict_prim_path[str(skeleton_path)],
                ) = Utils.get_tags_to_skeleton_dict(str(skeleton_path))
                self.skeleton_path_to_animator[str(skeleton_path)] = animator
                self.skeleton_path_to_skeleton_root_path[str(skeleton_path)] = str(skeleton_root_path)

    def _get_prim_transform_fabric(self, rtstage, prim_path):
        prim = rtstage.GetPrimAtPath(str(prim_path))
        pos = carb.Float3(0, 0, 0)
        rot = carb.Float4(0, 0, 0, 0)
        if prim.GetAttribute("_worldPosition").IsValid():
            gf_pos = prim.GetAttribute("_worldPosition").Get()
            pos = carb.Float3(gf_pos[0], gf_pos[1], gf_pos[2])
        if prim.GetAttribute("_worldOrientation").IsValid():
            gf_quat = prim.GetAttribute("_worldOrientation").Get()
            gf_i = gf_quat.GetImaginary()
            rot = carb.Float4(gf_i[0], gf_i[1], gf_i[2], gf_quat.GetReal())
        return pos, rot

    def _get_prim_transform_timeline(self, stage, prim_path):
        prim = stage.GetPrimAtPath(prim_path)
        xform = UsdGeom.Xformable(prim)
        timeline = omni.timeline.get_timeline_interface()
        time_code = Usd.TimeCode(timeline.get_current_time() * timeline.get_ticks_per_second())
        world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time_code)
        gf_pos = world_transform.ExtractTranslation()
        pos = carb.Float3(gf_pos[0], gf_pos[1], gf_pos[2])
        gf_quat = world_transform.ExtractRotation().GetQuat()
        gf_i = gf_quat.GetImaginary()
        rot = carb.Float4(gf_i[0], gf_i[1], gf_i[2], gf_quat.GetReal())
        return pos, rot

    # check whether the semantic lable belong to an recorded character
    def is_character(self, semantic_label):
        return semantic_label in self.label_to_head_pos.keys()

    # update character's translation,rotation, joint position and store them into buffer
    def update_character_pos(self):

        for skeleton_path in self.skeleton_path_to_animator.keys():
            char_pos = None
            char_rot = None
            spot_set = None
            joint_pos_3d = []
            animator = self.skeleton_path_to_animator[skeleton_path]
            label = self.skeleton_path_to_label[str(skeleton_path)]

            if animator is None:
                skel_root_path = str(self.skeleton_path_to_skeleton_root_path[skeleton_path])
                self.skeleton_path_to_animator[skeleton_path] = ag.get_character(skel_root_path)
                animator = self.skeleton_path_to_animator[skeleton_path]

            if animator:
                tags_to_joint_dict = self.skeleton_path_to_tags_dict[str(skeleton_path)]
                char_pos, char_rot = Utils.get_character_transform(animator)
                head_pos, head_rot = Utils.get_character_joint_transform(animator, tags_to_joint_dict["Head"])
                # estimate character's head position base on current head joint's location
                spot_set = Utils.calculate_character_head_position(head_pos)
                # traverse character's joint list ana get rotation and location of each joint
                for joint_key in Utils.skeleton_joint_list:
                    joint_pos, joint_rot = Utils.get_character_joint_transform(
                        animator, tags_to_joint_dict[str(joint_key)]
                    )
                    joint_pos_3d.append(joint_pos)
            else:
                # Read from fabric stage when anim graph is not found
                stage = omni.usd.get_context().get_stage()
                rtstage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
                tags_dict_prim_path = self.skeleton_path_to_tags_dict_prim_path[str(skeleton_path)]
                char_pos, char_rot = self._get_prim_transform_timeline(stage, str(skel_root_path))
                head_pos, head_rot = self._get_prim_transform_fabric(rtstage, str(tags_dict_prim_path["Head"]))
                spot_set = Utils.calculate_character_head_position(head_pos)
                for joint_key in Utils.skeleton_joint_list:
                    joint_pos, joint_rot = self._get_prim_transform_fabric(rtstage, tags_dict_prim_path[str(joint_key)])
                    joint_pos_3d.append(joint_pos)

            joint_set = joint_pos_3d

            if len(self.label_to_joint_pos[label]) > self.frame_delay:
                self.label_to_joint_pos[label].popleft()

            self.label_to_joint_pos[label].append(joint_set)
            character_transform = {"pos": char_pos, "rot": char_rot}

            if len(self.label_to_head_pos[label]) > self.frame_delay:
                self.label_to_head_pos[label].popleft()
                self.label_to_character_pos[label].popleft()
            self.label_to_head_pos[label].append(spot_set)
            self.label_to_character_pos[label].append(character_transform)

    # create random color for instance segmentation
    def create_random_color_dict(self, exist_label_dict):
        color_dict = {}
        character_label_list = list(self.label_to_head_pos.keys())
        number_of_character = len(character_label_list)
        # avoid too similar color in the Kitti label list
        registered_color = exist_label_dict.values()
        registered_color_modified = [(a, b, c) for (a, b, c, d) in registered_color]
        colors_list = Utils.generate_colors(self.segmentation_seed, number_of_character, registered_color_modified)
        for i in range(0, number_of_character):
            color_dict[character_label_list[i]] = colors_list[i]
        return color_dict

    # return 3d and 2d position of the character joints
    def get_joint_information(self, semantic_label, view_proj_mat, rp_width, rp_height):
        ## get character's 3d and 2d joint information:
        joint_3d = self.label_to_joint_pos[semantic_label][0]
        joint_2d_raw = Utils.convert_to_2d(joint_3d, view_proj_mat, rp_width, rp_height)
        joint_2d = []
        for joint in joint_2d_raw:
            if Utils.in_scope(joint, [rp_width, 0, rp_height, 0]):
                joint_2d.append(joint)
            else:
                joint_2d.append([-1, -1])
        return joint_3d, joint_2d

    # check the show ratio of the head
    def check_head_occlusion(self, semantic_label, character_tight_box, view_proj_mat, rp_width, rp_height):

        head_bound_box = Utils.convert_to_2d(
            self.label_to_head_pos[semantic_label][0], view_proj_mat, rp_width, rp_height
        )
        x_max = -1
        x_min = 10000
        y_max = -1
        y_min = 10000
        for head_pos in head_bound_box:
            if head_pos[0] > x_max:
                x_max = head_pos[0]
            if head_pos[0] < x_min:
                x_min = head_pos[0]
            if head_pos[1] > y_max:
                y_max = head_pos[1]
            if head_pos[1] < y_min:
                y_min = head_pos[1]
        head_box = [x_max, x_min, y_max, y_min]
        larger = Utils.larger
        in_bound, true_head_tight_box = Utils.in_the_bounding(head_box, character_tight_box, larger)
        ratio = 100
        if in_bound == 0 or in_bound == 1:
            ratio = Utils.compute_shown_ratio(true_head_tight_box, head_box)
        else:
            ratio = 0
        # if the show ratio is larger than 0.8, then head is visible
        return ratio > 0.8

    # get character's position and rotation
    def get_character_transform(self, semantic_label):
        character_position = self.label_to_character_pos[semantic_label][0]["pos"]
        character_rotation = self.label_to_character_pos[semantic_label][0]["rot"]
        return character_position, character_rotation

    # check whether the character can pass visibility threshold checking
    def valid_character(self, semantic_label, tight_box, loose_box, viewport_box, view_proj_mat):

        # character label is semantic_label
        # place holder for the bounding box used to hanle trancated character case
        calculated_box = [10, 10, 10, 10]
        # check whether character is trancated, out of image, or within image
        in_bound, true_body_box = Utils.check_character_box(tight_box, viewport_box)
        rp_width = viewport_box[0]
        rp_height = viewport_box[2]
        character_width = true_body_box[0] - true_body_box[1]
        character_height = true_body_box[2] - true_body_box[3]

        ## calculate character's width and height show ratio. Edge case such as truncated character would be discussed later
        width_show_ratio = character_width / (loose_box[0] - loose_box[1] + Utils.EPS)
        height_show_ratio = character_height / (loose_box[2] - loose_box[3] + Utils.EPS)

        # if the character is not in the image, then we filter them out directly
        if in_bound == 2:
            return False, None

        ## get character's head position and check whether it is shown in the tight box
        face_is_show = self.check_head_occlusion(semantic_label, tight_box, view_proj_mat, rp_width, rp_height)
        shoulder_is_show = False
        lowerbody_is_shown = True

        joint_3d, joint_2d = self.get_joint_information(semantic_label, view_proj_mat, rp_width, rp_height)

        shoulder_pos = []
        shoulder_pos.append(joint_2d[Utils.get_joint_index("Left_Shoulder")])
        shoulder_pos.append(joint_2d[Utils.get_joint_index("Right_Shoulder")])

        lowerbody_pos = []
        lowerbody_pos.append(joint_2d[Utils.get_joint_index("Left_Knee")])
        lowerbody_pos.append(joint_2d[Utils.get_joint_index("Right_Knee")])

        ## get character's shoulder position, check whether shoulder is shown
        for pos in shoulder_pos:
            if (
                pos[0] < true_body_box[0]
                and pos[0] > true_body_box[1]
                and pos[1] < true_body_box[2]
                and pos[1] > true_body_box[3]
            ):
                shoulder_is_show = True

        for pos in lowerbody_pos:
            if not (
                pos[0] < true_body_box[0]
                and pos[0] > true_body_box[1]
                and pos[1] < true_body_box[2]
                and pos[1] > true_body_box[3]
            ):
                lowerbody_is_shown = False

        if in_bound == 1:
            character_position, character_rotation = self.get_character_transform(semantic_label)
            # if character is truncated, then we need to calculate character's bounding box from skeleton joints and then we update the height and wdith show ratio
            calculated_box = Utils.recalculate_loose_bounding_box(
                joint_3d, character_position, character_rotation, view_proj_mat, rp_width, rp_height
            )
            width_show_ratio = character_width / (calculated_box[0] - calculated_box[1] + Utils.EPS)
            height_show_ratio = character_height / (calculated_box[2] - calculated_box[3] + Utils.EPS)

        if in_bound == 0:
            # if character is not truncated, then it need to meet the requirements of both width and height show ratio
            # height threshold checking
            if height_show_ratio < self.valid_unoccluded_threshold:

                if not face_is_show:
                    # if character's head is not shown
                    return False, None
                else:
                    if not shoulder_is_show:
                        # unoccluded ratio is less than 22% percent
                        if height_show_ratio < 0.22:
                            return False, None
                        # shoulder and head are not shown, return directly
                        elif tight_box[3] > (loose_box[3] + (loose_box[2] - loose_box[3]) / 4):
                            return False, None
            # width threshold checking
            if width_show_ratio < self.valid_unoccluded_threshold:
                if lowerbody_is_shown or tight_box[2] > (loose_box[3] + 7 * (loose_box[2] - loose_box[3]) / 10):
                    return False, None
        if in_bound == 1:
            # if character is truncated, then it need meet either the requirement of width or height show ratio
            if (
                height_show_ratio < self.valid_unoccluded_threshold
                and width_show_ratio < self.valid_unoccluded_threshold
            ):
                if not face_is_show:
                    return False, None
                else:
                    if not shoulder_is_show:
                        if height_show_ratio < 0.22:
                            return False, None
                        elif tight_box[3] > (loose_box[3] + (loose_box[2] - loose_box[3]) / 4):
                            return False, None

        return True, true_body_box
