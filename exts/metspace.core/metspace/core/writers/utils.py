import json
import math
import random

import carb
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import omni.isaac.core.utils.numpy.rotations as rot_utils
import omni.usd
from PIL import Image
from pxr import AnimGraphSchema, Gf, RetargetingSchema, Usd, UsdGeom, UsdSkel
from scipy.spatial.transform import Rotation as R


class Utils:

    EPS = 1e-5

    skeleton_joint_list = [
        "Pelvis",
        "Head",
        "Left_Shoulder",
        "Left_Elbow",
        "Left_Hand",
        "Right_Shoulder",
        "Right_Elbow",
        "Right_Hand",
        "Left_Thigh",
        "Left_Knee",
        "Left_Foot",
        "Left_Toe",
        "Right_Thigh",
        "Right_Knee",
        "Right_Foot",
        "Right_Toe",
    ]

    def larger(a, b):
        return a > b

    def add3(a, b):
        return carb.Float3(a[0] + b[0], a[1] + b[1], a[2] + b[2])

    def sub3(a, b):
        return carb.Float3(a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def close_enough(a, b):
        return abs(a - b) <= 4

    def scale3(v, f):
        return carb.Float3(v[0] * f, v[1] * f, v[2] * f)

    def normalize3(v):
        gf_v = Gf.Vec3d(v[0], v[1], v[2])
        gf_v_normalized = gf_v.GetNormalized()
        return carb.Float3(gf_v_normalized[0], gf_v_normalized[1], gf_v_normalized[2])

    def get_trig(deg):
        return math.cos(math.radians(deg)), math.sin(math.radians(deg))

    def rot_x(deg):
        c, s = Utils.get_trig(deg)
        return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])

    def rot_y(deg):
        c, s = Utils.get_trig(deg)
        return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])

    def rot_z(deg):
        c, s = Utils.get_trig(deg)
        return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

    def get_joint_index(joint_name):
        index = Utils.skeleton_joint_list.index(joint_name)
        return index

    def get_tags_to_skeleton_dict(target_skeleton_path):
        skel_prim_path = target_skeleton_path
        stage = omni.usd.get_context().get_stage()
        skel_prim = stage.GetPrimAtPath(str(skel_prim_path))
        control_rig_api = RetargetingSchema.ControlRigAPI(skel_prim)
        tags = control_rig_api.GetRetargetTagsAttr().Get()

        skeleton = UsdSkel.Skeleton(skel_prim)
        joints_attr = skeleton.GetJointsAttr().Get()
        joint_node_list = []
        joint_path_list = []
        for joint_path in joints_attr:
            joint_node_list.append(joint_path.split("/")[-1])
            joint_path_list.append("{}/{}".format(str(skel_prim_path), str(joint_path)))

        joints_length = len(joint_node_list)

        tag_dict = {}
        tag_dict_path = {}
        for i in range(0, joints_length):
            if tags[i] is not None and tags[i] != "":
                tag_dict[tags[i]] = joint_node_list[i]
                tag_dict_path[tags[i]] = joint_path_list[i]

        for retarget_joint_tag in Utils.skeleton_joint_list:
            if retarget_joint_tag not in tag_dict.keys():
                carb.log_error(
                    "Error: character skeleton"
                    + str(target_skeleton_path)
                    + " is not retarget completely! Retarget joint tag: "
                    + str(retarget_joint_tag)
                    + " is missing !"
                )

        return tag_dict, tag_dict_path

    # calculate character's scale
    def get_scale(bbox_3d_data):
        x_min = bbox_3d_data["x_min"]
        x_max = bbox_3d_data["x_max"]
        y_min = bbox_3d_data["y_min"]
        y_max = bbox_3d_data["y_max"]
        z_min = bbox_3d_data["z_min"]
        z_max = bbox_3d_data["z_max"]

        scale_x = x_max - x_min
        scale_y = y_max - y_min
        scale_z = z_max - z_min

        return [scale_x + 0.14, scale_y + 0.2, scale_z]

    # calcuate character's transform
    def get_character_transform(c):
        pos = carb.Float3(0, 0, 0)
        rot = carb.Float4(0, 0, 0, 0)
        c.get_world_transform(pos, rot)
        return pos, rot

    # get character's joint transform with joint name and character's animgraph
    def get_character_joint_transform(c, joint_name):
        pos = carb.Float3(0, 0, 0)
        rot = carb.Float4(0, 0, 0, 0)
        c.get_joint_transform(joint_name, pos, rot)
        return pos, rot

    # estimate character's head position with its neck joint node
    def calculate_character_head_position(neck_location):
        result = []
        # draw a cube according to the position of the head
        result.append(carb.Float3(neck_location[0], neck_location[1] + 0.055, neck_location[2] + 0.03))
        result.append(carb.Float3(neck_location[0], neck_location[1] - 0.055, neck_location[2] + 0.03))
        result.append(carb.Float3(neck_location[0] + 0.055, neck_location[1], neck_location[2] + 0.03))
        result.append(carb.Float3(neck_location[0] - 0.055, neck_location[1], neck_location[2] + 0.03))
        result.append(carb.Float3(neck_location[0], neck_location[1] + 0.055, neck_location[2] + 0.13))
        result.append(carb.Float3(neck_location[0], neck_location[1] - 0.055, neck_location[2] + 0.13))
        result.append(carb.Float3(neck_location[0] + 0.055, neck_location[1], neck_location[2] + 0.13))
        result.append(carb.Float3(neck_location[0] - 0.055, neck_location[1], neck_location[2] + 0.13))

        return result

    # convert rotation quat to angle in degree
    def convert_to_angle(quat_rot):
        rot_in_angle = Gf.Rotation(Gf.Quatd(quat_rot.w, quat_rot.x, quat_rot.y, quat_rot.z))
        zaxis = rot_in_angle.GetAxis()[2]
        rot_angle = rot_in_angle.GetAngle()
        if zaxis < 0:
            rot_angle = -rot_angle
        return rot_angle

    def compute_shown_ratio(box_tight, box_loose):
        area_tight = (box_tight[0] - box_tight[1]) * (box_tight[2] - box_tight[3])
        area_loose = (box_loose[0] - box_loose[1]) * (box_loose[2] - box_loose[3])
        area_ratio = area_tight / (area_loose + Utils.EPS)
        return area_ratio

    def check_character_box(target_tight_box, view_port_box):
        close_enough = Utils.close_enough
        in_the_bound, true_tight_box = Utils.in_the_bounding(target_tight_box, view_port_box, close_enough)

        return in_the_bound, true_tight_box

    # check whether character is:
    # inside image
    # truncated on the edge
    # out of the image
    def in_the_bounding(target_scope, bounding_scope, fn):

        in_the_bound = 0

        target_x_max = target_scope[0]
        target_x_min = target_scope[1]
        target_y_max = target_scope[2]
        target_y_min = target_scope[3]

        x_max = bounding_scope[0]
        x_min = bounding_scope[1]
        y_max = bounding_scope[2]
        y_min = bounding_scope[3]

        if fn(x_min, target_x_max) or fn(target_x_min, x_max) or fn(y_min, target_y_max) or fn(target_y_min, y_max):
            in_the_bound = 2
            return in_the_bound, [-1, -1, -1, -1]

        if fn(target_x_max, x_max):
            in_the_bound = 1
            target_x_max = x_max

        if fn(x_min, target_x_min):
            in_the_bound = 1
            target_x_min = x_min

        if fn(target_y_max, y_max):
            in_the_bound = 1
            target_y_max = y_max

        if fn(y_min, target_y_min):
            in_the_bound = 1
            target_y_min = y_min

        return in_the_bound, [target_x_max, target_x_min, target_y_max, target_y_min]

    def close_enough(a, b):
        return abs(a - b) <= 4

    # convert a list of 3d points to 2d image coordinate
    def convert_to_2d(spot, view_proj_mat, rp_width, rp_height):
        point_homo = np.pad(spot, ((0, 0), (0, 1)), constant_values=1.0)
        joint_pos2d = np.dot(point_homo, view_proj_mat)
        joint_pos2d = joint_pos2d / (joint_pos2d[..., -1:])
        joint_pos2d = 0.5 * (joint_pos2d[..., :2] + 1)
        joint_pos2d *= np.array([-rp_width, rp_height])
        joint_pos2d = np.array([0, rp_height]) - joint_pos2d
        return joint_pos2d.tolist()

    def in_scope(spot, tight_box):
        if spot[0] < tight_box[0] and spot[0] > tight_box[1] and spot[1] < tight_box[2] and spot[1] > tight_box[3]:
            return True
        return False

    # calculate 3d bounding box information
    def generate_bounding_box_information(camera_view, camera_projection, metadata, width, height):
        camera_xform = camera_view

        # calculate character's transform matrix
        obj_xform = np.identity(4)
        obj_xform[:3, :3] = Utils.rot_z(metadata["rot_deg_z"])
        xlate = metadata["translate"]
        obj_xform[3, 0] = xlate[0]
        obj_xform[3, 1] = xlate[1]
        obj_xform[3, 2] = xlate[2]

        obj_extent = np.array(metadata["scale"]) / 2

        # calculate bounding boxs vertex translate in the model space:
        model_pts = [
            (0, 0, 0),
            (-obj_extent[0], obj_extent[1], 0),
            (-obj_extent[0], -obj_extent[1], 0),
            (-obj_extent[0], obj_extent[1], 2 * obj_extent[2]),
            (-obj_extent[0], -obj_extent[1], 2 * obj_extent[2]),
            (obj_extent[0], obj_extent[1], 0),
            (obj_extent[0], -obj_extent[1], 0),
            (obj_extent[0], obj_extent[1], 2 * obj_extent[2]),
            (obj_extent[0], -obj_extent[1], 2 * obj_extent[2]),
        ]

        # convert vertexs'3d location in world coordination to 2d and camera coordination
        screen_camera_pts = [
            Utils.model_to_2d(pt, obj_xform, camera_xform, camera_projection, width, height) for pt in model_pts
        ]

        # extract character 's camera space rotation quaterion from character's camera space transform matrix
        quaternion_xyzw = R.from_matrix(np.transpose((obj_xform @ camera_xform)[:3, :3])).as_quat()
        quaternion = [quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1], quaternion_xyzw[2]]
        rotation_in_degree_numpy = rot_utils.quats_to_euler_angles(np.array(quaternion))

        # convert rotation to degree
        rotation_in_degree = [
            rotation_in_degree_numpy[0] * 57.296,
            rotation_in_degree_numpy[1] * 57.296,
            rotation_in_degree_numpy[2] * 57.296,
        ]
        object_dict = {
            "keypoints_3d": [list(pt[1]) for pt in screen_camera_pts],
            "rotate_in_degree": rotation_in_degree,
            "location": list(screen_camera_pts[0][1]),
            "projected_cuboid": [list(pt[0]) for pt in screen_camera_pts],
            "scale": [metadata["scale"][0], metadata["scale"][1], metadata["scale"][2]],
        }

        return object_dict

    def model_to_2d(pt, obj_xform, camera_xform, proj_mat, screen_width, screen_height):
        model_space = np.array([pt[0], pt[1], pt[2], 1])
        world_space = model_space @ obj_xform
        # camera_space = world_space @ np.linalg.inv(camera_xform)
        camera_space = world_space @ camera_xform
        ndc_space = camera_space @ proj_mat
        ndc_space /= ndc_space[3]
        x = (1 + ndc_space[0]) / 2 * screen_width
        y = (1 - ndc_space[1]) / 2 * screen_height
        return (int(x), int(y)), (camera_space)

    def get_trig(deg):
        return math.cos(math.radians(deg)), math.sin(math.radians(deg))

    def rot_z(deg):
        c, s = Utils.get_trig(deg)
        return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

    # Handle trancated character:
    # recalculate character' 2d loose bounding box with character's joint position
    def recalculate_loose_bounding_box(joint_3d_list, character_pos, character_rot, view_proj_mat, rp_width, rp_height):
        char_pos = character_pos
        char_rot = character_rot
        forward_max = 0
        right_max = 0
        up_max = 0
        forward_min = 100
        right_min = 100
        # calculate character's facing direction
        character_rot_in_degree = Utils.convert_to_angle(char_rot)
        character_rot_normal = character_rot_in_degree - 90
        forward_direction = [
            math.cos(math.radians(character_rot_in_degree)),
            math.sin(math.radians(character_rot_in_degree)),
        ]
        right_direction = [math.cos(math.radians(character_rot_normal)), math.sin(math.radians(character_rot_normal))]

        # traverse character's position to calculate the scale of the character
        for pos in joint_3d_list:
            vector_2d = [pos[0] - char_pos[0], pos[1] - char_pos[1]]
            forward_project = vector_2d[0] * forward_direction[0] + vector_2d[1] * forward_direction[1]
            right_project = vector_2d[0] * right_direction[0] + vector_2d[1] * right_direction[1]
            height = pos[2]
            if height > up_max:
                up_max = height
            if right_project > right_max:
                right_max = right_project
            if forward_project > forward_max:
                forward_max = forward_project
            if right_project < right_min:
                right_min = right_project
            if forward_project < forward_min:
                forward_min = forward_project

        # offsets are added to the character according to difference between skeleton and mesh
        forward_max = forward_max + 0.05
        forward_min = forward_min - 0.05
        right_max = right_max + 0.1
        right_min = right_min - 0.1
        up_max = up_max + 0.125
        forward_value = carb.Float3(forward_direction[0], forward_direction[1], 0)
        forward = Utils.add3(Utils.scale3(forward_value, forward_max), char_pos)
        backward = Utils.add3(Utils.scale3(forward_value, forward_min), char_pos)
        right_value = carb.Float3(right_direction[0], right_direction[1], 0)
        right = Utils.add3(Utils.scale3(right_value, right_max), char_pos)
        left = Utils.add3(Utils.scale3(right_value, right_min), char_pos)
        up = carb.Float3(char_pos[0], char_pos[1], char_pos[2] + up_max)
        vertex_3d_list = [forward, backward, right, left, up]
        vertex_2d_list = Utils.convert_to_2d(vertex_3d_list, view_proj_mat, rp_width, rp_height)
        x_max = -10000
        x_min = 10000
        y_max = -10000
        y_min = 10000

        for joint_2d in vertex_2d_list:
            if joint_2d[0] > x_max:
                x_max = joint_2d[0]
            if joint_2d[0] < x_min:
                x_min = joint_2d[0]
            if joint_2d[1] > y_max:
                y_max = joint_2d[1]
            if joint_2d[1] < y_min:
                y_min = joint_2d[1]
        return [x_max, x_min, y_max, y_min]

    # calculate how close two color are
    def color_distance(c1, c2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

    # check whether color is to close to colors in the list
    def is_too_close(new_color, colors, threshold=25):
        return any(Utils.color_distance(new_color, color) < threshold for color in colors)

    def generate_colors(seed, length, exclude_colors=[]):
        random.seed(seed)
        colors = []
        last_color = (0, 0, 0)
        max_attempts = 25
        while len(colors) < length and max_attempts > 0:
            new_color = []
            for value in last_color:
                delta = random.choice([random.randint(-255, -128), random.randint(128, 255)])
                new_value = (value + delta) % 256  # ensure the generated color in (0,255)
                new_color.append(new_value)
            new_color_tuple = tuple(new_color)
            if not Utils.is_too_close(new_color_tuple, exclude_colors):
                last_color = new_color_tuple
                colors.append((last_color[0], last_color[1], last_color[2], 255))  # append the transparent
                exclude_colors.append((last_color[0], last_color[1], last_color[2]))
                max_attempts = 25
            else:
                max_attempts -= 1

        if length > len(colors):
            carb.log_error("Unable to generate required number of randomized colors.")

        return colors

    # calculate character's intrinsic information
    def get_camera_info(width, height, focal, horizontalApeture):

        aspect_ratio = width / height
        pinhole_ratio = 2 * focal * width / height / horizontalApeture
        fx = width * pinhole_ratio / aspect_ratio / 2
        fy = height * pinhole_ratio / 2
        cx = width / 2
        cy = height / 2
        metadata = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
        return metadata


class Objectron_Utils:
    def get_character_transform(c):
        pos = carb.Float3(0, 0, 0)
        rot = carb.Float4(0, 0, 0, 0)
        c.get_world_transform(pos, rot)
        return pos, rot

    def get_character_joint_transform(c, joint_name):
        pos = carb.Float3(0, 0, 0)
        rot = carb.Float4(0, 0, 0, 0)
        c.get_joint_transform(joint_name, pos, rot)
        return pos, rot

    def get_camera_transform(camera_prim_path):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(camera_prim_path)
        camera_xform = omni.usd.get_world_transform_matrix(prim)
        return camera_xform

    def get_rotation_matrix_3d(rotation_xyz):
        return Utils.rot_x(rotation_xyz[0]) @ Utils.rot_y(rotation_xyz[1]) @ Utils.rot_z(rotation_xyz[2])

    def read(path):
        with open(path) as infile:
            json_data = json.load(infile)
            return json_data

    def to_array(mat):
        return [[i for i in row] for row in mat]

    def get_projection_matrix(camera_intrinsics):
        n, f = camera_intrinsics["near_clip"], camera_intrinsics["far_clip"]
        P = (n + f) / (n - f)
        Q = 2 * n * f / (n - f)
        screen_width, screen_height = camera_intrinsics["screen_width"], camera_intrinsics["screen_height"]
        pinhole_ratio = camera_intrinsics["pinhole_ratio"]
        return np.transpose(
            np.array(
                [
                    [pinhole_ratio / screen_width * screen_height, 0, 0, 0],
                    [0, pinhole_ratio, 0, 0],
                    [0, 0, P, Q],
                    [0, 0, -1, 0],
                ]
            )
        )

    def model_to_2d(pt, obj_xform, camera_xform, proj_mat, screen_width, screen_height):

        model_space = np.array([pt[0], pt[1], pt[2], 1])
        # calculate the world space position by cross product those two
        world_space = model_space @ obj_xform

        camera_space = world_space @ np.linalg.inv(camera_xform)
        ndc_space = camera_space @ proj_mat
        ndc_space /= ndc_space[3]
        x = (1 + ndc_space[0]) / 2 * screen_width
        y = (1 - ndc_space[1]) / 2 * screen_height
        return (int(x), int(y)), (camera_space[:3] @ Utils.rot_z(90))

    def projection_matrix_format(camera_projection):
        proj_mat_output = np.transpose(camera_projection)
        proj_mat_output[0][0], proj_mat_output[1][1] = proj_mat_output[1][1], proj_mat_output[0][0]
        return Objectron_Utils.to_array(proj_mat_output)

    def map_pts(pts):
        new_pts = []

        order = [0, 3, 1, 4, 2, 7, 5, 8, 6]
        for i in order:
            new_pts.append(np.array(pts[i]) @ Utils.rot_x(-90))
        return new_pts

    def generate_bounding_box_information(camera_intrinsics, camera_projection, metadata, width, height):

        # rot_xyz = camera_info["camera_rot_xyz"]
        xlate = camera_intrinsics["camera_xlate"] @ Utils.rot_x(-90)
        camera_xform = np.identity(4)
        camera_xform[:3, :3] = camera_intrinsics["camera_rotate"] @ Utils.rot_x(-90)
        camera_xform[3, 0] = xlate[0]
        camera_xform[3, 1] = xlate[1]
        camera_xform[3, 2] = xlate[2]

        obj_xform = np.identity(4)
        obj_xform[:3, :3] = Utils.rot_x(90) @ Utils.rot_z(metadata["rot_deg_z"]) @ Utils.rot_x(-90)
        xlate = metadata["translate"] @ Utils.rot_x(-90)
        obj_xform[3, 0] = xlate[0]
        obj_xform[3, 1] = xlate[1] + metadata["scale"][2] / 2
        obj_xform[3, 2] = xlate[2]

        obj_extent = np.array(metadata["scale"]) / 2

        model_pts = [
            (0, 0, 0),
            (-obj_extent[0], -obj_extent[1], -obj_extent[2]),
            (-obj_extent[0], -obj_extent[1], obj_extent[2]),
            (-obj_extent[0], obj_extent[1], -obj_extent[2]),
            (-obj_extent[0], obj_extent[1], obj_extent[2]),
            (obj_extent[0], -obj_extent[1], -obj_extent[2]),
            (obj_extent[0], -obj_extent[1], obj_extent[2]),
            (obj_extent[0], obj_extent[1], -obj_extent[2]),
            (obj_extent[0], obj_extent[1], obj_extent[2]),
        ]

        model_pts = Objectron_Utils.map_pts(model_pts)
        scale = metadata["scale"]
        scale[1], scale[2] = scale[2], scale[1]

        screen_camera_pts = [
            Objectron_Utils.model_to_2d(pt, obj_xform, camera_xform, camera_projection, width, height)
            for pt in model_pts
        ]
        object_dict = {
            "keypoints_3d": [list(pt[1]) for pt in screen_camera_pts],
            "quaternion_xyzw": list(
                R.from_matrix(
                    np.transpose(obj_xform[:3, :3] @ np.linalg.inv(camera_xform[:3, :3]) @ Utils.rot_z(90))
                ).as_quat()
            ),
            "location": list(screen_camera_pts[0][1]),
            "projected_cuboid": [list(pt[0]) for pt in screen_camera_pts],
            "scale": [float(scale[0]), float(scale[1]), float(scale[2])],
        }

        return object_dict

    def get_camera_info(
        camera_translate, camera_rotation, width, height, focal, horizontalApeture, near_clip, far_clip
    ):

        aspect_ratio = width / height
        pinhole_ratio = 2 * focal * width / height / horizontalApeture
        fx = width * pinhole_ratio / aspect_ratio / 2
        fy = height * pinhole_ratio / 2
        cx = width / 2
        cy = height / 2
        metadata = {
            "camera_xlate": camera_translate,
            "camera_rotate": camera_rotation,
            "screen_width": width,
            "screen_height": height,
            "pinhole_ratio": pinhole_ratio,
            "near_clip": near_clip,
            "far_clip": far_clip,
            "focal": focal,
            "horizontalApeture": horizontalApeture,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        }

        return metadata
