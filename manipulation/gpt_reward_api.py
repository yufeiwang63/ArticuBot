import pybullet as p
import numpy as np
from manipulation.utils import get_pc
from manipulation.grasping_utils import get_pc_and_normal, get_full_pc_aroung_obj
from manipulation.utils import take_round_images, rotate_point_around_axis, find_nearest_point_on_line, load_obj
import os
from scipy import ndimage
import json
import random
import matplotlib.pyplot as plt

def compute_obj_to_center_dist(simulator, obj_a, obj_b):
    obj_a_center = get_position(simulator, obj_a)
    obj_b_bbox_min, obj_b_bbox_max = get_bounding_box(simulator, obj_b)
    obj_b_center = (obj_b_bbox_min + obj_b_bbox_max) / 2
    return np.linalg.norm(obj_a_center - obj_b_center)

def get_initial_joint_angle(simulator, object_name, joint_name):
    object_name = object_name.lower()
    return simulator.initial_joint_angle[object_name][joint_name]

def get_initial_pos_orient(simulator, object_name):
    object_name = object_name.lower()
    return simulator.initial_pos[object_name], np.array(p.getEulerFromQuaternion(simulator.initial_orient[object_name]))

### success check functions
def gripper_close_to_object_link(simulator, object_name, link_name):
    link_pc = get_link_pc(simulator, object_name, link_name)
    gripper_pos, _ = get_eef_pose(simulator)
    distance = np.linalg.norm(link_pc.reshape(-1, 3) - gripper_pos.reshape(1, 3), axis=1)
    if np.min(distance) < 0.06:
        return True
    return False

def gripper_close_to_object(simulator, object_name):
    object_pc, _ = get_pc_and_normal(simulator, object_name)
    gripper_pos, _ = get_eef_pose(simulator)
    distance = np.linalg.norm(object_pc.reshape(-1, 3) - gripper_pos.reshape(1, 3), axis=1)
    if np.min(distance) < 0.06:
        return True
    return False

def check_grasped(self, object_name, link_name=None):
    object_name = object_name.lower()
    grasped_object_name, grasped_link_name = get_grasped_object_and_link_name(self)
    if link_name is None:
        return grasped_object_name == object_name
    else:
        return grasped_object_name == object_name and grasped_link_name == link_name

def get_grasped_object_name(simulator):
    grasped_object_id = simulator.suction_obj_id
    if grasped_object_id is None:
        return None
    
    id_to_name = {v: k for k, v in simulator.urdf_ids.items()}
    return id_to_name[grasped_object_id]

def get_grasped_object_and_link_name(simulator):
    grasped_object_id = simulator.suction_obj_id
    grasped_link_id = simulator.suction_contact_link
    if grasped_object_id is None or grasped_link_id is None:
        return None, None
    
    id_to_name = {v: k for k, v in simulator.urdf_ids.items()}
    grasped_obj_name = id_to_name[grasped_object_id]

    if grasped_link_id == -1:
        return grasped_obj_name, "base"
    
    joint_info = p.getJointInfo(grasped_object_id, grasped_link_id, physicsClientId=simulator.id)
    link_name = joint_info[12].decode("utf-8")

    return grasped_obj_name, link_name

def get_joint_limit(simulator, object_name, custom_joint_name):
    object_name = object_name.lower()
    object_id = simulator.urdf_ids[object_name]
    num_joints = p.getNumJoints(object_id, physicsClientId=simulator.id)
    
    urdf_joint_name = custom_joint_name
    max_joint_val = 0
    min_joint_val = 0
    for j_id in range(num_joints):
        joint_info = p.getJointInfo(object_id, j_id, physicsClientId=simulator.id)
        if joint_info[1].decode("utf-8") == urdf_joint_name:
            max_joint_val = joint_info[9]
            min_joint_val = joint_info[8]
            break
    
    if min_joint_val < max_joint_val:
        return min_joint_val, max_joint_val
    else:
        return max_joint_val, min_joint_val

def get_position(simulator, object_name):
    object_name = object_name.lower()
    object_id = simulator.urdf_ids[object_name]
    return np.array(p.getBasePositionAndOrientation(object_id, physicsClientId=simulator.id)[0])

def get_mc_position(simulator, object_name):
    object_name = object_name.lower()
    object_id = simulator.urdf_ids[object_name]
    bounding_box_low, bounding_box_high = simulator.get_aabb(object_id)
    return (bounding_box_low + bounding_box_high) / 2


def get_velocity(simulator, object_name):
    object_name = object_name.lower()
    object_id = simulator.urdf_ids[object_name]
    return np.array(p.getBaseVelocity(object_id, physicsClientId=simulator.id)[0])

def get_orientation(simulator, object_name):
    object_name = object_name.lower()
    object_id = simulator.urdf_ids[object_name]
    return np.array(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(object_id, physicsClientId=simulator.id)[1]))

def get_eef_pose(simulator):
    robot_eef_pos, robot_eef_orient = simulator.robot.get_pos_orient(simulator.robot.right_end_effector)
    return np.array(robot_eef_pos).flatten(), np.array(p.getEulerFromQuaternion(robot_eef_orient)).flatten()

def get_finger_pos(simulator):
    left_finger_joint_pos =  p.getLinkState(simulator.robot.body, simulator.robot.right_gripper_indices[0], physicsClientId=simulator.id)[0]
    right_finger_joint_pos = p.getLinkState(simulator.robot.body, simulator.robot.right_gripper_indices[1], physicsClientId=simulator.id)[0]
    return np.array(left_finger_joint_pos), np.array(right_finger_joint_pos)

def get_finger_distance(simulator): 
    left_finger_joint_angle = p.getJointState(simulator.robot.body, simulator.robot.right_gripper_indices[0], physicsClientId=simulator.id)[0]
    right_finger_joint_angle = p.getJointState(simulator.robot.body, simulator.robot.right_gripper_indices[1], physicsClientId=simulator.id)[0]
    return left_finger_joint_angle + right_finger_joint_angle

def get_bounding_box(simulator, object_name):
    object_name = object_name.lower()
    object_id = simulator.urdf_ids[object_name]
    if object_name != "init_table":
        return simulator.get_aabb(object_id)
    else:
        return simulator.table_bbox_min, simulator.table_bbox_max

def get_bounding_box_link(simulator, object_name, link_name):
    object_name = object_name.lower()
    link_id = get_link_id_from_name(simulator, object_name, link_name)
    object_id = simulator.urdf_ids[object_name]
    return simulator.get_aabb_link(object_id, link_id)
    
def get_joint_state(simulator, object_name, custom_joint_name):
    object_name = object_name.lower()
    object_id = simulator.urdf_ids[object_name]
    num_joints = p.getNumJoints(object_id, physicsClientId=simulator.id)

    urdf_joint_name = custom_joint_name
    for i in range(num_joints):
        joint_info = p.getJointInfo(object_id, i, physicsClientId=simulator.id)
        if joint_info[1].decode("utf-8") == urdf_joint_name:
            joint_index = i
            break
        
    return np.array(p.getJointState(object_id, joint_index, physicsClientId=simulator.id)[0])

def get_link_state(simulator, object_name, custom_link_name):
    object_name = object_name.lower()
    object_id = simulator.urdf_ids[object_name]
    urdf_link_name = custom_link_name
    link_id = get_link_id_from_name(simulator, object_name, urdf_link_name)
    link_pos, link_orient = p.getLinkState(object_id, link_id, physicsClientId=simulator.id)[:2]
    return np.array(link_pos)
    

def get_link_pose(simulator, object_name, custom_link_name):
    object_name = object_name.lower()
    object_id = simulator.urdf_ids[object_name]
    urdf_link_name = custom_link_name
    link_id = get_link_id_from_name(simulator, object_name, urdf_link_name)
    link_pos, link_orient = p.getLinkState(object_id, link_id, physicsClientId=simulator.id)[:2]
    return np.array(link_pos), np.array(link_orient)

def get_link_pc(simulator, object_name, custom_link_name):
    object_name = object_name.lower()
    urdf_link_name = custom_link_name 
    link_com, all_pc = render_to_get_link_com(simulator, object_name, urdf_link_name)

    return all_pc

def get_groundtruth_link_pc(simulator, object_name, custom_link_name):
    object_name = object_name.lower()
    urdf_link_name = custom_link_name
    link_pc = render_to_get_groundtruth_link_com(simulator, object_name, urdf_link_name)
    return link_pc

def set_joint_value(simulator, object_name, joint_name, joint_value="max"):
    object_name = object_name.lower()
    object_id = simulator.urdf_ids[object_name]
    num_joints = p.getNumJoints(object_id, physicsClientId=simulator.id)
    joint_index = None
    max_joint_val = 0
    min_joint_val = 0
    for j_id in range(num_joints):
        joint_info = p.getJointInfo(object_id, j_id, physicsClientId=simulator.id)
        if joint_info[1].decode("utf-8") == joint_name:
            joint_index = j_id
            max_joint_val = joint_info[9]
            min_joint_val = joint_info[8]
            break
    
    if joint_value == 'max':
        p.resetJointState(object_id, joint_index, max_joint_val, physicsClientId=simulator.id)
    elif joint_value == 'min':
        p.resetJointState(object_id, joint_index, min_joint_val, physicsClientId=simulator.id)
    else:
        p.resetJointState(object_id, joint_index, joint_value, physicsClientId=simulator.id)


def in_bbox(simulator, pos, bbox_min, bbox_max):
    if (pos[0] <= bbox_max[0] and pos[0] >= bbox_min[0] and \
        pos[1] <= bbox_max[1] and pos[1] >= bbox_min[1] and \
        pos[2] <= bbox_max[2] and pos[2] >= bbox_min[2]):
        return True
    return False

def grasped(simulator, object_name):
    if object_name in simulator.grasped_object_list:
        return True
    return False

def render_to_get_link_com(simulator, object_name, urdf_link_name):    
    ### make all other objects invisiable
    prev_rgbas = []
    object_id = simulator.urdf_ids[object_name]
    for obj_name, obj_id in simulator.urdf_ids.items():
        if obj_name != object_name:
            num_links = p.getNumJoints(obj_id, physicsClientId=simulator.id)
            for link_idx in range(-1, num_links):
                prev_rgba = p.getVisualShapeData(obj_id, link_idx, physicsClientId=simulator.id)[0][14:18]
                prev_rgbas.append(prev_rgba)
                p.changeVisualShape(obj_id, link_idx, rgbaColor=[0, 0, 0, 0], physicsClientId=simulator.id)

    ### center camera to the target object
    env_prev_view_matrix, env_prev_projection_matrix = simulator.view_matrix, simulator.projection_matrix
    camera_width = 640
    camera_height = 480
    obj_id = object_id
    min_aabb, max_aabb = simulator.get_aabb(obj_id)
    camera_target = (max_aabb + min_aabb) / 2
    distance = np.linalg.norm(max_aabb - min_aabb) * 1.2
    elevation = 30

    ### get a round of images of the target object
    rgbs, depths, view_matrices, projection_matrices = take_round_images(
        simulator, camera_target, distance, elevation, 
        camera_width=camera_width, camera_height=camera_height, 
        z_near=0.01, z_far=10,
        return_camera_matrices=True)

    ### make the target link invisiable
    link_id = get_link_id_from_name(simulator, object_name, urdf_link_name)
    # import pdb; pdb.set_trace()
    prev_link_rgba = p.getVisualShapeData(obj_id, link_id, physicsClientId=simulator.id)[0][14:18]
    p.changeVisualShape(obj_id, link_id, rgbaColor=[0, 0, 0, 0], physicsClientId=simulator.id)

    ### get a round of images of the target object with link invisiable
    rgbs_link_invisiable, depths_link_invisible, _, _ = take_round_images(
        simulator, camera_target, distance, elevation,
        camera_width=camera_width, camera_height=camera_height, 
        z_near=0.01, z_far=10, return_camera_matrices=True
    )

    ### use subtraction to get the link mask
    max_num_diff_pixels = 0
    best_idx = 0
    for idx, (depth, depth_) in enumerate(zip(depths, depths_link_invisible)):
        diff_image = np.abs(depth - depth_)
        diff_pixels = np.sum(diff_image > 0)
        if diff_pixels > max_num_diff_pixels:
            max_num_diff_pixels = diff_pixels
            best_idx = idx
    best_mask = np.abs(depths[best_idx] - depths_link_invisible[best_idx]) > 0
    # best_mask = np.any(best_mask)


    ### get the link mask center
    center = ndimage.measurements.center_of_mass(best_mask)
    center = [int(center[0]), int(center[1])]

    ### back project the link mask center to get the link com in 3d coordinate
    best_pc = get_pc(projection_matrices[best_idx], view_matrices[best_idx], depths[best_idx], camera_width, camera_height)
    
    pt_idx = center[0] * camera_width + center[1]
    link_com = best_pc[pt_idx]
    best_pc = best_pc.reshape((camera_height, camera_width, 3))
    all_pc = best_pc[best_mask]


    ### reset the object and link rgba to previous values, and the simulator view matrix and projection matrix
    p.changeVisualShape(obj_id, link_id, rgbaColor=prev_link_rgba, physicsClientId=simulator.id)

    cnt = 0
    object_id = simulator.urdf_ids[object_name]
    for obj_name, obj_id in simulator.urdf_ids.items():
        if obj_name != object_name:
            num_links = p.getNumJoints(obj_id, physicsClientId=simulator.id)
            for link_idx in range(-1, num_links):
                p.changeVisualShape(obj_id, link_idx, rgbaColor=prev_rgbas[cnt], physicsClientId=simulator.id)
                cnt += 1

    simulator.view_matrix, simulator.projection_matrix = env_prev_view_matrix, env_prev_projection_matrix

    ### add a safety check here in case the rendering fails
    bounding_box = get_bounding_box_link(simulator, object_name, urdf_link_name)
    if not in_bbox(simulator, link_com, bounding_box[0], bounding_box[1]):
        link_com = (bounding_box[0] + bounding_box[1]) / 2

    return link_com, all_pc

def render_to_get_groundtruth_link_com(simulator, object_name, urdf_link_name):
    ### make all other objects invisiable
    prev_rgbas = []
    object_id = simulator.urdf_ids[object_name]
    for obj_name, obj_id in simulator.urdf_ids.items():
        if obj_name != object_name:
            num_links = p.getNumJoints(obj_id, physicsClientId=simulator.id)
            for link_idx in range(-1, num_links):
                prev_rgba = p.getVisualShapeData(obj_id, link_idx, physicsClientId=simulator.id)[0][14:18]
                prev_rgbas.append(prev_rgba)
                p.changeVisualShape(obj_id, link_idx, rgbaColor=[0, 0, 0, 0], physicsClientId=simulator.id)

    ### center camera to the target object
    env_prev_view_matrix, env_prev_projection_matrix = simulator.view_matrix, simulator.projection_matrix
    camera_width = 640
    camera_height = 480
    obj_id = object_id
    min_aabb, max_aabb = simulator.get_aabb(obj_id)
    camera_target = (max_aabb + min_aabb) / 2
    distance = np.linalg.norm(max_aabb - min_aabb) * 1.2
    elevation = 30


    ### make only the target link visible
    link_id = get_link_id_from_name(simulator, object_name, urdf_link_name)
    
    for joint_idx in range(p.getNumJoints(object_id, physicsClientId=simulator.id)):
        if joint_idx != link_id:
            prev_rgba = p.getVisualShapeData(obj_id, link_idx, physicsClientId=simulator.id)[0][14:18]
            prev_rgbas.append(prev_rgba)
            p.changeVisualShape(object_id, joint_idx, rgbaColor=[0, 0, 0, 0], physicsClientId=simulator.id)

    ### get a round of images of the target object
    pc = get_full_pc_aroung_obj(simulator, object_name, distance=distance, elevation=elevation, camera_height=camera_height, camera_width=camera_width, ignore_pc_below_table=False)
    link_pc = pc

    ### reset the object and link rgba to previous values, and the simulator view matrix and projection matrix
    cnt = 0
    object_id = simulator.urdf_ids[object_name]
    for obj_name, obj_id in simulator.urdf_ids.items():
        if obj_name != object_name:
            num_links = p.getNumJoints(obj_id, physicsClientId=simulator.id)
            for link_idx in range(-1, num_links):
                p.changeVisualShape(obj_id, link_idx, rgbaColor=prev_rgbas[cnt], physicsClientId=simulator.id)
                cnt += 1

    simulator.view_matrix, simulator.projection_matrix = env_prev_view_matrix, env_prev_projection_matrix

    return link_pc



def get_link_id_from_name(simulator, object_name, link_name):
    object_id = simulator.urdf_ids[object_name]
    num_joints = p.getNumJoints(object_id, physicsClientId=simulator.id)
    joint_index = None
    for i in range(num_joints):
        joint_info = p.getJointInfo(object_id, i, physicsClientId=simulator.id)
        if joint_info[12].decode("utf-8") == link_name:
            joint_index = i
            break

    return joint_index


def get_joint_id_from_name(simulator, object_name, joint_name):
    object_id = simulator.urdf_ids[object_name]
    num_joints = p.getNumJoints(object_id, physicsClientId=simulator.id)
    joint_index = None
    for i in range(num_joints):
        joint_info = p.getJointInfo(object_id, i, physicsClientId=simulator.id)
        if joint_info[1].decode("utf-8") == joint_name:
            joint_index = i
            break

    return joint_index

def bbox_in_bbox(bbox_a_min, bbox_a_max, bbox_b_min, bbox_b_max):
    # check if bbox_a is in bbox_b
    if (bbox_a_min[0] >= bbox_b_min[0] and bbox_a_min[1] >= bbox_b_min[1] and bbox_a_min[2] >= bbox_b_min[2] and \
        bbox_a_max[0] <= bbox_b_max[0] and bbox_a_max[1] <= bbox_b_max[1] and bbox_a_max[2] <= bbox_b_max[2]):
        return True
    return False

def get_gripper_pos(simulator):
    left_finger_pos = np.array(p.getLinkState(simulator.robot.body, simulator.robot.right_gripper_indices[0], physicsClientId=simulator.id)[0])
    right_finger_pos = np.array(p.getLinkState(simulator.robot.body, simulator.robot.right_gripper_indices[1], physicsClientId=simulator.id)[0])
    return left_finger_pos, right_finger_pos

def get_gripper_joint(simulator):
    return p.getJointState(simulator.robot.body, simulator.robot.right_gripper_indices[0], physicsClientId=simulator.id)[0]

def sample_point_inside_triangle(v1,v2,v3):
    r1 = random.uniform(0, 1)
    r2 = random.uniform(0, 1)
    while r1 + r2 >= 1:
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
    r3 = 1 - r1 - r2

    # Calculate the point using barycentric coordinates
    x = r1 * v1[0] + r2 * v2[0] + r3 * v3[0]
    y = r1 * v1[1] + r2 * v2[1] + r3 * v3[1]
    z = r1 * v1[2] + r2 * v2[2] + r3 * v3[2]
    return [x, y, z]

def sample_point_inside_triangle(v1,v2,v3):
    r1 = random.uniform(0, 1)
    r2 = random.uniform(0, 1)
    while r1 + r2 >= 1:
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
    r3 = 1 - r1 - r2

    # Calculate the point using barycentric coordinates
    x = r1 * v1[0] + r2 * v2[0] + r3 * v3[0]
    y = r1 * v1[1] + r2 * v2[1] + r3 * v3[1]
    z = r1 * v1[2] + r2 * v2[2] + r3 * v3[2]
    return [x, y, z]

# NOTE: hard-coded for now, should make it more general in the future
def get_handle_pos(simulator, obj_name, return_median=True, handle_pts_obj_frame=None, mobility_info=None, return_info=False, target_object='handle'):
    obj_name = obj_name.lower()
    scaling = simulator.simulator_sizes[obj_name]

    # get the parent frame of the revolute joint.
    obj_id = simulator.urdf_ids[obj_name] 

    # axis in parent frame, transform everything to world frame
    if mobility_info is None:
        urdf_path = simulator.urdf_paths[obj_name]
        parent_dir = os.path.dirname(urdf_path)
        start_idx = parent_dir.find("data/dataset")
        parent_dir = parent_dir[start_idx:]
        parent_dir = os.path.join(os.environ["PROJECT_DIR"], parent_dir)
        mobility_info = json.load(open(f"{parent_dir}/mobility_v2.json", "r"))
    
    # return a list of handle points in world frame
    ret_handle_pt_list = []
    ret_joint_idx_list = []

    joint_name = None
    parent_joint_name = None
    handle_idx = 0
    all_handle_pts_object_frame = []
    for idx, joint_info in enumerate(mobility_info):
        all_parts = [part["name"] for part in joint_info["parts"]]
        if target_object in all_parts:
            all_ids = [part["id"] for part in joint_info["parts"]]
            index = all_parts.index(target_object)
            handle_id = all_ids[index]
            joint_name = "joint_{}".format(joint_info["id"])
            parent_joint_name = "joint_{}".format(joint_info["parent"])
            joint_data = joint_info['jointData']
            axis_body = np.array(joint_data["axis"]["origin"]) * scaling
            axis_dir_body = np.array(joint_data["axis"]["direction"])
            joint_limit = joint_data["limit"]
            if joint_limit['a'] > joint_limit['b']:
                axis_dir_body = -axis_dir_body

            joint_idx = get_joint_id_from_name(simulator, obj_name, joint_name) # this is the joint id in pybullet
            parent_joint_idx = get_joint_id_from_name(simulator, obj_name, parent_joint_name) # this is the joint id in pybullet
            
            parent_link_state = p.getLinkState(obj_id, parent_joint_idx, physicsClientId=simulator.id) # NOTE: the handle link id should be dependent on the object urdf.
            # parent_link_state = p.getLinkState(obj_id, joint_idx, physicsClientId=simulator.id) # NOTE: the handle link id should be dependent on the object urdf.
            link_urdf_world_pos, link_urdf_world_orn = parent_link_state[0], parent_link_state[1]
            # this is the transformation from the parent frame to the world frame. 
            T_body_to_world = np.eye(4) # transformation from the parent body frame to the world frame
            T_body_to_world[:3, :3] = np.array(p.getMatrixFromQuaternion(link_urdf_world_orn)).reshape(3, 3)
            T_body_to_world[:3, 3] = link_urdf_world_pos
            
            axis_world = T_body_to_world[:3, :3] @ axis_body + T_body_to_world[:3, 3]   
            axis_pt2_body = np.array(axis_body) + axis_dir_body
            axis_end_world = T_body_to_world[:3, :3] @ axis_pt2_body + T_body_to_world[:3, 3]
            axis_dir_world = axis_end_world - axis_world

            # get the handle points in world frame
            if handle_pts_obj_frame is None:
                handle_obj_path = f"{parent_dir}/parts_render/{handle_id}{target_object}.obj" # NOTE: this path should be dependent on the object. 
                handle_pts, handle_faces = load_obj(handle_obj_path) # this is in object frame

                handle_pts = handle_pts * scaling
                # add more dense points around handle
                added_points = []
                for f in handle_faces:
                    v1,v2,v3 = f
                    v1 = handle_pts[v1-1]
                    v2 = handle_pts[v2-1]
                    v3 = handle_pts[v3-1]
                    a = np.linalg.norm(v1-v2)
                    b = np.linalg.norm(v2-v3)
                    c = np.linalg.norm(v3-v1)
                    s = (a+b+c) / 2
                    temp = max(0, s*(s-a)*(s-b)*(s-c))
                    surface = np.sqrt(temp)
                    num_points = surface * 1e6
                    num_points = int(num_points)
                    num_points = np.clip(num_points, 0, 5)
                    added_points.extend([sample_point_inside_triangle(v1,v2,v3) for _ in range(num_points)])

                if added_points != []:
                    added_points = np.array(added_points)
                    handle_pts = np.concatenate((handle_pts, added_points), axis=0)
                    
                all_handle_pts_object_frame.append(handle_pts)
                    
            else:
                handle_pts = handle_pts_obj_frame[handle_idx]
            
            
            # transform this to the world frame using the object *base*'s position and orientation
            handle_points_world = T_body_to_world[:3, :3] @ handle_pts.T + T_body_to_world[:3, 3].reshape(3, 1) # 3 x N
            if return_median:
                handle_point_median = np.median(handle_points_world, axis=1)
            else:
                handle_point_median = handle_points_world.T

            # find the projection of the handle point to the rotation axis, in world frame. 
            project_on_rotation_axis = find_nearest_point_on_line(axis_world, axis_end_world, handle_point_median)
            # p.addUserDebugLine(project_on_rotation_axis, handle_point_median, [1, 0, 0], 25, 0)

            # TODO: GPT can parse the mobility.json to get the joint name. 
            joint_info = p.getJointInfo(obj_id, joint_idx, physicsClientId=simulator.id)
            joint_type = joint_info[2]
            
            if joint_type == p.JOINT_REVOLUTE:
                rotation_angle = p.getJointState(obj_id, joint_idx, physicsClientId=simulator.id)[0] # NOTE: this joint id should be dependent on the object urdf.
                rotated_handle_pt_local = rotate_point_around_axis(handle_point_median - project_on_rotation_axis, axis_dir_world, rotation_angle)
                rotated_handle_pt = project_on_rotation_axis + rotated_handle_pt_local
            elif joint_type == p.JOINT_PRISMATIC:
                translation = p.getJointState(obj_id, joint_idx, physicsClientId=simulator.id)[0]
                rotated_handle_pt = handle_point_median + axis_dir_world * translation
                
            # import pdb; pdb.set_trace()
            # rotated_handle_pt = handle_points_world.T

            if return_median:
                ret_handle_pt_list.append(rotated_handle_pt.flatten())
            else:
                ret_handle_pt_list.append(rotated_handle_pt)
            ret_joint_idx_list.append(joint_idx)
            
            handle_idx += 1
            
    if return_info:
        return ret_handle_pt_list, ret_joint_idx_list, all_handle_pts_object_frame, mobility_info
    
    return ret_handle_pt_list, ret_joint_idx_list