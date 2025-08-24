import pybullet as p
import os
import numpy as np
import open3d as o3d
from manipulation.motion_planning_utils import motion_planning
from manipulation.grasping_utils import align_gripper_z_with_normal, align_gripper_x_and_z
from manipulation.utils import *
import scipy
import time
from termcolor import cprint
import fpsample
from multiprocessing import Pool
import pickle
import json

MOTION_PLANNING_TRY_TIMES=100
SAMPLE_ORIENTATION_NUM=3
PARALLEL_POOL_NUM=40
HANDLE_FPS_NUM_POINT=15

def get_save_path(simulator):
    state_save_path = os.path.join(simulator.primitive_save_path, "states")
    if not os.path.exists(state_save_path):
        os.makedirs(state_save_path)
    return simulator.primitive_save_path

def approach_object_link_parallel(simulator, object_name, link_name, debug=False):    
    save_path = get_save_path(simulator)
    ori_simulator_state = save_env(simulator, None)
    object_name = object_name.lower()
    link_name = link_name.lower()
    link_pc, view, projection, img = simulator.get_link_pc(object_name, link_name)
    object_pc = link_pc
    
    pcd = o3d.geometry.PointCloud() 
    pcd.points = o3d.utility.Vector3dVector(object_pc)
    pcd.estimate_normals()
    object_normal = np.asarray(pcd.normals)

    all_handle_pos, handle_joint_id, axis_world, axis_end_world = simulator.get_handle_pos(return_median=False, custom_joint_name=simulator.handle_name)
    threshold = 0.02
        
    handle_pc, handle_joint_id, handle_median, _ = get_link_handle(all_handle_pos, handle_joint_id, link_pc, threshold=threshold)
    if handle_pc.shape[0] <= 5:
        print("No handle point cloud found, return")
        return [], []

    axis = axis_world[0]
    axis_end = axis_end_world[0]
    handle_orientation = get_handle_orient(handle_pc)
    distance_pc = pc_to_line_distance(handle_pc, axis, axis_end)

    # clip the handle point cloud to get graspable points
    max_distance = np.max(distance_pc)
    handle_clip_factor = (0.0, 1.0, 0.0, 1.0, 0.0)
    selected_idx_screw= np.where((distance_pc > max_distance * handle_clip_factor[0]) & (distance_pc < max_distance * handle_clip_factor[1]))[0]
    max_z = np.max(handle_pc[:, 2])
    min_z = np.min(handle_pc[:, 2])
    selected_idx_height = np.where((handle_pc[:, 2] > min_z + (max_z - min_z) * handle_clip_factor[2]) & (handle_pc[:, 2] < min_z + (max_z - min_z) * handle_clip_factor[3]))[0]
    selected_idx = np.intersect1d(selected_idx_screw, selected_idx_height)

    handle_pc = handle_pc[selected_idx]
    # return if there is no handle point cloud
    if handle_pc.shape[0] <= 5:
        print("No handle point cloud found, return")
        return [], []

    handle_dir = estimate_line_direction(handle_pc)
    handle_pc_project = np.dot(handle_pc, handle_dir)
    min_handle_pc_project = np.min(handle_pc_project)
    max_handle_pc_project = np.max(handle_pc_project)
    selected_idx = np.where((handle_pc_project > min_handle_pc_project + (max_handle_pc_project - min_handle_pc_project) * handle_clip_factor[4]) & (handle_pc_project < min_handle_pc_project + (max_handle_pc_project - min_handle_pc_project) * (1-handle_clip_factor[4])))[0]
    
    handle_pc = handle_pc[selected_idx]
    handle_median = np.median(handle_pc, axis=0)
   
    # use fps to get a bunch of trying points
    fps_point = HANDLE_FPS_NUM_POINT 
    handle_fps_num_point = min(fps_point, len(handle_pc))
    h = min(3, int(np.log2(handle_fps_num_point)))
    kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(handle_pc, handle_fps_num_point, h=h)
    to_try_handle_points = handle_pc[kdline_fps_samples_idx] 
    to_try_handle_points = np.concatenate([to_try_handle_points, handle_median.reshape(1, 3)], axis=0)
    
    # find the necessary args
    args = []

    # parallel motion planning to search each fps handle point
    env_kwargs = {
        "task_config": simulator.config_path, 
        "env_name": "articulated",
        "task_name": simulator.task_name, 
        "restore_state_file": simulator.restore_state_file, 
        "render": False if not debug else True, 
        "randomize": False, 
        "obj_id": simulator.obj_id, 
    }
    
    
    horizontal_grasp = True if handle_orientation == 'vertical' else False
    
    ## first compute some parameters to use
    mp_target_poses = []
    real_target_poses = []
    target_orientations = []

    for target_pos in to_try_handle_points:
        nearest_point_idx = np.argmin(np.linalg.norm(object_pc - target_pos.reshape(1, 3), axis=1))
        align_normal = object_normal[nearest_point_idx]
        # compute the normal of the point sampled from the handle point cloud
        
        low, high = simulator.get_bounding_box(object_name)
        com = (low + high) / 2
        line = com - target_pos
        if np.dot(line, align_normal) > 0:
            align_normal = -align_normal
        
        for normal in [align_normal]:
            real_target_pos = target_pos + normal * -0.02
            mp_target_pos = target_pos + normal * 0.04

            for orientation_idx in range(SAMPLE_ORIENTATION_NUM):
                # target_orientation = align_gripper_z_with_normal(-normal, horizontal=horizontal_grasp, randomize=True).as_quat()
                target_orientation = align_gripper_x_and_z(handle_dir, -normal, randomize=True).as_quat()
                mp_target_poses.append(mp_target_pos)
                real_target_poses.append(real_target_pos)
                target_orientations.append(target_orientation)
                
            # target_orientation_1 = align_gripper_z_with_normal(-normal, horizontal=horizontal_grasp, randomize=False, flip=False).as_quat()
            target_orientation_1 = align_gripper_x_and_z(handle_dir, -normal, randomize=False, flip=False).as_quat()
            # target_orientation_2 = align_gripper_z_with_normal(-normal, horizontal=horizontal_grasp, randomize=False, flip=True).as_quat()
            target_orientation_2 = align_gripper_x_and_z(handle_dir, -normal, randomize=False, flip=True).as_quat()
            mp_target_poses.append(mp_target_pos); mp_target_poses.append(mp_target_pos) 
            real_target_poses.append(real_target_pos); real_target_poses.append(real_target_pos)
            target_orientations.append(target_orientation_1); target_orientations.append(target_orientation_2)  

    # project the target position to the img
    def project_point(points_world, view_matrix, proj_matrix, width=640, height=480):
        # Ensure matrices are in correct shape and column-major order
        view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
        proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")

        # Compose projection matrix: projection * view
        proj_view_matrix = proj_matrix @ view_matrix

        # Add 1 to make points homogeneous if needed
        points_world = np.append(points_world, 1)
        # print(points_world.shape)
        # Transform points to clip space
        points_clip = (proj_view_matrix @ points_world.T).T
        # print(points_clip.shape)
        # Perspective divide to get NDC
        ndc = points_clip[:3] / points_clip[3]

        # NDC to pixel coordinates
        x = (ndc[0] + 1) * 0.5 * width
        y = (1 - ndc[1]) * 0.5 * height  # flip Y

        return x,y
    
    ### debug plot to see which points are we trying to grasp
    import cv2
    for pos in to_try_handle_points:
        screen_x, screen_y = project_point(pos, view, projection)
        # print("screen pos: ", screen_x, screen_y)
        img = img.astype(np.uint8)
        cv2.circle(img, (int(screen_x), int(screen_y)), radius=2, color=(0, 255, 0), thickness=-1)
        cv2.imwrite("img.png", img)
    
    args = [[env_kwargs, object_name, real_target_poses[it], mp_target_poses[it], target_orientations[it],\
            handle_pc, handle_joint_id, save_path, ori_simulator_state, it, link_name] for it in range(len(target_orientations))]

    if debug:
        results = parallel_motion_planning(args[0])
        results = [results]
    else:
        with Pool(processes=PARALLEL_POOL_NUM) as pool:
            results = pool.map(parallel_motion_planning, args)

    door_opened_ratios = np.array([x[0][0] for x in results])
    door_opened_angles = np.array([x[0][1] for x in results])
    grasp_scores = [x[1] for x in results]
    all_traj_states = [x[2] for x in results]
    all_traj_rgbs = [x[3] for x in results]
    all_stage_lengths = [x[4] for x in results]
    all_motion_planning_path_translation_lengths = [x[5] for x in results]
    all_motion_planning_path_rotation_lengths = [x[6] for x in results]

    ratio_threshold = 0.7
    if len(door_opened_ratios) > 0 and np.max(door_opened_ratios) > 0.1:
        best_idx = None
        if not np.sum(door_opened_ratios > ratio_threshold) > 0:
            best_idx = np.argmax(door_opened_ratios)
        else:
            # NOTE: maybe optimize orientation length as well. 
            best_rank = 100000
            path_translation_length_rank = np.argsort(all_motion_planning_path_translation_lengths)
            path_rotation_length_rank = np.argsort(all_motion_planning_path_rotation_lengths)
            grasping_score_rank = np.argsort(-np.array(grasp_scores))
            for idx, score in enumerate(door_opened_ratios):
                if score > ratio_threshold and path_translation_length_rank[idx] + grasping_score_rank[idx] < best_rank:
                    best_idx = idx
                    best_rank = path_translation_length_rank[idx] + grasping_score_rank[idx]
            
        best_score = grasp_scores[best_idx]
        with open(os.path.join(save_path, "best_score.txt"), "w") as f:
            f.write(str(best_score))
            
        # store the best env states
        state_files = []
        for t_idx, state in enumerate(all_traj_states[best_idx]):
            save_state_path = os.path.join(save_path, "states",  "state_{}.pkl".format(t_idx))
            state_files.append(save_state_path)
            with open(save_state_path, 'wb') as f:
                pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)
        
        # get the opened angle of the last state
        joint_limit_low, joint_limit_high = p.getJointInfo(simulator.urdf_ids[object_name], handle_joint_id, physicsClientId=simulator.id)[8:10]
        best_opened_angle = door_opened_angles[best_idx]
        with open(os.path.join(save_path, "opened_angle.txt"), "w") as f:
            f.write(str(best_opened_angle) + "\n")
            f.write(str(joint_limit_low) + "\n")
            f.write(str(joint_limit_high) + "\n")
        simulator.reset(ori_simulator_state)
        
        best_stage_length = all_stage_lengths[best_idx]
        with open(os.path.join(save_path, "stage_lengths.json"), "w") as f:
            json.dump(best_stage_length, f, indent=4)
                
        return all_traj_rgbs[best_idx], state_files
    
    with open(os.path.join(save_path, "best_score.txt"), "w") as f:
        f.write(str(0))
    
    joint_limit_low, joint_limit_high = p.getJointInfo(simulator.urdf_ids[object_name], handle_joint_id, physicsClientId=simulator.id)[8:10]
    with open(os.path.join(save_path, "opened_angle.txt"), "w") as f:
        f.write(str(0) + "\n")
        f.write(str(joint_limit_low) + "\n")
        f.write(str(joint_limit_high) + "\n")
            
    load_env(simulator, state=ori_simulator_state)
    save_env(simulator, os.path.join(save_path,  "state_{}.pkl".format(0)))
    rgbs = [simulator.render()]
    state_files = [os.path.join(save_path,  "state_{}.pkl".format(0))]
    return rgbs, state_files

def reach_till_contact(simulator, real_target_pos, target_orientation, return_contact_pose=False):
    intermediate_states = []
    rgbs = []
    cur_eef_pos, _ = simulator.robot.get_pos_orient(simulator.robot.right_end_effector)
    moving_vector = real_target_pos - cur_eef_pos
    delta_movement = 0.005
    movement_steps = int(np.linalg.norm(moving_vector) / delta_movement) + 1
    moving_direction = moving_vector / np.linalg.norm(moving_vector)
    target_orient_euler = p.getEulerFromQuaternion(target_orientation)
    for t in range(movement_steps):
        ik_indices = [_ for _ in range(len(simulator.robot.right_arm_joint_indices))]
        target_pos = cur_eef_pos + moving_direction * delta_movement * (t + 1)
        simulator.take_direct_action(np.array([*target_pos, *target_orient_euler, simulator.robot.finger_fully_open_joint_angle]))
        rgb = simulator.render()
        rgbs.append(rgb)
        state = save_env(simulator)
        intermediate_states.append(state)
        
        collision = False
        points_left_finger = p.getContactPoints(bodyA=simulator.robot.body, linkIndexA=simulator.robot.right_gripper_indices[0], physicsClientId=simulator.id)
        points_right_finger = p.getContactPoints(bodyA=simulator.robot.body, linkIndexA=simulator.robot.right_gripper_indices[1], physicsClientId=simulator.id)
        points_hand = p.getContactPoints(bodyA=simulator.robot.body, linkIndexA=8, physicsClientId=simulator.id)
        points = points_left_finger + points_right_finger + points_hand
        collision_points_a = [points[_][5] for _ in range(len(points))]
        if len(collision_points_a) > 0:
            p.addUserDebugPoints(collision_points_a, [[0, 1, 0] for _ in range(len(collision_points_a))], 12, 0.55, physicsClientId=simulator.id)
        if points:
            # Handle contact between suction with a rigid object.
            for point in points:
                obj_id, contact_link, contact_position_on_obj = point[2], point[4], point[6]
                if obj_id == simulator.urdf_ids['plane'] or obj_id == simulator.robot.body:
                    pass
                else:
                    collision = True    
                    if return_contact_pose:
                        cur_eef_pos, cur_eef_orient = simulator.robot.get_pos_orient(simulator.robot.right_end_effector)
                        return cur_eef_pos, cur_eef_orient
                    break
            
        if collision:
            # recover to the state where contact has not been made
            if len(intermediate_states) >= 3:
                simulator.reset(reset_state=intermediate_states[-3])
            break
    
    if len(intermediate_states) >= 3:
        return intermediate_states[:-2], rgbs[:-2]
    else:
        return intermediate_states, rgbs

def close_gripper(simulator, handle_pc):
    intermediate_states = []
    rgbs = []
    close_steps = 40 
    left_collision = False
    right_collision = False
    close_joint_angle = simulator.robot.finger_fully_close_joint_angle
    
    for t in range(close_steps):
        agent = simulator.robot
        agent.set_gripper_open_position(agent.right_gripper_indices, [close_joint_angle, close_joint_angle], set_instantly=False)
        p.stepSimulation(physicsClientId=simulator.id)
        state = save_env(simulator)
        intermediate_states.append(state)
        rgb = simulator.render()
        rgbs.append(rgb)
        
        # NOTE: update the score such that after closing, both gripper is in contact with the handle itself.
        points_left_finger = p.getContactPoints(bodyA=simulator.robot.body, linkIndexA=simulator.robot.right_gripper_indices[0], physicsClientId=simulator.id)
        points_right_finger = p.getContactPoints(bodyA=simulator.robot.body, linkIndexA=simulator.robot.right_gripper_indices[1], physicsClientId=simulator.id)

        if points_left_finger:
            collision_points_b = [points_left_finger[_][5] for _ in range(len(points_left_finger))]
            dist_collision_to_handle = scipy.spatial.distance.cdist(collision_points_b, handle_pc).min(axis=1)
            if np.sum(dist_collision_to_handle < 0.01) > 0:
                left_collision = True
        if points_right_finger:
            collision_points_b = [points_right_finger[_][5] for _ in range(len(points_right_finger))]
            dist_collision_to_handle = scipy.spatial.distance.cdist(collision_points_b, handle_pc).min(axis=1)
            if np.sum(dist_collision_to_handle < 0.01) > 0:
                right_collision = True
                
        if left_collision and right_collision:
            break
        
        
    return intermediate_states, rgbs, left_collision, right_collision

def open_gripper(simulator):
    intermediate_states = []
    rgbs = []
    open_steps = 40 
    open_joint_angle = simulator.robot.finger_fully_open_joint_angle
    for t in range(open_steps):
        agent = simulator.robot
        agent.set_gripper_open_position(agent.right_gripper_indices, [open_joint_angle, open_joint_angle], set_instantly=False)
        p.stepSimulation(physicsClientId=simulator.id)
        state = save_env(simulator)
        intermediate_states.append(state)
        rgb = simulator.render()
        rgbs.append(rgb)

        if p.getJointState(agent.body, agent.right_gripper_indices[0], physicsClientId=simulator.id)[0] > 0.037:
            break
    
    return intermediate_states, rgbs

def open_door(simulator, object_name, link_name, handle_joint_id):
    intermediate_states = []
    rgbs = []
    
    eef_pos, eef_orient = simulator.robot.get_pos_orient(simulator.robot.right_end_effector)
    link_pos, link_orient = simulator.get_link_pose(object_name, link_name)
    world_to_link = p.invertTransform(link_pos, link_orient)
    # EEf in link frame remains the same as the link frame rotates
    eef_in_link = p.multiplyTransforms(world_to_link[0], world_to_link[1], eef_pos, eef_orient) 

    joint_limit = p.getJointInfo(simulator.urdf_ids[object_name], handle_joint_id, physicsClientId=simulator.id)[8:10]
    ori_joint_angle = p.getJointState(simulator.urdf_ids[object_name], handle_joint_id, physicsClientId=simulator.id)[0]
    eef_poses = []
    timesteps = 100 
    
    ratio = 0.8
    cur_joint_angle = p.getJointState(simulator.urdf_ids[object_name], handle_joint_id, physicsClientId=simulator.id)[0]
    target = joint_limit[0] + ratio * (joint_limit[1] - joint_limit[0])
    cur_move_amount = target - cur_joint_angle
    full_move_amount = ratio * (joint_limit[1] - joint_limit[0])
    timesteps = int(timesteps * np.abs(cur_move_amount) / full_move_amount)
    for t in range(1, timesteps):
        joint_angle = cur_joint_angle + (target - cur_joint_angle) * t / timesteps
        p.resetJointState(simulator.urdf_ids[object_name], handle_joint_id, joint_angle, physicsClientId=simulator.id)
        new_link_pos, new_link_orient = simulator.get_link_pose(object_name, link_name)
        # new_link_pos, new_link_orient is the transformation from link coordinate to world coordinate
        new_eef_pos, new_eef_orient = p.multiplyTransforms(new_link_pos, new_link_orient, eef_in_link[0], eef_in_link[1])
        eef_poses.append([new_eef_pos, new_eef_orient])
        
    
    p.resetJointState(simulator.urdf_ids[object_name], handle_joint_id, ori_joint_angle, physicsClientId=simulator.id)
    for t in range(len(eef_poses)):
        pos, orient = eef_poses[t]
        ik_indices = [_ for _ in range(len(simulator.robot.right_arm_joint_indices))]
        ik_joint_angles = simulator.robot.ik(simulator.robot.right_end_effector, 
                                        pos, orient, 
                                        ik_indices=ik_indices)

        ik_joint_angles = list(ik_joint_angles) + [0, 0]
        ik_joints = simulator.robot.right_arm_joint_indices + list(simulator.robot.right_gripper_indices)
        
        for _ in range(2):
            p.setJointMotorControlArray(simulator.robot.body, jointIndices=ik_joints, 
                                        controlMode=p.POSITION_CONTROL, targetPositions=ik_joint_angles, physicsClientId=simulator.id)
            p.stepSimulation(physicsClientId=simulator.id)
        
        rgb = simulator.render()
        rgbs.append(rgb)
        state = save_env(simulator)
        intermediate_states.append(state)
    
    final_joint_angle = p.getJointState(simulator.urdf_ids[object_name], handle_joint_id, physicsClientId=simulator.id)[0]
    # NOTE: change to return the ratio of the door opened to the upper limit
    joint_limit_high = joint_limit[1]
    final_joint_angle_ratio = (final_joint_angle - joint_limit[0]) / (joint_limit_high - joint_limit[0])
    return intermediate_states, rgbs, (final_joint_angle_ratio, final_joint_angle)

def parallel_motion_planning(args):
    debug = False
    np.random.seed(time.time_ns() % 2**32)
    
    env_kwargs, object_name, real_target_pos, mp_target_pos, target_orientation, \
        handle_pc, handle_joint_id, save_path, ori_simulator_state, \
        it, link_name = args
        
    stage_length = {}
    object_name = object_name.lower()
    
    simulator, _ = build_up_env(
        **env_kwargs
    )
    simulator.reset(ori_simulator_state)
    p.addUserDebugLine([0, 0, 0], [1, 0, 0], [1, 0, 0], lineWidth=2, lifeTime=0, physicsClientId=simulator.id)
    p.addUserDebugLine([0, 0, 0], [0, 1, 0], [0, 1, 0], lineWidth=2, lifeTime=0, physicsClientId=simulator.id)
    p.addUserDebugLine([0, 0, 0], [0, 0, 1], [0, 0, 1], lineWidth=2, lifeTime=0, physicsClientId=simulator.id)

    intermediate_states = []
    rgbs = []
    
    # open gripper before reaching the handle
    open_gripper_states, open_gripper_rgbs = open_gripper(simulator)
    intermediate_states += open_gripper_states
    rgbs += open_gripper_rgbs

    all_objects = list(simulator.urdf_ids.keys())
    all_objects.remove("robot")
    obstacles = [simulator.urdf_ids[x] for x in all_objects]
    allow_collision_links = []
    cur_eef_pos, cur_eef_orient = simulator.robot.get_pos_orient(simulator.robot.right_end_effector)
    translation_length = np.linalg.norm(mp_target_pos - cur_eef_pos)
    rotation_length = 2 * np.arccos(np.abs(np.dot(target_orientation, cur_eef_orient)))
    rotation_length = np.rad2deg(rotation_length)
    translation_steps = int(translation_length / 0.004) + 1
    rotation_steps = int(rotation_length / 1.8) + 1
    interpolation_steps = max(translation_steps, rotation_steps)
    
    if debug:
        p.addUserDebugPoints([mp_target_pos], [[1, 0, 0]], 12, 0, physicsClientId=simulator.id)
        p.addUserDebugPoints([real_target_pos], [[1, 0, 0]], 12, 0, physicsClientId=simulator.id)
    
    res, path, path_translation_length, path_rotation_length = motion_planning(
        simulator, mp_target_pos, target_orientation, obstacles=obstacles, allow_collision_links=allow_collision_links, save_path=save_path, 
        smooth_path=True, interpolation_num=interpolation_steps)
    
    if res:
        stage_length['reach_handle'] = len(path) + len(open_gripper_states)
        
        if debug:
            import pdb; pdb.set_trace()

        for idx, q in enumerate(path):
            simulator.robot.set_joint_angles(simulator.robot.right_arm_joint_indices, q)
            rgb = simulator.render()
            rgbs.append(rgb)
            state = save_env(simulator)
            intermediate_states.append(state)

        if debug:
            import pdb; pdb.set_trace()

        # reach till contact is made, and get the number of handle points between the two fingers
        reach_to_concatc_states, reach_to_contact_rgbs = reach_till_contact(simulator, real_target_pos, target_orientation)
        intermediate_states += reach_to_concatc_states
        rgbs += reach_to_contact_rgbs
        stage_length['reach_to_contact'] = len(reach_to_contact_rgbs)
        
        # get a score for this grasping pose, which is the number of handle points between the two fingers
        cur_eef_pos, cur_eef_orient = simulator.robot.get_pos_orient(simulator.robot.right_end_effector)

        score = get_pc_num_within_gripper(cur_eef_pos, cur_eef_orient, handle_pc)
        
        # if not point is being grasped we directly return a failed score
        if score == 0:
            return (-1, -1), -1, [], [], {}, np.inf, np.inf

        # # close gripper
        if debug:
            import pdb; pdb.set_trace()
            
        close_states, close_rgbs, left_collision, right_collision = close_gripper(simulator, handle_pc)
        if not (left_collision and right_collision):
            score = 0
        intermediate_states += close_states
        rgbs += close_rgbs
        stage_length['close_gripper'] = len(close_states)
        

        cprint("iteration {} score {}".format(it, score), "green")

        # # pull out following the rotation axis
        if debug:
            import pdb; pdb.set_trace()
            
        open_door_states, open_door_rgbs, final_joint_angle_ratio = open_door(simulator, object_name, link_name, handle_joint_id)
        
        intermediate_states += open_door_states
        rgbs += open_door_rgbs
        stage_length['open_door'] = len(open_door_states)

        cprint(f"final joint angle ratio: {final_joint_angle_ratio}", "green")
        simulator.close()
        return final_joint_angle_ratio, score, intermediate_states, rgbs, stage_length, path_translation_length, path_rotation_length

    simulator.close()
    return (-1, -1), -1, [], [], {}, np.inf, np.inf

