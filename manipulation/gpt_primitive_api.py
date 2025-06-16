import pybullet as p
import os
import numpy as np
import open3d as o3d
# from manipulation.motion_planning_utils import motion_planning
from manipulation.grasping_utils import get_pc_and_normal, align_gripper_z_with_normal, align_gripper_x_with_normal
from manipulation.gpt_reward_api import get_link_pc, get_bounding_box, get_link_id_from_name, get_groundtruth_link_pc
from manipulation.utils import save_env, load_env, build_up_env
from manipulation.gpt_reward_api import (
    get_link_pc, get_bounding_box, get_link_id_from_name, get_handle_pos, get_link_pose,
)
from manipulation.utils import save_env, load_env
import scipy
import time
import copy
from termcolor import cprint
import fpsample
from multiprocessing import Pool
# import pickle5 as pickle
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


def release_grasp(simulator):
    # simulator.deactivate_suction()
    save_path = get_save_path(simulator)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    rgbs = []
    states = []
    for t in range(20):
        p.stepSimulation()
        rgbs.append(simulator.render())
        state_save_path = os.path.join(save_path, "states", "state_{}.pkl".format(t))
        save_env(simulator, state_save_path)
        states.append(state_save_path)

    return rgbs, states

def grasp_object(simulator, object_name):
    ori_state = save_env(simulator, None)
    p.stepSimulation()
    object_name = object_name.lower()
    save_path = get_save_path(simulator)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # if the target object is already grasped.  
    points = p.getContactPoints(bodyA=simulator.robot.body, linkIndexA=simulator.suction_id, physicsClientId=simulator.id)
    if points:
        for point in points:
            obj_id, contact_link = point[2], point[4]
            if obj_id == simulator.urdf_ids[object_name]:
                # simulator.activate_suction()
                rgbs = []
                states = []
                for t in range(10):
                    p.stepSimulation()
                    rgbs.append(simulator.render())
                    state_save_path = os.path.join(save_path, "states", "state_{}.pkl".format(t))
                    save_env(simulator, state_save_path)
                    states.append(state_save_path)
                return rgbs, states

    rgbs, states = approach_object(simulator, object_name)
    base_t = len(rgbs)
    if base_t > 1:
        for t in range(10):
            # simulator.activate_suction()
            p.stepSimulation()
            rgbs.append(simulator.render())
            state_save_path = os.path.join(save_path, "states", "state_{}.pkl".format(t + base_t))
            save_env(simulator, state_save_path)
            states.append(state_save_path)
    else:
        # directy reset the state
        load_env(simulator, state=ori_state)

    return rgbs, states

def grasp_object_link(simulator, object_name, link_name):
    # ori_state = save_env(simulator, None)
    # p.stepSimulation()
    # object_name = object_name.lower()
    # save_path = get_save_path(simulator)
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    
    # # if the target object link is already grasped.  
    # points = p.getContactPoints(bodyA=simulator.robot.body, linkIndexA=simulator.suction_id, physicsClientId=simulator.id)
    # if points:
    #     for point in points:
    #         obj_id, contact_link = point[2], point[4]
    #         if obj_id == simulator.urdf_ids[object_name] and contact_link == get_link_id_from_name(simulator, object_name, link_name):
    #             # simulator.activate_suction()
    #             rgbs = []
    #             states = []
    #             for t in range(10):
    #                 p.stepSimulation()
    #                 rgbs.append(simulator.render()[0])
    #                 state_save_path = os.path.join(save_path, "states", "state_{}.pkl".format(t))
    #                 save_env(simulator, state_save_path)
    #                 states.append(state_save_path)
    #             return rgbs, states


    # rgbs, states = approach_object_link(simulator, object_name, link_name)
    # base_t = len(rgbs)
    # if base_t > 1:
    #     # simulator.activate_suction()
    #     for t in range(10):
    #         p.stepSimulation()
    #         rgbs.append(simulator.render()[0])
    #         state_save_path = os.path.join(save_path, "states", "state_{}.pkl".format(t + base_t))
    #         save_env(simulator, state_save_path)
    #         states.append(state_save_path)
    # else:
    #     # directy reset the state
    #     load_env(simulator, state=ori_state)

    # return rgbs, states
    return approach_object_link(simulator, object_name, link_name)

def approach_object(simulator, object_name, dynamics=False):
    save_path = get_save_path(simulator)
    ori_state = save_env(simulator, None)
    # simulator.deactivate_suction()
    release_rgbs = []
    release_states = []
    release_steps = 20
    for t in range(release_steps):
        p.stepSimulation()
        rgb = simulator.render()
        release_rgbs.append(rgb)
        state_save_path = os.path.join(save_path, "states", "state_{}.pkl".format(t))
        save_env(simulator, state_save_path)
        release_states.append(state_save_path)

    object_name = object_name.lower()
    it = 0
    object_name = object_name.lower()
    object_pc, object_normal = get_pc_and_normal(simulator, object_name)
    low, high = get_bounding_box(simulator, object_name)
    com = (low + high) / 2
    current_joint_angles = simulator.robot.get_joint_angles(indices=simulator.robot.right_arm_joint_indices)
    

    while True:
        random_point = object_pc[np.random.randint(0, object_pc.shape[0])]
        random_normal = object_normal[np.random.randint(0, object_normal.shape[0])]

        ### adjust the normal such that it points outwards the object.
        ### TODO: make sure the normal points outwards the object.
        line = com - random_point
        if np.dot(line, random_normal) > 0:
            random_normal = -random_normal
            
        for normal in [random_normal, -random_normal]:
            simulator.robot.set_joint_angles(simulator.robot.right_arm_joint_indices, current_joint_angles)

            target_pos = random_point
            real_target_pos = target_pos + normal * 0
            if simulator.robot_name in ["panda", "sawyer"]:
                target_orientation = align_gripper_z_with_normal(-normal).as_quat()
                mp_target_pos = target_pos + normal * 0.03
            elif simulator.robot_name in ['ur5', 'fetch']:
                target_orientation = align_gripper_x_with_normal(-normal).as_quat()
                if simulator.robot_name == 'ur5':
                    mp_target_pos = target_pos + normal * 0.07
                elif simulator.robot_name == 'fetch':
                    mp_target_pos = target_pos + normal * 0.07

            all_objects = list(simulator.urdf_ids.keys())
            all_objects.remove("robot")
            obstacles = [simulator.urdf_ids[x] for x in all_objects]
            allow_collision_links = []
            res, path, path_length = motion_planning(simulator, mp_target_pos, target_orientation, obstacles=obstacles, allow_collision_links=allow_collision_links)

            if res:
                rgbs = release_rgbs
                intermediate_states = release_states
                for idx, q in enumerate(path):
                    if not dynamics:
                        simulator.robot.set_joint_angles(simulator.robot.right_arm_joint_indices, q)
                        p.stepSimulation()
                    else:
                        for _ in range(3):
                            simulator.robot.control(simulator.robot.right_arm_joint_indices, q, simulator.robot.motor_gains, forces=5 * 240.)
                            p.stepSimulation()

                    rgb = simulator.render()
                    rgbs.append(rgb)
                    save_state_path = os.path.join(save_path,  "states", "state_{}.pkl".format(idx + release_steps))
                    save_env(simulator, save_state_path)
                    intermediate_states.append(save_state_path)

                base_idx = len(intermediate_states)
                for t in range(20):
                    ik_indices = [_ for _ in range(len(simulator.robot.right_arm_joint_indices))]
                    ik_joints = simulator.robot.ik(simulator.robot.right_end_effector, 
                                                    real_target_pos, target_orientation, 
                                                    ik_indices=ik_indices)
                    p.setJointMotorControlArray(simulator.robot.body, jointIndices=simulator.robot.right_arm_joint_indices, 
                                                controlMode=p.POSITION_CONTROL, targetPositions=ik_joints,
                                                forces=[5*240] * len(simulator.robot.right_arm_joint_indices), physicsClientId=simulator.id)
                    p.stepSimulation()
                    rgb = simulator.render()
                    rgbs.append(rgb)
                    save_state_path = os.path.join(save_path, "states" , "state_{}.pkl".format(base_idx + t))
                    save_env(simulator, save_state_path)
                    intermediate_states.append(save_state_path)

                    # TODO: check if there is already a collision. if so, break.
                    collision = False
                    points = p.getContactPoints(bodyA=simulator.robot.body, linkIndexA=simulator.suction_id, physicsClientId=simulator.id)
                    if points:
                        # Handle contact between suction with a rigid object.
                        for point in points:
                            obj_id, contact_link, contact_position_on_obj = point[2], point[4], point[6]
                            
                            if obj_id == simulator.urdf_ids['plane'] or obj_id == simulator.robot.body:
                                pass
                            else:
                                collision = True
                                break
                    if collision:
                        break

                return rgbs, intermediate_states
        
            it += 1
            if it > MOTION_PLANNING_TRY_TIMES:
                print("failed to execute the primitive")
                load_env(simulator, state=ori_state)
                save_env(simulator, os.path.join(save_path,  "state_{}.pkl".format(0)))
                rgbs = [simulator.render()]
                state_files = [os.path.join(save_path,  "state_{}.pkl".format(0))]
                return rgbs, state_files

def approach_object_link(simulator, object_name, link_name, dynamics=False, grasp_handle=True, 
                         execute_opening_primitive=True):
    return approach_object_link_parallel(simulator, object_name, link_name)

def get_link_handle(all_handle_pos, handle_joint_id, link_pc):
    handle_median_points = np.array([np.median(handle_pos, axis=0) for handle_pos in all_handle_pos]).reshape(-1, 3)
    distance_handle_median_to_link_pc = scipy.spatial.distance.cdist(handle_median_points, link_pc)
    min_distance = np.min(distance_handle_median_to_link_pc, axis=1)
    min_distance_handle_idx = np.argmin(min_distance)
    handle_joint_id = handle_joint_id[min_distance_handle_idx]
    handle_pc = all_handle_pos[min_distance_handle_idx]
    handle_median = handle_median_points[min_distance_handle_idx]
    
    threshold = 0.02
    pc_to_handle_distance = scipy.spatial.distance.cdist(link_pc, handle_pc).min(axis=1)
    handle_pc = link_pc[pc_to_handle_distance < threshold]
    
    return handle_pc, handle_joint_id, handle_median, min_distance_handle_idx

def approach_object_link_parallel(simulator, object_name, link_name):    
    save_path = get_save_path(simulator)
    ori_simulator_state = save_env(simulator, None)
    object_name = object_name.lower()
    link_name = link_name.lower()
    link_pc = get_link_pc(simulator, object_name, link_name)
    object_pc = link_pc
    pcd = o3d.geometry.PointCloud() 
    pcd.points = o3d.utility.Vector3dVector(object_pc)
    pcd.estimate_normals()
    object_normal = np.asarray(pcd.normals)

    all_handle_pos, handle_joint_id = get_handle_pos(simulator, object_name, return_median=False)
    # from matplotlib import pyplot as plt
    # from manipulation.grasping_utils import voxelize_pc
    # ax = plt.axes(projection='3d')
    # link_pc = voxelize_pc(link_pc, voxel_size=0.01)
    # ax.scatter(link_pc[:, 0], link_pc[:, 1], link_pc[:, 2], c='r', s=1)
    # for handle_pos in all_handle_pos:
    #     handle_pos = voxelize_pc(handle_pos, voxel_size=0.01)
    #     ax.scatter(handle_pos[:, 0], handle_pos[:, 1], handle_pos[:, 2], c='b', s=1)
    # plt.show()
    # import pdb; pdb.set_trace()
    handle_pc, handle_joint_id, handle_median, _ = get_link_handle(all_handle_pos, handle_joint_id, link_pc)

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
        "solution_path": simulator.solution_path,
        "task_name": simulator.task_name, 
        "restore_state_file": simulator.restore_state_file, 
        "render": False, 
        # "render": True, 
        "randomize": False, 
        "obj_id": simulator.obj_id, 
    }
    
    handle_orientation = get_handle_orient(handle_pc)
    horizontal_grasp = True if handle_orientation == 'vertical' else False
    
    ## first compute some parameters to use
    mp_target_poses = []
    real_target_poses = []
    target_orientations = []
    for target_pos in to_try_handle_points:
        nearest_point_idx = np.argmin(np.linalg.norm(object_pc - target_pos.reshape(1, 3), axis=1))
        align_normal = object_normal[nearest_point_idx]
        
        low, high = get_bounding_box(simulator, object_name)
        com = (low + high) / 2
        line = com - target_pos
        if np.dot(line, align_normal) > 0:
            align_normal = -align_normal
            
        for normal in [align_normal]:
            real_target_pos = target_pos + normal * -0.02
            mp_target_pos = target_pos + normal * 0.04

            for orientation_idx in range(SAMPLE_ORIENTATION_NUM):
                target_orientation = align_gripper_z_with_normal(-normal, horizontal=horizontal_grasp, randomize=True).as_quat()
                mp_target_poses.append(mp_target_pos)
                real_target_poses.append(real_target_pos)
                target_orientations.append(target_orientation)
                
            target_orientation_1 = align_gripper_z_with_normal(-normal, horizontal=horizontal_grasp, randomize=False, flip=False).as_quat()
            target_orientation_2 = align_gripper_z_with_normal(-normal, horizontal=horizontal_grasp, randomize=False, flip=True).as_quat()
            mp_target_poses.append(mp_target_pos); mp_target_poses.append(mp_target_pos) 
            real_target_poses.append(real_target_pos); real_target_poses.append(real_target_pos)
            target_orientations.append(target_orientation_1); target_orientations.append(target_orientation_2)     

    # env_kwargs, object_name, real_target_pos, mp_target_pos, target_orientation, \
    #     handle_pc, handle_joint_id, save_path, ori_simulator_state, \
    #     it, link_name = args
    args = [[env_kwargs, object_name, real_target_poses[it], mp_target_poses[it], target_orientations[it],\
            handle_pc, handle_joint_id, save_path, ori_simulator_state, it, link_name] for it in range(len(target_orientations))]

    # results = parallel_motion_planning(args[0])
    # results = [results]
    with Pool(processes=PARALLEL_POOL_NUM) as pool:
        results = pool.map(parallel_motion_planning, args)
    # final_joint_angle, score, intermediate_states, rgbs, stage_length, path_length

    door_opened_scores = np.array([x[0] for x in results])
    grasp_scores = [x[1] for x in results]
    all_traj_states = [x[2] for x in results]
    all_traj_rgbs = [x[3] for x in results]
    all_stage_lengths = [x[4] for x in results]
    all_motion_planning_path_translation_lengths = [x[5] for x in results]
    all_motion_planning_path_rotation_lengths = [x[6] for x in results]


    ratio_threshold = 0.7
    if len(door_opened_scores) > 0 and np.max(door_opened_scores) > 0.1:
        best_idx = None
        if not np.sum(door_opened_scores > ratio_threshold) > 0:
            best_idx = np.argmax(door_opened_scores)
        else:
            # TODO: optimize orientation length as well. 
            best_rank = 100000
            path_translation_length_rank = np.argsort(all_motion_planning_path_translation_lengths)
            path_rotation_length_rank = np.argsort(all_motion_planning_path_rotation_lengths)
            grasping_score_rank = np.argsort(-np.array(grasp_scores))
            for idx, score in enumerate(door_opened_scores):
                # if score > 0.8 and path_translation_length_rank[idx] + path_rotation_length_rank[idx] + grasping_score_rank[idx] < best_rank:
                if score > ratio_threshold and path_translation_length_rank[idx] + grasping_score_rank[idx] < best_rank:
                    best_idx = idx
                    # best_rank = path_translation_length_rank[idx] + path_rotation_length_rank[idx] + grasping_score_rank[idx]
                    best_rank = path_translation_length_rank[idx] + grasping_score_rank[idx]
            
        best_opened_angle = door_opened_scores[best_idx]
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
    
    print("handle joint id: ", handle_joint_id)
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
        simulator.take_direct_action(np.array([*target_pos, *target_orient_euler, 0.04]))
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
                if obj_id == simulator.urdf_ids['plane'] or obj_id == simulator.robot.body or (simulator.use_table and obj_id == simulator.table):
                    pass
                else:
                    # print("collision detected")
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
    after_collision_steps = 0
    close_joint_angle = 0.
    for t in range(close_steps):
        agent = simulator.robot
        for _ in range(2):
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

def open_door(simulator, object_name, link_name, handle_joint_id):
    intermediate_states = []
    rgbs = []
    
    eef_pos, eef_orient = simulator.robot.get_pos_orient(simulator.robot.right_end_effector)
    link_pos, link_orient = get_link_pose(simulator, object_name, link_name)
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
        new_link_pos, new_link_orient = get_link_pose(simulator, object_name, link_name)
        # new_link_pos, new_link_orient is the transformation from link coordinate to world coordinate
        new_eef_pos, new_eef_orient = p.multiplyTransforms(new_link_pos, new_link_orient, eef_in_link[0], eef_in_link[1])
        eef_poses.append([new_eef_pos, new_eef_orient])
        
    
    p.resetJointState(simulator.urdf_ids[object_name], handle_joint_id, ori_joint_angle, physicsClientId=simulator.id)
    for t in range(len(eef_poses)):
        pos, orient = eef_poses[t]
        # new way of control
        # target_orient_euler = p.getEulerFromQuaternion(orient)
        # simulator.take_direct_action(np.array([*pos, *target_orient_euler, 0]))
        
        # old way of control
        ik_indices = [_ for _ in range(len(simulator.robot.right_arm_joint_indices))]
        ik_joint_angles = simulator.robot.ik(simulator.robot.right_end_effector, 
                                        pos, orient, 
                                        ik_indices=ik_indices)
        ik_joint_angles = list(ik_joint_angles)  + [0, 0]
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
    final_joint_angle = (final_joint_angle - joint_limit[0]) / (joint_limit_high - joint_limit[0])
    return intermediate_states, rgbs, final_joint_angle

def parallel_motion_planning(args):
    np.random.seed(time.time_ns() % 2**32)
    
    env_kwargs, object_name, real_target_pos, mp_target_pos, target_orientation, \
        handle_pc, handle_joint_id, save_path, ori_simulator_state, \
        it, link_name = args
        
    stage_length = {}
    object_name = object_name.lower()
    
    simulator, _ = build_up_env(
        **env_kwargs
    )
    # load_env(simulator, state=ori_simulator_state)
    simulator.reset(ori_simulator_state)
    
    intermediate_states = []
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
    res, path, path_translation_length, path_rotation_length = motion_planning(
        simulator, mp_target_pos, target_orientation, obstacles=obstacles, allow_collision_links=allow_collision_links, save_path=save_path, 
        smooth_path=True, interpolation_num=interpolation_steps)
    
    if res:
        stage_length['reach_handle'] = len(path)
        
        rgbs = []
        for idx, q in enumerate(path):
            simulator.robot.set_joint_angles(simulator.robot.right_arm_joint_indices, q)
            rgb = simulator.render()
            rgbs.append(rgb)
            state = save_env(simulator)
            intermediate_states.append(state)

        
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
            return -1, -1, [], [], {}, np.inf, np.inf

        # # close gripper
        close_states, close_rgbs, left_collision, right_collision = close_gripper(simulator, handle_pc)
        if not (left_collision and right_collision):
            score = 0
        intermediate_states += close_states
        rgbs += close_rgbs
        stage_length['close_gripper'] = len(close_states)
        
        cprint("iteration {} score {}".format(it, score), "green")

        # # pull out following the rotation axis
        open_door_states, open_door_rgbs, final_joint_angle = open_door(simulator, object_name, link_name, handle_joint_id)
        
        intermediate_states += open_door_states
        rgbs += open_door_rgbs
        stage_length['open_door'] = len(open_door_states)

        cprint(f"final joint angle: {final_joint_angle}", "green")
        return final_joint_angle, score, intermediate_states, rgbs, stage_length, path_translation_length, path_rotation_length
        
    return -1, -1, [], [], {}, np.inf, np.inf

def open_gripper(simulator):
    intermediate_states = []
    rgbs = []
    steps = 10
    open_joint_angle = 0.05
    for t in range(steps):
        agent = simulator.robot
        # NOTE: control till reached the target joint angle
        for _ in range(2):
            agent.set_gripper_open_position(agent.right_gripper_indices, [open_joint_angle, open_joint_angle], set_instantly=False)
        p.stepSimulation(physicsClientId=simulator.id)
        state = save_env(simulator)
        intermediate_states.append(state)
        rgb = simulator.render()
        rgbs.append(rgb)
        current_joint_angle = agent.get_joint_angles(agent.right_gripper_indices)
        if np.abs(current_joint_angle[0] - 0.04) < 0.0001:
            break
        
    return intermediate_states, rgbs

def get_pc_num_within_gripper(cur_eef_pos, cur_eef_orient, pc_points):
    
    cur_pos, cur_orient = cur_eef_pos, cur_eef_orient

    X_GW = p.invertTransform(cur_pos, cur_orient)
    translation = np.array(X_GW[0])
    rotation = np.array(p.getMatrixFromQuaternion(X_GW[1])).reshape(3, 3)
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation ### this is the transformation from world frame to gripper frame

    pc_homogeneous = np.hstack((pc_points, np.ones((pc_points.shape[0], 1))))  # Convert to homogeneous coordinates Nx4
    pc_transformed_homogeneous = T @ pc_homogeneous.T # 4x4 @ 4xN = 4xN
    p_GC = pc_transformed_homogeneous[:3, :] # 3xN

    ### Crop to a region inside of the finger box.
    crop_min = [-0.02, -0.06, -0.01] 
    crop_max = [0.02, 0.06, 0.01]
    indices = np.all(
        (
            crop_min[0] <= p_GC[0, :],
            p_GC[0, :] <= crop_max[0],
            crop_min[1] <= p_GC[1, :],
            p_GC[1, :] <= crop_max[1],
            crop_min[2] <= p_GC[2, :],
            p_GC[2, :] <= crop_max[2],
        ),
        axis=0,
    )
    
    within_bbox_handle_pc = pc_points[indices]
    if len(within_bbox_handle_pc) == 0:
        # print("no points are within the gripper")
        return 0
    score = np.sum(indices) 
    # print("score is: ", score)
    return score

def get_handle_orient(handle_pc):
    # get axis aligned bounding box of the handle pc
    min_xyz = np.min(handle_pc, axis=0)
    max_xyz = np.max(handle_pc, axis=0)
    x_range = max_xyz[0] - min_xyz[0]
    y_range = max_xyz[1] - min_xyz[1]
    z_range = max_xyz[2] - min_xyz[2]
    horizontal_range = np.max([x_range, y_range])
    vertical_range = z_range
    if horizontal_range > vertical_range:
        handle_orient = "horizontal"
    else:
        handle_orient = "vertical"
    
    return handle_orient