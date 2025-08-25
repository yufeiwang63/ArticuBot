import numpy as np
from manipulation.utils import build_up_env
from manipulation.utils import load_env, rotation_transfer_6D_to_matrix, rotation_transfer_matrix_to_6D
import os
import pickle
import yaml
import tqdm
import time
from manipulation.robogen_wrapper import RobogenPointCloudWrapper
from termcolor import cprint
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import json
from scipy.spatial.transform import Rotation as R
import pickle
from manipulation.utils import save_numpy_as_gif
from collections import defaultdict


def sort_states_file_by_file_number(state_path):
    # all the file are named as state_0.pkl, state_1.pkl, ...
    ret_files = []
    for file in os.listdir(state_path):
        if file.startswith("state_") and file.endswith(".pkl"):
            ret_files.append(file)

    ret_files = sorted(ret_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    return ret_files

def get_all_expert_angles(all_experiments):
    ratios = []
    opened_angles = []
    for experiment_path in all_experiments:
        opened_angle_file = os.path.join(experiment_path, "opened_angle.txt")
        if os.path.exists(opened_angle_file): # for some perturbed trajectories, we did not really continue openeing the handle. 
            with open(opened_angle_file, "r") as f:
                angles = f.readlines()
                opened_angle = float(angles[0].lstrip().rstrip())
                min_angle = float(angles[1].lstrip().rstrip())
                max_angle = float(angles[-1].lstrip().rstrip())
                opened_angle -= min_angle
                ratio = opened_angle / (max_angle - min_angle)
                ratios.append(ratio)
                opened_angles.append(opened_angle)
            
    return np.array(ratios), np.array(opened_angles)

def extract_pc_states_for_all_trajectories(pool_args):
    task_config_path, solution_path, env_name, exp_name, experiments, save_path, angle_threshold, args = pool_args
       
    if exp_name is None:
        experiment_folder = os.path.join(solution_path, "experiment")
    else:
        experiment_folder = os.path.join(solution_path, "experiment", exp_name)
    if not os.path.exists(experiment_folder):
        print("Experiment folder does not exist: ", experiment_folder)
        if exp_name is not None:
            experiment_folder = solution_path
        else:
            experiment_folder = os.path.join(solution_path, exp_name)
        
    all_traj_stage_lengths = []
    all_traj_store_label_paths = []
    obs_keys = ['point_cloud', 'agent_pos', 'gripper_pcd', 'goal_gripper_pcd', 'displacement_gripper_to_object']
    all_traj_obs_dict_of_list = defaultdict(list)
    all_view_matrices = []
    all_proj_matrices = []
    
    for experiment in experiments:
        if "meta" in experiment:
            continue
        
        expert_states = []
        experiment_path = os.path.join(experiment_folder, experiment)
        task_config_path = os.path.join(experiment_path, "task_config.yaml")
        states_path = os.path.join(experiment_path, "states")
        
        states = sort_states_file_by_file_number(states_path)      
        expert_states.extend([os.path.join(states_path, x) for x in states])
        if len(expert_states) == 0:
            print("No states found for experiment continue")
            continue
        
        stage_lengths = os.path.join(experiment_path, "stage_lengths.json")
        with open(stage_lengths, "r") as f:
            stage_lengths = json.load(f)

        reach_till_contact_idx = stage_lengths['reach_handle'] + stage_lengths["reach_to_contact"]
        open_time_idx = stage_lengths['reach_handle'] + stage_lengths["reach_to_contact"] + stage_lengths["close_gripper"]

        if len(expert_states) - open_time_idx < 5: # if the opening time is too short, skip this trajectory
            print("Opening time is too short, continue")
            continue
        
        opened_angle_file = os.path.join(experiment_path, "opened_angle.txt")
        if os.path.exists(opened_angle_file): 
            with open(opened_angle_file, "r") as f:
                angles = f.readlines()
                opened_angle = float(angles[0].lstrip().rstrip())
                min_angle = float(angles[1].lstrip().rstrip())
                max_angle = float(angles[-1].lstrip().rstrip())
                opened_angle -= min_angle
                ratio = opened_angle / (max_angle - min_angle)
            if opened_angle < angle_threshold or ratio < args.min_opened_ratio:
                print("not open enough, continue")
                continue

        # already saved the data
        if os.path.exists(os.path.join(save_path, experiment)):
            print("Already saved the data, continue")
            continue

        bad_experiment = False
        camera_detected = True
        beg = time.time()
            
        config = yaml.safe_load(open(task_config_path, "r"))
        for config_dict in config:
            if 'name' in config_dict:
                object_name = config_dict['name'].lower()

        simulator, _ = build_up_env(
            task_config=task_config_path,
            env_name=env_name,
            restore_state_file=None,
            render=False,
            randomize=False,
            obj_id=0,
        )
        simulator = RobogenPointCloudWrapper(simulator, 
            object_name,
            num_points=args.pointcloud_num,
            observation_mode=args.observation_mode, 
            noise_real_world_pcd=args.noise_real_world_pcd,
            real_world_camera=args.real_world_camera
        )

        traj_list = defaultdict(list)
        
        reach_till_contact_idx = reach_till_contact_idx // args.combine_action_steps
        open_time_idx = open_time_idx // args.combine_action_steps

        expert_states = expert_states[::args.combine_action_steps]
        load_env(simulator._env, load_path=expert_states[0])
        # random set the camera of the environment
        if args.randomize_camera:
            simulator.reset_random_cameras()
        
        if simulator.check_handle_observed_in_pc() < 5:
            print("Handle not observed in the point cloud, continue")
            camera_detected = False
            simulator._env.close()

        if camera_detected:
            for t_idx, state in enumerate(tqdm.tqdm(expert_states)):
                load_env(simulator._env, load_path=state)
                
                # only object is the opposite to add_distractors
                only_object = True
                observation = simulator._get_observation(only_object=only_object)  
                rgb = simulator._env.render()

                traj_list['rgb'].append(rgb)
                for key in obs_keys:
                    traj_list[key].append(observation[key].tolist())
                    
            goal_gripper_pcd_at_grasping = traj_list['gripper_pcd'][open_time_idx]
            goal_gripper_pcd_at_end = traj_list['gripper_pcd'][-1]
            for t in range(open_time_idx):
                traj_list['goal_gripper_pcd'][t] = goal_gripper_pcd_at_grasping
            for t in range(open_time_idx, len(traj_list['gripper_pcd'])):
                traj_list['goal_gripper_pcd'][t] = goal_gripper_pcd_at_end
            
            simulator._env.close()
    
        end = time.time()
        cprint(f"Finished extracting data from trajectory with length: {len(expert_states) // args.combine_action_steps} time cost {end - beg}", "green")

        if not bad_experiment and camera_detected:
            all_traj_stage_lengths.append(stage_lengths)
            all_traj_store_label_paths.append(experiment_path)
            for key in obs_keys + ['rgb']:
                all_traj_obs_dict_of_list[key].append(traj_list[key])
            all_view_matrices.append(simulator.view_matrices)
            all_proj_matrices.append(simulator.project_matrices)
        else:
            label_path = os.path.join(experiment_path, "label.json")
            print("handle not in camera!!")
            try:
                with open(label_path, "w") as f:
                    json.dump({"good_traj": False, "failure reason": "simulation error during door opening"}, f)        
            except:
                pass
    
    return all_traj_obs_dict_of_list, all_traj_stage_lengths, all_traj_store_label_paths, all_view_matrices, all_proj_matrices
    
def extract_demos_from_a_directory(dirtory_path, exp_name=None, env_name=None, extract_name=None, save_path=None):
    
    demo_rgb_save_path = os.path.join(save_path, "demo_rgbs")
    if not os.path.exists(demo_rgb_save_path):
        os.makedirs(demo_rgb_save_path)

    all_demo_paths = []
    task_path = extract_name
    solution_path = os.path.join(dirtory_path, task_path)
    files_and_folders = os.listdir(solution_path)
    task_config_path = None
    for file_or_folder in files_and_folders:
        if file_or_folder.endswith(".yaml"):
            task_config_path = os.path.join(dirtory_path, task_path, file_or_folder)
            
    if exp_name is None:
        experiment_folder = os.path.join(solution_path, "experiment")
    else:
        experiment_folder = os.path.join(solution_path, "experiment", exp_name)

    if not os.path.exists(experiment_folder):
        print("Experiment folder does not exist: ", experiment_folder)
        if exp_name is not None:
            experiment_folder = solution_path
        else:
            experiment_folder = os.path.join(solution_path, exp_name)
            
    all_experiments = [x for x in os.listdir(experiment_folder) if os.path.isdir(os.path.join(experiment_folder, x)) and not x.startswith(".")]
    all_experiments = sorted(all_experiments)
    print("all experiments to render: ", all_experiments)
    # filter out successful experiments
    success_experiments = []
    for exp in all_experiments:
        state_path = os.path.join(experiment_folder, exp, "states")
        if os.path.exists(state_path):
            if len(os.listdir(state_path)) > 1 and os.path.exists(os.path.join(experiment_folder, exp, "all.gif")):
                success_experiments.append(exp)
    
    all_experiments = success_experiments
    num_experiment = len(all_experiments)
    
    _, all_expert_opened_angles = get_all_expert_angles([os.path.join(experiment_folder, exp) for exp in all_experiments])
    angle_threshold = np.quantile(all_expert_opened_angles, 0.1)
    
    num_experiment = min(num_experiment, args.num_experiment)        
    batch_size = 1
    num_batch = (num_experiment - 1) // batch_size + 1
    
    for batch_idx in range(num_batch):
        beg_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_experiment)
        
        all_traj_obs_dict_of_list, all_traj_stage_lengths, all_traj_store_label_paths, all_view_matrices, all_proj_matrices = extract_pc_states_for_all_trajectories(
            [
                task_config_path, solution_path, env_name, exp_name, 
                all_experiments[beg_idx:end_idx], 
                save_path, 
                angle_threshold, args
            ])

        all_traj_pc = all_traj_obs_dict_of_list['point_cloud']
        all_traj_pos_ori = all_traj_obs_dict_of_list['agent_pos']
        all_traj_rgbs = all_traj_obs_dict_of_list['rgb']
        all_traj_gripper_pcds = all_traj_obs_dict_of_list['gripper_pcd']
        all_traj_goal_gripper_pcd = all_traj_obs_dict_of_list['goal_gripper_pcd']
        all_traj_displacement_gripper_to_object = all_traj_obs_dict_of_list['displacement_gripper_to_object']
        
        print("all traj_pc length: ", len(all_traj_pc))
        
        for traj_idx in tqdm.tqdm(range(len(all_traj_pc)), total=len(all_traj_pc)):

            traj_pc, traj_pos_ori, traj_gripper_pcd = all_traj_pc[traj_idx], all_traj_pos_ori[traj_idx], all_traj_gripper_pcds[traj_idx]
            traj_stage_length, traj_store_label_path  = all_traj_stage_lengths[traj_idx], all_traj_store_label_paths[traj_idx]
            # traj_dp3_pc = all_traj_dp3_pc[traj_idx]
            traj_goal_gripper_pcd, traj_displacement_gripper_to_object = all_traj_goal_gripper_pcd[traj_idx], all_traj_displacement_gripper_to_object[traj_idx]
            
            print(f"starting to save traj {traj_idx} with length {len(traj_pc)}")
            
            good_traj = True
            failure_reason = "null"

            traj_actions = []
            quaternion_diffs = []
            opening_start_idx = traj_stage_length['reach_handle'] + traj_stage_length['reach_to_contact'] + traj_stage_length['close_gripper']
            after_contact_idx = traj_stage_length['reach_handle'] + traj_stage_length['reach_to_contact']
            opening_start_idx = opening_start_idx // args.combine_action_steps
            after_contact_idx = after_contact_idx // args.combine_action_steps
        
            filtered_pcs = []
            filtered_pos_oris = []
            filtered_gripper_pcds = []
            filtered_rgbs = []
            # filtered_dp3_pcs = []
            filtered_goal_gripper_pcds = []
            filtered_displacement_gripper_to_objects = []
            
            base_pos = traj_pos_ori[0][:3]
            base_ori_6d = traj_pos_ori[0][3:9]
            base_finger_angle = traj_pos_ori[0][9]
            base_rgb = all_traj_rgbs[traj_idx][0]
            base_gripper_pcd = traj_gripper_pcd[0]
            base_pc = traj_pc[0]
            base_pos_ori = traj_pos_ori[0]
            base_goal_gripper_pcd = traj_goal_gripper_pcd[0]
            base_displacement_gripper_to_object = traj_displacement_gripper_to_object[0]
            
            for i in range(len(traj_pos_ori) - 1):
                cur_pos = traj_pos_ori[i][:3]
                target_pos = traj_pos_ori[i+1][:3]

                single_step_delta_pos = np.array(target_pos) - np.array(cur_pos)
                
                # if single step translation is too large, ignore this trajectory
                if np.linalg.norm(single_step_delta_pos) > 0.02 * args.combine_action_steps:
                    good_traj = False
                    failure_reason = "delta movement too large"
                    print("not good traj due to delta movement too large")
                    break
                
                delta_pos = np.array(target_pos) - np.array(base_pos)

                cur_ori_6d = traj_pos_ori[i][3:9]
                
                target_ori_6d = traj_pos_ori[i+1][3:9]
                cur_ori_matrix = rotation_transfer_6D_to_matrix(cur_ori_6d)
                base_ori_matrix = rotation_transfer_6D_to_matrix(base_ori_6d)
                target_ori_matrix = rotation_transfer_6D_to_matrix(target_ori_6d)

                delta_ori_matrix = base_ori_matrix.T @ target_ori_matrix
                delta_ori_6d = rotation_transfer_matrix_to_6D(delta_ori_matrix)
                
                cur_ori_quat =  R.from_matrix(cur_ori_matrix).as_quat()
                base_ori_quat =  R.from_matrix(base_ori_matrix).as_quat()
                target_ori_quat = R.from_matrix(target_ori_matrix).as_quat()
                quat_diff = np.arccos(2 * np.dot(base_ori_quat, target_ori_quat)**2 - 1)
                one_step_quaternion_diff = np.arccos(2 * np.dot(cur_ori_quat, target_ori_quat)**2 - 1)
                quaternion_diffs.append(quat_diff)
                
                # if single step rotation is too large, ignore this trajectory
                if np.abs(one_step_quaternion_diff) > 0.085 * args.combine_action_steps:
                    good_traj = False
                    failure_reason = "delta quaternion too large"
                    print("not good due to delta quaternion too large")
                    break
                                        
                target_finger_angle = traj_pos_ori[i+1][9]
                delta_finger_angle = target_finger_angle - base_finger_angle
                        
                filter_action = False
                if i >= after_contact_idx and i < opening_start_idx:
                    if np.abs(delta_finger_angle) < args.min_finger_angle_diff and args.filter_close_zero_action:
                        filter_action = True
                        
                if filter_action:
                    continue
                else:
                    action = delta_pos.tolist() + delta_ori_6d.tolist() + [delta_finger_angle]
                    traj_actions.append(action)
                    filtered_pcs.append(base_pc)
                    filtered_gripper_pcds.append(base_gripper_pcd)
                    filtered_pos_oris.append(base_pos_ori)
                    filtered_rgbs.append(base_rgb)
                    # filtered_dp3_pcs.append(base_dp3_pc)
                    filtered_goal_gripper_pcds.append(base_goal_gripper_pcd)
                    filtered_displacement_gripper_to_objects.append(base_displacement_gripper_to_object)
                    
                    base_pc = traj_pc[i+1]
                    base_gripper_pcd = traj_gripper_pcd[i+1]
                    base_pos_ori = traj_pos_ori[i+1]
                    base_rgb = all_traj_rgbs[traj_idx][i+1]
                    base_pos = target_pos
                    base_ori_6d = target_ori_6d
                    base_finger_angle = target_finger_angle
                    # base_dp3_pc = traj_dp3_pc[i+1]
                    base_displacement_gripper_to_object = traj_displacement_gripper_to_object[i+1]
                    base_goal_gripper_pcd = traj_goal_gripper_pcd[i+1]
                    
        
            # plot the delta translation action distribution
            if traj_idx % 5 == 0:        
                try:
                    save_numpy_as_gif(np.array(filtered_rgbs), os.path.join(demo_rgb_save_path, "demo_" + str(traj_idx) + ".gif"))
                    plt.close("all")
                except:
                    pass

            path = os.path.join(traj_store_label_path, "label.json")
            if not os.path.exists(path):
                try:
                    with open(path, 'w') as f:
                        json.dump({"good_traj": good_traj, "failure reason": failure_reason}, f)
                except:
                    pass

            print(f"finished parsing traj, right before saving data, good_traj is {good_traj}")
            if good_traj:
                for i in range(len(traj_actions)):
                    if i >= after_contact_idx:
                        traj_actions[i][-1] = args.close_gripper_action
                        
                experiment_path = traj_store_label_path
                all_demo_paths.append(experiment_path)
                data_save_path = os.path.join(save_path, experiment_path.split("/")[-1])
                beg = time.time()
                save_data_pickle(filtered_pcs, filtered_pos_oris, filtered_gripper_pcds, 
                        traj_actions, None, 
                        filtered_goal_gripper_pcds, filtered_displacement_gripper_to_objects, 
                        data_save_path)
                end = time.time()
                cprint("Finished saving data to {} using time {}".format(data_save_path, end-beg), "green")
                camera_param_save_path = os.path.join(data_save_path, "camera_params.npz")
                np.savez(camera_param_save_path,
                    view_matrix=np.array(all_view_matrices[traj_idx]),
                    proj_matrix=np.array(all_proj_matrices[traj_idx])
                )
                del filtered_pcs, filtered_pos_oris, filtered_gripper_pcds, traj_actions, filtered_displacement_gripper_to_objects, filtered_goal_gripper_pcds
        
        save_example_pointcloud([traj_pc[0] for traj_pc in all_traj_pc], save_path)
        del all_traj_pc, all_traj_pos_ori, all_traj_rgbs,  all_traj_gripper_pcds,  all_traj_stage_lengths, all_traj_store_label_paths
            
    with open(os.path.join(save_path, "all_demo_path.txt"), "w") as f:
        f.write("\n".join(all_demo_paths))


def save_data_pickle(pc_list, state_list, gripper_pcd_list, action_list, 
              dp3_pc_list,
              goal_gripper_pcd, 
              displacement_gripper_to_object,
              save_dir):

    state_arrays = np.array(state_list)
    point_cloud_arrays = np.array(pc_list)
    action_arrays = np.array(action_list)
    gripper_pcd_arrays = np.array(gripper_pcd_list)
    
    if goal_gripper_pcd is not None:
        goal_gripper_pcd = np.array(goal_gripper_pcd)
    if displacement_gripper_to_object is not None:
        displacement_gripper_to_object = np.array(displacement_gripper_to_object)
    if dp3_pc_list is not None:
        dp3_pc_list = np.array(dp3_pc_list)
        
    traj_len = len(state_list)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for t_idx in range(traj_len):
        step_save_dir = os.path.join(save_dir, str(t_idx) + ".pkl")
            
        pickle_data = {}
        pickle_data['state'] = state_arrays[t_idx][None, :]
        pickle_data['point_cloud'] = point_cloud_arrays[t_idx][None, :]
        pickle_data['action'] = action_arrays[t_idx][None, :]
        pickle_data['gripper_pcd'] = gripper_pcd_arrays[t_idx][None, :]
        if goal_gripper_pcd is not None:
            pickle_data['goal_gripper_pcd'] = goal_gripper_pcd[t_idx][None, :]
        if displacement_gripper_to_object is not None:
            pickle_data['displacement_gripper_to_object'] = displacement_gripper_to_object[t_idx][None, :]
        if dp3_pc_list is not None:
            pickle_data['dp3_point_cloud'] = dp3_pc_list[t_idx][None, :]
        
        with open(step_save_dir, 'wb') as f:
            pickle.dump(pickle_data, f)


    del state_arrays, point_cloud_arrays, gripper_pcd_arrays, action_arrays
    if goal_gripper_pcd is not None:
        del goal_gripper_pcd
    if displacement_gripper_to_object is not None:
        del displacement_gripper_to_object
    if dp3_pc_list is not None:
        del dp3_pc_list

def save_example_pointcloud(pc_list, save_dir, name='example_pointcloud'):
    if len(pc_list) > 10:
        idxes = np.random.choice(len(pc_list), 10)
    else:
        idxes = range(len(pc_list))
    save_dir = os.path.join(save_dir, "example_pointcloud")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, idx in enumerate(idxes):
        point_cloud = np.array(pc_list[idx])
        ax = plt.axes(projection='3d')
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
        ax.view_init(azim=-90, elev=10)
        plt.savefig(os.path.join(save_dir, "example_pc_" + str(i) + ".png"))
        plt.close()


def main(folder_name, save_path, exp_name=None, env_name=None, extract_name=None):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    meta_info = {
        "folder_name": folder_name,
        "exp_name": exp_name,
    }
    meta_info.update(args.__dict__)
    with open(os.path.join(save_path, "meta_info.json"), "w") as f:
        json.dump(meta_info, f, indent=4)
    
    extract_demos_from_a_directory(folder_name, exp_name=exp_name, env_name=env_name, extract_name=extract_name, save_path=save_path)
    

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--pointcloud_num", type=int, default=4500)
    args.add_argument("--observation_mode", type=str, default='act3d_goal_displacement_gripper_to_object')
    args.add_argument("--filter_close_zero_action", type=int, default=1)
    args.add_argument("--min_finger_angle_diff", type=float, default=0.001)
    args.add_argument("--close_gripper_action", type=float, default=-0.006)
    args.add_argument("--combine_action_steps", type=int, default=2)
    args.add_argument("--min_opened_ratio", type=float, default=0.2)
    args.add_argument("--randomize_camera", type=int, default=0)
    args.add_argument("--noise_real_world_pcd", type=int, default=0)
    args.add_argument("--real_world_camera", type=int, default=0)
    
    args.add_argument("--save_path", type=str, required=True)
    args.add_argument("--folder_name", type=str, required=True)
    args.add_argument("--exp_name", type=str, default=None)
    args.add_argument("--env_name", type=str, default="articulated")
    args.add_argument("--extract_name", type=str, default=None)
    args.add_argument("--num_experiment", type=int, default=10000)
    args = args.parse_args()

    main(args.folder_name, args.save_path, exp_name=args.exp_name, env_name=args.env_name, extract_name=args.extract_name)
