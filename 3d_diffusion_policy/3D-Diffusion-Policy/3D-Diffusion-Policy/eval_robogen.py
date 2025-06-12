import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
import pybullet as p
import numpy as np
from copy import deepcopy
import sys
from termcolor import cprint
import tqdm
import json
import time
import yaml
import pickle as pkl
import argparse
from typing import List, Optional
from collections import deque
from manipulation.robogen_wrapper import RobogenPointCloudWrapper
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from train_ddp import TrainDP3Workspace
from diffusion_policy_3d.common.pytorch_util import dict_apply
from manipulation.utils import build_up_env, save_numpy_as_gif

def construct_env(cfg, config_file, solution_path, task_name, init_state_file, 
                  real_world_camera=False, noise_real_world_pcd=False,
                  randomize_camera=False):
    env, _ = build_up_env(
                    config_file,
                    solution_path,
                    task_name,
                    init_state_file,
                    render=False, 
                    horizon=600,
            )
            
    object_name = "StorageFurniture".lower()
    env.reset()
    pointcloud_env = RobogenPointCloudWrapper(env, object_name, 
                                                num_points=cfg.task.env_runner.num_point_in_pc,
                                                observation_mode=cfg.task.env_runner.observation_mode,
                                                real_world_camera=real_world_camera,
                                                noise_real_world_pcd=noise_real_world_pcd,
                                            )
        
    if randomize_camera:
        pointcloud_env.reset_random_cameras()
        
    env = MultiStepWrapper(pointcloud_env, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps, 
                        max_episode_steps=600, reward_agg_method='sum')
    return env

def prepare_env(experiment_folder, experiment_path, all_experiments):
    all_substeps_path = os.path.join(experiment_folder, "substeps.txt")
    with open(all_substeps_path, "r") as f:
        substeps = f.readlines()
        first_step = substeps[0].lstrip().rstrip()        

    expert_opened_angles = []
    init_state_files = []
    config_files = []
    for experiment in all_experiments:
        if "meta" in experiment:
            continue
        
        first_step_folder = first_step.replace(" ", "_") + "_primitive"
        first_step_folder = os.path.join(experiment_path, experiment, first_step_folder)
        if os.path.exists(os.path.join(first_step_folder, "label.json")):
            with open(os.path.join(first_step_folder, "label.json"), 'r') as f:
                label = json.load(f)
            if not label['good_traj']: continue
            
        first_step_states_path = os.path.join(first_step_folder, "states")
        expert_states = os.listdir(first_step_states_path)
        if len(expert_states) == 0:
            continue
            
        expert_opened_angle_file = os.path.join(experiment_path, experiment, first_step_folder, "opened_angle.txt")
        if os.path.exists(expert_opened_angle_file):
            with open(expert_opened_angle_file, "r") as f:
                angles = f.readlines()
                expert_opened_angle = float(angles[0].lstrip().rstrip())
                max_angle = float(angles[-1].lstrip().rstrip())
                ratio = expert_opened_angle / max_angle

        expert_opened_angles.append(expert_opened_angle)
        init_state_file = os.path.join(first_step_states_path, "state_0.pkl")
        init_state_files.append(init_state_file)
        config_file = os.path.join(experiment_path, experiment, "task_config.yaml")
        config_files.append(config_file)
                
    return config_files, init_state_files, expert_opened_angles

def high_level_policy_infer(parallel_input_dict, high_level_policy, output_obj_pcd_only=True):
    with torch.no_grad():
        pointcloud = parallel_input_dict['point_cloud'][:, -1, :, :]
        gripper_pcd = parallel_input_dict['gripper_pcd'][:, -1, :]
        inputs = torch.cat([pointcloud, gripper_pcd], dim=1)
            
        if args.add_one_hot_encoding:
            # for pointcloud, we add (1, 0)
            # for gripper_pcd, we add (0, 1)
            pointcloud_one_hot = torch.zeros(pointcloud.shape[0], pointcloud.shape[1], 2).float().to(pointcloud.device)
            pointcloud_one_hot[:, :, 0] = 1
            pointcloud_ = torch.cat([pointcloud, pointcloud_one_hot], dim=2)
            gripper_pcd_one_hot = torch.zeros(gripper_pcd.shape[0], gripper_pcd.shape[1], 2).float().to(pointcloud.device)
            gripper_pcd_one_hot[:, :, 1] = 1
            gripper_pcd_ = torch.cat([gripper_pcd, gripper_pcd_one_hot], dim=2)
            inputs = torch.cat([pointcloud_, gripper_pcd_], dim=1) # B, N+4, 5
            
        inputs = inputs.to('cuda')
        inputs_ = inputs.permute(0, 2, 1)
        outputs = high_level_policy(inputs_)
        weights = outputs[:, :, -1] # B, N
        outputs = outputs[:, :, :-1] # B, N, 12
        if output_obj_pcd_only:
            weights = weights[:, :-4]
            outputs = outputs[:, :-4, :]
            inputs = inputs[:, :-4, :]

        B, N, _ = outputs.shape
        outputs = outputs.view(B, N, 4, 3)
                
        outputs = outputs + inputs[:, :, :3].unsqueeze(2)
        weights = torch.nn.functional.softmax(weights, dim=1)
        outputs = outputs * weights.unsqueeze(-1).unsqueeze(-1)
        outputs = outputs.sum(dim=1)
        outputs = outputs.unsqueeze(1)
        
    return outputs
            
def run_eval_non_parallel(cfg, low_level_policy, high_level_policy, 
                          save_path, exp_beg_idx=0,
                          exp_end_idx=1000, 
                          horizon=150,  
                          exp_beg_ratio=None, exp_end_ratio=None,
                          dataset_index=None, 
                          output_obj_pcd_only=False, 
                          update_goal_freq=1, 
                          real_world_camera=False, 
                          noise_real_world_pcd=False,
                          randomize_camera=False):
    
    ### loop through each test object
    for dataset_idx, (experiment_folder, experiment_name) in enumerate(zip(cfg.task.env_runner.experiment_folder, cfg.task.env_runner.experiment_name)):

        if dataset_index is not None:
            dataset_idx = dataset_index

        init_state_files = []
        config_files = []
        experiment_folder = "{}/{}".format(os.environ['PROJECT_DIR'], experiment_folder)
        experiment_name = experiment_name
        experiment_path = os.path.join(experiment_folder, "experiment", experiment_name)
        all_experiments = os.listdir(experiment_path)
        all_experiments = sorted(all_experiments)
        config_files, init_state_files, expert_opened_angles = prepare_env(experiment_folder, experiment_path, all_experiments)
        
        opened_joint_angles = {}

        if exp_end_ratio is not None:
            exp_end_idx = int(exp_end_ratio * len(config_files))
        if exp_beg_ratio is not None:
            exp_beg_idx = int(exp_beg_ratio * len(config_files))

        config_files = config_files[exp_beg_idx:exp_end_idx]
        init_state_files = init_state_files[exp_beg_idx:exp_end_idx]
        expert_opened_angles = expert_opened_angles[exp_beg_idx:exp_end_idx]
        
        ### loop through each test configuration of the object
        for exp_idx, (config_file, init_state_file) in enumerate(zip(config_files, init_state_files)):
                
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            solution_path = [x['solution_path'] for x in config if "solution_path" in x][0]
            all_substeps_path = os.path.join(os.environ['PROJECT_DIR'], solution_path, "substeps.txt")
            with open(all_substeps_path, "r") as f:
                substeps = f.readlines()
                first_step = substeps[0].lstrip().rstrip()
                task_name = first_step.replace(" ", "_")
            
            env = construct_env(cfg, config_file, solution_path, task_name, init_state_file, real_world_camera, noise_real_world_pcd, 
                                randomize_camera)
            
            obs = env.reset()
            rgb = env.env.render()
            info = env.env._env._get_info()
            all_rgbs = [rgb]
            last_goal = None
            for t in range(1, horizon):
                parallel_input_dict = obs
                parallel_input_dict = dict_apply(parallel_input_dict, lambda x: torch.from_numpy(x).to('cuda'))
                for key in obs:
                    parallel_input_dict[key] = parallel_input_dict[key].unsqueeze(0)
                
                ### infer the high-level policy to get the predicted goal
                if t == 1 or t % update_goal_freq == 0:
                    predicted_goal = high_level_policy_infer(parallel_input_dict, high_level_policy, output_obj_pcd_only=output_obj_pcd_only)
                    last_goal = predicted_goal
                else:
                    predicted_goal = last_goal
                    
                ### run the low-level policy to get the robot eef delta transformations    
                np_predicted_goal = predicted_goal.detach().to('cpu').numpy()
                predicted_goal = predicted_goal.repeat(1, 2, 1, 1)
                parallel_input_dict['goal_gripper_pcd'] = predicted_goal
                with torch.no_grad():
                    batched_action = low_level_policy.predict_action(parallel_input_dict)
                np_batched_action = dict_apply(batched_action, lambda x: x.detach().to('cpu').numpy())
                np_batched_action = np_batched_action['action']
                
                ### step the environment with the low-level action
                obs, reward, done, info = env.step(np_batched_action.squeeze(0))
                env.env.goal_gripper_pcd = np_predicted_goal.squeeze(0)[0].reshape(4, 3)
                rgb = env.env.render()
                all_rgbs.append(rgb)
            
            env.env._env.close()

            ### save statistics
            opened_joint_angles[config_file] = \
            {
                "final_door_joint_angle": float(info['opened_joint_angle'][-1]), 
                "expert_door_joint_angle": expert_opened_angles[exp_idx], 
                "initial_joint_angle": float(info['initial_joint_angle'][-1]),
                "ik_failure": float(info['ik_failure'][-1]),
                'grasped_handle': float(info['grasped_handle'][-1]),
                "exp_idx": exp_idx, 
            }
                    
            with open("{}/opened_joint_angles_{}.json".format(save_path, dataset_idx), "w") as f:
                json.dump(opened_joint_angles, f, indent=4)
            
            gif_save_exp_name = experiment_folder.split("/")[-2]
            gif_save_folder = "{}/{}".format(save_path, gif_save_exp_name)
            if not os.path.exists(gif_save_folder):
                os.makedirs(gif_save_folder, exist_ok=True)
            gif_save_path = "{}/{}_{}.gif".format(gif_save_folder, exp_idx, 
                    float(info["improved_joint_angle"][-1]))
            
            save_numpy_as_gif(np.array(all_rgbs), gif_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--low_level_exp_dir', type=str, default=None)
    parser.add_argument('--low_level_ckpt_name', type=str, default=None)
    parser.add_argument("--high_level_ckpt_name", type=str, default=None)
    parser.add_argument("--eval_exp_name", type=str, default=None)
    parser.add_argument('--output_obj_pcd_only', type=int, default=1)
    parser.add_argument("--update_goal_freq", type=int, default=1)
    parser.add_argument("--noise_real_world_pcd", type=int, default=0)
    parser.add_argument("--randomize_camera", type=int, default=0)
    parser.add_argument("--real_world_camera", type=int, default=0)
    parser.add_argument('--add_one_hot_encoding', type=int, default=0)
    args = parser.parse_args()
    
    ### load low-level policy
    exp_dir = args.low_level_exp_dir
    checkpoint_name = args.low_level_ckpt_name
    with hydra.initialize(config_path='diffusion_policy_3d/config'):  # same config_path as used by @hydra.main
        recomposed_config = hydra.compose(
            config_name="dp3.yaml",  # same config_name as used by @hydra.main
            overrides=OmegaConf.load("{}/.hydra/overrides.yaml".format(exp_dir)),
        )
    cfg = recomposed_config
    workspace = TrainDP3Workspace(cfg)
    checkpoint_dir = "{}/checkpoints/{}".format(exp_dir, checkpoint_name)
    workspace.load_checkpoint(path=checkpoint_dir, )
    low_level_policy = deepcopy(workspace.model)
    if workspace.cfg.training.use_ema:
        low_level_policy = deepcopy(workspace.ema_model)
    low_level_policy.eval()
    low_level_policy.reset()
    low_level_policy = low_level_policy.to('cuda')
    
    ### load the high-level policy
    load_model_path = args.high_level_ckpt_name    
    num_class = 13 
    input_channel = 5 if args.add_one_hot_encoding else 3
    from weighted_displacement_model.model_invariant import PointNet2_super
    high_level_policy = PointNet2_super(num_classes=num_class, input_channel=input_channel).to("cuda")
    high_level_policy.load_state_dict(torch.load(load_model_path))
    high_level_policy.eval()
    
    
    ### prepare the evaluation environment
    cfg.task.env_runner.experiment_name = ['0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first' for _ in range(10)]
    cfg.task.env_runner.experiment_folder = [
        'data/diverse_objects/open_the_door_40147/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_44817/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_44962/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45132/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45219/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45243/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45332/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45378/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45384/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45463/task_open_the_door_of_the_storagefurniture_by_its_handle'
    ]
    cfg.task.env_runner.demo_experiment_path = [None for _ in range(10)]
    
    
    ### dump evaluation configuration
    checkpoint_dir = "{}/checkpoints/{}".format(exp_dir, checkpoint_name)
    save_path = "data/{}".format(args.eval_exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    checkpoint_info = {
        "low_level_policy": checkpoint_dir,
        "low_level_policy_checkpoint": checkpoint_name,
        "high_level_policy_checkpoint": args.high_level_ckpt_name,
    }
    checkpoint_info.update(args.__dict__)
    with open("{}/checkpoint_info.json".format(save_path), "w") as f:
        json.dump(checkpoint_info, f, indent=4)
    
    cfg.task.env_runner.observation_mode = "act3d_goal_displacement_gripper_to_object"
    cfg.task.dataset.observation_mode = "act3d_goal_displacement_gripper_to_object"
    run_eval_non_parallel(
            cfg, low_level_policy, high_level_policy,
            save_path, 
            horizon=35,
            exp_beg_idx=0,
            exp_end_idx=25,
            output_obj_pcd_only=args.output_obj_pcd_only,
            update_goal_freq=args.update_goal_freq,
            real_world_camera=args.real_world_camera,
            noise_real_world_pcd=args.noise_real_world_pcd,
            randomize_camera=args.randomize_camera,
    )

