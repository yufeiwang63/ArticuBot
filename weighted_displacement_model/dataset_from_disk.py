import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import zarr
import os
from termcolor import cprint
import numpy as np
from tqdm import tqdm
import pickle
import random
        
from weighted_displacement_model.all_data import *
from scripts.datasets.randomize_partition_10_obj import *
from scripts.datasets.randomize_partition_50_obj import *
from scripts.datasets.randomize_partition_100_obj import *
from scripts.datasets.randomize_partition_200_obj import *

class PointNetDatasetFromDisk(torch.utils.data.Dataset):
    def __init__(self, all_obj_paths, beg_ratio=0, end_ratio=0.9, use_all_data=False):
        self.all_obj_paths = all_obj_paths
        self.beg_ratio = beg_ratio
        self.end_ratio = end_ratio
        self.use_all_data = use_all_data

        self.all_trajectory_path = []
        self.episode_idx_to_obj_id = {}
        self.obj_id_to_all_episodes_indices = {}

        episode_idx = 0
        ### loop through all training objects to get all demonstration trajectories
        for obj_path in all_obj_paths:
            all_subfolder = os.listdir(obj_path)
            for s in ['action_dist', 'demo_rgbs', 'all_demo_path.txt', 'meta_info.json', 'example_pointcloud']:
                if s in all_subfolder:
                    all_subfolder.remove(s)
            all_subfolder = sorted(all_subfolder)
            beg = int(beg_ratio * len(all_subfolder))
            end = int(end_ratio * len(all_subfolder))
            if not self.use_all_data:
                end = min(end, 75)
            all_subfolder = all_subfolder[beg:end]
            self.all_trajectory_path += [os.path.join(obj_path, s) for s in all_subfolder]
            this_obj_episode_beg = episode_idx
            for s in all_subfolder:
                self.episode_idx_to_obj_id[episode_idx] = obj_path
                episode_idx += 1
            this_obj_episode_end = episode_idx
            self.obj_id_to_all_episodes_indices[obj_path] = [i for i in range(this_obj_episode_beg, this_obj_episode_end)]            

        cprint('Preparing all pickle data', 'green')
        self.episode_lengths = []
        for idx, traj_path in enumerate(tqdm(self.all_trajectory_path)):
            all_substeps = os.listdir(traj_path)
            self.episode_lengths.append(len(all_substeps))
                
        self.episode_lengths = np.array(self.episode_lengths)
        self.accumulated_episode_lengths = np.cumsum(self.episode_lengths)
        cprint(f'Finished preparing all pickle data with total datapoints: {self.accumulated_episode_lengths[-1]}', 'green')

    def __len__(self):
        return self.accumulated_episode_lengths[-1]
    
    def read_pickle_data(self, episode_idx, step_idx):
        step_path = os.path.join(self.all_trajectory_path[episode_idx], str(step_idx) + '.pkl')
        with open(step_path, 'rb') as f:
            data = pickle.load(f)
        pointcloud = data['point_cloud'][:][0].astype(np.float32)
        gripper_pcd = data['gripper_pcd'][:][0].astype(np.float32)
        goal_gripper_pcd = data['goal_gripper_pcd'][:][0].astype(np.float32)
        return pointcloud, gripper_pcd, goal_gripper_pcd

    def __getitem__(self, idx):
        
        episode_idx = np.searchsorted(self.accumulated_episode_lengths, idx, side='right')
        start_idx = idx - self.accumulated_episode_lengths[episode_idx]
        if start_idx < 0:
            start_idx += self.episode_lengths[episode_idx]
            
        pointcloud, gripper_pcd, goal_gripper_pcd = self.read_pickle_data(episode_idx, start_idx)
        return pointcloud, gripper_pcd, goal_gripper_pcd
    

def get_dataset_from_pickle(all_obj_paths=None, beg_ratio=0, end_ratio=0.9, use_all_data=False, dataset_prefix=None, num_train_objects=200):
    
    assert dataset_prefix is not None, "dataset_prefix must be provided for loading the data from disk"
    if not dataset_prefix.startswith('/'):
        dataset_prefix = os.path.join(os.environ['PROJECT_DIR'], dataset_prefix)
    
    if all_obj_paths is None:
        if num_train_objects == 'debug':
            all_obj_paths = [f'{dataset_prefix}/0628-act3d-obj-47570-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1']
        ### without camera randomizations
        elif num_train_objects == '10':
            all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(10)]
        elif num_train_objects == '50':
            all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(50)]
        elif num_train_objects == '100':
            all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(100)]
        elif num_train_objects == '200':
            all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(200)]
        elif num_train_objects == '300':
            all_trajectory_path_part_1 = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(246)]
            all_subfolders = sorted(os.listdir(dataset_prefix))
            object_other_categories_no_cam_rand = [x for x in all_subfolders if "1121-other-cat-no-cam-rand" in x]
            all_trajectory_path_part_2 = [f"{dataset_prefix}/{x}" for x in object_other_categories_no_cam_rand]
            num = len(all_trajectory_path_part_2)
            all_obj_paths = all_trajectory_path_part_1 + all_trajectory_path_part_2
        ### with camera randomizations
        elif num_train_objects == "camera_random_10_obj_high_level":
            all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["camera_random_10_save_data_name_{}".format(i)]) for i in range(20)]
        elif num_train_objects == 'camera_random_50_obj_high_level':
            all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["camera_random_50_save_data_name_{}".format(i)]) for i in range(87)]
        elif num_train_objects == 'camera_random_100_obj_high_level':
            all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["camera_random_100_save_data_name_{}".format(i)]) for i in range(175)]
        elif num_train_objects == 'camera_random_200_obj_high_level':
            all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["camera_random_200_save_data_name_{}".format(i)]) for i in range(350)]
        elif num_train_objects == 'camera_random_500_obj_high_level' or num_train_objects == "500_object_high_level":
            all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(462)]
        ### with noisy point cloud to simulate real-world depth camera noise
        elif num_train_objects == '500_plus_all_real_world_clean_distorted_goal':
            dataset_prefix = "/scratch/yufeiw2/dp3_demo_clean_distorted_goal"
            non_real_world_camera_500_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(463)]
            real_world_camera_500_paths = os.listdir("/scratch/yufeiw2/dp3_demo_real_world_noise_pcd_clean_distorted_goal")
            real_world_camera_500_paths = sorted(real_world_camera_500_paths)
            real_world_camera_500_paths = [os.path.join("/scratch/yufeiw2/dp3_demo_real_world_noise_pcd_clean_distorted_goal", x) for x in real_world_camera_500_paths]
            all_obj_paths = non_real_world_camera_500_paths + real_world_camera_500_paths
        else:
            raise ValueError('num_train_objects not supported')
        
    dataset = PointNetDatasetFromDisk(all_obj_paths, beg_ratio, end_ratio, use_all_data=use_all_data)    
    return dataset