from typing import Dict
import torch
import time
import numpy as np
import copy
import os
from tqdm import tqdm
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.common.sampler_disk import SequenceSampler
from termcolor import cprint
import random
import copy
from weighted_displacement_model.all_data import *
from scripts.datasets.randomize_partition_50_obj import *
from scripts.datasets.randomize_partition_100_obj import *
from scripts.datasets.randomize_partition_200_obj import *
from diffusion_policy_3d.common.replay_buffer_disk import ReplayBuffer

import pybullet as p
from manipulation.utils import rotation_transfer_6D_to_matrix_batch, rotation_transfer_matrix_to_6D_batch

def get_zarry_paths(zarr_path):
    dataset_prefix = os.path.join(os.environ['PROJECT_DIR'], 'data', 'dp3_demo_combined_2_step_0')
    if zarr_path == 'debug':
        all_zarr_paths = [os.path.join(dataset_prefix, '0628-act3d-obj-47570-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1')]
        
    if zarr_path == '10_object_low_level':
        all_zarr_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(10)]
    if zarr_path == '50_object_low_level':
        all_zarr_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(50)]
    if zarr_path == '100_object_low_level':
        all_zarr_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(100)]
    if zarr_path == "200_object_low_level":
        all_zarr_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(200)]
    if zarr_path == "300_object_low_level": 
        all_zarr_paths_part_1 = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(246)]
        all_subfolders = sorted(os.listdir(dataset_prefix))
        object_other_categories_no_cam_rand = [x for x in all_subfolders if "1121-other-cat-no-cam-rand" in x]
        all_zarr_paths_part_2 = [f"{dataset_prefix}/{x}" for x in object_other_categories_no_cam_rand]
        all_zarr_paths = all_zarr_paths_part_1 + all_zarr_paths_part_2
        
    return all_zarr_paths

class RobogenDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.1,
            train_ratio=0.9,
            max_train_episodes=None,
            task_name=None,
            observation_mode='act3d_goal_mlp',
            enumerate=True,
            is_pickle=True,
            dataset_keys=None,
            augmentation_pcd=False,
            augmentation_scale=False,
            scale_scene_by_pcd=False,
            augmentation_rot=False,
            prediction_target='action',
            **kwargs
            ):
        super().__init__()

        self.task_name = task_name
        self.observation_mode = observation_mode
        self.augmentation_rot = augmentation_rot
        self.augmentation_pcd = augmentation_pcd
        self.augmentation_scale = augmentation_scale
        self.scale_scene_by_pcd = scale_scene_by_pcd
        self.is_pickle = is_pickle
        self.prediction_target = prediction_target
        
        if dataset_keys is None:
            keys = ['state', 'action', 'point_cloud']
            if 'act3d' in observation_mode:
                keys += ['gripper_pcd']
                if 'goal' in observation_mode:
                    keys += ['goal_gripper_pcd']
                if 'displacement_gripper_to_object' in observation_mode:
                    keys += ['displacement_gripper_to_object']
        else:
            cprint(f"specifying dataset_keys: {dataset_keys}", "red")
            keys = dataset_keys
        
        self.keys_ = keys
        
        # try to get kept_in_disk from kwargs, if not, set it to False
        self.kept_in_disk = True
        self.load_per_step = kwargs.get('load_per_step', True)
        cprint("loading dataset in disk, need a lot of I/O", "red")
                    
        if type(zarr_path) == list:
            all_zarr_paths = copy.deepcopy(zarr_path)
        else:
            all_zarr_paths = get_zarry_paths(zarr_path)
        
        all_paths = []
        train_masks = []
        val_masks = []
        for zarr_path in tqdm(all_zarr_paths):
            all_subfolder = os.listdir(zarr_path)
            for string in ["action_dist", "demo_rgbs", "all_demo_path.txt", "meta_info.json", 'example_pointcloud', '.zgroup']:
                if string in all_subfolder:
                    all_subfolder.remove(string)
            all_subfolder = sorted(all_subfolder)
            n_episodes = len(all_subfolder)
            num_load_episodes = kwargs.get('num_load_episodes', n_episodes)
            num_load_episodes = min(num_load_episodes, n_episodes)
            all_subfolder = all_subfolder[:num_load_episodes]
            zarr_paths = []
            for subfolder in all_subfolder:
                if len(os.listdir(os.path.join(zarr_path, subfolder))) > 10:
                    zarr_paths.append(os.path.join(zarr_path, subfolder))
            all_paths += zarr_paths
            folder_train_mask = np.zeros(num_load_episodes, dtype=bool)
            folder_train_mask[:int(num_load_episodes*train_ratio)] = True
            train_masks.append(folder_train_mask)
            folder_val_mask = np.zeros(num_load_episodes, dtype=bool)
            folder_val_mask[-int(num_load_episodes*val_ratio):] = True
            val_masks.append(folder_val_mask)
        
        cprint(f'keep in disk and load per step, load_per_step:{self.load_per_step}', 'green')
        self.replay_buffer = ReplayBuffer.copy_from_multiple_path(all_paths, keys=keys, load_per_step=self.load_per_step, 
                                                                is_pickle=self.is_pickle, target_action=self.prediction_target)
        self.action_welford = self.replay_buffer.action_welford
        self.pcd_welford = self.replay_buffer.pcd_welford
        self.agent_pos_welford = self.replay_buffer.agent_pos_welford
        train_mask = np.concatenate(train_masks)
        self.val_mask = np.concatenate(val_masks)
        
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after 

        cprint('dataset has been loaded', 'green')
            
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.train_mask = self.val_mask
        return val_set
    

    def get_normalizer(self, mode='limits', **kwargs):
        normalizer = LinearNormalizer()
        keys = ['action']
        for key in keys:
            if key == 'action':
                welford = self.action_welford
            if key == 'point_cloud' or key == 'gripper_pcd':
                welford = self.pcd_welford
            if key == 'agent_pos':
                welford = self.agent_pos_welford
            
            input_min = welford.get_min()
            input_max = welford.get_max()
            input_mean = welford.get_mean()
            input_std = welford.get_std()
            input_range = input_max - input_min
            range_eps = 1e-4
            output_min = -1
            output_max = 1
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
            scale = torch.from_numpy(scale).float()
            offset = torch.from_numpy(offset).float()
            this_params = torch.nn.ParameterDict({
                'scale': scale,
                'offset': offset,
                'input_stats': torch.nn.ParameterDict({
                    'min': input_min,
                    'max': input_max,
                    'mean': input_mean,
                    'std': input_std
                })
            })
            for p in this_params.values():
                p.requires_grad = False
                
            if key == 'action':
                normalizer.params_dict[self.prediction_target] = this_params
            else:
                normalizer.params_dict[key] = this_params

        if self.augmentation_rot:
            value = self.action_welford.get_max_norm_3d()
            value = torch.from_numpy(value).float()
            additional_params = torch.nn.ParameterDict({
                'max_norm_3d': value
            })
            for p in additional_params.values():
                p.requires_grad = False

            normalizer.params_dict['additional_params'] = additional_params

        return normalizer
    
    def __len__(self) -> int:
        return len(self.sampler)
    
    def _sample_to_data(self, sample):

        # get data
        agent_pos = copy.deepcopy(sample['state'][:,])
        point_cloud = copy.deepcopy(sample['point_cloud'][:,])
        action = copy.deepcopy(sample['action'])
        
        # augmentation
        if self.augmentation_pcd:
            point_cloud = point_cloud + np.random.normal(0, 0.003, point_cloud.shape) # [AugTODO] add more 
                    
        if self.augmentation_rot:
            # random rotation
            random_trans = np.identity(4)
            random_zrot = (np.random.rand() * 2 - 1) * 10 * np.pi / 180 # -10 degree to 10 degree in raduis
            
            ###########################################
            if debug:
                random_zrot = 45 * np.pi / 180 
            ###########################################

            random_rotmat = p.getMatrixFromQuaternion(p.getQuaternionFromEuler([0, 0, random_zrot]))
            random_rotmat = np.asarray(random_rotmat).reshape(3, 3)
            random_trans[:3, :3] = random_rotmat

            # agent pos
            agent_trans = np.identity(4).repeat(self.horizon, 1)
            pos_index = np.asarray([4*i+3 for i in range(self.horizon)]).astype(np.uint16)
            agent_trans[:3, pos_index] = agent_pos[:, :3].T
            rot_index = np.asarray([[4*i, 4*i+1, 4*i+2] for i in range(self.horizon)]).astype(np.uint16).reshape(-1)
            agent_trans[:3, rot_index] = rotation_transfer_6D_to_matrix_batch(agent_pos[:,3:9]) # should be 6D rotation representation
            agent_trans = random_trans @ agent_trans
            agent_pos[:, :3] = agent_trans[:3, pos_index].T
            agent_pos[:, 3:9] = rotation_transfer_matrix_to_6D_batch(agent_trans[:3, rot_index].T)

            # point cloud
            point_cloud_homo = np.ones((point_cloud.shape[0] * point_cloud.shape[1], 4))
            point_cloud_homo[:,:3] = point_cloud.reshape((-1, 3))
            point_cloud = (point_cloud_homo @ random_trans.T)[:, :3]
            point_cloud = point_cloud.reshape(self.horizon, -1, 3)

            # action
            action[:,:3] = action[:,:3] @ random_rotmat.T

            if 'act3d' in self.observation_mode:

                gripper_pcd_copy = copy.deepcopy(gripper_pcd)
                gripper_pcd_homo = np.ones((gripper_pcd.shape[0] * gripper_pcd.shape[1], 4))
                gripper_pcd_homo[:,:3] = gripper_pcd.reshape((-1, 3))
                gripper_pcd = (gripper_pcd_homo @ random_trans.T)[:, :3]
                gripper_pcd = gripper_pcd.reshape(self.horizon, -1, 3)

                if 'goal' in self.observation_mode:
                    goal_gripper_pcd_homo = np.ones((goal_gripper_pcd.shape[0] * goal_gripper_pcd.shape[1], 4))
                    goal_gripper_pcd_homo[:,:3] = goal_gripper_pcd.reshape((-1, 3))
                    goal_gripper_pcd = (goal_gripper_pcd_homo @ random_trans.T)[:, :3]
                    goal_gripper_pcd = goal_gripper_pcd.reshape(self.horizon, -1, 3)
                
                if 'displacement_gripper_to_object' in self.observation_mode:
                    goal_gripper_to_pcd = gripper_pcd_copy + displacement_gripper_to_object
                    goal_gripper_to_pcd_homo = np.ones((goal_gripper_to_pcd.shape[0] * goal_gripper_to_pcd.shape[1], 4))
                    goal_gripper_to_pcd_homo[:,:3] = goal_gripper_to_pcd.reshape((-1, 3))
                    goal_gripper_to_pcd = (goal_gripper_to_pcd_homo @ random_trans.T)[:, :3]
                    goal_gripper_to_pcd = goal_gripper_to_pcd.reshape(self.horizon, -1, 3)
                    displacement_gripper_to_object = goal_gripper_to_pcd - gripper_pcd
            

        if self.augmentation_scale:

            max_difference = 0.2
            random_scale = 1 + max_difference * (2 * np.random.rand() - 1) # [1 - max_difference, 1 + max_difference]

            point_cloud[...,:3] *= random_scale
            agent_pos[...,:3] *= random_scale
            action[...,:3] *= random_scale
            
            if 'act3d' in self.observation_mode:
                gripper_pcd[...,:3] *= random_scale
                if 'goal' in self.observation_mode:
                    goal_gripper_pcd[...,:3] *= random_scale
                if 'displacement_gripper_to_object' in self.observation_mode:
                    displacement_gripper_to_object[...,:3] *= random_scale
            
            elif 'act3d_pointnet' == self.observation_mode:
                gripper_pcd[...,:3] *= random_scale

        if self.scale_scene_by_pcd:

            max_scale = np.max(np.linalg.norm(point_cloud, axis=-1))

            point_cloud[...,:3] /= max_scale
            agent_pos[...,:3] /= max_scale
            action[...,:3] /= max_scale

            if 'act3d' in self.observation_mode:
                gripper_pcd[...,:3] /= max_scale
                if 'goal' in self.observation_mode:
                    goal_gripper_pcd[...,:3] /= max_scale
                if 'displacement_gripper_to_object' in self.observation_mode:
                    displacement_gripper_to_object[...,:3] /= max_scale
            
            elif 'act3d_pointnet' == self.observation_mode:
                gripper_pcd[...,:3]  /= max_scale
                    
        # assign to dict
        data = {
            'obs': {
                'point_cloud': point_cloud.astype(np.float32), # T, 1280, 
                'agent_pos': agent_pos.astype(np.float32), # T, D_pos
            },
            'action': action.astype(np.float32)
        }

        for key in self.keys_:
            if key not in ['state', 'action', 'point_cloud']:
                data['obs'][key] = copy.deepcopy(sample[key][:,].astype(np.float32))
                
        if self.prediction_target == 'delta_to_goal_gripper':
            data['obs']['delta_to_goal_gripper'] = data['obs']['goal_gripper_pcd'] - data['obs']['gripper_pcd']
        
        return data

    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
