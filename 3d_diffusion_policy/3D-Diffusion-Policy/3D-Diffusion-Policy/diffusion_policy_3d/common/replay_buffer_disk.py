from typing import Union, Dict, Optional
import os
import math
import numbers
import zarr
import pickle # [DebugPickle]
import numcodecs
import numpy as np
from functools import cached_property
from termcolor import cprint
from collections import defaultdict
import pickle
from tqdm import tqdm

class WelfordOnlineStatistics:
    def __init__(self):
        self.mean = None
        self.variance = None
        self.n = 0
        self.min = None
        self.max = None

        # [DebugNormalize] [Chialiang]
        self.max_norm_3d = 0.0

    def add(self, data):
        """
        data: numpy array [n, d]
        """
        if self.mean is None:
            self.mean = np.mean(data, axis=0)
            self.variance = np.var(data, axis=0)
            self.min = np.min(data, axis=0)
            self.max = np.max(data, axis=0)
            self.n = data.shape[0]
        else:
            new_n = self.n + data.shape[0]
            new_mean = (self.mean * self.n + np.sum(data, axis=0)) / new_n
            # new_variance = (self.variance * self.n + np.sum((data - self.mean) ** 2, axis=0)) / new_n
            # new_variance = (self.variance * self.n + np.sum((data - new_mean) ** 2, axis=0)) / new_n
            new_variance = (self.variance * self.n + np.sum((data - self.mean) * (data - new_mean), axis=0)) / new_n
            new_min = np.minimum(self.min, np.min(data, axis=0))
            new_max = np.maximum(self.max, np.max(data, axis=0))
            self.mean = new_mean
            self.variance = new_variance
            self.min = new_min
            self.max = new_max
            self.n = new_n

            # [DebugNormalize] [Chialiang]
            self.max_norm_3d = max(self.max_norm_3d, np.max(np.linalg.norm(data[...,:3], axis=-1)))
    
    def get_mean(self):
        return self.mean
    
    def get_variance(self):
        return self.variance
    
    def get_min(self):
        return self.min
    
    def get_max(self):
        return self.max
    
    def get_std(self):
        return np.sqrt(self.variance)
    
    # [DebugNormalize] [Chialiang]
    def get_max_norm_3d(self):
        return np.array([self.max_norm_3d])

class ReplayBuffer:
    """
    Zarr-based temporal datastructure.
    Every time we need data, we load from disk.
    """
    def __init__(self, all_path_list, episode_lengths, accumulated_episode_lengths, keys, action_welford, pcd_welford=None, agent_pos_welford=None, load_per_step=True, is_pickle=False):
        self.all_path_list = all_path_list
        self.episode_lengths = episode_lengths
        self.accumulated_episode_lengths = accumulated_episode_lengths
        self.keys_ = keys
        self.action_welford = action_welford
        self.pcd_welford = pcd_welford
        self.agent_pos_welford = agent_pos_welford
        self.load_per_step = load_per_step

        self.is_pickle = is_pickle

    @classmethod
    def copy_from_multiple_path(self, path_list, load_per_step=False, keys=None, only_reach_stage=False, is_pickle=False, target_action='action', dp3=False):
        """
        restore the path_list as well as the length of every episode
        """
        if only_reach_stage:
            cprint("loading the data before reaching the goal, no goal switching!!", 'red')
        else:
            cprint("loading the full episode data", 'red')

        if is_pickle:
            cprint("loading the data from pickle files", 'blue')
        else:
            cprint("loading the data from zarr files", 'blue')
        self.all_path_list = path_list
        self.is_pickle = is_pickle

        episode_lengths = []

        action_welford = WelfordOnlineStatistics()
        if dp3:
            pcd_welford = WelfordOnlineStatistics()
            agent_pos_welford = WelfordOnlineStatistics()

        for idx, zarr_path  in enumerate(tqdm(path_list)):

            if not load_per_step:
                if is_pickle:
                    with open(zarr_path, 'rb') as f:
                        data = pickle.load(f)
                    if keys is None:
                        keys = data.keys()
                        self.keys_ = list(keys)
                    else:
                        self.keys_ = keys
                    episode_lengths.append(len(data[keys[0]][:]))
                    if target_action == 'action':
                        action = data['action'][:]
                    elif target_action == 'delta_to_goal_gripper':
                        action = (data['goal_gripper_pcd'][:] - data['gripper_pcd'][:]).flatten()
                    action_welford.add(action)
                else:
                    group = zarr.open(zarr_path, 'r')
                    src_store = group.store
                
                    # numpy backend
                    src_root = zarr.group(src_store)

                    if keys is None:
                        keys = src_root['data'].keys()
                        self.keys_ = list(keys)
                    else:
                        self.keys_ = keys

                    # print("episode {} lenght {}".format(idx, len(src_root['data'][keys[0]][:])))
                    episode_lengths.append(len(src_root['data'][keys[0]][:]))

                    action = None
                    if target_action == 'action':
                        action = src_root['data']['action'][:]
                    elif target_action == 'delta_to_goal_gripper':
                        action = (data['goal_gripper_pcd'][:] - data['gripper_pcd'][:]).flatten()
                    elif target_action == 'goal_gripper_pcd':
                        action = data['goal_gripper_pcd'][:]
                    action_welford.add(action)
            else:
                
                # episode_lengths.append(len(all_substeps))

                first_goal = None

                if is_pickle:
                    all_substeps = os.listdir(zarr_path)
                    all_substeps = sorted(all_substeps, key=lambda x: int(x.split('.')[0])) # ex: 0.pkl -> 0
                    for i, substep in enumerate(all_substeps):
                        substep_path = os.path.join(zarr_path, substep)
                        try:
                            with open(substep_path, 'rb') as f:
                                data = pickle.load(f)
                        except:
                            print(substep_path)
                        if keys is None:
                            keys = data.keys()
                            self.keys_ = list(keys)
                        else:
                            self.keys_ = keys
                        
                        if target_action == 'action':
                            action = data['action'][:]  
                        elif target_action == 'delta_to_goal_gripper':
                            action = (data['goal_gripper_pcd'][:] - data['gripper_pcd'][:]).reshape(1, -1)
                        elif target_action == 'goal_gripper_pcd':
                            action = data['goal_gripper_pcd'][:]

                        current_goal = data['goal_gripper_pcd'][:]
                        if first_goal is None:
                            first_goal = current_goal
                        elif not np.allclose(first_goal, current_goal) and only_reach_stage:
                            episode_lengths.append(i)
                            break

                        action_welford.add(action)
                        
                        if dp3:
                            pcd = data['point_cloud']
                            gripper_pcd = data['gripper_pcd']
                            
                            pcd_welford.add(pcd.squeeze(0))
                            pcd_welford.add(gripper_pcd.squeeze(0))
                            
                            agent_pos = data['state'][:]

                            if np.isnan(pcd).any() or np.isnan(gripper_pcd).any() or np.isnan(agent_pos).any():
                                print(substep_path)
                            
                            agent_pos_welford.add(agent_pos)
                            
                else:         
                    all_substeps = os.listdir(zarr_path)
                    all_substeps = sorted(all_substeps, key=lambda x: int(x))
                    for i, substep in enumerate(all_substeps):
                        substep_path = os.path.join(zarr_path, substep)
                        group = zarr.open(substep_path, 'r')
                        src_store = group.store
                        src_root = zarr.group(src_store)

                        if keys is None:
                            keys = src_root['data'].keys()
                            self.keys_ = list(keys)
                        else:
                            self.keys_ = keys

                        if target_action == 'action':
                            action = src_root['data']['action'][:]
                        elif target_action == 'delta_to_goal_gripper':
                            action = (data['goal_gripper_pcd'][:] - data['gripper_pcd'][:]).flatten()
                        

                        current_goal = src_root['data']['goal_gripper_pcd'][:]
                        if first_goal is None:
                            first_goal = current_goal
                        elif not np.allclose(first_goal, current_goal) and only_reach_stage:
                            episode_lengths.append(i)
                            break

                        action_welford.add(action)

                if not only_reach_stage:
                    episode_lengths.append(len(all_substeps))
                    
        # exit()
        
        self.episode_lengths = np.array(episode_lengths)
        self.accumulated_episode_lengths = np.cumsum(self.episode_lengths)
        
        # we might need the min, max, mean, std of every data
        # TODO: add more statistics
        self.action_welford = action_welford
        if dp3:
            self.pcd_welford = pcd_welford
            self.agent_pos_welford = agent_pos_welford
        else:
            self.pcd_welford = self.agent_pos_welford = None

        return ReplayBuffer(self.all_path_list, self.episode_lengths, self.accumulated_episode_lengths, self.keys_, self.action_welford, self.pcd_welford, self.agent_pos_welford, load_per_step=load_per_step, is_pickle=is_pickle)

    def get_data_disk(self, start_idx, end_idx):
        """
        get data from disk
        """
        if not self.load_per_step:
            # find the corresponding zarr file
            start_episode_idx = np.searchsorted(self.accumulated_episode_lengths, start_idx, side='right')
            end_episode_idx = np.searchsorted(self.accumulated_episode_lengths, end_idx, side='right')
            if start_episode_idx == end_episode_idx:
                # only one zarr file
                start_idx = start_idx - self.accumulated_episode_lengths[start_episode_idx]
                end_idx = end_idx - self.accumulated_episode_lengths[start_episode_idx]
                ret_data = self.get_data_single_zarr(self.all_path_list[start_episode_idx], start_idx, end_idx)
            else:
                # two zarr files
                start_idx = start_idx - self.accumulated_episode_lengths[start_episode_idx]
                ret_data = self.get_data_single_zarr(self.all_path_list[start_episode_idx], start_idx, None)
                # end_idx = end_idx - self.accumulated_episode_lengths[end_episode_idx]
                # ret_data_end = self.get_data_single_zarr(self.all_path_list[end_episode_idx], None, end_idx)
                # for key in ret_data.keys():
                #     ret_data[key] = np.concatenate([ret_data[key], ret_data_end[key]], axis=0)
        else:
            # find the corresponding zarr file
            start_episode_idx = np.searchsorted(self.accumulated_episode_lengths, start_idx, side='right')
            end_episode_idx = np.searchsorted(self.accumulated_episode_lengths, end_idx, side='right')
            if start_episode_idx == end_episode_idx:
                # only one zarr file
                start_idx = start_idx - self.accumulated_episode_lengths[start_episode_idx]
                end_idx = end_idx - self.accumulated_episode_lengths[start_episode_idx]
                ret_data = self.get_data_single_zarr_per_step(self.all_path_list[start_episode_idx], start_idx, end_idx, start_episode_idx)
            else:
                # two zarr files
                start_idx = start_idx - self.accumulated_episode_lengths[start_episode_idx]
                ret_data = self.get_data_single_zarr_per_step(self.all_path_list[start_episode_idx], start_idx, None, start_episode_idx)
        return ret_data
    
    def get_data_single_zarr_per_step(self, zarr_path, start_idx, end_idx, zarr_idx=None):
        ret_data = defaultdict(list)
        if zarr_idx is None:
            all_steps = len(os.listdir(zarr_path))
        else:
            all_steps = self.episode_lengths[zarr_idx]

        start_idx = start_idx + all_steps if start_idx < 0 else start_idx        
        if end_idx is None:
            end_idx = all_steps
        else:
            end_idx = end_idx + all_steps if end_idx < 0 else end_idx
        
        for step_idx in range(start_idx, end_idx):
            if self.is_pickle:
                step_path = os.path.join(zarr_path, f'{step_idx}.pkl')
                with open(step_path, 'rb') as f:
                    data = pickle.load(f)
                for key in self.keys_:
                    ret_data[key].append(data[key][:])
            else:
                step_path = os.path.join(zarr_path, str(step_idx))
                group = zarr.open(step_path, 'r')
                src_store = group.store

                # numpy backend
                src_root = zarr.group(src_store)
                for key in self.keys_:
                    ret_data[key].append(src_root['data'][key][:])
        
        for key in self.keys_:
            ret_data[key] = np.concatenate(ret_data[key], axis=0)
                
        return ret_data
    
    def get_data_single_zarr(self, zarr_path, start_idx, end_idx):
        """
        get data from a single zarr file
        """
        if self.is_pickle:
            with open(zarr_path, 'rb') as f:
                data = pickle.load(f)
            ret_data = dict()
            for key in self.keys_:
                if end_idx is None:
                    ret_data[key] = data[key][:][start_idx:]
                elif start_idx is None:
                    ret_data[key] = data[key][:][:end_idx]
                else:
                    ret_data[key] = data[key][:][start_idx:end_idx]
        else:
            group = zarr.open(zarr_path, 'r')
            src_store = group.store

            # numpy backend
            src_root = zarr.group(src_store)
            ret_data = dict()
            for key in self.keys_:
                if end_idx is None:
                    ret_data[key] = src_root['data'][key][:][start_idx:]
                elif start_idx is None:
                    ret_data[key] = src_root['data'][key][:][:end_idx]
                else:
                    ret_data[key] = src_root['data'][key][:][start_idx:end_idx]
        return ret_data
    
    ## SOME APIs
    @property
    def episode_lengths(self):
        return self.episode_lengths
    
    @property
    def episode_ends(self):
        return self.accumulated_episode_lengths
    
    def keys(self):
        return self.keys_
    
    def __contains__(self, key):
        return key in self.keys_
    
    @property
    def n_steps(self):
        if len(self.episode_ends) == 0:
            return 0
        return self.episode_ends[-1]
    
    @property
    def n_episodes(self):
        return len(self.episode_ends)
    
