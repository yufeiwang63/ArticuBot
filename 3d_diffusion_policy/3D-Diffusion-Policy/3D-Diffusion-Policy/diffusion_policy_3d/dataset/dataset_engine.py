from collections import defaultdict, Counter
import itertools
import math
import einops
from pathlib import Path
import random
from time import time

import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# from .utils import loader, Resize, Rotate, TrajectoryInterpolator
from .utils import Resize, Rotate, TrajectoryInterpolator

# chialiang
import zarr, glob, os, copy
import numpy as np

class ChainedDiffusorDataset(Dataset):
    """RLBench dataset."""

    def __init__(
        self,
        # required
        # root,
        zarr_path=['/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45448-chained-diffuser'],
        # dataset specification
        cache_size=0,
        num_iters=None,
        # for augmentations
        training=True,
        gripper_loc_bounds=np.array([[-1, -1, -1], [1, 1, 1]]),
        image_rescale=(1.0, 1.0),
        point_cloud_rotate_yaw_range=0.0,
        # for trajectories
        interpolation_length=50,
        action_dim=7,  # elements of action trajectory to regress
        # for chialiang
        val_ratio=0.1,
        train_ratio=0.9,
        **kwargs
    ):
        self._cache = {}
        self._cache_size = cache_size
        self._num_iters = num_iters
        self._training = training
        self._action_dim = action_dim

        assert val_ratio < 1 and train_ratio < 1 and (val_ratio + train_ratio) <= 1 + 1e-6

        # for chialiang
        self.load_per_step = kwargs.get('load_per_step', False)

        # For trajectory optimization, initialize interpolation tools
        self._interpolate_traj = TrajectoryInterpolator(
            use=True,
            interpolation_length=interpolation_length
        )

        # If training, initialize augmentation classes
        if self._training:
            self._resize = Resize(scales=image_rescale)
            self._rotate = Rotate(
                gripper_loc_bounds=gripper_loc_bounds,
                yaw_range=point_cloud_rotate_yaw_range
            )
            assert point_cloud_rotate_yaw_range == 0.

        # File-names of episodes per-task and variation
        self._data_dirs = []
        # episodes_by_task = defaultdict(list)
        # for root, (task, var) in itertools.product(self._root, taskvar):
        all_zarr_paths = copy.deepcopy(zarr_path)

        self.train_paths = []
        self.val_paths = []
        for zarr_path in tqdm(all_zarr_paths):

            all_subfolder = os.listdir(zarr_path) # a list of trajectories names
            # import pdb; pdb.set_trace()
            for string in ["action_dist", "demo_rgbs", "all_demo_path.txt", "meta_info.json", 'example_pointcloud', '.zgroup']:
                if string in all_subfolder:
                    all_subfolder.remove(string)
            all_subfolder = sorted(all_subfolder)
            n_episodes = len(all_subfolder)
            num_load_episodes = kwargs.get('num_load_episodes', n_episodes)
            num_load_episodes = min(num_load_episodes, n_episodes)
            all_subfolder = all_subfolder[:num_load_episodes]
            zarr_roots = [os.path.join(zarr_path, subfolder) for subfolder in all_subfolder]

            for i, zarr_root in enumerate(zarr_roots):
                zarr_paths_in_one_root = glob.glob(f'{zarr_root}/*')

                if i < train_ratio * len(zarr_roots):
                    for zarr_path in zarr_paths_in_one_root:
                        if os.path.isdir(zarr_path):
                            self.train_paths.append(zarr_path)
                elif i >= (1 - val_ratio) * len(zarr_roots):
                    for zarr_path in zarr_paths_in_one_root:
                        if os.path.isdir(zarr_path):
                            self.val_paths.append(zarr_path)
                else:
                    pass # not included for training and validation
            
        self._num_episodes = len(self.train_paths)
            
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.train_paths = self.val_paths
        val_set._num_episodes = len(val_set)
        return val_set

    def __getitem__(self, episode_id):

        zarr_path = self.train_paths[episode_id]

        # Load episode

        group = zarr.open(zarr_path, mode='r')

        # one trajectory
        data_group = group['data']
        # print(list(data_group.keys()))

        # pcd = torch.from_numpy(np.asarray(data_group['point_cloud'])).squeeze()
        current_gripper = torch.from_numpy(np.asarray(data_group['init_pose'])).squeeze()
        target_gripper = torch.from_numpy(np.asarray(data_group['target_pose'])).squeeze()
        feature_map = torch.from_numpy(np.asarray(data_group['feature_map'])).squeeze()
        pcd = feature_map[...,2:]
        visible_rgb = einops.rearrange(feature_map, 'n_am h w c -> n_am c h w')
        visible_pcd = einops.rearrange(pcd, 'n_am h w c -> n_am c h w')

        pcd_mask = torch.from_numpy(np.asarray(data_group['pcd_mask'])).squeeze()
        trajectory = torch.from_numpy(np.asarray(data_group['trajectory'])).squeeze()

        trajectory = self._interpolate_traj(trajectory) 
        trajectory_mask = torch.zeros(trajectory.shape[0], dtype=torch.uint8)
        trajectory_mask[0] = 1 # first
        trajectory_mask[:-1] = 1 # last

        # [CDTODO] augmentation

        # print(f'pcd: {pcd.shape}')
        # print(f'current_gripper: {current_gripper.shape}')
        # print(f'target_gripper: {target_gripper.shape}')
        # print(f'feature_map: {feature_map.shape}')
        # print(f'pcd_mask: {pcd_mask.shape}')
        # print(f'trajectory: {trajectory.shape}')
        # print(f'trajectory_mask: {trajectory_mask.shape}')
        # print()
        # exit(0)

        data = {
            'obs': {
                "trajectory": trajectory[:,:self._action_dim].to(torch.float32), 
                "trajectory_mask": trajectory_mask.bool(), 
                "visible_rgb": visible_rgb,  
                "visible_pcd": visible_pcd,  
                # "visible_pcd": pcd,  
                "pcd_mask": pcd_mask,  # [TODO] actually not been used 
                "curr_gripper": current_gripper[:self._action_dim],
                "goal_gripper": target_gripper[:self._action_dim], 
            },
            'action': trajectory.to(torch.float32)
        }

        return data

        # episode_id %= self._num_episodes
        # task, variation, file = self._episodes[episode_id]

        # # Load episode
        # episode = self.read_from_cache(file)
        # if episode is None:
        #     return None

        # # Dynamic chunking so as not to overload GPU memory
        # chunk = random.randint(
        #     0, math.ceil(len(episode[0]) / self._max_episode_length) - 1
        # )

        # # Get frame ids for this chunk
        # frame_ids = episode[0][
        #     chunk * self._max_episode_length:
        #     (chunk + 1) * self._max_episode_length
        # ]

        # # Get the image tensors for the frame ids we got
        # states = torch.stack([
        #     episode[1][i] if isinstance(episode[1][i], torch.Tensor)
        #     else torch.from_numpy(episode[1][i])
        #     for i in frame_ids
        # ])

        # # Camera ids
        # if episode[3]:
        #     cameras = list(episode[3][0].keys())
        #     assert all(c in cameras for c in self._cameras)
        #     index = torch.tensor([cameras.index(c) for c in self._cameras])
        #     # Re-map states based on camera ids
        #     states = states[:, index]

        # # Split RGB and XYZ
        # rgbs = states[:, :, 0]
        # pcds = states[:, :, 1]
        # rgbs = self._unnormalize_rgb(rgbs)

        # # Get action tensors for respective frame ids
        # action = torch.cat([episode[2][i] for i in frame_ids])

        # # # Sample one instruction feature
        # # if self._instructions:
        # #     instr = random.choice(self._instructions[task][variation])
        # #     instr = instr[None].repeat(len(rgbs), 1, 1)
        # # else:
        # #     instr = torch.zeros((rgbs.shape[0], 53, 512))

        # # Get gripper tensors for respective frame ids
        # gripper = torch.cat([episode[4][i] for i in frame_ids])

        # # gripper history
        # gripper_history = torch.stack([
        #     torch.cat([episode[4][max(0, i-2)] for i in frame_ids]),
        #     torch.cat([episode[4][max(0, i-1)] for i in frame_ids]),
        #     gripper
        # ], dim=1)

        # # Low-level trajectory
        # traj, traj_lens = None, 0
        # if self._return_low_lvl_trajectory:
        #     traj_items = [
        #         self._interpolate_traj(episode[5][i]) for i in frame_ids
        #     ]
        #     max_l = max(len(item) for item in traj_items)
        #     traj = torch.zeros(len(traj_items), max_l, 8)
        #     traj_lens = torch.as_tensor(
        #         [len(item) for item in traj_items]
        #     )
        #     for i, item in enumerate(traj_items):
        #         traj[i, :len(item)] = item
        #     traj_mask = torch.zeros(traj.shape[:-1])
        #     for i, len_ in enumerate(traj_lens.long()):
        #         traj_mask[i, len_:] = 1

        # # Augmentations
        # if self._training:
        #     pcds, gripper, action, traj = self._rotate(
        #         pcds, gripper, action, None, traj
        #     )
        #     if traj is not None:
        #         for t, tlen in enumerate(traj_lens):
        #             traj[t, tlen:] = 0
        #     modals = self._resize(rgbs=rgbs, pcds=pcds)
        #     rgbs = modals["rgbs"]
        #     pcds = modals["pcds"]

        # ret_dict = {
        #     "task": [task for _ in frame_ids],
        #     "rgbs": rgbs,  # e.g. tensor (n_frames, n_cam, 3+1, H, W)
        #     "pcds": pcds,  # e.g. tensor (n_frames, n_cam, 3, H, W)
        #     "action": action[..., :self._action_dim],  # e.g. tensor (n_frames, 8), target pose
        #     # "instr": instr,  # a (n_frames, 53, 512) tensor
        #     "curr_gripper": gripper[..., :self._action_dim],
        #     "curr_gripper_history": gripper_history[..., :self._action_dim]
        # }
        # if self._return_low_lvl_trajectory:
        #     ret_dict.update({
        #         "trajectory": traj[..., :self._action_dim],  # e.g. tensor (n_frames, T, 8)
        #         "trajectory_mask": traj_mask.bool()  # e.g. tensor (n_frames, T)
        #     })
        # return ret_dict

    def __len__(self):
        return len(self.train_paths)
    

if __name__=='__main__':
    dataset = ChainedDiffusorDataset()# Testing with DataLoader
    dataloader = DataLoader(
                    dataset,
                    batch_size=30,
                    num_workers=15,
                    shuffle=True,
                    pin_memory=True,
                    persistent_workers=False,
                )
    
    for batch in dataloader:

        for k in batch['obs']:
            print(k, batch['obs'][k].shape)
        print('action', batch['action'].shape)