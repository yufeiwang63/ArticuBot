import wandb
import numpy as np
import torch
import tqdm 
import pybullet as p
import numpy as np
from copy import deepcopy
import gym
from gym import spaces
import open3d as o3d
import matplotlib.pyplot as plt
import time
from termcolor import cprint
import fpsample
import os
import json
import pickle
import cv2
import scipy
from scipy.interpolate import RectBivariateSpline
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
from manipulation.utils import get_pc, get_pc_in_camera_frame, rotation_transfer_6D_to_matrix, rotation_transfer_matrix_to_6D, add_sphere, get_pixel_location, get_matrix_from_pos_rot
from manipulation.gpt_reward_api import get_handle_pos, get_link_pc
from manipulation.gpt_primitive_api import get_pc_num_within_gripper

class RobogenPointCloudWrapper:
    def __init__(self, 
                 env, 
                 object_name, 
                 rpy_mean_list=[[0, 0, -45], [0, 0, -135]], 
                 seed=None, 
                 num_points=4500,
                 horizon=400,
                 observation_mode=None,
                 camera_height=480,
                 camera_width=640,
                 camera_elevation=30,
                 only_object=True,
                 noise_real_world_pcd=False,
                 real_world_camera=False,
            ):
        np.random.seed(time.time_ns() % 2**32)
        if seed is not None:
            np.random.seed(seed)
            
        self.noise_real_world_pcd = noise_real_world_pcd
        self.real_world_camera = real_world_camera

        self._env = env
        self._object_name = object_name
        self.horizon = horizon
        
        self.num_points = num_points
        self.observation_mode = observation_mode
        self.camera_elevation = camera_elevation
        self.camera_width = camera_width
        self.camera_height = camera_height

        
        ### setup observation space
        self.action_low = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        self.action_high = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32)
        self.observation_space = spaces.Dict({
            'point_cloud': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 1280, 3), dtype=np.float32),
            'agent_pos': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 10), dtype=np.float32), # pos(3) + orient(6) + joint_angle(1): we use 6D representation for orientation
            'gripper_pcd': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 4, 3), dtype=np.float32), # pos(3) + orient(6) + joint_angle(1): we use 6D representation for orientation
        })

        if 'goal' in observation_mode:
            self.observation_space['goal_gripper_pcd'] = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 4, 3), dtype=np.float32) # pos(3) + orient(6) + joint_angle(1): we use 6D representation for orientation
        if 'displacement_gripper_to_object' in observation_mode:
            self.observation_space['displacement_gripper_to_object'] = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 4, 3), dtype=np.float32) # pos(3) + orient(6) + joint_angle(1): we use 6D representation for orientation

        if 'dp3' in observation_mode:
            self.observation_space = spaces.Dict({
                'point_cloud': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 1280, 3), dtype=np.float32),
                'agent_pos': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 10), dtype=np.float32), # pos(3) + orient(6) + joint_angle(1): we use 6D representation for orientation
            })
        elif 'act3d' in observation_mode:
            self.observation_space = spaces.Dict({
                'point_cloud': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 1280, 3), dtype=np.float32),
                'agent_pos': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 10), dtype=np.float32), # pos(3) + orient(6) + joint_angle(1): we use 6D representation for orientation
                'gripper_pcd': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 4, 3), dtype=np.float32), # pos(3) + orient(6) + joint_angle(1): we use 6D representation for orientation
            })
            if 'goal' in observation_mode:
                self.observation_space['goal_gripper_pcd'] = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 4, 3), dtype=np.float32) # pos(3) + orient(6) + joint_angle(1): we use 6D representation for orientation
            if 'displacement_gripper_to_object' in observation_mode:
                self.observation_space['displacement_gripper_to_object'] = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 4, 3), dtype=np.float32) # pos(3) + orient(6) + joint_angle(1): we use 6D representation for orientation
        else :
            raise NotImplementedError

        ### setup cameras
        for name in self._env.urdf_ids: # randomly center at an object
            if name in ['robot', 'plane', 'init_table']: continue
            obj_id = self._env.urdf_ids[name]
            min_aabb, max_aabb = self._env.get_aabb(obj_id)
            center = (min_aabb + max_aabb) / 2
            self.mean_camera_target = center 
            self.mean_distance = np.linalg.norm(max_aabb - min_aabb) * 1.15
            break
        
        self.rpy_mean_list = [[-10, 0, -45], [-10, 0, -135]]
        self.mean_distance = np.linalg.norm(max_aabb - min_aabb) * 0.9
        self.camera_height = 256
        self.camera_width = 256

        self.depth_near = 0.01
        self.depth_far = 100
        self.view_matrices = []
        self.project_matrices = []
        for rpy_mean in self.rpy_mean_list:
            rpy = np.array(rpy_mean)
            camera_center = self.mean_camera_target
            distance = self.mean_distance
            view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=camera_center, distance=distance, yaw=rpy[2], pitch=rpy[0], roll=rpy[1], upAxisIndex=2, physicsClientId=env.id)
            project_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1 ,nearVal=self.depth_near, farVal=self.depth_far, physicsClientId=env.id)
            self.view_matrices.append(view_matrix)
            self.project_matrices.append(project_matrix)
        if self.real_world_camera:
            self.randomize_real_world_camera()

        self.time_step = 0
        
        ### load the ground-truth goal from the evaluation trajectory
        if ("act3d_goal" in self.observation_mode):
            config_path = self._env.config_path
            task_name = self._env.task_name
            parent_path = os.path.dirname(config_path)
            state_path = os.path.join(parent_path, "{}_primitive".format(task_name), "states") 
                
            stage_lengths_json_file = os.path.join(parent_path, "{}_primitive".format(task_name), 'stage_lengths.json')
            with open(stage_lengths_json_file, 'r') as f:
                stage_lengths = json.load(f)
            open_begin_t_idx = stage_lengths['reach_handle'] + stage_lengths['reach_to_contact'] + stage_lengths['close_gripper']
            all_time_steps = stage_lengths['reach_handle'] + stage_lengths['reach_to_contact'] + stage_lengths['close_gripper'] + stage_lengths['open_door']

            goal_1_state = os.path.join(state_path, "state_{}.pkl".format(open_begin_t_idx))
            goal_2_state = os.path.join(state_path, "state_{}.pkl".format(all_time_steps - 1))
            
            # NOTE: load the goal state, reset the robot to there, record the eef pose as the goal.
            with open(goal_1_state, 'rb') as f:
                goal_1_state = pickle.load(f)
            with open(goal_2_state, 'rb') as f:
                goal_2_state = pickle.load(f)
            
            self._env.reset(reset_state=goal_1_state)
            grasping_eef_pc = self.get_gripper_pc()

            self._env.reset(reset_state=goal_2_state)
            final_eef_pc = self.get_gripper_pc()
            
            self.grasping_goal = grasping_eef_pc
            self.final_goal = final_eef_pc
            self.grasped_handle = False
            self.goal_gripper_pcd = None

        self.only_object = only_object
        
    def randomize_real_world_camera(self):
        self.camera_width = 640
        self.camera_height = 576
        
        camera_ids = [0, 3]
        view_matrices = []
        project_matrices = []
        camera_calibration_folder = os.path.join(os.environ["PROJECT_DIR"], 'data/real_world')
        self.camera_eyes = []
        for camera_id in camera_ids:
            camera_parameter_file = os.path.join(camera_calibration_folder, "cam{}_calibration.npz".format(camera_id))
            data = np.load(camera_parameter_file)
            camera_extrinsic = data['T'] # 4x4


            camera_eye = camera_extrinsic[:3, 3]
            camera_eye[2] = 1.0
            camera_target = [0.7, 0, 0.4]
            camera_eye = camera_eye + np.random.normal(0, 0.1, 3)
            camera_target = camera_target + np.random.normal(0, 0.1, 3)
            self.camera_eyes.append(camera_eye)

            view_matrix = p.computeViewMatrix(camera_eye, camera_target, [0, 0, 1])
            project_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=640/576 ,nearVal=self.depth_near, 
                                                            farVal=self.depth_far, physicsClientId=self._env.id)
            view_matrices.append(view_matrix)
            project_matrices.append(project_matrix)

        self.view_matrices = view_matrices
        self.project_matrices = project_matrices
        
        self._env.view_matrix = self.view_matrices[0]
        self._env.projection_matrix = self.project_matrices[0]

    def reset_random_cameras(self):
        # do a while loop to sample a new camera view
        try_times = 0
        # get handle point cloud
        link_pc = get_link_pc(self._env, self._object_name, 'link_0')
        all_handle_pos, handle_joint_id = get_handle_pos(self._env, self._object_name, return_median=False)
        handle_pc = np.concatenate(all_handle_pos, axis=0)
        while try_times < 5000:
            view_matrices = []
            project_matrices = []
            try_times += 1
            distance = np.random.uniform(0.8, 1.2) * self.mean_distance + np.random.normal(0, 0.05, 1)
            camera_center = self.mean_camera_target + np.random.normal(0, 0.05, 3)
            for _ in range(2):
                rpy = np.zeros(3)
                rpy[0] = np.random.uniform(-20, 20)
                rpy[1] = np.random.uniform(-40, 0)
                if np.random.uniform() > 0.5:
                    rpy[2] = np.random.uniform(-110, -160)
                else:
                    rpy[2] = np.random.uniform(-20, -70)
                view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=camera_center, distance=distance, yaw=rpy[2], pitch=rpy[0], roll=rpy[1], upAxisIndex=2, physicsClientId=self._env.id)
                project_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=640/480 ,nearVal=self.depth_near, farVal=self.depth_far, physicsClientId=self._env.id)
                view_matrices.append(view_matrix)
                project_matrices.append(project_matrix)
            self.view_matrices = view_matrices
            self.project_matrices = project_matrices
            if self.check_handle_observed_in_pc(handle_pc=handle_pc) > 5:
                self._env.projection_matrix = project_matrices[0]
                self._env.view_matrix = view_matrices[0]
                break
        if try_times >= 5000:
            raise ValueError("Cannot find a camera view that has handle points in the point cloud")

    def reset(self, **kwargs):
        if "act3d_goal" in self.observation_mode:
            self.grasped_handle = False
        self._env.reset(**kwargs)
        self._env._get_info()
        self.time_step = 0
        if "goal" in self.observation_mode:
            self.grasped_handle = False
        return self._get_observation(only_object=self.only_object)
    
    def check_handle_observed_in_pc(self, handle_pc=None):
        # given the current camera view, check if the handle is observed in the point cloud
        # return the number of points that are close to the handle
        if handle_pc is None:
            # get handle point cloud
            link_pc = get_link_pc(self._env, self._object_name, 'link_0')
            all_handle_pos, handle_joint_id = get_handle_pos(self._env, self._object_name, return_median=False)
            handle_pc = np.concatenate(all_handle_pos, axis=0)
        pcs = []
        rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices = self.take_images_around_object(self._env, self._object_name.lower(), camera_elevation=self.camera_elevation,
                                            return_camera_matrices=True, camera_height=self.camera_height, camera_width=self.camera_width,)
        for rgb, depth, segmask, view_matrix, project_matrix in zip(rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices):
            pc = get_pc(proj_matrix=project_matrix, view_matrix=view_matrix, depth=depth, width=self.camera_width, height=self.camera_height, mask_infinite=False)
            
            segmask_obj_id = segmask & ((1 << 24) - 1)
            object_mask = np.zeros_like(depth).astype(np.float32)
            object_mask[segmask_obj_id == self._env.urdf_ids[self._object_name]] = 1
            object_mask_ = np.flatnonzero(object_mask.flatten())
            pcs.append(pc[object_mask_])
            
        pcs = np.concatenate(pcs, axis=0)
        pcd_distance = cdist(pcs, handle_pc)  # Shape: (M, N)

        # Find the minimum distance for each point in pcs
        min_distance = np.min(pcd_distance, axis=1)  # Shape: (M,)

        # Filter the distances that are less than 0.02
        min_distance = min_distance[min_distance < 0.02]
        
        return min_distance.shape[0]
    
    def step(self, action, render=True):

        pos, orient = self._env.robot.get_pos_orient(self._env.robot.right_end_effector)
        
        ### new_pos = current_pos + action[:3]
        pos = pos + np.array(action[:3])
        
        ### new_orient = current_orient @ delta_orient, which is represented using 6D representation as action[3:9]
        current_rotate_matrix = np.array(p.getMatrixFromQuaternion(orient)).reshape(3, 3)            
        delta_orient = action[3:9]
        delta_rotate_matrix = rotation_transfer_6D_to_matrix(delta_orient)
        after_rotate_matrix = current_rotate_matrix @ delta_rotate_matrix
        orient = R.from_matrix(after_rotate_matrix).as_quat()
        euler = p.getEulerFromQuaternion(orient)

        ### new finger joint = current_finger_joint + action[9]
        cur_joint_angle = p.getJointState(self._env.robot.body, self._env.robot.right_gripper_indices[0], physicsClientId=self._env.id)
        target_joint_angle = action[9] + cur_joint_angle[0]
        
        action = pos.tolist() + list(euler) + [target_joint_angle]
        self._env.take_direct_action(action)
        
        reward, success = self._env.compute_reward()

        info = self._env._get_info()            
        done = self._env.time_step >= self.horizon
        obs = self._get_observation(render=render, only_object=self.only_object)
        self.time_step += 1
        return obs, reward, done, info
    
    def get_gripper_pc(self):
        # get the point cloud of the gripper
        right_finger_pos, _ = self._env.robot.get_pos_orient(self._env.robot.right_gripper_indices[0])
        left_finger_pos, _ = self._env.robot.get_pos_orient(self._env.robot.right_gripper_indices[1])
        right_hand_pos, _ = self._env.robot.get_pos_orient(self._env.robot.right_hand)
        eef_pos, _ = self._env.robot.get_pos_orient(self._env.robot.right_end_effector)
        gripper_pc = np.array([right_hand_pos, right_finger_pos, left_finger_pos, eef_pos]).reshape(-1, 3)
        return gripper_pc.astype(np.float32)
    
    def _get_act3d_observation(self, rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices, using_torch=False, only_object=True):
        obs_dict_input = {}
        
        ### get robot state: eef pos, orient (6D), and finger joint angle
        pos, orient = self._env.robot.get_pos_orient(self._env.robot.right_end_effector)
        rotate_matrix = p.getMatrixFromQuaternion(orient)
        orient = rotation_transfer_matrix_to_6D(rotate_matrix)
        cur_joint_angle = p.getJointState(self._env.robot.body, self._env.robot.right_gripper_indices[0], physicsClientId=self._env.id)
        pos_ori = pos.tolist() + orient.tolist() + [cur_joint_angle[0]]
        
        pcs = []
        gripper_pcd = []
        pcd_mask_indices = []
        for rgb, depth, segmask, view_matrix, project_matrix in zip(rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices):
            
            ### get the object pcd
            pc = get_pc(proj_matrix=project_matrix, view_matrix=view_matrix, depth=depth, width=self.camera_width, height=self.camera_height, mask_infinite=False)
            segmask_obj_id = segmask & ((1 << 24) - 1)
            robot_mask = np.zeros_like(depth).astype(np.float32)
            robot_mask[segmask_obj_id == self._env.urdf_ids['robot']] = 1
            object_mask = np.zeros_like(depth).astype(np.float32)
            object_mask[segmask_obj_id == self._env.urdf_ids[self._object_name]] = 1
            object_mask_ = np.flatnonzero(object_mask.flatten())
            pcs.append(pc[object_mask_])
                
            ### get the gripper 4 points
            gripper_pc = self.get_gripper_pc()
            gripper_pcd.append(gripper_pc)
    
            ### read the ground-truth goal from the evaluation trajectory
            if 'goal' in self.observation_mode:
                # add goal as part of the observation. During policy eval with a learned high-level policy,
                # this will be overwritten with the predicted goal from the high-level policy during eval         
                if self._env.grasped_handle:
                    goal_gripper_pcd = self.final_goal
                else:
                    goal_gripper_pcd = self.grasping_goal
                self.goal_gripper_pcd = goal_gripper_pcd
            
        point_cloud = np.concatenate(pcs, axis=0)
        
        ### perform whatever we do in the real world to remove the noise in the point cloud
        if self.noise_real_world_pcd:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
            
            real_world_radius = 0.02
            real_world_nb_neighbors = 20
            real_world_std_ratio = np.random.uniform(0.4, 0.6)
            real_world_voxel_size = 0.002
            ratio = np.random.uniform(0.3, 0.95)
            real_world_nb_points = int(((real_world_radius / real_world_voxel_size) ** 2) * ratio)

            beg = time.time()
            pcd = pcd.voxel_down_sample(real_world_voxel_size)
            
            beg = time.time()
            pcd2, indices = pcd.remove_radius_outlier(nb_points=real_world_nb_points, radius=real_world_radius)
            
            beg = time.time()
            pcd3, indices = pcd2.remove_statistical_outlier(nb_neighbors=real_world_nb_neighbors, std_ratio=real_world_std_ratio)
                    
            point_cloud = np.array(pcd3.points)
            distance_to_camera_eye_1 = np.linalg.norm(self.camera_eyes[0] - point_cloud, axis=1)
            distance_to_camera_eye_2 = np.linalg.norm(self.camera_eyes[1] - point_cloud, axis=1)
            point_cloud = point_cloud[distance_to_camera_eye_1 > 0.1]
            point_cloud = point_cloud[distance_to_camera_eye_2 > 0.1]
        
        num_points = self.num_points

        ### perform fps on the point clouds
        if point_cloud.shape[0] < num_points:
            to_add_points_num = num_points - point_cloud.shape[0]
            random_sampled_points = np.random.choice(point_cloud.shape[0], to_add_points_num, replace=True)
            point_cloud = np.concatenate([point_cloud, point_cloud[random_sampled_points]], axis=0)
        
        try:
            ### I am not exactly sure why, it seems that sometimes the installed fpsample does not have this function ... 
            h = min(9, np.log2(num_points))
            kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(point_cloud[:, :3], num_points, h=h)
        except:
            kdline_fps_samples_idx = fpsample.fps_npdu_kdtree_sampling(point_cloud[:, :3], num_points)
        kdline_fps_samples_idx = np.array(sorted(kdline_fps_samples_idx))
        point_cloud = point_cloud[kdline_fps_samples_idx]
           
        point_cloud = point_cloud.tolist()
        
        ### build the observation dictionary
        obs_dict_input['point_cloud'] = np.array(point_cloud).astype(np.float32)
        obs_dict_input['agent_pos'] = np.array(pos_ori).astype(np.float32)
        
        obs_dict_input['gripper_pcd'] = gripper_pcd[0].astype(np.float32)
        if 'goal' in self.observation_mode:
            obs_dict_input['goal_gripper_pcd'] = goal_gripper_pcd
  
        if 'displacement_gripper_to_object' in self.observation_mode:
            gripper_pcd = obs_dict_input['gripper_pcd']
            object_pcd = obs_dict_input['point_cloud']
            distance = scipy.spatial.distance.cdist(gripper_pcd, object_pcd)
            min_distance_obj_idx = np.argmin(distance, axis=1)
            closest_point = object_pcd[min_distance_obj_idx]
            displacement = closest_point - gripper_pcd
            obs_dict_input['displacement_gripper_to_object'] = displacement.astype(np.float32)
                
        return obs_dict_input
    
            
    def add_edge_artifacts(self, depth_map):
        """
        Apply edge artifacts to a depth map using correlated depth noise via bilinear interpolation.
        
        Args:
            depth_map (numpy.ndarray): The input depth map of size (H, W).
        
        Returns:
            numpy.ndarray: The depth map with edge artifacts applied.
        """
        H, W = depth_map.shape
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))

        # Generate random shifts for each grid point
        shifts_x = np.random.normal(0, 0.5, size=(H, W))
        shifts_y = np.random.normal(0, 0.5, size=(H, W))
        
        # Apply shifts with a probability of 0.8
        mask = np.random.rand(H, W) < 0.8
        shifted_x = grid_x + shifts_x * mask
        shifted_y = grid_y + shifts_y * mask
        
        # Ensure shifted coordinates stay within valid bounds
        shifted_x = np.clip(shifted_x, 0, W - 1)
        shifted_y = np.clip(shifted_y, 0, H - 1)
        
        # Perform bilinear interpolation between original depth values and shifted grid
        interpolator = RectBivariateSpline(np.arange(H), np.arange(W), depth_map)
        adjusted_depth_map = interpolator(shifted_y, shifted_x, grid=False)

        return adjusted_depth_map

    def add_random_holes(self, depth_map):
        """
        Add random holes to a depth map to simulate irregularities in real-world depth maps.
        
        Args:
            depth_map (numpy.ndarray): The input depth map of size (H, W).
        
        Returns:
            numpy.ndarray: The depth map with random holes applied.
        """
        if np.random.rand() > 0.5:
            # Skip random hole generation with probability 0.5
            return depth_map

        H, W = depth_map.shape
        
        # Create a random mask from U(0,1)
        random_mask = np.random.uniform(0, 1, size=(H, W))
        
        # Apply Gaussian blur to smooth the mask
        smoothed_mask = cv2.GaussianBlur(random_mask, (5, 5), sigmaX=1, sigmaY=1)
        
        # Normalize the mask to the range [0, 1]
        smoothed_mask = (smoothed_mask - smoothed_mask.min()) / (smoothed_mask.max() - smoothed_mask.min())
        
        # Randomly sample a threshold from U(0.6, 0.9)
        threshold = np.random.uniform(0.6, 0.9)
        
        # Zero out pixels where mask values exceed the threshold
        depth_map_with_holes = np.copy(depth_map)
        depth_map_with_holes[smoothed_mask > threshold] = self.depth_near
        
        return depth_map_with_holes

            
    def augment_depth_image(self, depth_images):
        final_images = []
        for image in depth_images:
            
            real_depth = self.get_real_depth(image)
            
            max_depth = np.max(real_depth[real_depth < self.depth_far * 0.9])
            real_depth[real_depth > self.depth_far * 0.9] = max_depth
            beg = time.time()
            edge_augmented_image = self.add_edge_artifacts(real_depth)
            beg = time.time()
            hole_augmented_image = self.add_random_holes(edge_augmented_image)
            bullet_depth = self.get_bullet_depth(hole_augmented_image)
            
            final_images.append(bullet_depth)
        
        return final_images
    
    def _get_observation(self, render=True, using_torch=False, only_object=True):
        ### NOTE: when taking the images, the robot can occlude the object
        rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices = \
            self.take_images_around_object(self._env, self._object_name.lower(), camera_elevation=self.camera_elevation,
                                            return_camera_matrices=True, camera_height=self.camera_height, camera_width=self.camera_width)
        
        if self.noise_real_world_pcd:
            depths = self.augment_depth_image(depths)
        
        if only_object:
            segmented_depths = []
            for depth, segmask in zip(depths, segmasks):
                segmask_obj_id = segmask & ((1 << 24) - 1)
                object_mask = np.zeros_like(depth).astype(np.float32)
                object_mask[segmask_obj_id == self._env.urdf_ids[self._object_name.lower()]] = 1
                object_mask = object_mask.reshape(self.camera_height, self.camera_width)
                # remove the robot point cloud
                depth = depth * object_mask
                depth[depth == 0] = self.depth_far
                segmented_depths.append(depth)
            depths = segmented_depths

        act3d_obs_dict = self._get_act3d_observation(rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices, using_torch=using_torch, only_object=only_object)            
        return act3d_obs_dict

    def get_real_depth(self, depth):
        near = self.depth_near
        far = self.depth_far
        depth = far * near / (far - (far - near) * depth)
        return depth
    
    def get_bullet_depth(self, real_depth):
        near = self.depth_near
        far = self.depth_far
        depth = (far - far * near / real_depth) / (far - near)
        return depth
    
    
    def render(self):
        if 'goal' not in self.observation_mode or 'dp3' in self.observation_mode or self.goal_gripper_pcd is None:
            return self._env.render()
        else:
            image = self._env.render()
            image = np.array(image)
            for point in self.goal_gripper_pcd:
                pixel_x, pixel_y, _ = get_pixel_location(self._env.projection_matrix, self._env.view_matrix, point, self._env.camera_width, self._env.camera_height)
                color = (0, 0, 255)  # Red color in BGR
                thickness = 2
                radius = 5
                image = cv2.circle(image, (pixel_x, pixel_y), radius, color, thickness)
            return image

    
    def take_images_around_object(self, env, object_name, camera_elevation=30, return_camera_matrices=False, camera_height=480, camera_width=640):
        rgbs = []
        depths = []
        segmasks = []
        view_camera_matrices = []
        project_camera_matrices = []
        
        for view_matrix, project_matrix in zip(self.view_matrices, self.project_matrices):
            w, h, img, depth, segmask = p.getCameraImage(camera_width, camera_height, view_matrix, project_matrix, 
                                                         flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX, renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=env.id)
            img = np.reshape(img, (h, w, 4))[:, :, :3]
            depth = np.reshape(depth, (h, w))
            rgbs.append(img)
            depths.append(depth)
            segmasks.append(segmask)
            view_camera_matrices.append(view_matrix)
            project_camera_matrices.append(project_matrix)

        return rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices


   