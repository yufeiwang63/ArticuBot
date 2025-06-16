import numpy as np
import pybullet as p
import gym
from gym.utils import seeding
from gym import spaces
import pickle
import yaml
import os.path as osp
from collections import defaultdict
import matplotlib.pyplot as plt
import open3d 
from termcolor import cprint
import time
import scipy
import os
from manipulation.gpt_primitive_api import get_pc_num_within_gripper
from manipulation.utils import parse_config, load_env, save_env, radial_shift
from manipulation.gpt_reward_api import get_joint_id_from_name, get_handle_pos, get_link_pc
from manipulation.gpt_primitive_api import get_link_handle
from scipy.spatial.transform import Rotation as R
from manipulation.panda import Panda
from typing import List, Optional

class SimpleEnv(gym.Env):
    def __init__(self, 
                    dt=1/240, 
                    config_path=None, 
                    gui=False, 
                    control_step=2, 
                    horizon=250, 
                    restore_state_file=None, 
                    vhacd=True, # if to perform vhacd on the object for better collision detection for pybullet
                    randomize=0, # if to randomize the scene
                    obj_id=0, # which object to choose to use from the candidates
                    mobile=False,
                    task_name=None,
                    open_gripper_at_reset=True,
                    ik_limit=True, 
                ):
        
        super().__init__()
        
        # Task
        self.config_path = config_path
        self.task_name = task_name
        self.restore_state_file = restore_state_file
        self.control_step = control_step
        self.horizon = horizon
        self._max_episode_steps = horizon
        self.gui = gui
        self.solution_path = None        
        self.success = False # not really used, keeped for now
        self.primitive_save_path = None # to be used for saving the primitives execution results
        self.randomize = randomize
        self.obj_id = obj_id # which object to choose to use from the candidates
        self.open_gripper_at_reset = open_gripper_at_reset
        
        # robot
        self.ik_limit = ik_limit

        # physics
        self.gravity = -9.81
        self.vhacd = vhacd
                
        if self.gui:
            try:
                self.id = p.connect(p.GUI)
            except:
                self.id = p.connect(p.DIRECT)
        else:
            self.id = p.connect(p.DIRECT)

        self.asset_dir = osp.join(osp.dirname(osp.realpath(__file__)), "assets/")
        p.setTimeStep(dt, physicsClientId=self.id)

        self.init_state = None
        self.handle_joint = None
        self.grasped_handle = False
        self.seed()
        self.set_scene()
        self.view_matrix = None
        self.projection_matrix = None
        self.setup_camera_rpy()

        self.action_low = np.array([-1, -1, -1, -1, -1, -1, -1])
        self.action_high = np.array([1, 1, 1, 1, 1, 1, 1])

        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32) 
        self.base_action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32) 
        self.num_objects = len(self.urdf_ids) - 2 # exclude plane, robot

        num_obs = 100 ### just some fake numbers
        self.base_num_obs = 10 ### just some fake numbers
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_obs, ), dtype=np.float32) 
        self.base_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.base_num_obs, ), dtype=np.float32)

        self.time_step = 0
        self.success = False
        self.control_rgbs = []
        self.init_joint_angle = None
        self.ik_failure = False

    def get_gripper_pc(self):
        # get the point cloud of the gripper
        right_finger_pos, _ = self.robot.get_pos_orient(self.robot.right_gripper_indices[0])
        left_finger_pos, _ = self.robot.get_pos_orient(self.robot.right_gripper_indices[1])
        right_hand_pos, _ = self.robot.get_pos_orient(self.robot.right_hand)
        eef_pos, _ = self.robot.get_pos_orient(self.robot.right_end_effector)
        gripper_pc = np.array([right_hand_pos, right_finger_pos, left_finger_pos, eef_pos]).reshape(-1, 3)
        return gripper_pc.astype(np.float32)

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random()

    def get_aabb(self, id):
        num_joints = p.getNumJoints(id, physicsClientId=self.id)
        min_aabbs, max_aabbs = [], []
        for link_idx in range(-1, num_joints):
            min_aabb, max_aabb = p.getAABB(id, link_idx, physicsClientId=self.id)
            min_aabbs.append(list(min_aabb))
            max_aabbs.append(list(max_aabb))
        min_aabb = np.min(np.concatenate(min_aabbs, axis=0).reshape(-1, 3), axis=0)
        max_aabb = np.max(np.concatenate(max_aabbs, axis=0).reshape(-1, 3), axis=0)
        return min_aabb, max_aabb
    
    def get_aabb_link(self, id, link_id):
        min_aabb, max_aabb = p.getAABB(id, link_id, physicsClientId=self.id)
        return np.array(min_aabb), np.array(max_aabb)

    def clip_x_bbox_within_workspace(self, robot_pos, ori_pos, on_table , min_bbox, max_bbox):
        x_near_low = robot_pos[0] - 0.4
        x_near_high = robot_pos[0] + 0.4
        offset = 0
        if min_bbox[0] > robot_pos[0]:
            if min_bbox[0] < x_near_high:
                offset = x_near_high - min_bbox[0]
        elif max_bbox[0] < robot_pos[0]:
            if max_bbox[0] > x_near_low:
                offset = x_near_low - max_bbox[0]

        pos = ori_pos.copy()
        pos[0] += offset
        
        if not on_table:
            return pos
        else:
            # Object is on table, should be within table's bounding box
            new_pos = pos.copy()
            new_pos[:2] = np.clip(new_pos[:2], self.table_bbox_min[:2], self.table_bbox_max[:2])
            return new_pos

    def get_robot_base_pos(self):
        robot_base_pos = [0, 0, 0]
        return robot_base_pos
    
    def get_robot_init_joint_angles(self, robot_init_joint_angles=None):
        if robot_init_joint_angles is None:
            init_joint_angles = [0 for _ in range(len(self.robot.right_arm_joint_indices))]

            init_joint_angles[3] = -0.4
            init_joint_angles[5] = 0.4
            return init_joint_angles  
        return robot_init_joint_angles

    def set_scene(
        self,
        reset_state=None,
    ):
        ### simulation preparation
        p.resetSimulation(physicsClientId=self.id)
        if self.gui:
            p.resetDebugVisualizerCamera(cameraDistance=1.75, cameraYaw=-25, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.4], physicsClientId=self.id)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.id)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.id)
        p.setRealTimeSimulation(0, physicsClientId=self.id)
        p.setGravity(0, 0, self.gravity, physicsClientId=self.id)

        ### load restore state
        restore_state = None
        if self.restore_state_file is not None:
            with open(self.restore_state_file, 'rb') as f:
                restore_state = pickle.load(f)

        ### load and parse task config (including semantically meaningful distractor objects)
        self.urdf_paths = {}
        self.urdf_types = {}
        self.init_positions = {}
        self.init_orientations = {}
        self.on_tables = {}
        self.simulator_sizes = {}
        self.is_distractor = {
            "robot": 0,
            "plane": 0,
        }
        urdf_paths, urdf_sizes, urdf_positions, urdf_orientations, urdf_names, urdf_types, urdf_on_table, urdf_movables, urdf_crop_sizes, \
            use_table, articulated_init_joint_angles, spatial_relationships, robot_initial_joint_angles = self.load_and_parse_config(restore_state)

        ### load plane 
        planeId = p.loadURDF(osp.join(self.asset_dir, "plane", "plane.urdf"), physicsClientId=self.id)

        ### create and load a robot
        self.robot_base_pos = self.load_robot(restore_state, robot_initial_joint_angles=robot_initial_joint_angles)

        self.urdf_ids = {
            "robot": self.robot.body,
            "plane": planeId,
        }
        

        ### load each object from the task config
        self.load_object(urdf_paths, urdf_sizes, urdf_positions, urdf_orientations, urdf_names, urdf_types, urdf_on_table, urdf_movables, urdf_crop_sizes)

        ### if a state is passed in, restore the state
        if reset_state is not None:
            load_env(self, state=reset_state)
            return

        ### after first set scene, the init state will be stored, and can be restored here, skipping the following steps to save time
        if self.init_state is not None:
            load_env(self, state=self.init_state)
            return
        
        ### adjusting object positions
        ### place the lowest point on the object to be the height where GPT specifies
        object_height = self.adjust_object_positions(self.robot_base_pos)

        ### resolve collisions between objects
        self.resolve_collision(self.robot_base_pos, object_height, spatial_relationships)

        ### set all object's joint angles to the lower joint limit
        self.set_to_default_joint_angles()

        ### overwrite joint angles specified by GPT
        self.handle_gpt_joint_angle(articulated_init_joint_angles)
          
        # open the gripper at reset 
        if self.open_gripper_at_reset:
            for _ in range(20):
                self.robot.set_gripper_open_position(self.robot.right_gripper_indices, [0.04, 0.04], set_instantly=False)

        ### stabilize the scene
        for _ in range(500):
            p.stepSimulation(physicsClientId=self.id)

        ### restore to a state if provided
        if self.restore_state_file is not None:
            load_env(self, self.restore_state_file)

        ### record initial joint angles and positions
        self.record_initial_joint_and_pose()

        ### Enable debug rendering
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
 
        self.init_state = save_env(self)

        
    def load_robot(self, restore_state, robot_initial_joint_angles=None):
        robot_classes = {
            "panda": Panda,
        }
        robot_names = list(robot_classes.keys())
        self.robot_name = robot_names[np.random.randint(len(robot_names))]
        if restore_state is not None and "robot_name" in restore_state:
            self.robot_name = restore_state['robot_name']
        self.robot_class = robot_classes[self.robot_name]
      
        # Create robot
        self.robot = self.robot_class()
        self.robot.init(self.asset_dir, self.id, self.np_random, fixed_base=True, ik_limit=self.ik_limit)
        self.agents = [self.robot]
        self.suction_id = self.robot.right_gripper_indices[0]

        # Set robot base position & orientation, and joint angles
        robot_base_pos = self.get_robot_base_pos()
        robot_base_orient = [0, 0, 0, 1]
        self.robot_base_orient = robot_base_orient
        self.robot.set_base_pos_orient(robot_base_pos, robot_base_orient)
        init_joint_angles = self.get_robot_init_joint_angles(robot_initial_joint_angles)
        self.robot.set_joint_angles(self.robot.right_arm_joint_indices, init_joint_angles)    
                
        self.robot.set_gravity(0, 0, 0)
        
        return robot_base_pos        
    
    def load_and_parse_config(self, restore_state):        
        self.config = None
        while self.config is None:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        for obj in self.config:
            if "solution_path" in obj:
                self.solution_path = obj["solution_path"]
                break
        
        ### parse config
        urdf_paths, urdf_sizes, urdf_positions, urdf_orientations, urdf_names, urdf_types, urdf_on_table, \
            use_table, urdf_crop_sizes, articulated_init_joint_angles, spatial_relationships, distractor_config_path, urdf_movables, \
                robot_initial_joint_angles = parse_config(self.config, 
                        use_bard=True, obj_id=self.obj_id,
                        use_vhacd=True)
        if not use_table:
            urdf_on_table = [False for _ in urdf_on_table]
        urdf_names = [x.lower() for x in urdf_names]
        for name in urdf_names:
            self.is_distractor[name] = 0
        
        if restore_state is not None:
            if "urdf_paths" in restore_state:
                self.urdf_paths = {}
                for urdf_name in restore_state['urdf_paths']:
                    urdf_path = restore_state['urdf_paths'][urdf_name]
                    start_idx = urdf_path.find("data/dataset")
                    urdf_path = urdf_path[start_idx:]
                    urdf_path = os.path.join(os.environ["PROJECT_DIR"], urdf_path)
                    self.urdf_paths[urdf_name] = urdf_path
                urdf_paths = []
                for name in urdf_names:
                    if name in self.urdf_paths:
                        urdf_paths.append(self.urdf_paths[name])
            if "object_sizes" in restore_state:
                self.simulator_sizes = restore_state['object_sizes']
                urdf_sizes = [self.simulator_sizes[name] for name in urdf_names]
                
        return urdf_paths, urdf_sizes, urdf_positions, urdf_orientations, urdf_names, urdf_types, urdf_on_table, urdf_movables, urdf_crop_sizes, \
            use_table, articulated_init_joint_angles, spatial_relationships, robot_initial_joint_angles
        
    def load_object(self, urdf_paths, urdf_sizes, urdf_positions, urdf_orientations, urdf_names, urdf_types, urdf_on_table, urdf_movables, urdf_crop_sizes):
        for path, size, pos, urdf_ori, name, type, on_table, moveable, is_crop_size in zip(urdf_paths, urdf_sizes, urdf_positions, urdf_orientations, urdf_names, urdf_types, urdf_on_table, urdf_movables, urdf_crop_sizes):
            
            name = name.lower()
            # by default, all objects movable, except the urdf files
            use_fixed_base = (type == 'urdf' and not self.is_distractor[name])
            if type == 'urdf' and moveable: # if gpt specified the object is movable, then it is movable
                use_fixed_base = False
            
            if not moveable:
                use_fixed_base = True
            
            if type == 'urdf' and is_crop_size:
                size = min(size, 1.2)
                size = max(size, 0.075) # if the object is too small, current gripper cannot really manipulate it.
            
            x_orient = np.pi/2 if type == 'mesh' else 0 # handle different coordinate axis by objaverse and partnet-mobility
            if self.randomize or self.is_distractor[name]:
                orientation = p.getQuaternionFromEuler([x_orient, 0, self.np_random.uniform(-np.pi/3, np.pi/3)], physicsClientId=self.id)
            else:
                orientation = p.getQuaternionFromEuler([x_orient, 0, 0], physicsClientId=self.id)

            # combine the orientation from the config file
            urdf_mat = R.from_quat(urdf_ori)
            ori_mat = R.from_quat(orientation)
            orientation = urdf_mat * ori_mat
            orientation = orientation.as_quat()

            if name == 'stepping_stone':
                orientation = urdf_ori
                use_fixed_base = True

            if not on_table:
                load_pos = pos
            else: # change to be table coordinate
                table_xy_range = self.table_bbox_max[:2] - self.table_bbox_min[:2]
                obj_x = self.table_bbox_min[0] + pos[0] * table_xy_range[0]
                obj_y = self.table_bbox_min[1] + pos[1] * table_xy_range[1]
                obj_z = self.table_height + pos[2]
                load_pos = [obj_x, obj_y, obj_z]
            try:
                id = p.loadURDF(path, basePosition=load_pos, baseOrientation=orientation, physicsClientId=self.id, useFixedBase=use_fixed_base, globalScaling=size)
            except:
                path = osp.join(os.environ["PROJECT_DIR"], path)
                id = p.loadURDF(path, basePosition=load_pos, baseOrientation=orientation, physicsClientId=self.id, useFixedBase=use_fixed_base, globalScaling=size)

            # scale size 
            if name in self.simulator_sizes:
                p.removeBody(id, physicsClientId=self.id)
                saved_size = self.simulator_sizes[name]
                id = p.loadURDF(path, basePosition=load_pos, baseOrientation=orientation, physicsClientId=self.id, useFixedBase=use_fixed_base, globalScaling=saved_size)
            elif size == -1:
                id = p.loadURDF(path, basePosition=load_pos, baseOrientation=orientation, physicsClientId=self.id, useFixedBase=use_fixed_base)
                min_aabb, max_aabb = self.get_aabb(id)
                actual_size = np.linalg.norm(max_aabb - min_aabb)
                self.simulator_sizes[name] = np.sqrt(actual_size)
            else:
                min_aabb, max_aabb = self.get_aabb(id)
                actual_size = np.linalg.norm(max_aabb - min_aabb)
                if np.abs(actual_size - size) > 0.05:
                    p.removeBody(id, physicsClientId=self.id)
                    id = p.loadURDF(path, basePosition=load_pos, baseOrientation=orientation, physicsClientId=self.id, useFixedBase=use_fixed_base, globalScaling=size ** 2 / actual_size)
                    self.simulator_sizes[name] = size ** 2 / actual_size
                else:
                    self.simulator_sizes[name] = size

            self.urdf_ids[name] = id
            self.urdf_paths[name] = path
            self.urdf_types[name] = type
            self.init_positions[name] = np.array(load_pos)
            self.init_orientations[name] = orientation
            self.on_tables[name] = on_table
    
    def adjust_object_positions(self, robot_base_pos):
        object_height = {}
        for name, id in self.urdf_ids.items():
            if name == 'robot' or name == 'plane' or name == 'init_table': continue
            min_aabb, max_aabb = self.get_aabb(id)
            min_z = min_aabb[2]
            object_height[id] = 2 * self.init_positions[name][2] - min_z
            pos, orient = p.getBasePositionAndOrientation(id, physicsClientId=self.id)
            new_pos = np.array(pos) 
            new_pos = self.clip_x_bbox_within_workspace(robot_base_pos, new_pos, self.on_tables[name], min_aabb, max_aabb)
            new_pos[2] = object_height[id]
            p.resetBasePositionAndOrientation(id, new_pos, orient, physicsClientId=self.id)
            self.init_positions[name] = new_pos
        
        return object_height
        
    def resolve_collision(self, robot_base_pos, object_height, spatial_relationships):
        collision = True
        collision_cnt = 1
        while collision:
            if collision_cnt % 50 == 0: # if collision is not resolved every 50 iterations, we randomly reset object's position
                for name, id in self.urdf_ids.items():
                    if name == 'robot' or name == 'plane' or name == "init_table": continue
                    pos = self.init_positions[name]
                    _, orient = p.getBasePositionAndOrientation(id, physicsClientId=self.id)
                    new_pos = np.array(pos) + np.random.uniform(-0.2, 0.2, size=3)
                    min_aabb, max_aabb = self.get_aabb(id)
                    new_pos = self.clip_x_bbox_within_workspace(robot_base_pos, new_pos, self.on_tables[name], min_aabb, max_aabb)
                    new_pos[2] = object_height[id]
                    p.resetBasePositionAndOrientation(id, new_pos, orient, physicsClientId=self.id)
                    p.stepSimulation(physicsClientId=self.id)

            push_directions = defaultdict(list) # store the push direction for each object

            # detect collisions between objects 
            detected_collision = False
            for name, id in self.urdf_ids.items():
                if name == 'robot' or name == 'plane' or name == 'init_table': continue
                for name2, id2 in self.urdf_ids.items():
                    if name == name2 or name2 == 'robot' or name2 == 'plane' or name2 == 'init_table': continue

                    # if gpt specifies obj a and obj b should have some special relationship, then skip collision resolution
                    skip = False
                    for spatial_relationship in spatial_relationships:
                        words = spatial_relationship.lower().split(",")
                        words = [word.strip().lstrip() for word in words]
                        if name in words and name2 in words:
                            skip = True
                            break

                    if skip: continue
                    
                    contact_points = p.getClosestPoints(id, id2, 0.01, physicsClientId=self.id)
                    if len(contact_points) > 0:
                        contact_point = contact_points[0]
                        push_direction = contact_point[7]
                        push_direction = np.array([push_direction[0], push_direction[1], push_direction[2]])

                        # both are distractors or both are not, push both objects away
                        if (self.is_distractor[name] and self.is_distractor[name2]) or \
                            (not self.is_distractor[name] and not self.is_distractor[name2]):
                            push_directions[id].append(-push_direction)
                            push_directions[id2].append(push_direction)
                        # only 1 is distractor, only pushes the distractor
                        if self.is_distractor[name] and not self.is_distractor[name2]:
                            push_directions[id].append(push_direction)
                        if not self.is_distractor[name] and self.is_distractor[name2]:
                            push_directions[id2].append(-push_direction)
                        
                        detected_collision = True

            # collisions between robot and objects, only push object away
            for name, id in self.urdf_ids.items():
                if name == 'robot' or name == 'plane' or name == 'init_table': 
                    continue

                contact_points = p.getClosestPoints(self.robot.body, id, 0.05, physicsClientId=self.id)
                if len(contact_points) > 0:
                    contact_point = contact_points[0]
                    push_direction = contact_point[7]
                    push_direction = np.array([push_direction[0], push_direction[1], push_direction[2]])
                    push_directions[id].append(-push_direction)
                    detected_collision = True

            # move objects
            push_distance = 0.1
            for id in push_directions:
                for direction in push_directions[id]:
                    pos, orient = p.getBasePositionAndOrientation(id, physicsClientId=self.id)
                    new_pos = np.array(pos) + push_distance * direction    
                    min_aabb, max_aabb = self.get_aabb(id)
                    new_pos = self.clip_x_bbox_within_workspace(robot_base_pos, new_pos, self.on_tables[name], min_aabb, max_aabb)
                    new_pos[2] = object_height[id]

                    p.resetBasePositionAndOrientation(id, new_pos, orient, physicsClientId=self.id)
                    p.stepSimulation(physicsClientId=self.id)

            collision = detected_collision
            collision_cnt += 1

            if collision_cnt > 1000:
                break
    
    def record_initial_joint_and_pose(self):
        self.initial_joint_angle = {}
        for name in self.urdf_ids:        
            obj_id = self.urdf_ids[name.lower()]
            if name == 'robot' or name == 'plane' or name == "init_table": continue
            if self.urdf_types[name.lower()] == 'urdf':
                self.initial_joint_angle[name] = {}
                num_joints = p.getNumJoints(obj_id, physicsClientId=self.id)
                for joint_idx in range(num_joints):
                    joint_name = p.getJointInfo(obj_id, joint_idx, physicsClientId=self.id)[1].decode("utf-8")
                    joint_angle = p.getJointState(obj_id, joint_idx, physicsClientId=self.id)[0]
                    self.initial_joint_angle[name][joint_name] = joint_angle
        
        self.initial_pos = {}
        self.initial_orient = {}
        for name in self.urdf_ids:
            obj_id = self.urdf_ids[name.lower()]
            if name == 'robot' or name == 'plane' or name == "init_table": continue
            pos, orient = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.id)
            self.initial_pos[name] = pos
            self.initial_orient[name] = orient
            
        self.initial_eef_pos = self.robot.get_pos_orient(self.robot.right_end_effector)[0]
        
    def set_to_default_joint_angles(self):
        for obj_name in self.urdf_ids:
            if obj_name == 'robot' or obj_name == 'plane' or obj_name == "init_table": continue
            obj_id = self.urdf_ids[obj_name]
            num_joints = p.getNumJoints(obj_id, physicsClientId=self.id)
            for joint_idx in range(num_joints):
                joint_limit_low, joint_limit_high = p.getJointInfo(obj_id, joint_idx, physicsClientId=self.id)[8:10]
                if joint_limit_low > joint_limit_high:
                    joint_limit_low, joint_limit_high = joint_limit_high, joint_limit_low
                joint_val = joint_limit_low
                p.resetJointState(obj_id, joint_idx, joint_val, physicsClientId=self.id)          

    def handle_gpt_joint_angle(self, articulated_init_joint_angles):
        for name in articulated_init_joint_angles:
            obj_id = self.urdf_ids[name.lower()]

            if "set_joint_angle_joint_id" not in articulated_init_joint_angles[name].keys():
                for joint_name, joint_angle in articulated_init_joint_angles[name].items():
                    joint_idx = get_joint_id_from_name(self, name.lower(), joint_name)
                    joint_limit_low, joint_limit_high = p.getJointInfo(obj_id, joint_idx, physicsClientId=self.id)[8:10]
                    if joint_limit_low > joint_limit_high:
                        joint_limit_low, joint_limit_high = joint_limit_high, joint_limit_low
                    if 'random' not in joint_angle:
                        joint_angle = joint_limit_low
                    else:
                        joint_angle = self.np_random.uniform(joint_limit_low, joint_limit_high)
            else:
                # TODO: account for cases when there are multiple joints to be set.
                p.resetJointState(obj_id, articulated_init_joint_angles[name]["set_joint_angle_joint_id"], 
                              articulated_init_joint_angles[name]['set_joint_angle_joint_angle'], physicsClientId=self.id)

    def reset(self, reset_state=None, object_name='StorageFurniture', open_gripper_at_reset=False):
        self.grasped_handle = False
        self.set_scene(reset_state)
            
        self.time_step = 0
        self.success = False
        object_name = object_name.lower()
        friction = 5
        num_links = p.getNumJoints(self.urdf_ids[object_name], physicsClientId=self.id)
        for l_id in range(num_links):
            p.changeDynamics(self.urdf_ids[object_name], l_id, lateralFriction=friction, physicsClientId=self.id)
            p.changeDynamics(self.urdf_ids[object_name], l_id, rollingFriction=friction, physicsClientId=self.id)
            p.changeDynamics(self.urdf_ids[object_name], l_id, spinningFriction=friction, physicsClientId=self.id)

        # increase the friction for firmly grasping the handle
        p.changeDynamics(self.robot.body, self.robot.right_gripper_indices[0], lateralFriction=friction, physicsClientId=self.id)
        p.changeDynamics(self.robot.body, self.robot.right_gripper_indices[1], lateralFriction=friction, physicsClientId=self.id)
        p.changeDynamics(self.robot.body, self.robot.right_gripper_indices[0], rollingFriction=friction, physicsClientId=self.id)
        p.changeDynamics(self.robot.body, self.robot.right_gripper_indices[1], rollingFriction=friction, physicsClientId=self.id)
        p.changeDynamics(self.robot.body, self.robot.right_gripper_indices[0], spinningFriction=friction, physicsClientId=self.id)
        p.changeDynamics(self.robot.body, self.robot.right_gripper_indices[1], spinningFriction=friction, physicsClientId=self.id)

        self.ik_failure = False
        
        if open_gripper_at_reset:
            for _ in range(40):
                self.robot.set_gripper_open_position(self.robot.right_gripper_indices, [0.04, 0.04], set_instantly=False)
                p.stepSimulation()

        return self._get_obs()

    def setup_camera(self, camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], fov=60, camera_width=640, camera_height=480):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.view_matrix = p.computeViewMatrix(camera_eye, camera_target, [0, 0, 1], physicsClientId=self.id)
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, camera_width / camera_height, 0.01, 100, physicsClientId=self.id)
    
    def setup_camera_rpy(self, camera_target=None, distance=1.6, rpy=[0, -30, -30], fov=60, camera_width=640, camera_height=480):
        self.camera_width = camera_width
        self.camera_height = camera_height
        if camera_target is None:
            for name in self.urdf_ids: # randomly center at an object
                if name in ['robot', 'plane', 'init_table']: continue
                obj_id = self.urdf_ids[name]
                min_aabb, max_aabb = self.get_aabb(obj_id)
                center = (min_aabb + max_aabb) / 2
                camera_target = center 
                break

        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(camera_target, distance, rpy[2], rpy[1], rpy[0], 2, physicsClientId=self.id)
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, camera_width / camera_height, 0.01, 100, physicsClientId=self.id)

    def render(self, return_depth=False, mode=None):
        assert self.view_matrix is not None, 'You must call env.setup_camera() or env.setup_camera_rpy() before getting a camera image'
        w, h, img, depth, segmask = p.getCameraImage(self.camera_width, self.camera_height, 
            self.view_matrix, self.projection_matrix, 
            renderer=p.ER_BULLET_HARDWARE_OPENGL, 
            physicsClientId=self.id)
        img = np.reshape(img, (h, w, 4))[:, :, :3]
        depth = np.reshape(depth, (h, w))

        if return_depth:
            return img, depth
        else:
            return img
         
    def take_direct_action(self, actions, gains=None, forces=None, save_img_interval=0, ik_try_times=25, far_target=False):
        if gains is None:
            gains = [a.motor_gains for a in self.agents]
        elif type(gains) not in (list, tuple): 
            gains = [gains]*len(self.agents)
        if forces is None:
            forces = [a.motor_forces for a in self.agents]
        elif type(forces) not in (list, tuple):
            forces = [forces]*len(self.agents)

        np.random.seed(time.time_ns() % 2**32)

        self.control_rgbs = []
        action_index = 0
        for i, agent in enumerate(self.agents):
            agent_action_len = self.base_action_space.shape[0] 
            action = np.copy(actions[action_index:action_index+agent_action_len])
            action_index += agent_action_len
            translation = action[:3] ### target position
            rotation = action[3:6] ### target orientation in euler angle
            finger_joint_angle = action[6] ### gripper joint angle
            original_joint_angles = agent.get_joint_angles(agent.all_joint_indices)

            ik_indices = [_ for _ in range(len(agent.right_arm_ik_indices))]
            
            pos = translation
            orient = p.getQuaternionFromEuler(rotation)

            ### try to get a ik solution using tracIK
            tracIK_solutions = agent.ik_tracik_franka(pos, orient, ik_indices)
            
            ### try to get a list of IK solutions using the bullet's default ik solver
            bullet_solutions = []
            old_state = save_env(self)
            ik_indices = [_ for _ in range(len(self.robot.right_arm_joint_indices))]
            for try_idx in range(ik_try_times):
                if try_idx > 0: 
                    new_joint_angles = original_joint_angles[ik_indices] + np.random.uniform(-0.3, 0.3, size=len(ik_indices))
                    self.robot.set_joint_angles(ik_indices, new_joint_angles)

                ik_joint_angles = self.robot.ik(self.robot.right_end_effector, pos, orient, ik_indices=ik_indices, max_iterations=10000, residualThreshold=1e-4)
                if np.all(ik_joint_angles >= self.robot.ik_lower_limits[ik_indices]) and np.all(ik_joint_angles <= self.robot.ik_upper_limits[ik_indices]):
                    bullet_solutions.append(ik_joint_angles)

            ### choose ik solutions that are close to current joint angles
            load_env(self, state = old_state)
            all_possible_solutions = tracIK_solutions + bullet_solutions
            if len(all_possible_solutions) > 0:
                all_possible_solutions = np.array(all_possible_solutions).reshape(-1, len(ik_indices))
                distance_to_cur_angle = np.linalg.norm(all_possible_solutions - original_joint_angles[agent.controllable_joint_indices].reshape(1, -1), axis=1)
                min_idx = np.argmin(distance_to_cur_angle)
                min_joint_distance = distance_to_cur_angle[min_idx]
                best_joint_angles = all_possible_solutions[min_idx]
                agent_joint_angles = best_joint_angles
                ik_success = min_joint_distance < 0.3
            else:
                ik_success = False

            self.ik_failure = (not ik_success) or self.ik_failure
            
            it = 0
            if ik_success:
                control_total = 50
                ### control gripper
                for _ in range(2):
                    agent.set_gripper_open_position(agent.right_gripper_indices, [finger_joint_angle, finger_joint_angle], set_instantly=False)
                    p.stepSimulation(physicsClientId=self.id) 
                    if save_img_interval > 0:
                        rgb = self.render()
                        self.control_rgbs.append(rgb)

                cur_joint_angles = agent.get_joint_angles(agent.controllable_joint_indices)

                old_err = 1e10
                ### control until the joint angles are close enough to the target joint angles
                while True:
                    agent.control(agent.controllable_joint_indices, agent_joint_angles)
                    cur_joint_angles = agent.get_joint_angles(agent.controllable_joint_indices)
                    err = np.linalg.norm(cur_joint_angles - agent_joint_angles) 
                    if err < 1e-4:
                        break

                    old_err = err
                    if save_img_interval > 0 and it % save_img_interval == 0:
                        rgb = self.render()
                        self.control_rgbs.append(rgb)

                    if it > control_total:
                        break
                    
                    it += 1
                    p.stepSimulation(physicsClientId=self.id) 
                    
                
                end = time.time()
            else:
                pass    
    
    def get_control_rgbs(self):
        return self.control_rgbs

    def compute_reward(self):
        return 0, 0

    def _get_info(self):
        ### we focus on storagefurniture so it is hardcoded here
        object_name = 'storagefurniture'
        if self.handle_joint is None:
            all_handle_pos, all_handle_joint_id, handle_pts_obj_frame, mobility_info = get_handle_pos(self, object_name, return_median=False, return_info=True)
            self.handle_pts_obj_frame = handle_pts_obj_frame
            self.mobility_info = mobility_info
            link_name = "link_0"
            link_pc = get_link_pc(self, object_name, link_name)
            _, link_handle_joint_id, link_handle_median, min_link_idx = get_link_handle(all_handle_pos, all_handle_joint_id, link_pc)
            self.handle_joint = link_handle_joint_id
            self.handle_pos = link_handle_median
            self.min_link_idx = min_link_idx
            self.all_handle_points = all_handle_pos[min_link_idx]
        else:
            all_handle_pos, _ = get_handle_pos(self, object_name, return_median=False, handle_pts_obj_frame=self.handle_pts_obj_frame, mobility_info=self.mobility_info)
            handle_median_points = np.array([np.median(handle_pos, axis=0) for handle_pos in all_handle_pos]).reshape(-1, 3)
            self.handle_pos = handle_median_points[self.min_link_idx]
            self.all_handle_points = all_handle_pos[self.min_link_idx]
            
        opened_joint_angle = p.getJointState(self.urdf_ids[object_name], self.handle_joint, physicsClientId=self.id)[0]
        if self.init_joint_angle is None:
            self.init_joint_angle = opened_joint_angle
            
        cur_eef_pos, cur_eef_orient = self.robot.get_pos_orient(self.robot.right_end_effector)
        handle_points = self.all_handle_points
        num_handle_points_within_gripper = get_pc_num_within_gripper(cur_eef_pos, cur_eef_orient, handle_points)
        distance_eef_to_handle = np.linalg.norm(self.handle_pos.flatten() - cur_eef_pos.flatten())
        if num_handle_points_within_gripper > 0:
            points_left_finger = p.getContactPoints(bodyA=self.robot.body, linkIndexA=self.robot.right_gripper_indices[0], physicsClientId=self.id)
            points_right_finger = p.getContactPoints(bodyA=self.robot.body, linkIndexA=self.robot.right_gripper_indices[1], physicsClientId=self.id)
            if len(points_left_finger) > 0 and len(points_right_finger) > 0:
                contact_points_left = np.array([point[6] for point in points_left_finger])
                contact_points_right = np.array([point[6] for point in points_right_finger])
                left_distance = scipy.spatial.distance.cdist(handle_points, contact_points_left)
                right_distance = scipy.spatial.distance.cdist(handle_points, contact_points_right)
                min_distance_left = np.min(left_distance)
                min_distance_right = np.min(right_distance)
                if min_distance_left < 0.02 or min_distance_right < 0.02:
                    grasped_handle = True
                    self.grasped_handle = self.grasped_handle or grasped_handle
        
        right_finger_pos, _ = self.robot.get_pos_orient(self.robot.right_gripper_indices[0])
        left_finger_pos, _ = self.robot.get_pos_orient(self.robot.right_gripper_indices[1])
        finger_distance = np.linalg.norm(right_finger_pos - left_finger_pos)
        
        return {
            "opened_joint_angle": opened_joint_angle,
            "improved_joint_angle": opened_joint_angle - self.init_joint_angle,
            "handle_pos": self.handle_pos, 
            "initial_joint_angle": self.init_joint_angle,
            "ik_failure": self.ik_failure,
            "grasped_handle": self.grasped_handle,
            "finger_distance": finger_distance, 
        }

    def _get_obs(self):
        ### not really used, will be overridden by the robogenpointcloud wrapper class. 
        obs = np.zeros(self.base_observation_space.shape[0])
        return obs

    def disconnect(self):
        p.disconnect(self.id)

    def close(self):
        p.disconnect(self.id)
    
    
if __name__ == "__main__":
    from manipulation.utils import build_up_env
    env = SimpleEnv(
        config_path="data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46462_2024-03-27-23-35-10/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-4-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first/2024-05-11-00-58-12/task_config.yaml",
        gui=True,
        mobile=True,
        control_step=10,
        max_translation=0.05,
    )
    
    env.reset()
    
    for t in range(10000):
        p.stepSimulation(physicsClientId=env.id)