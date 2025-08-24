from manipulation.sim import SimpleEnv
from manipulation.primitive_api import *
import gym

handle_name_dict = {
    'bucket': 'handle',
    'faucet': 'switch',
    'foldingchair': 'seat',
    'laptop': 'screen_frame',
    'stapler': 'lid',
    'toilet': 'lid',
}

class articulated(SimpleEnv):

    def __init__(self, task_name, object_name, link_name, init_angle, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.object_name = object_name
        self.link_name = link_name
        self.handle_name = handle_name_dict.get(object_name, 'handle')  # default handle name
        self.init_angle = init_angle
        self.handle_joint_id = None
        # link_1 for stapler
        # link_0 for other objects

    def execute(self):
        rgbs, final_state = approach_object_link_parallel(self, self.object_name, self.link_name, debug=False) 
        return rgbs, final_state
    
    def get_handle_joint_id(self):
        if self.handle_joint_id is None:
            all_handle_pos, all_handle_joint_id, _, _ = self.get_handle_pos(return_median=False, custom_joint_name=self.handle_name)
            link_pc, _, _, _ = self.get_link_pc(self.object_name, self.link_name)
            _, self.handle_joint_id, _, _ = get_link_handle(all_handle_pos, all_handle_joint_id, link_pc)
        return self.handle_joint_id
    
    def set_handle(self, angle=0.5):
        handle_joint_id = self.get_handle_joint_id()
        # link_pc, _, _, _ = self.get_link_pc(self.object_name, self.link_name)
        # all_handle_pos, handle_joint_id, _, _ = self.get_handle_pos(return_median=False, custom_joint_name=self.handle_name)
        # _, handle_joint_id, _, _ = get_link_handle(all_handle_pos, handle_joint_id, link_pc, threshold=0.02)
        cur_angle = p.getJointState(self.urdf_ids[self.object_name], handle_joint_id, physicsClientId=self.id)[0]
        p.resetJointState(self.urdf_ids[self.object_name], handle_joint_id, cur_angle + angle, physicsClientId=self.id)

    def reset(self, reset_state=None, open_gripper_at_reset=False):
        super().reset(reset_state, self.object_name, open_gripper_at_reset)
        if reset_state is None and self.init_angle is not None and self.restore_state_file is None:
            self.set_handle(angle=self.init_angle) 

    # NOTE: hard-coded for now, should make it more general in the future
    def get_handle_pos(self, return_median=True, handle_pts_obj_frame=None, mobility_info=None, return_info=False, custom_joint_name=None):
        obj_name = self.object_name.lower()
        scaling = self.simulator_sizes[obj_name]

        # get the parent frame of the revolute joint.
        obj_id = self.urdf_ids[obj_name] 

        # axis in parent frame, transform everything to world frame
        if mobility_info is None:
            urdf_path = self.urdf_paths[obj_name]
            parent_dir = os.path.dirname(urdf_path)
            start_idx = parent_dir.find("data/dataset")
            parent_dir = parent_dir[start_idx:]
            parent_dir = os.path.join(os.environ["PROJECT_DIR"], parent_dir)
            # print("mobility_info path: ", parent_dir)
            mobility_info = json.load(open(f"{parent_dir}/mobility_v2.json", "r"))
        
        # return a list of handle points in world frame
        ret_handle_pt_list = []
        ret_joint_idx_list = []
        axis_world_list = []
        axis_end_world_list = []

        joint_name = None
        parent_joint_name = None
        handle_idx = 0
        all_handle_pts_object_frame = []
        for idx, joint_info in enumerate(mobility_info):
            all_parts = [part["name"] for part in joint_info["parts"]]
            if custom_joint_name in all_parts:
                all_ids = [part["id"] for part in joint_info["parts"]]
                index = all_parts.index(custom_joint_name)
                handle_id = all_ids[index]
                joint_name = "joint_{}".format(joint_info["id"])
                parent_joint_name = "joint_{}".format(joint_info["parent"])
                joint_data = joint_info['jointData']
                axis_body = np.array(joint_data["axis"]["origin"]) * scaling
                axis_dir_body = np.array(joint_data["axis"]["direction"])
                joint_limit = joint_data["limit"]
                if joint_limit['a'] > joint_limit['b']:
                    axis_dir_body = -axis_dir_body

                joint_idx = self.get_joint_id_from_name( obj_name, joint_name) # this is the joint id in pybullet
                parent_joint_idx = self.get_joint_id_from_name(obj_name, parent_joint_name) # this is the joint id in pybullet
                
                parent_link_state = p.getLinkState(obj_id, parent_joint_idx, physicsClientId=self.id) # NOTE: the handle link id should be dependent on the object urdf.
                # parent_link_state = p.getLinkState(obj_id, joint_idx, physicsClientId=simulator.id) # NOTE: the handle link id should be dependent on the object urdf.
                link_urdf_world_pos, link_urdf_world_orn = parent_link_state[0], parent_link_state[1]
                # this is the transformation from the parent frame to the world frame. 
                T_body_to_world = np.eye(4) # transformation from the parent body frame to the world frame
                T_body_to_world[:3, :3] = np.array(p.getMatrixFromQuaternion(link_urdf_world_orn)).reshape(3, 3)
                T_body_to_world[:3, 3] = link_urdf_world_pos
                
                axis_world = T_body_to_world[:3, :3] @ axis_body + T_body_to_world[:3, 3]   
                axis_pt2_body = np.array(axis_body) + axis_dir_body
                axis_end_world = T_body_to_world[:3, :3] @ axis_pt2_body + T_body_to_world[:3, 3]
                axis_dir_world = axis_end_world - axis_world

                # get the handle points in world frame
                if handle_pts_obj_frame is None:
                    handle_obj_path = f"{parent_dir}/parts_render/{handle_id}{custom_joint_name}.obj" # NOTE: this path should be dependent on the object. 
                    handle_pts, handle_faces = load_obj(handle_obj_path) # this is in object frame

                    handle_pts = handle_pts * scaling
                    # add more dense points around handle
                    added_points = []
                    for f in handle_faces:
                        v1,v2,v3 = f
                        v1 = handle_pts[v1-1]
                        v2 = handle_pts[v2-1]
                        v3 = handle_pts[v3-1]
                        a = np.linalg.norm(v1-v2)
                        b = np.linalg.norm(v2-v3)
                        c = np.linalg.norm(v3-v1)
                        s = (a+b+c) / 2
                        temp = max(0, s*(s-a)*(s-b)*(s-c))
                        surface = np.sqrt(temp)
                        num_points = surface * 1e6
                        num_points = int(num_points)
                        num_points = np.clip(num_points, 0, 5)
                        added_points.extend([sample_point_inside_triangle(v1,v2,v3) for _ in range(num_points)])

                    if added_points != []:
                        added_points = np.array(added_points)
                        handle_pts = np.concatenate((handle_pts, added_points), axis=0)
                        
                    all_handle_pts_object_frame.append(handle_pts)
                        
                else:
                    handle_pts = handle_pts_obj_frame[handle_idx]
                
                
                # transform this to the world frame using the object *base*'s position and orientation
                handle_points_world = T_body_to_world[:3, :3] @ handle_pts.T + T_body_to_world[:3, 3].reshape(3, 1) # 3 x N
                if return_median:
                    handle_point_median = np.median(handle_points_world, axis=1)
                else:
                    handle_point_median = handle_points_world.T

                # find the projection of the handle point to the rotation axis, in world frame. 
                project_on_rotation_axis = find_nearest_point_on_line(axis_world, axis_end_world, handle_point_median)
                # p.addUserDebugLine(project_on_rotation_axis, handle_point_median, [1, 0, 0], 25, 0)

                # TODO: GPT can parse the mobility.json to get the joint name. 
                joint_info = p.getJointInfo(obj_id, joint_idx, physicsClientId=self.id)
                joint_type = joint_info[2]
                
                if joint_type == p.JOINT_REVOLUTE:
                    rotation_angle = p.getJointState(obj_id, joint_idx, physicsClientId=self.id)[0] # NOTE: this joint id should be dependent on the object urdf.
                    rotated_handle_pt_local = rotate_point_around_axis(handle_point_median - project_on_rotation_axis, axis_dir_world, rotation_angle)
                    rotated_handle_pt = project_on_rotation_axis + rotated_handle_pt_local
                elif joint_type == p.JOINT_PRISMATIC:
                    translation = p.getJointState(obj_id, joint_idx, physicsClientId=self.id)[0]
                    rotated_handle_pt = handle_point_median + axis_dir_world * translation
                    
                # import pdb; pdb.set_trace()
                # rotated_handle_pt = handle_points_world.T

                if return_median:
                    ret_handle_pt_list.append(rotated_handle_pt.flatten())
                else:
                    ret_handle_pt_list.append(rotated_handle_pt)
                ret_joint_idx_list.append(joint_idx)
                axis_world_list.append(axis_world)
                axis_end_world_list.append(axis_end_world)
                handle_idx += 1
                
        if return_info:
            return ret_handle_pt_list, ret_joint_idx_list, all_handle_pts_object_frame, mobility_info
        
        return ret_handle_pt_list, ret_joint_idx_list, axis_world_list, axis_end_world_list
    
    def _get_info(self):
        # TODO: this should be implemented by GPT
        if self.handle_joint is None:
            all_handle_pos, all_handle_joint_id, handle_pts_obj_frame, mobility_info = self.get_handle_pos(return_median=False, return_info=True, custom_joint_name=self.handle_name)
            self.handle_pts_obj_frame = handle_pts_obj_frame
            self.mobility_info = mobility_info
            # link_name = "link_0"
            link_pc, _, _, _ = self.get_link_pc(self.object_name, self.link_name)
            # object_pc, _ = get_pc_and_normal(self, object_name)
            _, link_handle_joint_id, link_handle_median, min_link_idx = get_link_handle(all_handle_pos, all_handle_joint_id, link_pc)
            # _, link_handle_joint_id, link_handle_median, min_link_idx = get_link_handle(all_handle_pos, all_handle_joint_id, object_pc)
            self.handle_joint = link_handle_joint_id
            self.handle_pos = link_handle_median
            self.min_link_idx = min_link_idx
            self.all_handle_points = all_handle_pos[min_link_idx]
        else:
            all_handle_pos, _, _, _ = self.get_handle_pos(return_median=False, handle_pts_obj_frame=self.handle_pts_obj_frame, mobility_info=self.mobility_info, custom_joint_name=self.handle_name)
            handle_median_points = np.array([np.median(handle_pos, axis=0) for handle_pos in all_handle_pos]).reshape(-1, 3)
            self.handle_pos = handle_median_points[self.min_link_idx]
            self.all_handle_points = all_handle_pos[self.min_link_idx]
            
        opened_joint_angle = p.getJointState(self.urdf_ids[self.object_name], self.handle_joint, physicsClientId=self.id)[0]
        if self.init_joint_angle is None:
            self.init_joint_angle = opened_joint_angle
            grasped_handle = False
        else:
            grasped_handle = (self.last_joint_angle - opened_joint_angle) > 1e-6  # threshold to determine if the handle is grasped
        self.last_joint_angle = opened_joint_angle
        self.grasped_handle = self.grasped_handle or grasped_handle
            
        # cur_eef_pos, cur_eef_orient = self.robot.get_pos_orient(self.robot.right_end_effector)
        # handle_points = self.all_handle_points
        # num_handle_points_within_gripper = get_pc_num_within_gripper(cur_eef_pos, cur_eef_orient, handle_points)
        # cprint("num_handle_points_within_gripper: {}".format(num_handle_points_within_gripper), "red")
        # distance_eef_to_handle = np.linalg.norm(self.handle_pos.flatten() - cur_eef_pos.flatten())
        # grasped_handle = False
        # if num_handle_points_within_gripper > 0:
        #     # Compute vector between fingers
        #     left_finger_pos, _ = self.robot.get_pos_orient(self.robot.right_gripper_indices[0])
        #     right_finger_pos, _ = self.robot.get_pos_orient(self.robot.right_gripper_indices[1])
        #     finger_vec = right_finger_pos - left_finger_pos
        #     finger_length = np.linalg.norm(finger_vec)
            
        #     if finger_length < 1e-6:
        #         grasped_handle = False  # Prevent division by zero if fingers overlap
        #     else:
        #         finger_dir = finger_vec / finger_length  # Unit direction vector of gripper axis

        #         # Project handle points onto gripper axis to check if they are between fingers
        #         vecs = handle_points - left_finger_pos
        #         projections = np.dot(vecs, finger_dir)
        #         between_mask = np.logical_and(projections > -0.01, projections < finger_length + 0.01)
        #         handle_points_between_fingers = handle_points[between_mask]

        #         if handle_points_between_fingers.shape[0] > 0:
        #             # Compute distances from these points to each finger
        #             distance_left = np.linalg.norm(handle_points_between_fingers - left_finger_pos.reshape(1, 3), axis=1)
        #             distance_right = np.linalg.norm(handle_points_between_fingers - right_finger_pos.reshape(1, 3), axis=1)
        #             min_distance_left = np.min(distance_left)
        #             min_distance_right = np.min(distance_right)

        #             # Use a more lenient threshold (0.025 instead of 0.015)
        #             if min_distance_left < 0.05 and min_distance_right < 0.05:
        #                 grasped_handle = True
        #                 self.grasped_handle = self.grasped_handle or grasped_handle
            # points_left_finger = p.getContactPoints(bodyA=self.robot.body, linkIndexA=self.robot.right_gripper_indices[0], physicsClientId=self.id)
            # points_right_finger = p.getContactPoints(bodyA=self.robot.body, linkIndexA=self.robot.right_gripper_indices[1], physicsClientId=self.id)
            # print("points_left_finger: ", points_left_finger)
            # print("points_right_finger: ", points_right_finger)
            # if len(points_left_finger) > 0 and len(points_right_finger) > 0:
            #     contact_points_left = np.array([point[6] for point in points_left_finger])
            #     contact_points_right = np.array([point[6] for point in points_right_finger])
            #     left_distance = scipy.spatial.distance.cdist(handle_points, contact_points_left)
            #     right_distance = scipy.spatial.distance.cdist(handle_points, contact_points_right)
            #     min_distance_left = np.min(left_distance)
            #     min_distance_right = np.min(right_distance)
            #     # if min_distance_left < 0.015 and min_distance_right < 0.015:
            #     if min_distance_left < 0.01 or min_distance_right < 0.01:
            #         grasped_handle = True
            #         self.grasped_handle = self.grasped_handle or grasped_handle
        
        right_finger_pos, _ = self.robot.get_pos_orient(self.robot.right_gripper_indices[0])
        left_finger_pos, _ = self.robot.get_pos_orient(self.robot.right_gripper_indices[1])
        finger_distance = np.linalg.norm(right_finger_pos - left_finger_pos)
        
        return {
            "opened_joint_angle": opened_joint_angle,
            "improved_joint_angle": opened_joint_angle - self.init_joint_angle,
            "handle_pos": self.handle_pos, 
            "initial_joint_angle": self.init_joint_angle,
            "ik_failure": self.ik_failure,
            "oversized_joint_distance": self.oversized_joint_distance,
            "grasped_handle": self.grasped_handle,
            "current_grasped_handle": grasped_handle,
            "finger_distance": finger_distance, 
        }
            
gym.register(
    id='articulated-v0',
    entry_point=articulated,
)