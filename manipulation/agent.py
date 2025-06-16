import numpy as np
import pybullet as p
from termcolor import cprint
import os

class Agent:
    def __init__(self):
        self.base = -1
        self.body = None
        self.lower_limits = None
        self.upper_limits = None
        self.ik_lower_limits = None
        self.ik_upper_limits = None
        self.ik_joint_names = None

    def init(self, body, id, np_random, indices=None, ik_limit=True):
        self.body = body
        self.id = id
        self.np_random = np_random
        self.all_joint_indices = list(range(p.getNumJoints(body, physicsClientId=id)))
        if indices != -1:
            pass
        
        active_masks = [0,1,1,1,1,1,1,1,0,0,1]

        current_path = os.path.dirname(os.path.realpath(__file__))
        # self.franka_ikpy_chain = ikpy.chain.Chain.from_urdf_file(f"{current_path}/assets/panda_bullet/panda.urdf", base_elements=["panda_link0"], active_links_mask=active_masks)
        
        try:
            from tracikpy import TracIKSolver
            self.tracIK = True
            if ik_limit:
                self.franka_tracik_solver = TracIKSolver(f"{current_path}/assets/panda_ik/franka.urdf", "panda_link0", "panda_grasptarget", epsilon=1e-5, timeout=0.05)
            else:
                print("loading panda without joint limit!!!!!!!!!!!!!!!!!")
                self.franka_tracik_solver = TracIKSolver(f"{current_path}/assets/panda_ik/franka_no_limit.urdf", "panda_link0", "panda_grasptarget", epsilon=1e-5, timeout=0.05)
        except:
            self.tracIK = False
            print("tracikpy not installed, using just default pybullet ik solver")

    def control(self, indices, target_angles, gains=None, forces=None):
        if gains is not None:
            if type(gains) in [int, float]:
                gains = [gains]*len(indices)
            if type(forces) in [int, float]:
                forces = [forces]*len(indices)
        if gains is not None:
            p.setJointMotorControlArray(self.body, jointIndices=indices, controlMode=p.POSITION_CONTROL, targetPositions=target_angles, 
                                        positionGains=gains, forces=forces, 
                                        physicsClientId=self.id)
        else:
            p.setJointMotorControlArray(self.body, jointIndices=indices, controlMode=p.POSITION_CONTROL, targetPositions=target_angles, 
                                        physicsClientId=self.id)

    def get_joint_angles(self, indices=None):
        if indices is None:
            indices = self.all_joint_indices
        elif not indices:
            return []
        robot_joint_states = p.getJointStates(self.body, jointIndices=indices, physicsClientId=self.id)
        return np.array([x[0] for x in robot_joint_states])

    def get_joint_angles_dict(self, indices=None):
        return {j: a for j, a in zip(indices, self.get_joint_angles(indices))}

    def get_pos_orient(self, link):
        # Get the 3D position and orientation (4D quaternion) of a specific link on the body
        if link == self.base:
            pos, orient = p.getBasePositionAndOrientation(self.body, physicsClientId=self.id)
        else:
            pos, orient = p.getLinkState(self.body, link, physicsClientId=self.id)[:2]
        return np.array(pos), np.array(orient)

    def get_base_pos_orient(self):
        return self.get_pos_orient(self.base)

    def get_velocity(self, link):
        if link == self.base:
            return p.getBaseVelocity(self.body, physicsClientId=self.id)[0]
        return p.getLinkState(self.body, link, computeForwardKinematics=True, computeLinkVelocity=True, physicsClientId=self.id)[6]

    def get_euler(self, quaternion):
        return np.array(p.getEulerFromQuaternion(np.array(quaternion), physicsClientId=self.id))

    def get_quaternion(self, euler):
        return np.array(p.getQuaternionFromEuler(np.array(euler), physicsClientId=self.id))

    def get_motor_joint_states(self, joints=None):
        # Get the position, velocity, and torque for nonfixed joint motors
        joint_states = p.getJointStates(self.body, self.all_joint_indices if joints is None else joints, physicsClientId=self.id)
        joint_infos = [p.getJointInfo(self.body, i, physicsClientId=self.id) for i in (self.all_joint_indices if joints is None else joints)]
        motor_states = [j for j, i in zip(joint_states, joint_infos) if i[2] != p.JOINT_FIXED]
        motor_indices = [i[0] for j, i in zip(joint_states, joint_infos) if i[2] != p.JOINT_FIXED]
        motor_positions = [state[0] for state in motor_states]
        motor_velocities = [state[1] for state in motor_states]
        motor_torques = [state[3] for state in motor_states]
        return motor_indices, motor_positions, motor_velocities, motor_torques

    def get_joint_max_force(self, indices=None):
        if indices is None:
            indices = self.all_joint_indices
        joint_infos = [p.getJointInfo(self.body, i, physicsClientId=self.id) for i in indices]
        return [j[10] for j in joint_infos]

    def set_base_pos_orient(self, pos, orient):
        p.resetBasePositionAndOrientation(self.body, pos, orient if len(orient) == 4 else self.get_quaternion(orient), physicsClientId=self.id)
    
    def update_joint_limits(self, indices=None):
        if indices is None:
            indices = self.all_joint_indices
        self.lower_limits = dict()
        self.upper_limits = dict()
        self.ik_lower_limits = []
        self.ik_upper_limits = []
        self.ik_joint_names = []
        for j in indices:
            joint_info = p.getJointInfo(self.body, j, physicsClientId=self.id)
            joint_name = joint_info[1]
            joint_type = joint_info[2]
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            if lower_limit == 0 and upper_limit == -1:
                lower_limit = -1e10
                upper_limit = 1e10
                if joint_type != p.JOINT_FIXED:
                    # NOTE: IK only works on non fixed joints, so we build special joint limit lists for IK
                    self.ik_lower_limits.append(-2*np.pi)
                    self.ik_upper_limits.append(2*np.pi)
                    self.ik_joint_names.append([len(self.ik_joint_names)] + list(joint_info[:2]))
            elif joint_type != p.JOINT_FIXED:
                self.ik_lower_limits.append(lower_limit)
                self.ik_upper_limits.append(upper_limit)
                self.ik_joint_names.append([len(self.ik_joint_names)] + list(joint_info[:2]))
            self.lower_limits[j] = lower_limit
            self.upper_limits[j] = upper_limit
        self.ik_lower_limits = np.array(self.ik_lower_limits)
        self.ik_upper_limits = np.array(self.ik_upper_limits)

    def enforce_joint_limits(self, indices=None):
        if indices is None:
            indices = self.all_joint_indices
        joint_angles = self.get_joint_angles_dict(indices)
        if self.lower_limits is None or len(indices) > len(self.lower_limits):
            self.update_joint_limits()
        for j in indices:
            if joint_angles[j] < self.lower_limits[j]:
                p.resetJointState(self.body, jointIndex=j, targetValue=self.lower_limits[j], targetVelocity=0, physicsClientId=self.id)
            elif joint_angles[j] > self.upper_limits[j]:
                p.resetJointState(self.body, jointIndex=j, targetValue=self.upper_limits[j], targetVelocity=0, physicsClientId=self.id)

    def ik(self, target_joint, target_pos, target_orient, ik_indices, max_iterations=1000, residualThreshold=1e-4, use_current_as_rest=False, return_full_state=False):
        if target_orient is not None and len(target_orient) < 4:
            target_orient = self.get_quaternion(target_orient)
        if use_current_as_rest:
            ik_rest_poses = np.array(self.get_motor_joint_states()[1]).tolist()

        ik_lower_limits = self.ik_lower_limits 
        ik_upper_limits = self.ik_upper_limits 
        ik_joint_ranges = ik_upper_limits - ik_lower_limits

        if target_orient is not None:
            if use_current_as_rest:
                ik_joint_poses = np.array(p.calculateInverseKinematics(
                    self.body, target_joint, 
                    targetPosition=target_pos, targetOrientation=target_orient, 
                    lowerLimits=ik_lower_limits.tolist(), upperLimits=ik_upper_limits.tolist(), jointRanges=ik_joint_ranges.tolist(), 
                    restPoses=ik_rest_poses, 
                    maxNumIterations=max_iterations, 
                    residualThreshold=residualThreshold,
                    physicsClientId=self.id))
            else:
                ik_joint_poses = np.array(p.calculateInverseKinematics(self.body, 
                    target_joint, targetPosition=target_pos, targetOrientation=target_orient, 
                    maxNumIterations=max_iterations, 
                    residualThreshold=residualThreshold,
                    physicsClientId=self.id))
        else:
            if use_current_as_rest:
                ik_joint_poses = np.array(p.calculateInverseKinematics(self.body, target_joint, targetPosition=target_pos, restPoses=ik_rest_poses, maxNumIterations=max_iterations, physicsClientId=self.id))
            else:
                ik_joint_poses = np.array(p.calculateInverseKinematics(self.body, target_joint, targetPosition=target_pos, maxNumIterations=max_iterations, physicsClientId=self.id))            

        if return_full_state:
            return ik_joint_poses
        return ik_joint_poses[ik_indices]
    
    def ik_ikpy_franka(self, target_pos, target_orient, ik_indices):
        print(target_pos)
        target_pos = target_pos - np.array([1, 1, 0])
        import pdb; pdb.set_trace()

        # self.franka_ikpy_chain.active_links = ik_indices
        joint_angles = self.franka_ikpy_chain.inverse_kinematics(target_position=target_pos)
        real_frame = self.franka_ikpy_chain.forward_kinematics(joint_angles)
        cprint(real_frame, 'green')
        cprint(joint_angles, 'red')
        joint_angles = joint_angles[ik_indices]

        for i, j in enumerate(ik_indices):
            p.resetJointState(self.body, j, targetValue=joint_angles[i], targetVelocity=0, physicsClientId=self.id)

        import ikpy.utils.plot as plot_utils
        import matplotlib.pyplot as plt
        fig, ax = plot_utils.init_3d_figure()
        self.franka_ikpy_chain.plot(self.franka_ikpy_chain.inverse_kinematics(target_pos), ax, target=target_pos)
        plt.xlim(-0.5, 0.5)
        plt.ylim(-0.2, 0.2)
        plt.show()
         
        
        import pdb; pdb.set_trace()


    def ik_tracik_franka(self, target_pos, target_orient, ik_indices):
        if not self.tracIK:
            return []
        
        original_joint_angles = self.get_joint_angles(self.all_joint_indices)
        original_joint_angles = original_joint_angles[ik_indices]

        # get robot base position and orientation
        base_pos, base_orient = self.get_base_pos_orient()
        base_pos = np.array(base_pos)

        target_pos = target_pos - base_pos + np.array([0, 0, 0.05])
        target_orient = np.array(p.getMatrixFromQuaternion(target_orient)).reshape(3, 3)

        target_eef = np.eye(4)
        target_eef[:3, :3] = target_orient
        target_eef[:3, 3] = target_pos

        solutions = []
        ik_lower_limits = self.ik_lower_limits 
        ik_upper_limits = self.ik_upper_limits 
        ik_joint_ranges = ik_upper_limits - ik_lower_limits
        ik_lower_limits = ik_lower_limits + 0.05 * ik_joint_ranges
        ik_upper_limits = ik_upper_limits - 0.05 * ik_joint_ranges
            
        import time
        beg = time.time()
        for try_time in range(25): # try 100 times
            # TODO: sample different init joint angles
            if try_time == 0:
                ik_start_pose = original_joint_angles
            else:
                ik_start_pose = original_joint_angles + np.random.uniform(-0.3, 0.3, len(original_joint_angles))
            joint_angles = self.franka_tracik_solver.ik(target_eef, qinit=ik_start_pose[self.right_arm_joint_indices])
            if joint_angles is not None:
                solutions.append(joint_angles)
            if try_time == 0 and joint_angles is None:
                return []
        end = time.time()
        
        # cprint("IK time: {}".format(end - beg), "blue")
        return solutions        

        if len(solutions) == 0:
            # cprint('After 100 tries, tracIK failed', 'red')
            return solutions, False
        
        solution_np = np.array(solutions)
        distances = np.linalg.norm(solution_np - original_joint_angles, axis=1)
        joint_angles = solution_np[np.argmin(distances)]

        return joint_angles, True
        

    def print_joint_info(self, show_fixed=True):
        joint_names = []
        for j in self.all_joint_indices:
            info = p.getJointInfo(self.body, j, physicsClientId=self.id)
            if show_fixed or info[2] != p.JOINT_FIXED:
                print(info)
                joint_names.append((j, info[1]))
        print(joint_names)

    def set_joint_angles(self, indices, angles, use_limits=True, velocities=0):
        for i, (j, a) in enumerate(zip(indices, angles)):
            p.resetJointState(self.body, jointIndex=j, targetValue=min(max(a, self.lower_limits[j]), self.upper_limits[j]) if use_limits else a, targetVelocity=velocities if type(velocities) in [int, float] else velocities[i], physicsClientId=self.id)


    def set_gravity(self, ax=0.0, ay=0.0, az=-9.81):
        p.setGravity(ax, ay, az, physicsClientId=self.id)