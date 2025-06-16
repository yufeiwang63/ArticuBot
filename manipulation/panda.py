import os
import numpy as np
import pybullet as p
from .robot import Robot

class Panda(Robot):
    def __init__(self, controllable_joints='right', slider=False):
        right_arm_joint_indices = [0, 1, 2, 3, 4, 5, 6] # Controllable arm joints
        right_end_effector = 11 # Used to get the pose of the end effector
        right_gripper_indices = [9, 10] # Gripper actuated joints
        right_hand = 8 # TODO: check this
                
        super(Panda, self).__init__(controllable_joints, right_arm_joint_indices, right_end_effector, right_gripper_indices)
        self.right_hand = right_hand

    def init(self, directory, id, np_random, fixed_base=True, debug=False, ik_limit=True):
        if ik_limit:
            self.body = p.loadURDF(os.path.join(directory, 'panda_bullet', 'panda.urdf'), useFixedBase=fixed_base, basePosition=[-1, -1, 0.5], flags=p.URDF_USE_SELF_COLLISION, physicsClientId=id)
        else:
            print("loading panda without joint limit!!!!!!!!!!!!!!!!!")
            self.body = p.loadURDF(os.path.join(directory, 'panda_bullet', 'panda_no_limit.urdf'), useFixedBase=fixed_base, basePosition=[-1, -1, 0.5], flags=p.URDF_USE_SELF_COLLISION, physicsClientId=id)

        if debug:
            for i in range(p.getNumJoints(self.body, physicsClientId=id)):
                print(p.getJointInfo(self.body, i, physicsClientId=id))
                link_name = p.getJointInfo(self.body, i, physicsClientId=id)[12].decode('utf-8')
                joint_limits = p.getJointInfo(self.body, i, physicsClientId=id)[8:10]
                print("link_name: ", link_name)
        
        super(Panda, self).init(self.body, id, np_random, ik_limit=ik_limit)
