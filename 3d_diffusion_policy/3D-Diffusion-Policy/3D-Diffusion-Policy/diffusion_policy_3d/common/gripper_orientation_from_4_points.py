#import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from manipulation.infer_utils import matrix_to_rotation_6d_numpy, rotation_6d_to_matrix
from manipulation.utils import rotation_transfer_matrix_to_6D
def compute_plane_normal(gripper_pcd):
    x1 = gripper_pcd[0]
    x2 = gripper_pcd[1]
    x4 = gripper_pcd[3]
    v1 = x2 - x1
    v2 = x4 - x1
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)



def quaternion_to_rotation_matrix(quat):
    rotation = R.from_quat(quat)
    return rotation.as_matrix()

def rotation_matrix_to_quaternion(R_opt):
    rotation = R.from_matrix(R_opt)
    return rotation.as_quat()

def rotation_matrix_from_vectors(v1, v2):
    """
    Find the rotation matrix that aligns v1 to v2
    :param v1: A 3d "source" vector
    :param v2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to v1, aligns it with v2.
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    axis = np.cross(v1, v2)
    axis_len = np.linalg.norm(axis)
    if axis_len != 0:
        axis = axis / axis_len
    angle = np.arccos(np.clip(np.dot(v1, v2), -1, 1))

    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])

    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R


def _get_gripper_pos_orient_from_4_points(original_gripper_pcd, gripper_pcd, original_gripper_pos, original_gripper_orn, original_gripper_normal ):
    normal = compute_plane_normal(gripper_pcd)
    R1 = rotation_matrix_from_vectors(original_gripper_normal, normal)
    v1 = original_gripper_pcd[3] - original_gripper_pcd[0]
    v2 = gripper_pcd[3] - gripper_pcd[0]
    v1_prime = np.dot(R1, v1)
    R2 = rotation_matrix_from_vectors(v1_prime, v2)
    R = np.dot(R2, R1)
    gripper_pos = original_gripper_pos + gripper_pcd[3] - original_gripper_pcd[3]
    original_R = quaternion_to_rotation_matrix(original_gripper_orn)
    R = np.expand_dims(np.dot(R, original_R), axis = 0)
    #print("RRRRRRRRRR", R)
    gripper_orn = matrix_to_rotation_6d_numpy(R.swapaxes(1, 2))
    #print("Gripper Orn Initial", gripper_orn)
    #gripper_orn = matrix_to_rotation_6d_numpy(R)
    #gripper_orn = rotation_transfer_matrix_to_6D(R).reshape(1,-1)
    #print("Gripper Orn Initial", gripper_orn_alternate)
    #print("Gripper Orn Alternate", gripper_orn)
    #print("HELOOOOOOOOOOOOOOOOOOO", gripper_orn.shape)
    return gripper_pos, np.squeeze(gripper_orn, axis = 0)



def compute_plane_normal(gripper_pcd):
    x1 = gripper_pcd[0]
    x2 = gripper_pcd[1]
    x4 = gripper_pcd[3]
    v1 = x2 - x1
    v2 = x4 - x1
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)

def get_gripper_pos_orient_from_4_points_torch(gripper_pcd):
    #import pdb; pdb.set_trace();
    original_gripper_pcd = np.array([[ 0.10432111,  0.00228697,  0.8474241 ],
       [ 0.12816067, -0.04368229,  0.8114649 ],
       [ 0.08953098,  0.0484529 ,  0.80711854],
       [ 0.11198021,  0.00245327,  0.7828771 ]])
    original_gripper_pos = np.array([0.1119802 , 0.00245327, 0.78287711])
    original_gripper_orn = np.array([0.97841681, 0.19802945, 0.0581003 , 0.01045192])
    original_gripper_normal = compute_plane_normal(original_gripper_pcd)
    #print("HEREEEEEEEE", gripper_pcd.shape, original_gripper_pcd.shape)
    #gripper_pcd = gripper_pcd.T
    gripper_pos, gripper_orn = _get_gripper_pos_orient_from_4_points(original_gripper_pcd, gripper_pcd, original_gripper_pos, original_gripper_orn, original_gripper_normal)
    return np.concatenate((gripper_pos.reshape(3), gripper_orn.reshape(6)))


def get_points_from_pos_rotation_matrix(pos, orient):
    original_gripper_pcd = np.array([[ 0.10432111,  0.00228697,  0.8474241 ],
       [ 0.12816067, -0.04368229,  0.8114649 ],
       [ 0.08953098,  0.0484529 ,  0.80711854],
       [ 0.11198021,  0.00245327,  0.7828771 ]])
    original_gripper_orn = np.array([0.97841681, 0.19802945, 0.0581003 , 0.01045192])
    from manipulation.utils import rotation_transfer_6D_to_matrix
    absolute_rotation = rotation_transfer_6D_to_matrix(orient.numpy())
    #print("absolute_rotation", absolute_rotation)
    #import pdb; pdb.set_trace();
    #absolute_rotation_2 = rotation_6d_to_matrix(orient)
    #print("absolute_rotation_2", absolute_rotation_2)
    #print("absolute_rotation_2", absolute_rotation_2 - absolute_rotation)
    original_R = quaternion_to_rotation_matrix(original_gripper_orn)
    rotation_transfer = absolute_rotation * original_R.T
    original_pcd = original_gripper_pcd - original_gripper_pcd[3]
    rotated_pcd = np.dot(original_pcd, rotation_transfer.T)
    gripper_pcd = rotated_pcd + pos
    return gripper_pcd



gripper_pcd = np.array([[ 0.7666899, -0.16390935, 0.44825187],
    [ 0.8285312, -0.16042456, 0.4600023 ],
    [ 0.7625436, -0.1860767, 0.38937932],
    [ 0.8155203, -0.17972143, 0.40836987]])
gripper_10d = get_gripper_pos_orient_from_4_points_torch(gripper_pcd)
print(gripper_10d)
gripper_pcd_reconstructed = get_points_from_pos_rotation_matrix(gripper_10d[:3], torch.tensor(gripper_10d[3:]))
print(gripper_pcd_reconstructed)

