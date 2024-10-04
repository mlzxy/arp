from scipy.spatial.transform import Rotation as R
import numpy as np

eef_T_wrenchHead = np.loadtxt("eef_T_wrenchHead.txt") # (4, 4)

eef_sm_T_wrenchHead = np.loadtxt("eef_sm_T_wrenchHead.txt") # (4, 4)


def wrench_head_to_eef(wrenchHead_T_base: np.array) -> np.array: # (..., 4, 4)
    r"""
    convert wrench pose in the base's frame to the end-effector's pose in the base frame
    """
    eef_T_base = wrenchHead_T_base @ eef_T_wrenchHead
    return eef_T_base # (..., 4, 4)

def eef_to_wrench_head(eef_T_base: np.array) -> np.array: # (..., 4, 4)
    r"""
    convert end-effector pose in the base's frame to the wrench pose in the base frame
    """
    wrenchHead_T_base = eef_T_base @ np.linalg.inv(eef_T_wrenchHead)
    return wrenchHead_T_base # (..., 4, 4)

def pose_to_T(pose_vec: np.array) -> np.array:
    r"""
    pose_vec[np.array]: [x, y, z, qx, qy, qz, qw]
    ---
    T[np.array]: (4, 4)
    """
    T = np.eye(4)
    T[:3, 3] = pose_vec[:3]
    T[:3, :3] = R.from_quat(pose_vec[3:]).as_matrix()
    return T

def T_to_pose(T: np.array) -> np.array:
    r"""
    T[np.array]: (4, 4)
    ---
    pose_vec[np.array]: [x, y, z, qx, qy, qz, qw]
    """
    pose_vec = np.zeros(7)
    pose_vec[:3] = T[:3, 3]
    pose_vec[3:] = R.from_matrix(T[:3, :3]).as_quat()
    return pose_vec


def p3_to_p7(p3):
    x = np.zeros(7)
    x[:3] = p3
    x[-1]= 1.0
    return x


def wrench_pose7_to_eef_pose7(p7):
    return T_to_pose(wrench_head_to_eef(pose_to_T(p7)))


def wrench_pose3_to_eef_pose3(p3):
    p7 = p3_to_p7(p3)
    return T_to_pose(wrench_head_to_eef(pose_to_T(p7)))[:3]


def eef_pose7_to_wrench_pose7(p7):
    return T_to_pose(eef_to_wrench_head(pose_to_T(p7)))


def wrench_pose7_to_sm_eef_pose7(p7):
    return T_to_pose(pose_to_T(p7) @ np.linalg.inv(eef_sm_T_wrenchHead))

