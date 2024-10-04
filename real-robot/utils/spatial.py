import open3d as o3d
from scipy.spatial.transform import Rotation
import numpy as np
from numpy.linalg import inv


def to_o3d_pcd(pcd):
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(pcd)
    return source


def to_np_pcd(o3d_pcd):
    return np.asarray(o3d_pcd.points)



def voxel_grid(pcd, voxel_size=0.02):
    source = to_o3d_pcd(pcd)
    voxel_volume = o3d.geometry.VoxelGrid.create_from_point_cloud(source, voxel_size)
    return np.asarray([voxel_volume.origin + pt.grid_index*voxel_volume.voxel_size for pt in voxel_volume.get_voxels()])


def normalize_point_cloud_to_origin(pcd):
    center = pcd.mean(axis=0, keepdims=True)
    pcd = pcd - center
    return pcd, center.flatten()


def to_unit_length(v):
    return v / np.linalg.norm(v)


def R_2_X(R):
    X = np.eye(4)
    X[:3, :3] = R
    return X

    
def t_2_X(t):
    X = np.eye(4)
    if isinstance(t, np.ndarray): t = t.flatten()
    X[:3, -1] = t
    return X


def Rt_2_X(R, t):
    X = np.eye(4)
    X[:3, :3] = R
    if isinstance(t, np.ndarray): t = t.flatten()
    X[:3, -1] = t
    return X

def X_2_Rt(X):
    return X[:3, :3], X[:3, -1]

def to_homo_axis(pts):
    return np.concatenate([pts, np.ones((len(pts), 1))], axis=1)


def h_transform(T, pts):
    return (T @ to_homo_axis(pts).T).T[:, :3]

def r_transform(R, pts):
    return (R @ pts.T).T

def axis_angle_rotate(axis, radian):
    assert axis.shape == (3,)
    rot = Rotation.from_rotvec(to_unit_length(axis) * radian)
    return rot.as_matrix() 

def rotate_X(X, Pc, axis, angle):
    R, t = X_2_Rt(X)
    Rot_inv = inv(axis_angle_rotate(axis, angle))
    Rnew = Rot_inv @ R
    tnew = Rot_inv @ (t - Pc) + Pc
    return Rt_2_X(Rnew, tnew)


def get_matching_ratio(P_from, P_to, threshold=0.01):
    dist, _ = knn(P_from, P_to)
    matched_ratio = (dist <= threshold).sum() / len(P_from)
    return matched_ratio


def resolve_rotation_ambiguity(X, ref_points, points, ref_context_points, context_points, ambiguity_threshold=0.95):
    P = points
    P_n, P_c = normalize_point_cloud_to_origin(P)
    bbox = estimate_pca_box(P_n)
    
    P_ref = h_transform(X, ref_points)
    P_ref_n, P_ref_c = normalize_point_cloud_to_origin(P_ref)
    
    X_alternatives = [X, ]
    X_AXIS, Y_AXIS, Z_AXIS = 0, 1, 2

    for radian in [np.pi, np.pi/2, -np.pi/2]:
        for axis in [X_AXIS, Y_AXIS, Z_AXIS]:
            ratio = get_matching_ratio(P_ref_n, r_transform(axis_angle_rotate(bbox.R[axis], radian), P_n))
            # print('ambiguity ratio -> ', ratio)
            if ratio >= ambiguity_threshold:
                _X = rotate_X(X, P_c, bbox.R[axis], radian)
                if check_X_validity(_X):
                    X_alternatives.append(_X)
    
    if len(X_alternatives) == 1:
        return X_alternatives[0]
    
    distances = []
    Xs = []
    for _X in X_alternatives:
        distance = knn(h_transform(_X, ref_context_points), context_points)[0].mean()
        distances.append(distance)
        Xs.append(_X)
    
    ind = np.argmin(distances)
    # if distances[0] / (distances[ind] + 1e-6) > 2: 
    if ind != 0:
        print('ambiguity resolve to another X!')
    else:
        ind = 0
    chosen_X = Xs[ind]
    return chosen_X
    
    
    
def check_X_validity(X):
    plane = np.zeros((10, 10, 3))
    plane[:, :, 0], plane[:, :, 1] = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    plane[:, :, 2] = np.random.uniform(0.74, 0.76, size=(10, 10))
    plane = plane.reshape(-1, 3)
    X_plane = h_transform(X, plane)
    m = X_plane[:, 2].mean()
    return 0.73 <=  m <= 0.77


def pose7_to_X(pose):
    R = Rotation.from_quat(pose[3:]).as_matrix()
    t = pose[:3]
    X = np.zeros((4, 4))
    X[:3, :3] = R
    X[-1, -1] = 1
    X[:3, -1] = t
    return X


def X_to_pose7(X):
    t = X[:3, -1]
    q = Rotation.from_matrix(X[:3, :3]).as_quat()
    return np.concatenate([t, q])


def pose7_to_frame(pose, scale=0.1):
    pose = pose.copy()
    R = Rotation.from_quat(pose[3:]).as_matrix() * scale
    t = pose[:3]
    return np.array([t, R[0] + t, R[1] + t, R[2] + t])


def X_to_frame(X):
    return pose7_to_frame(X_to_pose7(X))


def frame_to_X(frame):
    frame = np.copy(frame)
    t, x, y, z = frame
    X = np.eye(4)
    X[:3, -1] = t
    x -= t
    y -= t
    z -= t
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    z = z / np.linalg.norm(z)
    X[0, :3] = x
    X[1, :3] = y
    X[2, :3] = z
    return X


def h_transform_X(T, X):
    return frame_to_X(h_transform(T, np.array(X_to_frame(X))))

def h_transform_pose(T, pose):
    return X_to_pose7(frame_to_X(h_transform(T, np.array(pose7_to_frame(pose)))))
    

def rotate_from_origin(pts, matrix):
    center = pts.mean(axis=0, keepdims=True) 
    pts -= center
    pts = r_transform(matrix, pts)
    pts += center
    return pts
    
