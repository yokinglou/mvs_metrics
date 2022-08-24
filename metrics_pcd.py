import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree as KDTree


def single_view_pc_evaluation(R_ply, G_ply):
    """
    Evaluation for single view point cloud.
    Input:
        R_ply: open3d reconstructed point cloud
        G_ply: open3d GT point cloud
    Return:
        acc_d, com_d, overall: distance based metrics
        acc_p, com_p, fscore: percentage based metrics
        norm: 2-norm between normals
    """
    acc_d, com_d, overall = distance_metrics(R_ply, G_ply)
    acc_p, com_p, fscore = percentage_metrics(R_ply, G_ply)
    norm = normal_eval(R_ply, G_ply)
    return [acc_d, com_d, overall, acc_p, com_p, fscore, norm]

# metrics for point cloud
def point_cloud_dist(src_ply, tar_ply, max_dist=1e5):
    """
    Calculate the closest distance from src to tar
    """
    dists = src_ply.compute_point_cloud_distance(tar_ply)
    dists = np.asarray(dists)
    dists = np.clip(dists, 0, 60)
    dists = dists[dists <= max_dist]
    return dists, np.mean(dists)

def distance_metrics(R, G):
    """
    Calculate the distance-based accuracy, completeness, overall score
    """
    _, acc = point_cloud_dist(R, G)
    _, com = point_cloud_dist(G, R)
    overall = (acc + com)/2
    return acc, com, overall

def percentage_metrics(R, G, max_dist=5):
    """
    Calculate the percentage-based accuracy, completeness, overall score
    """
    R_dists, _ = point_cloud_dist(R, G)
    G_dists, _ = point_cloud_dist(G, R)

    acc = np.sum((R_dists < max_dist)) * 100. / R_dists.shape[0]
    com = np.sum((G_dists < max_dist)) * 100. / G_dists.shape[0]

    fscore = 2 * acc * com / (acc + com)
    return acc, com, fscore

def normal_estimation(pcd):
    """
    Estimate the normals of the point cloud
    """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10))
    # pcd.orient_normals_towards_camera_location()
    # pcd.normals = o3d.utility.Vector3dVector(-1. * np.asarray(pcd.normals))
    # pcd.orient_normals_consistent_tangent_plane(20)
    pcd.normalize_normals()
    return pcd

def normal_eval(src, tar):
    """
    Find the nearest point in target point cloud and calculate the 2-norm between normals
    Input:
        src: open3d point cloud
        tar: open3d point cloud
    Return:
        norm: 2-norm between normals
    """
    src = normal_estimation(src)
    tar = normal_estimation(tar)

    src_p = np.asarray(src.points)
    src_n = np.asarray(src.normals)
    tar_p = np.asarray(tar.points)
    tar_n = np.asarray(src.normals)

    tar_tree = KDTree(tar_p)
    s_dist, s_idx = tar_tree.query(src_p, p=2, k=1, workers=8)
    matched_n = tar_n[s_idx]

    # 2-norm
    norm = np.sqrt(np.sum((src_n - matched_n) ** 2, axis=-1)).mean()
    return norm
