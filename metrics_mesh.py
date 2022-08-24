import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree as KDTree

def mesh_evaluation(src_path, 
                    tar_path, 
                    npoint=100000,
                    mnsd_k=100,
                    mnsd_thr=0.8,
                    overlap_thr=0.4,
                    tau=0.5):
    ''' 
    Input:
        src_path: src mesh file path
        tar_path: tar mesh file path
        mnsd_k: number of nerighbor points for overlap area search
        mnsd_thr: the threshold for overlapped points
        overlap_thr: the overlap areas ratio threshold
        tau: threshold for FScore
        npoint: number of points for mesh sampling
    Return:
        rmsd: mean distance from source query point to target nearest point
        mnsd: mean distance from overlap areas
        cd: chamfer distance
        fscore: FScore
        ncs: normal consistency score
    '''
    # load mesh
    src_mesh = trimesh.load(src_path)
    tar_mesh = trimesh.load(tar_path)
    # sample points from mesh uniformly and estimate the normals
    src_p, src_n = sample_normal(src_mesh, npoint)
    tar_p, tar_n = sample_normal(tar_mesh, npoint)
    # normalize normals
    src_n = src_n / np.linalg.norm(src_n, axis=-1, keepdims=True) 
    tar_n = tar_n / np.linalg.norm(tar_n, axis=-1, keepdims=True)
    # build KDTree for meshes
    src_tree = KDTree(src_p)
    tar_tree = KDTree(tar_p)

    # calculate rmsd, cd, fscore
    s_dist, s_idx = tar_tree.query(src_p, p=2, k=1, workers=8)
    t_dist, t_idx = src_tree.query(tar_p, p=2, k=1, workers=8)
    precision = np.sum(s_dist < tau).astype(float) / s_dist.shape[0]
    recall = np.sum(t_dist < tau).astype(float) / t_dist.shape[0]

    rmsd = np.sqrt(np.mean(s_dist**2))
    cd = 0.5*np.mean(s_dist) + 0.5*np.mean(t_dist)
    fscore = 2*precision*recall / (precision+recall) * 100

    # calculate mnsd
    _, src_nn_idx = src_tree.query(src_p, mnsd_k) # (N, K)
    src_nn = src_p[src_nn_idx] # (N, K, 3)
    src_nn_dist = s_dist[src_nn_idx]
    overlap = np.sum(src_nn_dist < mnsd_thr, axis=-1).astype(float) / mnsd_k
    mask = overlap > overlap_thr
    masked_nn_dist = src_nn_dist[mask]
    masked_nn_dist[masked_nn_dist>=mnsd_thr] = 0
    mnsd = masked_nn_dist.mean()    
    
    # calculate ncs
    s_normal_dot = (tar_n[s_idx] * src_n).sum(axis=-1)
    t_normal_dot = (src_n[t_idx] * tar_n).sum(axis=-1)

    ncs = 0.5*np.mean(np.abs(s_normal_dot)) + 0.5*np.mean(np.abs(t_normal_dot))

    print ("Root Mean Squared Distance:\t{}".format(rmsd))
    print ("Mean Neighbor Surface Distance:\t{}".format(mnsd))
    print ("Chamfer Distance:\t\t{}".format(cd))
    print ("FScore:\t\t\t\t{}".format(fscore))
    print ("Normal Consistency Score:\t{}".format(ncs))
    return rmsd, mnsd, cd, fscore, ncs

# single metric for mesh 
def chamfer_distance(src_ply, tar_ply):
    """
    Calculate the chamfer distance berween src and tar point cloud.
    For mesh, first uniformly sample points from surface by mesh.sample_points_uniformly().
    Input:
        src_ply: source open3d point cloud
        tar_ply: target open3d point cloud
    Return:
        dist: chamfer distance
    """
    s_dists = src_ply.compute_point_cloud_distance(tar_ply)
    s_dists = np.asarray(s_dists)
    s_dist = np.mean(s_dists)

    t_dists = tar_ply.compute_point_cloud_distance(src_ply)
    t_dists = np.asarray(t_dists)
    t_dist = np.mean(t_dists)

    return 0.5 * s_dist + 0.5 * t_dist

def FScore(src_ply, tar_ply, tau=0.5):
    """
    Calculate the fscore berween src and tar point cloud
    For mesh, first uniformly sample points from surface by mesh.sample_points_uniformly().
    Input:
        src_ply: source open3d point cloud
        tar_ply: target open3d point cloud
    Return:
        fscore
    """
    # precision
    s_dists = src_ply.compute_point_cloud_distance(tar_ply)
    s_dists = np.asarray(s_dists)
    precision = np.sum(s_dists < tau).astype(float) / s_dists.shape[0]
    
    # recall
    t_dists = tar_ply.compute_point_cloud_distance(src_ply)
    t_dists = np.asarray(t_dists)
    recall = np.sum(t_dists < tau).astype(float) / t_dists.shape[0]
    
    return 2*precision*recall / (precision+recall) * 100

def neural_feature_similarity(src, tar):
    """
    Calculate feature similarity between two point cloud.
    Given features have to be matched.
    Input:
        src: source features [N, C]
        tar: target features [N, C]
    """
    feats_dot = (src * tar).sum(axis=-1)
    src_norm = np.linalg.norm(src, axis=-1)
    tar_norm = np.linalg.norm(tar, axis=-1)
    return np.mean(feats_dot / (src_norm * tar_norm))

def sample_normal(mesh, npoint=1024):
    ''' 
    Sample a mesh, return points with normals.
    Input:
        mesh_path: input mesh file
        npoint: numbers of points to be sampled
    Return:
        sample_random: sampled points
        sample_normal: sampled normals
    '''
    mesh.fix_normals()  
    sample_random, index_random = trimesh.sample.sample_surface(mesh, npoint)
    sample_normal = mesh.face_normals[index_random]
    return sample_random, sample_normal


if __name__ == '__main__':
    mesh_path = 'data/640_512/mesh.ply'
    mesh_evaluation(mesh_path, mesh_path)