import numpy as np
import open3d as o3d
from utils import *
import cv2 
from skimage.metrics import structural_similarity

def single_view_depth_evaluation(img1, img2):
    _mse = mse(img1,img2)
    _rmse = rmse(img1,img2)
    _psnr = psnr(img1,img2)
    _ssim = ssim(img1,img2)
    _epe = epe(img1,img2)
    return [_mse, _rmse, _psnr, _ssim, _epe]

def single_view_pc_evaluation(R_ply, G_ply):
    acc_d, com_d, overall = distance_metrics(R_ply, G_ply)
    acc_p, com_p, fscore = percentage_metrics(R_ply, G_ply)
    norm = normal_eval(R_ply, G_ply)
    return [acc_d, com_d, overall, acc_p, com_p, fscore, norm]

def mse(img1, img2):
    """
    Calculates the mean square error (MSE) between two images
    """
    return np.square(np.subtract(img1,img2)).mean()

def rmse(img1, img2):
    """
    Calculates the root mean square error (RSME) between two images
    """
    return np.sqrt(mse(img1, img2))

def psnr(img1, img2):
    """
    Calculates the peak signal-to-noise ratio (PSNR) between two images
    """
    return cv2.PSNR(img1, img2, R=700)

def ssim(img1, img2):    
    """
    Calculates the structural similarity index measure (SSIM) between two images
    """
    return structural_similarity(img1, img2)

def epe(img1, img2):
    """
    Calculates the end point error (EPE) between two images
    """
    return (np.abs(np.subtract(img1, img2))).mean()

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
    """
    src = normal_estimation(src)
    tar = normal_estimation(tar)
    tar_tree = o3d.geometry.KDTreeFlann(tar)
    errs = []
    for i in range(len(src.points)):
        [k, idx, _] = tar_tree.search_knn_vector_3d(np.asarray(src.points[i]), 1)
        diff = np.asarray(src.normals)[i] - np.asarray(tar.normals)[idx]
        err = np.sqrt(np.sum(diff**2))
        errs.append(err)
    
    errs = np.array(errs)
    norm = np.mean(errs)
    return norm
    



