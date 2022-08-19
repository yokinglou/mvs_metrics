import torch
import open3d as o3d
import numpy as np
import re
from typing import Dict, List, Tuple

def read_pfm(filename: str) -> Tuple[np.ndarray, float]:
    """Read a depth map from a .pfm file

    Args:
        filename: .pfm file path string

    Returns:
        data: array of shape (H, W, C) representing loaded depth map
        scale: float to recover actual depth map pixel values
    """
    file = open(filename, "rb")  # treat as binary and read-only

    header = file.readline().decode("utf-8").rstrip()
    if header == "PF":
        color = True
    elif header == "Pf": # depth is Pf
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("utf-8"))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width, 1)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def read_ply(path):
    return o3d.io.read_point_cloud(path, format="ply")

def chamfer_distances(src, tar):
    """
    Input:
        src: source point cloud [B, N1, 3]
        tar: target point cloud [B, N2, 3]
    Return: 
        dists: [B, N1]
        mean_dists: [B,]
    """
    idx = knn_query(tar, src, 1)
    nearest = index_points(tar, idx)
    
    dists = (nearest - src).norm(dim=-1)
    mean_dists = dists.mean(dim=-1)
    return dists, mean_dists

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def knn_query(xyz, new_xyz, k):
    """
    Input:
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
        k: number of neighbours
    Return:
        knn.indices: grouped points index, [B, S, k]
    """
    dist = square_distance(new_xyz, xyz)
    knn = dist.topk(k, largest=False)

    return knn.indices

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

