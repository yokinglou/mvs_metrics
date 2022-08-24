import numpy as np
import open3d as o3d

from utils import *
import cv2 
from skimage.metrics import structural_similarity

def single_view_depth_evaluation(img1, img2):
    """
    Evaluation for depth image.
    Input:
        img1: depth image, np.array [M, N]
        img2: depth image, np.array [M, N]
    Return:
        _mse: mean square error
        _rmse: root mean square error
        _psnr: peak signal-to-noise ratio
        _ssim: structural similarity index measure
        _epe: end point error
    """
    _mse = mse(img1,img2)
    _rmse = rmse(img1,img2)
    _psnr = psnr(img1,img2)
    _ssim = ssim(img1,img2)
    _epe = epe(img1,img2)
    return [_mse, _rmse, _psnr, _ssim, _epe]

# metrics for depth image 
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




    



