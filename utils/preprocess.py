import os
import shutil
import natsort
import re
import operator
import SimpleITK as sitk
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

from skimage import transform

def calc_min_cube(gt_np, elasticity=0):
    nonzero_coordinate = np.nonzero(gt_np == 1)
    nonzero_coordinate = list(zip(*nonzero_coordinate))
    # print(nonzero_coordinate)
    nonz_coor = np.array(nonzero_coordinate)
    t1, x1, y1 = np.min(nonz_coor, axis=0) - elasticity
    t2, x2, y2 = np.max(nonz_coor, axis=0) + 1 + elasticity

    return t1, t2, x1, x2, y1, y2

def resize_64(img):
    t, h, w = img.shape[:]
    k1 = float(64 / h)
    k2 = float(64 / w)
    k3 = float(64 / t)
    k = min(k1, k2, k3)
    k=round(k,2)
    rs_img=transform.rescale(img,k)
    print(rs_img.shape)

    n_t,n_h,n_w=rs_img.shape

    rs_img=torch.from_numpy(rs_img)
    uni_th = torch.zeros((64, 64, 64), dtype=torch.float32)
    s1 = ceil((63 - n_t) / 2)
    s2 = ceil((63 - n_h) / 2)
    s3 = ceil((63 - n_w) / 2)

    uni_th[s1:s1 + n_t, s2:s2 + n_h, s3:s3 + n_w] = rs_img

    return uni_th

def crop_resize(src,dst,di_k=0):
    # mris=[mri for mri in os.listdir(src) if 'resamp_' not in mri]
    mris = [img for img in os.listdir(src) if 'gt' not in img]
    mris = natsort.natsorted(mris)

    gts = [img for img in os.listdir(src) if 'gt' in img]
    gts = natsort.natsorted(gts)

    print(len(mris))
    cnt = 0

    tr_path = dst + '/resize_64_tr/'
    itk_path = dst + '/resize_64_itk/'
    os.makedirs(tr_path, exist_ok=True)
    os.makedirs(itk_path, exist_ok=True)

    for i in range(0, len(mris)):
        mri_p = mris[i]
        gt_p = gts[i]

        # mri_tr_p=mri_p.split('.')[0]+'.pt'

        gt_itk = sitk.ReadImage(src + gt_p)
        gt_itk=sitk.BinaryDilate(gt_itk,di_k)

        gt_np = sitk.GetArrayFromImage(gt_itk)
        mri_itk = sitk.ReadImage(src + mri_p, sitk.sitkFloat32)
        mri_np = sitk.GetArrayFromImage(mri_itk)

        # t_c, h_c, w_c = calc_center(gt_np)

        t1, t2, x1, x2, y1, y2 = calc_min_cube(gt_np)
       
        mri_c_np = mri_np[t1:t2, x1:x2, y1:y2]

        
        mri_rs_tr=resize_64(mri_c_np)
        mri_rs_tr.squeeze_(0)

        
        torch.save(mri_rs_tr, tr_path + mri_p.split('.')[0] + '.pt')

      
        mri_rs_tr.squeeze_(0)
        mri_rs_np = mri_rs_tr.numpy()
        #print(mri_rs_np.shape)
        mri_rs_itk = sitk.GetImageFromArray(mri_rs_np)
        sitk.WriteImage(mri_rs_itk, itk_path + mri_p)
    
def sobel(image):

    image_float = sitk.Cast(image, sitk.sitkFloat32)
    sobel_op = sitk.SobelEdgeDetectionImageFilter()
    sobel_sitk = sobel_op.Execute(image_float)
    sobel_sitk = sitk.Cast(sobel_sitk, sitk.sitkInt16)
    #sitk.WriteImage(sobel_sitk, "sobel_sitk.mha")
    sobel_np=sitk.GetArrayFromImage(sobel_sitk)
    sobel_tr=torch.from_numpy(sobel_np)
    return sobel_tr


def sobel_itk(image):

    image_float = sitk.Cast(image, sitk.sitkFloat32)
    sobel_op = sitk.SobelEdgeDetectionImageFilter()
    sobel_sitk = sobel_op.Execute(image_float)
    sobel_sitk = sitk.Cast(sobel_sitk, sitk.sitkInt16)
    #sitk.WriteImage(sobel_sitk, "sobel_sitk.mha")
    
    return sobel_sitk

def use_sobel(mri_types,data_base_path,sobel_base,itk_base):
    for mri_type in mri_types:
        sobel_p=data_base_path+mri_type+'/'+sobel_base
        print(sobel_p)
        if not os.path.exists(sobel_p):
            os.makedirs(sobel_p)
        itk_p=data_base_path+mri_type+'/'+itk_base
        print(itk_p)
        mris=os.listdir(itk_p)
        for mri in mris:
            img=sitk.ReadImage(itk_p+mri)
            sobel=sobel_itk(img)
            sitk.WriteImage(sobel,sobel_p+mri)
