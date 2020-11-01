#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 20:12:09 2020

@author: zhouying
"""

import numpy as np
import cv2 as cv
import tifffile as tfl

R=np.array([[ 1., 1.94856493e-05, -1.52324792e-04],
            [-1.95053162e-05, 1.,-1.29114138e-04],
           [1.52322275e-04, 1.29117107e-04, 1. ]])          
T=np.array([ -4.14339018e+00, -2.38197036e-02, -1.90685259e-03 ])
cameraMatrix1=np.array([ [1.03530811e+03, 0., 5.96955017e+02],
                         [0., 1.03508765e+03,5.20410034e+02], 
                         [0., 0., 1. ]])
distCoeffs1=np.array([ -5.95157442e-04, -5.46629308e-04, 0., 0., 1.82959007e-03 ])
cameraMatrix2=np.array([[ 1.03517419e+03, 0., 6.88361877e+02],
                        [0., 1.03497900e+03,5.21070801e+02], 
                        [0., 0., 1. ]])
distCoeffs2=np.array([ -2.34280655e-04, -7.68933969e-04, 0., 0., 7.76395318e-04 ])
imageSize=(1280,1024)
# b=np.abs(T[0])
b=4.14353
# f=cameraMatrix1[0,0]
f=1035.033


path1='/Users/zhouying/Desktop/三维重建/DATASET/train/d3/k3/Left_Image.png'
path2='/Users/zhouying/Desktop/三维重建/DATASET/train/d3/k3/Right_Image.png'
savepath="/Users/zhouying/Desktop"

I1=cv.imread(path1)
I2=cv.imread(path2)


R1, R2, P1, P2, Q, validPixROI1, validPixROI2=\
    cv.stereoRectify(cameraMatrix1,distCoeffs1,
                     cameraMatrix2,distCoeffs2,imageSize,R,T,
                     flags=cv.CALIB_ZERO_DISPARITY, alpha=-1, newImageSize=(0,0))
left_map1, left_map2=cv.initUndistortRectifyMap(cameraMatrix1,
                                                distCoeffs1,R1,P1,imageSize,cv.CV_16SC2)
right_map1, right_map2=cv.initUndistortRectifyMap(cameraMatrix2,
                                                  distCoeffs2,R2,P2,imageSize,cv.CV_16SC2)




I1_rectified=cv.remap(I1,left_map1,left_map2,cv.INTER_LINEAR)
I2_rectified=cv.remap(I2,right_map1,right_map2,cv.INTER_LINEAR)





minDisparity = 0
numDisparities = 192
SADWindowSize = 3
P1 = 8 * 3 * SADWindowSize ** 2
# P1 = 10
P2 = 32 * 3 * SADWindowSize ** 2
# P2 = 200
disp12MaxDiff = 10
preFilterCap = 0
uniquenessRatio = 1
speckleWindowSize = 100
speckleRange = 10









gt_original=tfl.imread('/Users/zhouying/Desktop/三维重建/DATASET/train/d3/k3/left_depth_map.tiff')
gt3=gt_original[:,:,2]
gt_rectified=cv.remap(gt3,left_map1,left_map2,cv.INTER_LINEAR)
gt_norm = cv.normalize(gt_rectified[0:,numDisparities:], gt_rectified[0:,numDisparities:],
                        alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
cv.imwrite("/Users/zhouying/Desktop/gt_norm.png", gt_norm)











imgL = cv.cvtColor(I1_rectified, cv.COLOR_BGR2GRAY)
imgR = cv.cvtColor(I2_rectified, cv.COLOR_BGR2GRAY)

 
cv.imwrite("/Users/zhouying/Desktop/I1_rectified.png", I1_rectified)
cv.imwrite("/Users/zhouying/Desktop/I2_rectified.png", I2_rectified)


stereo = cv.StereoSGBM_create(minDisparity = minDisparity, 
                              numDisparities = numDisparities, 
                              blockSize = SADWindowSize, P1 = P1,
                              P2 = P2, disp12MaxDiff = disp12MaxDiff, 
                              preFilterCap = preFilterCap,
                              uniquenessRatio = uniquenessRatio, 
                              speckleWindowSize = speckleWindowSize, 
                              speckleRange = speckleRange,
                              mode = cv.StereoSGBM_MODE_HH)
disparity = stereo.compute(imgL, imgR).astype(np.float32)/16

gt_dp = b * f / gt_rectified

depth = b * f / disparity
depth[depth < 0] = np.nan




SA_error=np.abs(depth[:,numDisparities:]-gt_rectified[:,numDisparities:])
mask1 = np.isnan(SA_error)
mask = (1-mask1).astype(np.bool)
SA_error=sum(SA_error[mask])
SA_error=SA_error/(np.sum(mask))
print(SA_error)





cv.imwrite("/Users/zhouying/Desktop/disparity.tiff", disparity)
dp = cv.normalize(disparity[0:, numDisparities:], disparity[0:, numDisparities:], 
                  alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
# disp = np.concatenate((np.zeros(192,1024, dtype = np.uint8),dp), axis = 0)
cv.imwrite("/Users/zhouying/Desktop/disp.png", dp)


