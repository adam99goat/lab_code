#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:17:41 2020

@author: zhouying
"""
import numpy as np
import cv2 as cv
import os
import tifffile as tfl

train_path='/Users/zhouying/Desktop/train'
gt_path='/Users/zhouying/Desktop/GT'
# path1='/Users/zhouying/Desktop/Left_Image.png'
# path2='/Users/zhouying/Desktop/Right_Image.png'
savepath="/Users/zhouying/Desktop"


e_nums=[1,2,3,4,5,6,7]
k_nums=[1,2,3,4,5]
disparity_table=np.zeros((7,5))
depth_table=np.zeros((7,5))
valid_px=np.zeros((7,5))
# over_numDisparities=np.zeros((7,5))
for e_num in e_nums:
    os.makedirs("/Users/zhouying/Desktop/original_algorithm/e"+str(e_num))
    for k_num in k_nums:
        if e_num in [1,2,3]:
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
            b=4.14353
            f=1035.033
        elif e_num in [4,5]:
            R=np.array([[ 9.99999344e-01, 9.92073183e-06, -1.13043236e-03],
                        [-9.90405715e-06, 1., 1.47561313e-05],
                       [1.13043259e-03, -1.47449255e-05, 9.99999344e-01]])          
            T=np.array([ -4.36337757e+00, 2.11443733e-02, -4.78740744e-02 ])
            cameraMatrix1=np.array([ [1.07364844e+03, 0., 5.77499695e+02],
                                     [0., 1.07343433e+03, 5.24409790e+02], 
                                     [0., 0., 1. ]])
            distCoeffs1=np.array([ -7.47532817e-04, 1.77790504e-03, 1.39573240e-04, 0., 0. ])
            cameraMatrix2=np.array([[ 1.07266870e+03, 0., 6.76994690e+02],
                                    [0., 1.07244336e+03, 5.23896667e+02], 
                                    [0., 0., 1. ]])
            distCoeffs2=np.array([ -1.31973054e-03, 3.40759335e-03, 1.06380983e-04, 0., 0. ])
            b=np.abs(T[0])
            f=cameraMatrix1[0,0]
        else:
            R=np.array([[ 9.99999285e-01, -2.10470330e-06, -1.18983735e-03],
                        [2.19584786e-06, 1., 7.66011581e-05],
                       [1.18983723e-03, -7.66037119e-05, 9.99999285e-01]])          
            T=np.array([ -4.36411285e+00, 2.14479752e-02, -3.81391775e-03 ])
            cameraMatrix1=np.array([ [1.08697437e+03, 0., 5.86080322e+02],
                                     [0., 1.08676831e+03, 5.12475891e+02], 
                                     [0., 0., 1. ]])
            distCoeffs1=np.array([ -1.59616291e-03, 4.46009031e-03, 1.10806825e-04, 0., 0. ])
            cameraMatrix2=np.array([[ 1.08695398e+03, 0., 6.86717102e+02],
                                    [0., 1.08677051e+03, 5.11879486e+02], 
                                    [0., 0., 1. ]])
            distCoeffs2=np.array([ -1.11811305e-03, 3.16335540e-03, 8.72377423e-05, 0., 0. ])
            b=4.36414
            f=1086.769
        imageSize=(1280,1024)
        
        I1=cv.imread(train_path+'/d'+str(e_num)+'/k'+str(k_num)+'/Left_Image.png')
        I2=cv.imread(train_path+'/d'+str(e_num)+'/k'+str(k_num)+'/Right_Image.png')

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
        
        gt_original=tfl.imread(gt_path+'/d'+str(e_num)+'/k'+str(k_num)+'/left_depth_map.tiff')
        gt3=gt_original
        gt_rectified=gt3

        imgL = cv.cvtColor(I1_rectified, cv.COLOR_BGR2GRAY)
        imgR = cv.cvtColor(I2_rectified, cv.COLOR_BGR2GRAY)

        os.makedirs("/Users/zhouying/Desktop/evaluate/e"+str(e_num)+'/k'+str(k_num))

        cv.imwrite("/Users/zhouying/Desktop/evaluate/e"+str(e_num)+'/k'+str(k_num)+'/img_'+str(k_num-1)+'L.png', I1_rectified)
        cv.imwrite("/Users/zhouying/Desktop/evaluate/e"+str(e_num)+'/k'+str(k_num)+'/img_'+str(k_num-1)+'R.png', I2_rectified)
        cv.imwrite("/Users/zhouying/Desktop/original_algorithm/e"+str(e_num)+'/img_'+str(k_num-1)+'L.png', I1_rectified)

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
        
        disparity[disparity < 0] = np.nan
        valid_px[e_num-1,k_num-1] = np.sum(disparity > 0)/(1024*1280)
        
        gt_dp = b * f / gt_rectified
        gt_dp[gt_dp <= 0] = np.nan
        gt_dp[gt_dp == np.inf] = np.nan
        gt_dp[gt_dp > numDisparities] = np.nan
        gt_rectified[gt_dp > numDisparities] = np.nan
        depth = b * f / disparity
        
        SA_error=np.abs(depth[:,numDisparities:]-gt_rectified[:,numDisparities:])
        SAmask1 = np.isnan(SA_error)
        SAmask = (1-SAmask1).astype(np.bool)
        SA_error=sum(SA_error[SAmask])
        SA_error=SA_error/(np.sum(SAmask))
        depth_table[e_num-1,k_num-1] = SA_error
        
        AS_error=np.abs(disparity[:,numDisparities:]-gt_dp[:,numDisparities:])
        ASmask1 = np.isnan(AS_error)
        ASmask = (1-ASmask1).astype(np.bool)
        AS_error=sum(AS_error[ASmask])
        AS_error=AS_error/(np.sum(ASmask))
        disparity_table[e_num-1,k_num-1] = AS_error
        
        # valid_mask = SAmask.astype(np.int) * ASmask.astype(np.int)
        # valid_mask = valid_mask.astype(np.bool)
        # valid_px[e_num-1,k_num-1] = np.sum(valid_mask)/(1024*1280)
        
        cv.imwrite("/Users/zhouying/Desktop/original_algorithm/e"+str(e_num)+'/left_depth_map'+str(k_num-1)+'.tiff', depth)
        cv.imwrite("/Users/zhouying/Desktop/original_algorithm/e"+str(e_num)+'/left_disparity_map'+str(k_num-1)+'.tiff', disparity)
        dp = cv.normalize(disparity[0:, numDisparities:], disparity[0:, numDisparities:], 
                          alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        cv.imwrite("/Users/zhouying/Desktop/original_algorithm/e"+str(e_num)+'/img_'+str(k_num-1)+'.png', dp)