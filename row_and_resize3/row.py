#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 16:39:00 2020

@author: zhouying
"""

import numpy as np
import cv2 as cv
import os

savepath = '/Users/zhouying/Desktop/row_process'
trainpath = '/Users/zhouying/Desktop/row'
def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(('.png')):
                imagelist.append(os.path.join(parent, filename))
        return imagelist
   
file = get_img_file(trainpath)
file.sort()

for i in range(int(len(file)/2)):
    path1 = file[2*i]
    path2 = file[2*i+1]
    
    I1=cv.imread(path1)
    I2=cv.imread(path2)
    
    minDisparity = 0
    numDisparities = 64
    SADWindowSize = 3
    P1 = 8 * 3 * SADWindowSize ** 2
    P2 = 32 * 3 * SADWindowSize ** 2
    disp12MaxDiff = 10
    preFilterCap = 0
    uniquenessRatio = 1
    speckleWindowSize = 100
    speckleRange = 10
    
    imgL = cv.cvtColor(I1, cv.COLOR_BGR2GRAY)
    imgR = cv.cvtColor(I2, cv.COLOR_BGR2GRAY)
    
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
    
    cv.imwrite(savepath+'/left_disparity_map'+file[2*i][-11:-5]+'.tiff', disparity)
    disparity[disparity == np.nan] = 0
    dp = cv.normalize(disparity, disparity, 
                      alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    cv.imwrite(savepath+'/imgL_'+file[2*i][-11:-5]+'.png', dp)






# import numpy as np
# import cv2 as cv
# import os

# savepath = '/Users/zhouying/Desktop/row_process'
# trainpath = '/Users/zhouying/Desktop/row'
# def get_img_file(file_name):
#     imagelist = []
#     for parent, dirnames, filenames in os.walk(file_name):
#         for filename in filenames:
#             if filename.lower().endswith(('.png')):
#                 imagelist.append(os.path.join(parent, filename))
#         return imagelist
   
# file = get_img_file(trainpath)
# file.sort()

# for i in range(int(len(file)/2)):
#     path1 = file[2*i]
#     path2 = file[2*i+1]
    
#     I1=cv.imread(path1)
#     I2=cv.imread(path2)
    
#     minDisparity = 0
#     numDisparities = 64
#     SADWindowSize = 3
#     P1 = 8 * 3 * SADWindowSize ** 2
#     P2 = 32 * 3 * SADWindowSize ** 2
#     disp12MaxDiff = 10
#     preFilterCap = 0
#     uniquenessRatio = 1
#     speckleWindowSize = 100
#     speckleRange = 10
    
#     imgR = cv.cvtColor(I1, cv.COLOR_BGR2GRAY)
#     imgL = cv.cvtColor(I2, cv.COLOR_BGR2GRAY)
    
#     stereo = cv.StereoSGBM_create(minDisparity = -numDisparities, 
#                                   numDisparities = numDisparities, 
#                                   blockSize = SADWindowSize, P1 = P1,
#                                   P2 = P2, disp12MaxDiff = disp12MaxDiff, 
#                                   preFilterCap = preFilterCap,
#                                   uniquenessRatio = uniquenessRatio, 
#                                   speckleWindowSize = speckleWindowSize, 
#                                   speckleRange = speckleRange,
#                                   mode = cv.StereoSGBM_MODE_HH)
#     disparity = stereo.compute(imgL, imgR).astype(np.float32)/16
#     disparity = abs(disparity)
    
#     disparity[disparity > numDisparities] = np.nan
    
#     cv.imwrite(savepath+'/right_disparity_map'+file[2*i][-11:-5]+'.tiff', disparity)
#     disparity[disparity == np.nan] = 0
#     dp = cv.normalize(disparity, disparity, 
#                       alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
#     cv.imwrite(savepath+'/imgR_'+file[2*i][-11:-5]+'.png', dp)