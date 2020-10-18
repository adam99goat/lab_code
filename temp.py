# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cv2 as cv

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


path1='/Users/zhouying/Desktop/Left_Image.png'
path2='/Users/zhouying/Desktop/Right_Image.png'
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

imgL = cv.cvtColor(I1_rectified, cv.COLOR_BGR2GRAY)
imgR = cv.cvtColor(I2_rectified, cv.COLOR_BGR2GRAY)

 
cv.imwrite("/Users/zhouying/Desktop/I1_rectified.png", I1_rectified)
cv.imwrite("/Users/zhouying/Desktop/I2_rectified.png", I2_rectified)

stereo = cv.StereoSGBM_create(minDisparity = 0, numDisparities = 16, 
                              blockSize=5, P1 = 40,
                              P2= 160)
disparity = stereo.compute(imgL, imgR)
cv.imwrite("/Users/zhouying/Desktop/disparity.png", disparity)

