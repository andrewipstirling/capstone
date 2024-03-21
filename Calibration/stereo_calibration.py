import numpy as np
import cv2 as cv
import glob
import itertools

# List of camera IDs, should match folder names within stereo_images folder
cams = [1, 2, 3, 4, 5]

patternSize = (8, 6)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Intrinsic camera matrix and distortion coefficients used for both cameras
# Code can be edited to accept individual matrices for each camera, or to perform and return new intrinsic calibrations
cam_mat = cv.Mat(np.array([[1.56842921e+03, 0, 2.89275503e+02], 
                                [0, 1.57214434e+03, 2.21092150e+02], 
                                [0, 0, 1]]))
dist_coeffs = cv.Mat(np.array([[ 2.28769970e-02, -4.54632281e+00, -3.04424079e-03, -2.06207084e-03, 9.30400565e+01]]))
 
# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((patternSize[0]*patternSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:patternSize[0],0:patternSize[1]].T.reshape(-1,2)

# Returns stereo calibration of cam2 relative to cam1, where cam1 and cam2 are camera IDs present in the cams list
def stereoCalibration(cam1, cam2):
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints1 = [] # 2d points in image plane.
    imgpoints2 = [] # 2d points in image plane.
    
    images1 = sorted(glob.glob(f'stereo_images/{cam1}/*.jpg'))
    images2 = sorted(glob.glob(f'stereo_images/{cam2}/*.jpg'))
    
    for fname1, fname2 in zip(images1, images2):
        img1 = cv.imread(fname1)
        img2 = cv.imread(fname2)
        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    
        # Find the chess board corners
        ret1, corners1 = cv.findChessboardCorners(gray1, patternSize, None)
        ret2, corners2 = cv.findChessboardCorners(gray2, patternSize, None)
    
        # If found, add object points, image points (after refining them)
        if ret1 and ret2:
            objpoints.append(objp)
    
            refcorners1 = cv.cornerSubPix(gray1,corners1, (11,11), (-1,-1), criteria)
            refcorners2 = cv.cornerSubPix(gray2,corners2, (11,11), (-1,-1), criteria)
            
            imgpoints1.append(refcorners1)
            imgpoints2.append(refcorners2)
    
            # Draw and display the corners
            cv.drawChessboardCorners(img1, patternSize, corners2, ret1)
            cv.drawChessboardCorners(img2, patternSize, corners2, ret2)
            cv.imshow('img', cv.hconcat([img1, img2]))
            cv.waitKey(500)
            
        else: print('bad')
    
    cv.destroyAllWindows()

    _, _, _, _, _, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints1, imgpoints2, cam_mat, dist_coeffs, cam_mat, dist_coeffs, gray1.shape[::-1])
    print(f'R: {R}\n\
            T: {T}\n\
            E: {E}\n\
            F: {F}')
    
    return R, T, E, F

for cam1, cam2 in itertools.combinations(cams, 2):
    R, T, E, F = stereoCalibration(cam1, cam2)