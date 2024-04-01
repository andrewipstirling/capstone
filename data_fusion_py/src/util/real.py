import pose_estimation
import cv2
import numpy as np
import dodecaBoard

PLOTTING = True
  
cap = cv2.VideoCapture("udpsrc address=192.168.5.2 port=5000 ! application/x-rtp, clock-rate=90000, payload=96 ! rtph264depay ! h264parse ! avdec_h264 discard-corrupted-frames=true skip-frame=1 ! videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=1 drop=true sync=false", cv2.CAP_GSTREAMER)

poseEstimator = pose_estimation.pose_estimation(framerate=60, plotting=PLOTTING)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
arucoParams = cv2.aruco.DetectorParameters()
arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
arucoParams.cornerRefinementMaxIterations = 100
arucoParams.cornerRefinementMinAccuracy = 0.01
detector = cv2.aruco.ArucoDetector(poseEstimator.aruco_dict, arucoParams)

m = 33.2/2 # half of marker length (currently in mm)

# Single marker board
board_points = np.array([[[-m, m, 0],[m, m, 0],[m, -m, 0],[-m, -m, 0]]],dtype=np.float32)

ref_board = cv2.aruco.Board(board_points, aruco_dict, np.array([0]))
target_board = cv2.aruco.Board(board_points, aruco_dict, np.array([1]))

# Dodecahedron board
# dodecaLength = 40  # dodecahedron edge length in mm
# dodecaPoints = dodecaBoard.generate(dodecaLength)
# ref_board = cv2.aruco.Board(dodecaPoints, aruco_dict, np.arange(11))
# target_board = cv2.aruco.Board(dodecaPoints, aruco_dict, np.arange(11,22))

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()  # ret is True if frame is read correctly
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    corners, ids, rejected = detector.detectMarkers(frame)

    rel_trans, rel_rot, std_dev, ref_tvec = poseEstimator.estimate_pose_board(ref_board, target_board, corners, ids)
    # print(f'Translation: {rel_trans}, Rotation: {rel_rot}')
    if rel_trans is not None:
        print(f'X: {rel_trans[0]}, Y: {rel_trans[1]}, Z: {rel_trans[2]}', end='\r')
    
    overlayImg = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    
    if ids is not None:
        target_obj_pts, target_img_pts = target_board.matchImagePoints(corners,ids)
        target_val, target_rvec, target_tvec = cv2.solvePnP(target_obj_pts,target_img_pts,poseEstimator.cv_cam_mat,poseEstimator.cv_dist_coeffs,
                                                                        rvec=None,tvec=None,useExtrinsicGuess=False,flags=cv2.SOLVEPNP_ITERATIVE)
        overlayImg = cv2.drawFrameAxes(overlayImg, poseEstimator.cv_cam_mat, poseEstimator.cv_dist_coeffs, target_rvec, target_tvec, 50)
    
    cv2.imshow('frame', overlayImg)
    if cv2.pollKey() == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if PLOTTING:
    # poseEstimator.plot(trueTrans=[-155.2, 0, 0], trueRot=[0, 0, 0])
    poseEstimator.plot()
    avgPos = np.average(poseEstimator.total_distance, axis=0)
    print(f'Avg X: {avgPos[0]}, Y: {avgPos[1]}, Z: {avgPos[2]}')
