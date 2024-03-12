import pose_estimation
import cv2
import numpy as np

cap = cv2.VideoCapture("udpsrc address=192.168.5.2 port=5000 ! application/x-rtp, clock-rate=90000, payload=96 ! rtph264depay ! h264parse ! avdec_h264 discard-corrupted-frames=true skip-frame=1 ! videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=1 drop=true sync=false", cv2.CAP_GSTREAMER)

poseEstimator = pose_estimation.pose_estimation(framerate=60)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(poseEstimator.aruco_dict, arucoParams)

m = 33.2/2 # half of marker length (currently in mm)

board_points = np.array([[[-m, m, 0],[m, m, 0],[m, -m, 0],[-m, -m, 0]]],dtype=np.float32)

ref_board = cv2.aruco.Board(board_points, aruco_dict, np.array([0]))
target_board = cv2.aruco.Board(board_points, aruco_dict, np.array([1]))

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()  # ret is True if frame is read correctly
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    corners, ids, rejected = detector.detectMarkers(frame)
    rel_trans, rel_rot, std_dev = poseEstimator.estimate_pose_board(ref_board, target_board, corners, ids)
    print(f'Translation: {rel_trans}, Rotation: {rel_rot}')
    
    overlayImg = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.imshow('frame', overlayImg)
    if cv2.pollKey() == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()