import pose_estimation
import cv2

cap = cv2.VideoCapture("udpsrc address=192.168.5.2 port=5000 ! application/x-rtp, clock-rate=90000, payload=96 ! rtph264depay ! h264parse ! avdec_h264 discard-corrupted-frames=true skip-frame=1 ! videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=1 drop=true sync=false", cv2.CAP_GSTREAMER)

poseEstimator = pose_estimation.pose_estimation(framerate=60)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(poseEstimator.aruco_dict, arucoParams)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()  # ret is True if frame is read correctly
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    corners, ids, rejected = detector.detectMarkers(frame)
    rel_trans, rel_rot = poseEstimator.estimate_pose_marker(corners,ids, 0)
    print(f'Translation: {rel_trans}, Rotation: {rel_rot}')
    
    overlayImg = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.imshow('frame', overlayImg)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


"""
cap = cv2.capture

while (True):
    
    image = cap.read()
    corners, ids = cv2.aruco.findmarkers()
    rel_trans, rel_rot = poseEstimator.estimate_pose_marker(corners,ids, 0)
    
    
    

"""


