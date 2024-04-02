import kalman_filter.dodecaBoard as dodecaBoard
from kalman_filter.pose_estimation import pose_estimation
from kalman_filter.kalman import KalmanFilterCV
from scipy.spatial.transform import Rotation as R
import cv2
import numpy as np
import multiprocessing as mp
import time

### PRESS Q ON EACH WINDOW TO QUIT ###

ROS = True
# cams = [1,2,3,4,5] # Camera IDs that correspond to label on pi and port number 500X
cams = [1]
if ROS:
    import rospy
    from geometry_msgs.msg import Pose
    from gazebo_msgs.msg import ModelState

# caps = []
# for cam in cams[:]:
#     cap = cv2.VideoCapture(f"udpsrc address=192.168.5.2 port=500{cam} ! application/x-rtp, clock-rate=90000, payload=96 ! rtph264depay ! h264parse ! avdec_h264 discard-corrupted-frames=true skip-frame=1 ! videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=1 drop=true sync=false", cv2.CAP_GSTREAMER)
#     if not cap.isOpened():
#         print(f"Cannot open camera {cam}.")
#         cams.remove(cam)
#     else: caps.append(cap)
        

poseEstimator = pose_estimation(framerate=60, plotting=False)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
arucoParams = cv2.aruco.DetectorParameters()
arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
arucoParams.cornerRefinementMaxIterations = 100
arucoParams.cornerRefinementMinAccuracy = 0.01
detector = cv2.aruco.ArucoDetector(poseEstimator.aruco_dict, arucoParams)

# m = 33.2/2 # half of marker length (currently in mm)

# # Single marker board
# board_points = np.array([[[-m, m, 0],[m, m, 0],[m, -m, 0],[-m, -m, 0]]],dtype=np.float32)

# ref_board = cv2.aruco.Board(board_points, aruco_dict, np.array([0]))
# target_board = cv2.aruco.Board(board_points, aruco_dict, np.array([1]))

# Dodecahedron board
target_marker_size = 24  # dodecahedron edge length in mm
target_pentagon_size = 27.5
ref_marker_size = 35  # dodecahedron edge length in mm
ref_pentagon_size = 40
targetPoints = dodecaBoard.generate(target_marker_size, target_pentagon_size, (0, 0, 253), 'centre')
refPoints = dodecaBoard.generate(ref_marker_size, ref_pentagon_size, (0, 0, 109), 'centre')
target_board = cv2.aruco.Board(targetPoints, aruco_dict, np.arange(11))
ref_board = cv2.aruco.Board(refPoints, aruco_dict, np.arange(11,22))

def runCam(cam, childConn):
    frameTime = 1/60
    cap = cv2.VideoCapture(f"udpsrc address=192.168.5.2 port=500{cam} ! application/x-rtp, clock-rate=90000, payload=96 ! rtph264depay ! h264parse ! avdec_h264 discard-corrupted-frames=true skip-frame=1 ! videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=1 drop=true sync=false", cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print(f"Cannot open camera {cam}.")
        return
    
    while True:
        if cv2.pollKey() == ord('q'):
            cv2.destroyWindow(f'Camera {cam}')
            break
        
        capTime = time.time()
        ret, frame = cap.read()  # ret is True if frame is read correctly
        if not ret:
            print(f"Can't receive frame from camera {cam}.")
            break
        
        pose = None
        covariance = None
        
        corners, ids, rejected = detector.detectMarkers(frame)
        pose, covariance = poseEstimator.estimate_pose_board(ref_board, target_board, corners, ids)

        overlayImg = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.imshow(f'Camera {cam}', overlayImg)
        
        # if ids is not None:
        #     target_obj_pts, target_img_pts = target_board.matchImagePoints(corners,ids)
        #     if (target_img_pts is None):
        #         # print("Couldn't match object points...")
        #         pose=None
            
            
        #     elif (len(target_img_pts) < 3):
        #         # print("Not enough object points for SolvePnP")
        #         pose=None
        #     else:
        #         target_val, target_rvec, target_tvec = cv2.solvePnP(target_obj_pts,target_img_pts,poseEstimator.cv_cam_mat,poseEstimator.cv_dist_coeffs,
        #                                                                         rvec=None,tvec=None,useExtrinsicGuess=False,flags=cv2.SOLVEPNP_ITERATIVE)
        #         overlayImg = cv2.drawFrameAxes(overlayImg, poseEstimator.cv_cam_mat, poseEstimator.cv_dist_coeffs, target_rvec, target_tvec, 50)
            
        #         rel_rot_matrix, _ = cv2.Rodrigues(target_rvec)
        #         rel_rot_ypr = R.from_matrix(rel_rot_matrix).as_euler('ZYX',degrees=True)
        #         rel_rot_ypr = rel_rot_ypr.reshape((3,1))
                
        #         pose = np.vstack((target_tvec, rel_rot_ypr))
        #         covariance = np.zeros((6,6))
            
            
        # if pose is not None:
        #     print(f'X: {pose[0]}, Y: {pose[1]}, Z: {pose[2]}', end='\r')
            
        
        
        childConn.send((pose, covariance))
        
        # print(time.time()-capTime)
        # deltaTime = time.time() - capTime
        # if deltaTime < frameTime: time.sleep(frameTime-deltaTime)

    cap.release()
    childConn.close()

def update_kalman(kalman: KalmanFilterCV, poses: list, covars: list):
    final_pose = kalman.predict().reshape((12,1))[0:6]
    # poses = [pose_1,pose_2,pose_3,pose_4,pose_5]
    # poses = []
    # covars = [covar_1,covar_2,covar_3,covar_4,covar_5]
    # covars = []
    kalman_measurement = np.array([])
    covariance_matrix = np.array([])
    num_cameras = 0
    for i in range(len(poses)):
        if poses[i] is not None:
            num_cameras += 1
            if len(kalman_measurement) == 0:
                kalman_measurement = poses[i]
                covariance_matrix = covars[i]
            
            else:
                kalman_measurement = np.vstack((kalman_measurement,poses[i]))
                # Set size properly
                # [[1, 0],    [[1, 0, 0],
                #  [0, 1]] ->  [0, 1, 0]
                #              [0, 0, 1]]
                zero_block = np.zeros((covariance_matrix.shape[1],covars[i].shape[0]))
                covariance_matrix = np.block([[covariance_matrix,zero_block],
                                            [zero_block.T, covars[i]]])

    if num_cameras > 0:
        kalman_filter.set_measurement(y_k=kalman_measurement)
        kalman_filter.set_measurement_matrices(num_measurements=num_cameras, new_R=covariance_matrix)
        kalman.correct()

    return kalman, final_pose
    

def ros_publish(final_pose:np.ndarray, pose_msg):
    # if final_pose is not None:
    #     pose_msg.position.x = final_pose[0]
    #     pose_msg.position.y = final_pose[1]
    #     pose_msg.position.z = final_pose[2]
        
    #     euler = final_pose[3:].ravel()
    #     quat = R.from_euler(seq='ZYX',angles=euler,degrees=True).as_quat()
    #     pose_msg.orientation.x = quat[0]
    #     pose_msg.orientation.y = quat[1]
    #     pose_msg.orientation.z = quat[2]
    #     pose_msg.orientation.w = quat[3]

    if final_pose is not None:
        # Translate for origin of marker object in gazebo
        pose_msg.pose.position.x = (final_pose[0]/1000) + -0.023
        pose_msg.pose.position.y = (final_pose[1]/1000) + -0.051
        pose_msg.pose.position.z = (final_pose[2]/1000) + 0
        euler = final_pose[3:].ravel()
        quat = R.from_euler(seq='ZYX',angles=euler,degrees=True).as_quat()
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

    # else:
    #     pose_msg.pose.position.x = 0
    #     pose_msg.pose.position.y = 0
    #     pose_msg.pose.position.z
    #     pose_msg.pose.orientation.x = 0
    #     pose_msg.pose.orientation.y = 0
    #     pose_msg.pose.orientation.z = 0
    #     pose_msg.pose.orientation.w = 0
    
    return pose_msg

if __name__ == "__main__":
    if ROS: 
        rospy.init_node('pose_estimation', anonymous=True,log_level=rospy.INFO)
        # publisher = rospy.Publisher('kalman_filter/pose_estimate',Pose, queue_size=1)
        # pose_msg = Pose()
        
        publisher = rospy.Publisher('/gazebo/set_model_state',ModelState, queue_size=1)
        # publisher = rospy.Publisher('/gazebo/set_model_state',ModelState, queue_size=1, tcp_nodelay=True)
        pose_msg = ModelState()
        pose_msg.model_name = 'surgical_pointer'
        rate = rospy.Rate(60)
    
    kalman_filter = KalmanFilterCV(60)
    # kalman_filter.initiate_state(x0=np.zeros((6,1)))
    processes = []
    parentConns = []
    childConns = []
    
    for cam in cams:
        parentConn, childConn = mp.Pipe(False)  # True for two-way communication, False for one-way
        process = mp.Process(target=runCam, args=(cam, childConn))
        process.start()
        
        processes.append(process)
        parentConns.append(parentConn)
        childConns.append(childConn)
    
    # lastPublish = time.time()
    final_pose = np.zeros((6,1))
    
    while True:
        if True not in (process.is_alive() for process in processes): break
        if ROS and rospy.is_shutdown(): 
            cv2.destroyAllWindows()
            break
        
        poses = []
        covars = []
        
        for i, cam in enumerate(cams):
            pose, covar = parentConns[i].recv()
            if pose is not None:
                poses.append(pose)
                covars.append(covar)

        # if pose is not None:
        #     final_pose = np.median(poses, axis=0)
        # else:
        #     final_pose = np.zeros((6,1))
        
        if len(poses) > 0:
            if kalman_filter.has_been_initiated():
                kalman_filter, final_pose = update_kalman(kalman_filter, poses=poses, covars=covars)
            else:
                kalman_filter.initiate_state(x0=np.median(poses,axis=0))
                final_pose = kalman_filter.predict().reshape((12,1))[0:6]
        
            
        # print(final_pose)
        
        # currPublish = time.time()
        # diff = currPublish-lastPublish
        # print(diff)
        # lastPublish=currPublish

        if ROS:
            if len(poses)>0:
                pose_msg = ros_publish(final_pose, pose_msg)
                publisher.publish(pose_msg)
                
            rate.sleep()



    cv2.destroyAllWindows()
# poseEstimator.plot(trueTrans=[-155.2, 0, 0], trueRot=[0, 0, 0])
# poseEstimator.plot()
# avgPos = np.average(poseEstimator.total_distance, axis=0)
# print(f'Avg X: {avgPos[0]}, Y: {avgPos[1]}, Z: {avgPos[2]}')
