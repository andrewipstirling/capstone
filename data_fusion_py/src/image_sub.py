import rospy
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayDimension
from geometry_msgs.msg import PoseWithCovariance
from cv_bridge import CvBridge, CvBridgeError
from gazebo_msgs.msg import ModelStates
import geometry_msgs

PUBRATE = 10
PLOTTING = False

class ImageSub:
    # Class initialize
    def __init__(self,sub_topic_name:str,pub_topic_name:str) -> None:
        
        self.show_image = False
        self.img_msg = None
        self.bridge = CvBridge()
        self.cv_img_in = None
        self.cv_img_out = None

        self.camera_matrix = np.array([[2919.7999500495794, 0.0, 728.5], 
                                       [0.0, 2919.7999500495794, 544.5],
                                        [ 0.0, 0.0, 1.0]])
        self.cv_cam_mat = cv2.Mat(self.camera_matrix)
        self.dist_coeffs = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]
])
        self.cv_dist_coeffs = cv2.Mat(self.dist_coeffs)

        self.total_distance = []
        self.total_stddev = []
        self.total_rot = []

        self.encoding_type = 'bgr8'

        #create image subscriber object
        self.sub_image_raw = rospy.Subscriber(sub_topic_name,Image,self.image_cb) 
        self.sub_topic_name = sub_topic_name
        

        # self.pi_camera_info = rospy.Subscriber("/pi_camera/camera_info", CameraInfo, self.camera_info_cb)
        
        self.pub_image = rospy.Publisher('data_fusion/cv_image',Image,queue_size=1)

        self.gazebo_pos = None
        self.gazebo_ori = None
        self.total_ref_state = []
        self.total_gazebo_ori = []

        self.sub = rospy.Subscriber('/gazebo/model_states',ModelStates,self.pose_cb)

        # Board Information
        m = 0.04/2 # half of marker length
        c = 0.05/2 # half of cube length
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
        
        self.target_board_ids = np.array([0,1,2,3,4,5])
        self.ref_board_ids = np.array([6,7,8,9,10,11])
        # Size (6,4,3): [(6 markers), (4 corners of one marker), (3 xyz position of corner)]
        self.board_points = np.array([
            [[-m,-c,-m],[-m,-c,m],[m,-c,m],[m,-c,-m]], # ID: 0/6
            [[-m,-m,-c],[m,-m,-c],[m,m,-c],[-m,m,-c]], # ID: 1/7
            [[c,-m,-m],[c,-m,m],[c,m,m],[c,m,-m]], # ID: 2/8
            [[m,-m,c],[-m,-m,c],[-m,m,c],[m,m,c]], # ID: 3/9
            [[m,c,-m],[m,c,m],[-m,c,m],[-m,c,-m]], # ID: 4/10
            [[-c,m,-m],[-c,m,m],[-c,-m,m],[-c,-m,-m]]],dtype=np.float32) # ID: 5/11
        
        # print(self.board_points,self.board_points.shape)
        # Create reference and target boards
        self.ref_board = cv2.aruco.Board(self.board_points,self.aruco_dict,self.ref_board_ids)
        self.target_board = cv2.aruco.Board(self.board_points,self.aruco_dict,self.target_board_ids)

        self.rel_trans, self.rel_rot_mat, self.std_dev = np.zeros((3,1)),np.zeros((3,3)),np.zeros((6,1))
        
        self.dt = 1 / PUBRATE
        self.prev_ref_rvec = None
        self.ref_rvel = None
        self.prev_ref_tvec = None
        self.ref_tvel = None
        self.reference_corners = None
        self.reference_ids = None

        self.prev_target_tvec = None
        self.target_tvel = None
        self.prev_target_rvec = None
        self.target_rvel = None
        self.target_corners = None
        self.target_ids = None

        self.pub_pose_estimate = rospy.Publisher(pub_topic_name,PoseWithCovariance,queue_size=1)
        self.estimated_pose_success = 0
        self.pose_estimate_msg = PoseWithCovariance()

        rospy.loginfo("Succesfully connected" + cv2.__version__)

    def pose_cb(self, model_state_msg: ModelStates) -> None:
        if len(model_state_msg.pose)>0:
        
            pos = model_state_msg.pose[2].position
            ori_quat = model_state_msg.pose[2].orientation
            ori = np.array([ori_quat.x,ori_quat.y,ori_quat.z,ori_quat.w])
            rospy.loginfo_throttle(5,"True Position: %s", pos)
            
            self.gazebo_pos = [pos.x,pos.y,pos.z]
            # self.gazebo_ori = R.from_quat(ori).as_euler('xyz',degrees=True)

    
    def image_cb(self, img_msg: Image) -> None:
        # rospy.loginfo(img_msg.header)
        self.img_msg = img_msg

        return 
    
    def publish_pose(self):
        pos = self.rel_trans.ravel()
        quat = R.as_quat(R.from_matrix(self.rel_rot_mat)).ravel()
        
        succ = np.concatenate((self.std_dev.ravel(), np.ones(30)*self.estimated_pose_success))
        cov = list(succ.ravel())

        # print(cov)
        self.pose_estimate_msg.pose.position.x = pos[0]
        self.pose_estimate_msg.pose.position.y = pos[1]
        self.pose_estimate_msg.pose.position.z = pos[2]

        self.pose_estimate_msg.pose.orientation.x = quat[0]
        self.pose_estimate_msg.pose.orientation.y = quat[1]
        self.pose_estimate_msg.pose.orientation.z = quat[2]
        self.pose_estimate_msg.pose.orientation.w = quat[3]

        self.pose_estimate_msg.covariance = cov

        self.pub_pose_estimate.publish(self.pose_estimate_msg)
        return
        
    def find_markers(self):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.img_msg,desired_encoding=self.encoding_type)

        except CvBridgeError as e:
            rospy.logwarn(e)

        arucoParams = cv2.aruco.DetectorParameters()
        # arucoParams.polygonalApproxAccuracyRate = 0.01
        # arucoParams.useAruco3Detection = True
        detector = cv2.aruco.ArucoDetector(self.aruco_dict,arucoParams)
        
        corners, ids, rejected = detector.detectMarkers(cv_image)

        # self.filter_corners(corners,ids)
        
        if ids is not None and len(ids) > 0:
            ref_obj_pts, ref_img_pts  = self.ref_board.matchImagePoints(corners,ids)
            target_obj_pts, target_img_pts = self.target_board.matchImagePoints(corners,ids)
            
            if (target_img_pts is None or ref_img_pts is None):
                rospy.logwarn_throttle(5, "%s: Couldn't match points", self.sub_topic_name)
                self.estimated_pose_success = 0 # False
                self.rel_trans, self.rel_rot_mat, self.std_dev = np.zeros((3,1)),np.zeros((3,3)),np.zeros((6,1))
                return

            if (len(target_img_pts) < 3 or len(ref_img_pts) < 3):
                rospy.logwarn_throttle(5, "Not enough points for SolvePnp")
                self.estimated_pose_success = 0 # False
                self.rel_trans, self.rel_rot_mat, self.std_dev = np.zeros((3,1)),np.zeros((3,3)),np.zeros((6,1))
                return
            # Deprecated 
            # ref_obj_pts, ref_img_pts = cv2.aruco.getBoardObjectAndImagePoints(self.ref_board,corners,ids)
            # target_obj_pts, target_img_pts = cv2.aruco.getBoardObjectAndImagePoints(self.target_board,corners,ids)
            solve_flag = cv2.SOLVEPNP_ITERATIVE
            if self.prev_target_tvec is not None:
                # ref_val, ref_rvec, ref_tvec = cv2.aruco.estimatePoseBoard(corners,ids,self.ref_board,self.cv_cam_mat,
                #                                                           self.dist_coeffs,self.prev_ref_rvec, self.prev_ref_tvec,
                #                                                           useExtrinsicGuess=True)
                
                # target_val, target_rvec, target_tvec = cv2.aruco.estimatePoseBoard(corners,ids,self.target_board,self.cv_cam_mat,
                #                                                                    self.dist_coeffs,self.prev_target_rvec, 
                #                                                                    self.prev_target_tvec,useExtrinsicGuess=True)
                # ref_rvec_predict = self.prev_ref_rvec + (self.ref_rvel * self.dt)
                # ref_tvec_predict = self.prev_ref_tvec + (self.ref_tvel * self.dt)

                ref_val, ref_rvec, ref_tvec = cv2.solvePnP(ref_obj_pts,ref_img_pts,self.cv_cam_mat,self.cv_dist_coeffs,
                                                           rvec=self.prev_ref_rvec,tvec=self.prev_ref_tvec,useExtrinsicGuess=True,flags=solve_flag)
                
                # target_rvec_predict = self.prev_target_rvec + (self.target_rvel * self.dt)
                # target_tvec_predict = self.prev_target_tvec + (self.target_tvel * self.dt)
                target_val, target_rvec, target_tvec = cv2.solvePnP(target_obj_pts,target_img_pts,self.cv_cam_mat,self.cv_dist_coeffs,
                                                                    rvec=self.prev_target_rvec,tvec=self.prev_target_tvec,useExtrinsicGuess=True,flags=solve_flag)
                
                
                
            else:    
                # ref_val, ref_rvec, ref_tvec = cv2.aruco.estimatePoseBoard(cornerv_cs,ids,self.ref_board,
                #                                                           self.cv_cam_mat,self.dist_coeffs,rvec=None,tvec=None)
                # target_val, target_rvec, target_tvec = cv2.aruco.estimatePoseBoard(corners,ids,self.target_board,
                #                                                                    self.cam_mat,self.dist_coeffs, rvec=None,tvec=None)
                
                ref_val, ref_rvec, ref_tvec = cv2.solvePnP(ref_obj_pts,ref_img_pts,self.cv_cam_mat,self.cv_dist_coeffs,
                                                           rvec=None,tvec=None,useExtrinsicGuess=False,flags=solve_flag)
                
                target_val, target_rvec, target_tvec = cv2.solvePnP(target_obj_pts,target_img_pts,self.cv_cam_mat,self.cv_dist_coeffs,
                                                                    rvec=None,tvec=None,useExtrinsicGuess=False,flags=solve_flag)
                self.prev_ref_rvec = ref_rvec
                self.prev_ref_tvec = ref_tvec
                self.prev_target_rvec = target_rvec
                self.prev_target_tvec = target_tvec
                

            self.ref_tvel = (ref_tvec - self.prev_ref_tvec) / self.dt 
            self.ref_rvel = (ref_rvec - self.prev_ref_rvec) / self.dt

            self.target_tvel = (target_tvec - self.prev_target_tvec) / self.dt
            self.target_rvel = (target_rvec - self.prev_target_rvec) / self.dt

            self.prev_ref_rvec = ref_rvec
            self.prev_ref_tvec = ref_tvec
            self.prev_target_rvec = target_rvec
            self.prev_target_tvec = target_tvec

            if target_val != 0 and ref_val != 0:
                self.estimated_pose_success = 1
                
                # Put into matrix form, return rotation matrix and jacobian of rotation matrix
                target_rot_mat, _ = cv2.Rodrigues(target_rvec)
                ref_rot_mat, _= cv2.Rodrigues(ref_rvec)

                target_rot_mat = np.array(target_rot_mat)
                ref_rot_mat = np.array(ref_rot_mat)

                # Relative Translation Calculation
                # Go from object coordinates to world coordinates
                # T_{t//r} = T_{c//r} + (-T_{c//t})
                rel_trans = ref_tvec - target_tvec  # Camera's reference frame
                # Change into basis of reference frame
                self.rel_trans = ref_rot_mat.T @ rel_trans  # Reference's reference frame
                self.rel_trans[0] *= -1
                self.rel_trans[1] *= -1

                # solvePnp returns rotation of camera relative to marker !!
                # R_{t//r} = R_{t//c} @ R_{c//r}
                self.rel_rot_mat = target_rot_mat.T @ ref_rot_mat

                self.rel_rot = R.from_matrix(self.rel_rot_mat).as_euler('xyz',degrees=True)

                _ , tar_jacobian = cv2.projectPoints(target_obj_pts,target_rvec,target_tvec,self.cv_cam_mat,self.cv_dist_coeffs)

                
                sigma = np.linalg.inv(np.dot(tar_jacobian.T,tar_jacobian)[0:6,0:6])
                self.std_dev = np.sqrt(np.diag(np.abs(sigma)))

                
                
                rospy.loginfo_throttle(5,"Tool Pose: \n translation=%s \n rotation=%s", 
                                    self.rel_trans, self.rel_rot)
                rospy.loginfo_throttle(5,"Standard Deviation [x, y, z]: %s \n", self.std_dev[0:3])
                if PLOTTING:
                    self.total_distance.append(self.rel_trans)
                    self.total_stddev.append(self.std_dev)
                    self.total_rot.append(self.rel_rot)
                    self.total_ref_state.append(self.gazebo_pos)
                # self.total_gazebo_ori.append(self.gazebo_ori)

            else:
                rospy.logwarn_throttle(5, "Could not find any markers")
                self.estimated_pose_success = 0 # False
                self.rel_trans, self.rel_rot_mat, self.std_dev = np.zeros((3,1)),np.zeros((3,3)),np.zeros((6,1))

        else:
            # rospy.loginfo("Rejected markers: %s",len(rejected))
            rospy.logwarn_throttle(5,"Length of IDs is 0")
            self.estimated_pose_success = 0 # False
            self.rel_trans, self.rel_rot_mat, self.std_dev = np.zeros((3,1)),np.zeros((3,3)),np.zeros((6,1))



    def update(self):
        if self.img_msg is not None:
            # rospy.loginfo(self.img_msg.header)
            self.find_markers()
            self.publish_pose()
            # rospy.loginfo("Completed marker search")
            
        else:
            rospy.logwarn("No image message received yet")
    
    def get_dist(self):
        return self.total_distance, self.total_rot
    

def main():
    image_sub_1 = ImageSub(sub_topic_name='/pi_camera_1/image_raw',pub_topic_name='/pi_camera_1/pose_estimate')
    image_sub_2 = ImageSub(sub_topic_name='/pi_camera_2/image_raw',pub_topic_name='/pi_camera_2/pose_estimate')
    image_sub_3 = ImageSub(sub_topic_name='/pi_camera_3/image_raw',pub_topic_name='/pi_camera_3/pose_estimate')
    image_sub_4 = ImageSub(sub_topic_name='/pi_camera_4/image_raw',pub_topic_name='/pi_camera_4/pose_estimate')
    image_sub_5 = ImageSub(sub_topic_name='/pi_camera_5/image_raw',pub_topic_name='/pi_camera_5/pose_estimate')


    rospy.init_node('image_fusion', anonymous=True)
    # Run at 10 Hz
    rospy.Rate(PUBRATE)

    while not rospy.is_shutdown():
        image_sub_1.update()
        image_sub_2.update()
        image_sub_3.update()
        image_sub_4.update()
        image_sub_5.update()
        
    cv2.destroyAllWindows()
    # Plotting 
    if PLOTTING:
        total_distance, total_rot = image_sub_1.get_dist()
        total_distance = np.array(total_distance)
        total_rot = np.array(total_rot)
        t = np.linspace(0,len(total_distance))
        const = np.ones_like(total_distance[:,0])
        true_distance = np.array(image_sub_1.total_ref_state)
        true_orientation = np.array(image_sub_1.total_gazebo_ori)
        x = total_distance[:,1].flatten()
        y = total_distance[:,0].flatten()
        z = total_distance[:,2].flatten()
        
        true_x = true_distance[:,0]
        true_y = true_distance[:,1]
        true_z = true_distance[:,2] - 0.025
        plt.tight_layout()
        plt.subplot(2,1,1)
        plt.plot(x,color='red',linestyle='--',label='ref x')
        plt.plot(true_x,color='red',linestyle='-',label='true x')
        plt.plot(y,color='blue',linestyle='--',label='ref y')
        plt.plot(true_y,color='blue',linestyle='-',label='true y')
        plt.plot(z,color='green',linestyle='--',label='ref z')
        plt.plot(true_z,color='green',linestyle='-',label='true z')
        plt.title('Marker Position in Reference Frame')
        plt.ylabel('Position [m]')
        plt.xlabel('Frames')
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(total_rot[:,0],color='red',linestyle='--',label='roll')
        # plt.plot(true_orientation[:,0],color='red',linestyle='-')
        plt.plot(total_rot[:,1],color='blue',linestyle='--',label='pitch')
        # plt.plot(true_orientation[:,1],color='blue',linestyle='-')
        plt.plot(total_rot[:,2],color='green',linestyle='--',label='yaw')
        # plt.plot(true_orientation[:,2],color='green',linestyle='-')
        plt.title('Marker Rotation in Reference Frame')
        plt.ylabel('Angular Displacement [deg]')
        plt.xlabel('Frames')
        plt.tight_layout()
        plt.legend()
        
        # plt.savefig("/home/astirl/Documents/capstone/figs/marker_pos.png")
        plt.show()
        
        x_err = np.abs(x-true_x)
        y_err = np.abs(y-true_y)
        z_err = (z-true_z)
        x_err_rel = x_err / np.abs(true_x)
        y_err_rel = y_err / np.abs(true_y)
        std_dev = np.array(image_sub_1.total_stddev)
        plt.subplot(2,1,1)
        plt.tight_layout()
        plt.plot(x_err,color='red',linestyle='-',label='x error')
        plt.plot(y_err,color='blue',linestyle='-',label='y error')
        # plt.plot(z_err,color='green',linestyle='-',label='z error')
        plt.title('Absolute Error')
        plt.ylabel('[m]')
        plt.xlabel('Frames')
        plt.legend(loc='upper right')

        # plt.subplot(3,1,2)
        # plt.plot(x_err_rel,color='red',linestyle='-',label='x error')
        # plt.plot(y_err_rel,color='blue',linestyle='-',label='y error')
        # plt.title("Relative Error")
        # plt.ylabel('[%]')
        # plt.xlabel('Frames')
        # plt.legend(loc='upper right')

        plt.subplot(2,1,2)
        plt.plot(std_dev[:,0],color='red',linestyle='-',label='x std dev')
        plt.plot(std_dev[:,1],color='blue',linestyle='-',label='y std dev')
        plt.plot(std_dev[:,2],color='green',linestyle='-',label='z std dev')
        plt.title('Standard Deviation')
        plt.ylabel('[pixels]')
        plt.xlabel('Frames')
        plt.legend(loc='upper right')

        
        # plt.savefig("/home/astirl/Documents/capstone/figs/marker_error.png")
        plt.show()
    
    return

if __name__ == "__main__":
    try:
        main()

    except rospy.ROSInterruptException:
        pass