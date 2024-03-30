import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseWithCovariance
from geometry_msgs.msg import Pose
from kalman_filter.kalman import KalmanFilterCV
np.set_printoptions(suppress=True)


class Fuse:

    def __init__(self, simulation=True) -> None:
        if simulation:
            self.name_base = '/pi_camera_'
        else:
            self.name_base = '/real_cam_'
        
        self.sub_cam_1 = rospy.Subscriber(self.name_base + '1/pose_estimate',PoseWithCovariance,self.pose_cb_1)
        self.sub_cam_2 = rospy.Subscriber(self.name_base + '2/pose_estimate',PoseWithCovariance,self.pose_cb_2)
        self.sub_cam_3 = rospy.Subscriber(self.name_base + '3/pose_estimate',PoseWithCovariance,self.pose_cb_3)
        self.sub_cam_4 = rospy.Subscriber(self.name_base + '4/pose_estimate',PoseWithCovariance,self.pose_cb_4)
        self.sub_cam_5 = rospy.Subscriber(self.name_base + '5/pose_estimate',PoseWithCovariance,self.pose_cb_5)

        self.pub = rospy.Publisher('kalman_filter/pose_estimate',Pose, queue_size=1)
        self.pose_msg = Pose()

        self.pose_1 = np.zeros((6,1))
        self.pose_1_variance = np.zeros((6,6))
        self.pose_1_success = False
        self.pose_2 = np.zeros((6,1))
        self.pose_2_variance = np.zeros((6,6))
        self.pose_2_success = False
        self.pose_3 = np.zeros((6,1))
        self.pose_3_variance = np.zeros((6,6))
        self.pose_3_success = False
        self.pose_4 = np.zeros((6,1))
        self.pose_4_variance = np.zeros((6,6))
        self.pose_4_success = False
        self.pose_5 = np.zeros((6,1))
        self.pose_5_variance = np.zeros((6,6))
        self.pose_5_success = False

        self.pose_success = [False,False,False,False,False]
        self.new_update = [False,False,False,False,False]


        self.final_pose = np.zeros((6,1))

        self.kalman = KalmanFilterCV(freq=60)
        x0 = np.array([0.12,0.15,0.00,0.0,0.0,0.0])
        x0 = x0.reshape((6,1))
        self.kalman.initiate_state(x0)
    
    def pose_cb_1(self, pose:PoseWithCovariance) -> None:
        self.pose_1[0][0] = pose.pose.position.x
        self.pose_1[1][0] = pose.pose.position.y
        self.pose_1[2][0] = pose.pose.position.z

        ori = pose.pose.orientation
        ori_array = np.array([ori.x,ori.y,ori.z,ori.w])
        yaw, pitch, roll = R.from_quat(ori_array).as_euler('ZYX',degrees=False)
        
        self.pose_1[3][0] = yaw
        self.pose_1[4][0] = pitch
        self.pose_1[5][0] = roll

        covar = np.array(pose.covariance).reshape(36)
        np.fill_diagonal(self.pose_1_variance,covar[0:7])
        if covar[7:].all() == 0:
            self.pose_success[0] = False
            # self.new_update[0] = False
        else:
            self.pose_success[0] = True
            # self.new_update[0] = True
        
        return
    
    def pose_cb_2(self, pose:PoseWithCovariance) -> None:
        self.pose_2[0][0] = pose.pose.position.x
        self.pose_2[1][0] = pose.pose.position.y
        self.pose_2[2][0] = pose.pose.position.z

        ori = pose.pose.orientation
        ori_array = np.array([ori.x,ori.y,ori.z,ori.w])
        yaw, pitch, roll = R.from_quat(ori_array).as_euler('ZYX',degrees=False)
        
        self.pose_2[3][0] = yaw
        self.pose_2[4][0] = pitch
        self.pose_2[5][0] = roll

        covar = np.array(pose.covariance).reshape(36)
        np.fill_diagonal(self.pose_2_variance,covar[0:7])

        if covar[7:].all() == 0:
            self.pose_success[1] = False
            # self.new_update[1] = False
        else:
            self.pose_success[1] = True
            # self.new_update[1] = True

        return
    
    def pose_cb_3(self, pose:PoseWithCovariance) -> None:
        self.pose_3[0][0] = pose.pose.position.x
        self.pose_3[1][0] = pose.pose.position.y
        self.pose_3[2][0] = pose.pose.position.z

        ori = pose.pose.orientation
        ori_array = np.array([ori.x,ori.y,ori.z,ori.w])
        yaw, pitch, roll = R.from_quat(ori_array).as_euler('ZYX',degrees=False)
        
        self.pose_3[3][0] = yaw
        self.pose_3[4][0] = pitch
        self.pose_3[5][0] = roll

        covar = np.array(pose.covariance).reshape(36)
        np.fill_diagonal(self.pose_3_variance,covar[0:7])

        if covar[7:].all() == 0:
            self.pose_success[2] = False
            # self.new_update[2] = False
        else:
            self.pose_success[2] = True
            # self.new_update[3] = True

        return
    def pose_cb_4(self, pose:PoseWithCovariance) -> None:
        self.pose_4[0][0] = pose.pose.position.x
        self.pose_4[1][0] = pose.pose.position.y
        self.pose_4[2][0] = pose.pose.position.z

        ori = pose.pose.orientation
        ori_array = np.array([ori.x,ori.y,ori.z,ori.w])
        yaw, pitch, roll = R.from_quat(ori_array).as_euler('ZYX',degrees=False)
        
        self.pose_4[3][0] = yaw
        self.pose_4[4][0] = pitch
        self.pose_4[5][0] = roll

        covar = np.array(pose.covariance).reshape(36)
        np.fill_diagonal(self.pose_4_variance,covar[0:7])

        if covar[7:].all() == 0:
            self.pose_success[3] = False
            # self.new_update[3] = False
        else:
            self.pose_success[3] = True
            # self.new_update[3] = True
        
        return
    def pose_cb_5(self, pose:PoseWithCovariance) -> None:
        self.pose_5[0][0] = pose.pose.position.x
        self.pose_5[1][0] = pose.pose.position.y
        self.pose_5[2][0] = pose.pose.position.z

        ori = pose.pose.orientation
        ori_array = np.array([ori.x,ori.y,ori.z,ori.w])
        yaw, pitch, roll = R.from_quat(ori_array).as_euler('ZYX',degrees=False)
        
        self.pose_5[3][0] = yaw
        self.pose_5[4][0] = pitch
        self.pose_5[5][0] = roll

        covar = np.array(pose.covariance).reshape(36)
        np.fill_diagonal(self.pose_5_variance,covar[0:7])

        if covar[7:].all() == 0:
            self.pose_success[4] = False
            # self.new_update[4] = False
        else:
            self.pose_success[4] = True
            # self.new_update[4] = True
        return
    
    def update(self):

        # Predict using kalman filter
        self.final_pose = self.kalman.predict().reshape((12,1))[0:6]

        poses = [self.pose_1,self.pose_2,self.pose_3,self.pose_4,self.pose_5]
        covars = [self.pose_1_variance,self.pose_2_variance,self.pose_3_variance,self.pose_4_variance,self.pose_5_variance]

        kalman_measurement = np.array([])
        covariance_matrix = np.array([])
        num_cameras = 0
        for i in range(5):
            if self.pose_success[i]:
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
                

        # Add to Kalman Filter
        # self.kalman.set_measurement(y_k=np.median(poses,axis=0))
        if num_cameras > 0:
            self.kalman.set_measurement(y_k = kalman_measurement)
            self.kalman.set_measurement_matrices(num_measurements=num_cameras,new_R=covariance_matrix)
            self.kalman.correct()
            rospy.loginfo_throttle(2, "Found %s new measurements", num_cameras)
        else:
            rospy.loginfo_throttle(2,"No new measurements available")

    
        rospy.loginfo_throttle(2,"Final Pose %s", self.final_pose)
        # rospy.loginfo_throttle(2,"Median Pose %s", median)
        # rospy.loginfo_throttle(2,"Cameras Used %s", cameras_used)

        # rospy.loginfo_throttle(2,"Pose 1: %s",self.pose_1)
        # rospy.loginfo_throttle(2,"Pose 2: %s",self.pose_2)
        # rospy.loginfo_throttle(2,"Pose 3: %s",self.pose_3)
        # rospy.loginfo_throttle(2,"Pose 4: %s",self.pose_4)
        # rospy.loginfo_throttle(2,"Pose 5: %s",self.pose_5)

    def publish(self):
        self.pose_msg.position.x = self.final_pose[0]
        self.pose_msg.position.y = self.final_pose[1]
        self.pose_msg.position.z = self.final_pose[2]

        euler = self.final_pose[3:].ravel()
        quat = R.from_euler(seq='ZYX',angles=euler,degrees=False).as_quat()
        self.pose_msg.orientation.x = quat[0]
        self.pose_msg.orientation.y = quat[1]
        self.pose_msg.orientation.z = quat[2]
        self.pose_msg.orientation.w = quat[3]

        self.pub.publish(self.pose_msg)

def main():
    fuser = Fuse()
    rospy.init_node('pose_estimate_fusion',anonymous=False,log_level=rospy.INFO )
    rate = rospy.Rate(60)

    while not rospy.is_shutdown():
        fuser.update()
        fuser.publish()
        rate.sleep()
    
    return

if __name__ == "__main__":
    try:
        main()

    except rospy.ROSInterruptException:
        pass