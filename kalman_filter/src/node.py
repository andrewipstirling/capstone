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
        else:
            self.pose_success[0] = True
        
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
        else:
            self.pose_success[1] = True

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
        else:
            self.pose_success[2] = True

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
        else:
            self.pose_success[3] = True
        
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
        else:
            self.pose_success[4] = True
        
        return
    
    def update(self):
        # self.final_pose = self.pose_1 + self.pose_2 + self.pose_3 + self.pose_4 + self.pose_5
        # self.final_pose = self.final_pose / 5
        # rospy.loginfo("Final Pose = %s",self.final_pose)
        poses = np.array([self.pose_1,self.pose_2,self.pose_3,self.pose_4,self.pose_5])
        covars = np.array([self.pose_1_variance,self.pose_2_variance,self.pose_3_variance,self.pose_4_variance,self.pose_5_variance])
        # poses_toavg=[]
        # weights = []
        # median = np.median(poses,axis=0)
        # std = np.std(poses,axis=0)
        # for i in range(5):
        #     pose_i = poses[:][:][i]
        #     if self.pose_success[i]:
        #         if np.mean(np.abs( pose_i - median)) > np.mean(3*std):
        #             self.pose_success[i] = False
        #         else:
        #             print(np.diag(covars[:][:][i]))
        #             poses_toavg.append(pose_i)
        #             weights.append(1 / np.diag(covars[:][:][i]))

        # poses_toavg = np.array(poses_toavg)
        # weights = np.array(weights)
        # # Normalize weights to sum up to 1
        # weights /= np.sum(weights)
        # # Compute the weighted sum of arrays
        # self.weighted_pose = np.sum(poses_toavg * weights, axis=0).reshape((6,1))

        # Add to Kalman Filter
        self.kalman.get_measurement(y_k=np.median(poses,axis=0))
        self.final_pose = self.kalman.predict().reshape((12,1))[0:6]
        self.kalman.correct()

    
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
    
    return

if __name__ == "__main__":
    try:
        main()

    except rospy.ROSInterruptException:
        pass