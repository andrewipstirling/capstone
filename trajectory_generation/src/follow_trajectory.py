import rospy
import numpy as np
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

class PubTraj:
    
    def __init__(self) -> None:

        rospy.loginfo("Setting up follow_trajectory node")
        rospy.logerr("Starting node")
        self.pub_msg = None
        self.start_x = 0.12
        self.start_y = 0.15
        self.start_z = 0.025

        self.end_x = -0.12
        self.end_y = 0.151
        self.end_z = 0.0251
        self.x = np.arange(self.start_x,self.end_x,-1e-3,)
        self.y = np.arange(self.start_y,self.end_y,1e-3)
        self.z = np.arange(self.start_z,self.end_z,1e-3)

        
        
        self.max_count = max([len(self.x),len(self.y),len(self.z)])
        self.orientation_traj = None

        self.count = 0
        self.model_msg = ModelState()
        self.pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)

        rospy.loginfo("Succesfully connected with follow_trajectory node")
        rospy.loginfo("Publishing trajectory of \'aruco_box\' model...")
        

    def update(self):

        self.model_msg.model_name = 'aruco_box'

        if self.count < self.max_count:
            rospy.loginfo("Sending state")
            self.model_msg.pose.position.x = self.x[self.count]
            self.model_msg.pose.position.y = self.y[0]
            self.model_msg.pose.position.z = self.z[0]

            self.pub.publish(self.model_msg)
            self.count += 1
            # rospy.wait_for_service('/gazebo/set_model_state', SetModelState)
            # try: 
            #     set_state = rospy.ServiceProxy('/gazebo/model_state', SetModelState)
            #     response = set_state(self.model_msg)
                
            # except rospy.ServiceException as err:
            #     rospy.logerr("Service call failed: %s", err)
                



    
def main():
    rospy.init_node('publish_trajectory', anonymous=True, log_level=rospy.INFO)

    traj_pub = PubTraj()

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        traj_pub.update()
        
        rate.sleep()


    return


if __name__ == "__main__":
    try:
        main()

    except rospy.ROSInterruptException:
        pass
