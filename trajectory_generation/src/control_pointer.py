import rospy
import numpy as np
from gazebo_msgs.msg import ModelState
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SetModelState

class ControlPointer:

    def __init__(self) -> None:
        rospy.loginfo("Setting up control pointer node")

        self.model_msg = ModelState()
        self.model_msg.model_name = 'surgical_pointer'
        self.pub = rospy.Publisher('/gazebo/set_model_state',ModelState, queue_size=1)
        self.sub = rospy.Subscriber('/kalman_filter/pose_estimate',Pose,self.pose_cb)
        
        rospy.loginfo("Succesfully connected with control pointer")

    def pose_cb(self, pose_msg:Pose) -> None:
        if pose_msg is not None:
            # Translate for origin of marker object in gazebo
            self.model_msg.pose.position.x = (pose_msg.position.x/1000) + -0.023
            self.model_msg.pose.position.y = (pose_msg.position.y/1000) + -0.051
            self.model_msg.pose.position.z = (pose_msg.position.z/1000) + 0.032

            self.model_msg.pose.orientation.x = pose_msg.orientation.x
            self.model_msg.pose.orientation.y = pose_msg.orientation.y
            self.model_msg.pose.orientation.z = pose_msg.orientation.z
            self.model_msg.pose.orientation.w = pose_msg.orientation.w
        else:
            self.model_msg.pose.position.x = 0
            self.model_msg.pose.position.y = 0
            self.model_msg.pose.position.z = 0

            self.model_msg.pose.orientation.x = 0
            self.model_msg.pose.orientation.y = 0
            self.model_msg.pose.orientation.z = 0
            self.model_msg.pose.orientation.w = 0



    def update(self):
        # rospy.loginfo_throttle(2,"Read Pose Succesfully as: %s", self.model_msg)
        
        # Publish
        rospy.loginfo_throttle(10, "Succesfully published %s", self.model_msg)
        self.pub.publish(self.model_msg)

    
def main():
    surgical_pointer_control = ControlPointer()
    rospy.init_node('control_pointer', anonymous=False, log_level=rospy.INFO)
    rate = rospy.Rate(60)
    
    while not rospy.is_shutdown():
        surgical_pointer_control.update()
        rate.sleep()

    return

if __name__ == "__main__":
    try:
        main()

    except rospy.ROSInterruptException:
        pass