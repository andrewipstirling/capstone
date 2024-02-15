import rospy
import numpy as np
from gazebo_msgs.msg import ModelState
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetModelState

PUBRATE = 10

class PubTraj:
    
    def __init__(self) -> None:

        rospy.loginfo("Setting up follow_trajectory node")
        rospy.logerr("Starting node")
        self.pub_msg = None

        self.end_x = -0.12
        self.end_y = 0.15
        self.end_z = 0.025
        self.end = np.array([self.end_x,self.end_y,self.end_z])
        
        

        self.time_to_move = 10
        self.pub_rate = PUBRATE
        self.traj = None

        
        self.max_count = None
        self.orientation_traj = None

        self.count = 0
        self.model_msg = ModelState()
        self.pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)

        self.sub = rospy.Subscriber('/gazebo/model_states',ModelStates,self.pose_cb)
        self.has_start = False

        rospy.loginfo("Succesfully connected with follow_trajectory node")
        rospy.loginfo("Publishing trajectory of \'aruco_box\' model...")
        
    def pose_cb(self, model_state_msg: ModelStates) -> None:
        if not self.has_start:
            if len(model_state_msg.pose)>0:
                pos = model_state_msg.pose[2].position
                ori = model_state_msg.pose[2].orientation
                self.start_x = pos.x
                self.start_y = pos.y
                self.start_z = pos.z
                self.has_start = True
                self.orientation_traj = ori
                self.create_traj()

    def create_traj(self):
        self.start = np.array([self.start_x,self.start_y,self.start_z])

        self.traj = np.linspace(self.start,self.end,self.pub_rate * self.time_to_move)
        self.max_count = len(self.traj)

    def update(self):
        done = False
        if self.has_start :
            self.model_msg.model_name = 'aruco_box'

            if self.count < self.max_count:
                rospy.loginfo("Sending state")
                self.model_msg.pose.position.x = self.traj[self.count][0]
                self.model_msg.pose.position.y = self.traj[self.count][1]
                self.model_msg.pose.position.z = self.traj[self.count][2]
                self.model_msg.pose.orientation.x = self.orientation_traj.x
                self.model_msg.pose.orientation.y = self.orientation_traj.y
                self.model_msg.pose.orientation.z = self.orientation_traj.z
                self.model_msg.pose.orientation.w = self.orientation_traj.w

                self.pub.publish(self.model_msg)
                self.count += 1

            else:
                done = True
            
        return done
                
def main():
    rospy.init_node('publish_trajectory', anonymous=True, log_level=rospy.INFO)

    traj_pub = PubTraj()

    rate = rospy.Rate(PUBRATE)

    while not rospy.is_shutdown():
        done = traj_pub.update()
        rate.sleep()
        if done:
            rospy.loginfo("Finished trajectory, Shutting Down")
            rospy.signal_shutdown(reason="Finished Trajectory")

    return


if __name__ == "__main__":
    try:
        main()

    except rospy.ROSInterruptException:
        pass
