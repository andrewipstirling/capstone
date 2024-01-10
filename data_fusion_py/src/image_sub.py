import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class ImageSub:
    # Class initialize
    def __init__(self) -> None:
        self.img_msg = None

        #create image subscriber object
        self.sub_image_raw = rospy.Subscriber("/pi_camera/image_raw",Image,self.image_cb) 
        rospy.init_node('image_sub', anonymous=True)
        rospy.loginfo("Succesfully connected")
    
    def image_cb(self, img_msg: Image) -> None:
        # rospy.loginfo(img_msg.header)
        self.img_msg = img_msg
        return 

    def update(self):
        if self.img_msg is not None:
            rospy.loginfo(self.img_msg.header)
        else:
            rospy.logwarn("No image message received yet")
    

def main():
    image_sub = ImageSub()
    # Run at 10 Hz
    rospy.Rate(10)

    while not rospy.is_shutdown():
        image_sub.update()
    
    return

if __name__ == "__main__":
    try:
        main()

    except rospy.ROSInterruptException:
        pass