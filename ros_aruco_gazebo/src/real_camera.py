import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class RealCamera:

    def __init__(self,pub_topic_name:str) -> None:
        self.bridge = CvBridge()
        self.encoding_type = 'bgr8'
        self.publisher = rospy.Publisher(pub_topic_name,Image,queue_size=1)

    def publish(self, real_image):
        image_msg = self.bridge.cv2_to_imgmsg(real_image,encoding=self.encoding_type)
        self.publisher.publish(image_msg)
        return
    
def main():
    topic_name = '/real_cam_4/image_raw'
    real_camera_4 = RealCamera(topic_name)
    # Webcam
    # cap = cv2.VideoCapture("autovideosrc ! videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=1 drop=true", cv2.CAP_GSTREAMER)
    # Raspi cam 4
    cap = cv2.VideoCapture("udpsrc address=192.168.5.2 port=5000 ! application/x-rtp, clock-rate=90000, payload=96 ! rtph264depay ! h264parse ! avdec_h264 discard-corrupted-frames=true skip-frame=1 ! videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=1 drop=true sync=false", cv2.CAP_GSTREAMER)

    rospy.init_node('image_fusion', anonymous=False, log_level=rospy.INFO)
    # Run at 60 Hz
    rate = rospy.Rate(60)
        
    while not rospy.is_shutdown():
        if not cap.isOpened():
            rospy.logwarn("Cannot open camera")

        else:
            ret, frame = cap.read()
            if not ret:
                rospy.logwarn("Can't receive frame")
                print("Can't receive frame")
                break
            else:
                
                rospy.logdebug("Publishing to %s" , topic_name)
                rospy.loginfo_throttle(10,"Publishing to %s" , topic_name)
                real_camera_4.publish(frame)

        # cv2.imshow('frame', frame)
        # rospy.logdebug('Capture Opened: %s', cap.isOpened())
        # rate.sleep()

    cap.release()
    cv2.destroyAllWindows()
        
    

if __name__ == "__main__":
    try:
        main()

    except rospy.ROSInterruptException:
        pass