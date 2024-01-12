import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError


class ImageSub:
    # Class initialize
    def __init__(self) -> None:
        self.img_msg = None
        
        self.bridge = CvBridge()
        self.cv_img_in = None
        self.cv_img_out = None

        self.encoding_type = 'bgr8'

        #create image subscriber object
        self.sub_image_raw = rospy.Subscriber("/pi_camera/image_raw",Image,self.image_cb) 
        
        self.pi_camera_info = rospy.Subscriber("/pi_camera/camera_info", CameraInfo, self.camera_info_cb)
        
        self.pub_image = rospy.Publisher('data_fusion/cv_image',Image,queue_size=1)

        rospy.loginfo("Succesfully connected")
    
    def image_cb(self, img_msg: Image) -> None:
        # rospy.loginfo(img_msg.header)
        self.img_msg = img_msg

        return 
    
    def find_markers(self):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.img_msg,desired_encoding=self.encoding_type)
            # cv_image = cv2.transpose(cv_image)
            cv_image = cv2.flip(cv_image,0)
        except CvBridgeError as e:
            rospy.logwarn(e)

        # cv_image_out = cv_image.copy()
        
        # arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_100)
        # arucoParams = cv2.aruco.DetectorParameters_create()
        arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        arucoParams = cv2.aruco.DetectorParameters()
        # arucoParams.polygonalApproxAccuracyRate = 0.01
        # arucoParams.useAruco3Detection = True
        detector = cv2.aruco.ArucoDetector(arucoDict,arucoParams)
        
        corners, ids, rejected = detector.detectMarkers(cv_image)

        
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)

        else:
            # rospy.loginfo("Rejected markers: %s",len(rejected))
            if rejected is not None and len(rejected) > 0:
                rospy.loginfo("Rejected markers: %s",len(rejected))
                for r in rejected:
                    corners = r[0].astype(int)
                    cv2.polylines(cv_image, [corners], isClosed=True, color=(0, 0, 255), thickness=2)
                
        # (rows,cols,channels) = cv_image.shape
        # if cols > 60 and rows > 60 :
        #     cv2.circle(cv_image, (50,50), 10, 255)
    
        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)
        try:
            self.pub_image.publish(self.bridge.cv2_to_imgmsg(cv_image,self.encoding_type))
        except CvBridgeError as e:
            rospy.logwarn(e)


    def update(self):
        if self.img_msg is not None:
            # rospy.loginfo(self.img_msg.header)
            self.find_markers()
            # rospy.loginfo("Completed marker search")
            
        else:
            rospy.logwarn("No image message received yet")
    


def main():
    image_sub = ImageSub()

    rospy.init_node('image_fusion', anonymous=True)
    # Run at 10 Hz
    rospy.Rate(10)

    while not rospy.is_shutdown():
        image_sub.update()
    cv2.destroyAllWindows()
    return

if __name__ == "__main__":
    try:
        main()

    except rospy.ROSInterruptException:
        pass