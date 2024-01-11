import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class ImageSub:
    # Class initialize
    def __init__(self) -> None:
        self.img_msg = None
        
        self.bridge = CvBridge()
        self.cv_img_in = None
        self.cv_img_out = None

        #create image subscriber object
        self.sub_image_raw = rospy.Subscriber("/pi_camera/image_raw",Image,self.image_cb) 
        
        self.pub_image = rospy.Publisher('data_fusion/cv_image',Image,queue_size=1)

        rospy.loginfo("Succesfully connected")
    
    def image_cb(self, img_msg: Image) -> None:
        # rospy.loginfo(img_msg.header)
        self.img_msg = img_msg

        return 
    
    def find_markers(self):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.img_msg,desired_encoding='passthrough')
            cv_image = cv2.transpose(cv_image)
            cv_image = cv2.flip(cv_image,1)
        except CvBridgeError as e:
            rospy.logwarn(e)

        # cv_image_out = cv_image.copy()
        
        # arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_100)
        # arucoParams = cv2.aruco.DetectorParameters_create()
        arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
        arucoParams = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(arucoDict,arucoParams)
        try:
            corners, ids, rejected = detector.detectMarkers(cv_image)
        except CvBridgeError as e:
            rospy.logwarn(e)
        
        if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()
            # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(corners, ids):
                # extract the marker corners (which are always returned in
                # top-left, top-right, bottom-right, and bottom-left order)
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                # convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))
                # draw the bounding box of the ArUCo detection
                cv2.line(cv_image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(cv_image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(cv_image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(cv_image, bottomLeft, topLeft, (0, 255, 0), 2)

        else:
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
            self.pub_image.publish(self.bridge.cv2_to_imgmsg(cv_image,"passthrough"))
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