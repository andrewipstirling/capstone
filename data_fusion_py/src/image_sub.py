import rospy
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
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

        self.camera_matrix = np.array([[2919.7999500495794, 0.0, 728.5], 
                                       [0.0, 2919.7999500495794, 544.5],
                                        [ 0.0, 0.0, 1.0]])
        self.cv_cam_mat = cv2.Mat(self.camera_matrix)
        self.dist_coeffs = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]
])
        self.cv_dist_coeffs = cv2.Mat(self.dist_coeffs)
        self.rel_trans = 0
        self.rel_rot = 0

        self.total_distance = []
        self.total_rot = []

        self.encoding_type = 'bgr8'

        #create image subscriber object
        self.sub_image_raw = rospy.Subscriber("/pi_camera/image_raw",Image,self.image_cb) 

        # self.pi_camera_info = rospy.Subscriber("/pi_camera/camera_info", CameraInfo, self.camera_info_cb)
        
        self.pub_image = rospy.Publisher('data_fusion/cv_image',Image,queue_size=1)

        rospy.loginfo("Succesfully connected")
    
    def image_cb(self, img_msg: Image) -> None:
        # rospy.loginfo(img_msg.header)
        self.img_msg = img_msg

        return 
    
    # def camera_info_cb(self, cam_msg: CameraInfo) -> None:
    #     self.camera_matrix = np.array(cam_msg.K)
    #     np.reshape(self.camera_matrix,(3,3))
    #     self.cv_cam_mat = cv2.Mat(self.camera_matrix)
    #     self.dist_coeffs = np.array(cam_msg.D)
    #     self.cv_dist_coeffs = cv2.Mat(self.dist_coeffs)
        
    def find_markers(self):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.img_msg,desired_encoding=self.encoding_type)
            # cv_image = cv2.transpose(cv_image)
            cv_image = cv2.flip(cv_image,0)
            # cv_image = cv2.rotate(cv_image,cv2.ROTATE_90_COUNTERCLOCKWISE)
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
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners,0.032,
                                                                  self.cv_cam_mat,
                                                                  self.cv_dist_coeffs)
            if len(ids) == 2:
                # Tool Marker ID is '0'
                if(ids[0,0] == 0): # Tool found first
                    self.rel_trans = tvecs[0] - tvecs[1]
                    r1_mat = cv2.Rodrigues(rvecs[0])[0]
                    r2_mat = cv2.Rodrigues(rvecs[1])[0]
                    self.rel_rot = np.array(r1_mat).T * np.array(r2_mat)
                
                # Reference ID is '1'
                else: # Reference found first
                    
                    self.rel_trans = tvecs[1] - tvecs[0]
                    r1_mat = cv2.Rodrigues(rvecs[0])[0]
                    r2_mat = cv2.Rodrigues(rvecs[1])[0]
                    self.rel_rot = np.array(r2_mat).T * np.array(r1_mat)

                self.rel_rot = R.from_matrix(self.rel_rot).as_euler('xyz',degrees=True)
                
                rospy.loginfo_throttle(5,"ID %s: tvecs= %s", ids[0,0],tvecs[0])
                rospy.loginfo_throttle(5,"ID %s: tvecs= %s", ids[1,0],tvecs[1])
                rospy.loginfo_throttle(5,"Tool Pose: \n translation=%s \n rotation=%s", 
                                    self.rel_trans, self.rel_rot)
                
                
                self.total_distance.append(self.rel_trans)
                self.total_rot.append(self.rel_rot)
            else:
                rospy.logwarn_throttle(5, "Could not find two markers")

            # Draw Axes
            for i in range(len(ids)):
                cv2.drawFrameAxes(cv_image,self.cv_cam_mat,self.cv_dist_coeffs,rvecs[i],tvecs[i],0.02)
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
    
        cv2.imshow("Image Window", cv_image)
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
    
    def get_dist(self):
        return self.total_distance, self.total_rot

def main():
    image_sub = ImageSub()

    rospy.init_node('image_fusion', anonymous=True)
    # Run at 10 Hz
    rospy.Rate(10)

    while not rospy.is_shutdown():
        image_sub.update()
    cv2.destroyAllWindows()
    # Plotting 
    total_distance, total_rot = image_sub.get_dist()
    total_distance = np.array(total_distance).reshape((-1,3))
    total_rot = np.array(total_rot).reshape((-1,3))
    t = np.linspace(0,len(total_distance))
    const = np.ones_like(total_distance[:,0])
    plt.subplot(2,1,1)
    plt.plot(total_distance[:,0],color='red',linestyle='-',label='x')
    plt.plot(0.1*const,color='red',linestyle='--')
    plt.plot(total_distance[:,1],color='blue',linestyle='-',label='y')
    plt.plot(0.1*const,color='blue',linestyle='--')
    plt.plot(total_distance[:,2],color='green',linestyle='-',label='z')
    plt.plot(0.0*const,color='green',linestyle='--')
    plt.title('Marker Position in Reference Frame')
    plt.ylabel('Position [m]')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(total_rot[:,0],color='red',linestyle='-',label='roll')
    plt.plot(0*const,color='red',linestyle='--')
    plt.plot(total_rot[:,1],color='blue',linestyle='-',label='pitch')
    plt.plot(0*const,color='blue',linestyle='--')
    plt.plot(total_rot[:,2],color='green',linestyle='-',label='yaw')
    plt.plot(0.0*const,color='green',linestyle='--')
    plt.title('Marker Rotation in Reference Frame')
    plt.ylabel('Angular Displacement [deg]')
    plt.tight_layout()
    plt.legend()
    plt.savefig("figs/marker_pos.png")
    plt.show()

    plt.subplot(2,1,1)
    plt.plot(total_distance[:,0]-0.1,color='red',linestyle='-',label='x error')
    plt.plot(total_distance[:,1]-0.1,color='blue',linestyle='-',label='y error')
    plt.plot(total_distance[:,2],color='green',linestyle='-',label='z error')
    plt.title('Absolute Error')
    plt.ylabel('[m]')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot((total_distance[:,0]-0.1)*100/0.1,color='red',linestyle='-',label='x error')
    plt.plot((total_distance[:,1]-0.1)*100/0.1,color='blue',linestyle='-',label='y error')
    plt.title("Relative Error")
    plt.ylabel('[%]')
    plt.legend()
    plt.savefig("figs/marker_error.png")
    plt.show()
    return

if __name__ == "__main__":
    try:
        main()

    except rospy.ROSInterruptException:
        pass