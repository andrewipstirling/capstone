import pose_estimation
import cv2

def main():
    poseEstimator = pose_estimation.pose_estimation()
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    """
    cap = cv2.capture
    
    while (True):
        
        image = cap.read()
        corners, ids = cv2.aruco.findmarkers()
        rel_trans, rel_rot = poseEstimator.estimate_pose_marker(corners,ids, 0)
        
        
        
    
    """