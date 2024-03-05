import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


class pose_estimation:
    
    def __init__(self):
        # OpenCV Stuff
        self.camera_matrix = None
        self.dist_coeffs = None
        self.image = None
        self.marker_length = 0.04 # Side length of marker (currently set at 4cm)
        
        # Board Information
        self.m = self.marker_length/2 # half of marker length
        self.c = 0.05/2 # half of cube length
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
        
        # Pose Estimation Variables
        # Check this !!
        self.frame_rate = 75
        # Time between each frame
        self.dt = 1 / self.frame_rate
        self.prev_ref_rvec = None
        self.ref_rvel = None
        self.prev_ref_tvec = None
        self.ref_tvel = None
        self.reference_corners = None
        self.reference_ids = None

        self.prev_target_tvec = None
        self.target_tvel = None
        self.prev_target_rvec = None
        self.target_rvel = None
        self.target_corners = None
        self.target_ids = None
        
    def set_camera_params(self, camera_matrix: cv2.Mat, dist_coeffs: cv2.Mat) -> None:
        # Takes camera and distortion matrices from calibration as input
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        return
    
    def estimate_pose_board(self,reference_board:cv2.aruco.Board, target_board:cv2.aruco.Board, corners, ids):
        """
        Arguments:
        -----------
        reference_board: cv2.aruco.Board
            openCV Board class which defines custom board of reference object
        target_board: cv2.aruco.Board
            openCV Board class which defines custom board of target object
        corners: cv2.Matlike
            list of corners returned from aruco detector
        ids: cv2.Matlike
            list of ids returned from aruco detector
        Returns:
        -----------
        rel_trans: numpy.Array
        Relative translation between target and reference marker in world frame 
        stored as numpy array
        rel_rot_matrix: numpy.Array
        Relative orientation between target and reference marker in world frame stored
        as numpy matrix representing the rotation of target relative to reference.
        """
        if ids is not None and len(ids) > 0:
            ref_obj_pts, ref_img_pts = cv2.aruco.getBoardObjectAndImagePoints(reference_board,corners,ids)
            target_obj_pts, target_img_pts = cv2.aruco.getBoardObjectAndImagePoints(target_board,corners,ids)

            if self.prev_target_tvec is not None:
                # Estimate pose of Reference board
                ref_val, ref_rvec, ref_tvec = cv2.solvePnP(ref_obj_pts,ref_img_pts,self.cv_cam_mat,self.cv_dist_coeffs,
                                            rvec=self.prev_ref_rvec,tvec=self.prev_ref_tvec,useExtrinsicGuess=True,flags=cv2.SOLVEPNP_ITERATIVE)
                
                # Estimate Pose of Target Board
                target_val, target_rvec, target_tvec = cv2.solvePnP(target_obj_pts,target_img_pts,self.cv_cam_mat,self.cv_dist_coeffs,
                                            rvec=self.prev_target_rvec,tvec=self.prev_target_tvec,useExtrinsicGuess=True,flags=cv2.SOLVEPNP_ITERATIVE)
                   
            else:    
                # Pose of Reference Board
                ref_val, ref_rvec, ref_tvec = cv2.solvePnP(ref_obj_pts,ref_img_pts,self.cv_cam_mat,self.cv_dist_coeffs,
                                                           rvec=None,tvec=None,useExtrinsicGuess=False,flags=cv2.SOLVEPNP_ITERATIVE)
                # Pose of Target Board
                target_val, target_rvec, target_tvec = cv2.solvePnP(target_obj_pts,target_img_pts,self.cv_cam_mat,self.cv_dist_coeffs,
                                                                    rvec=None,tvec=None,useExtrinsicGuess=False,flags=cv2.SOLVEPNP_ITERATIVE)
                self.prev_ref_rvec = ref_rvec
                self.prev_ref_tvec = ref_tvec
                self.prev_target_rvec = target_rvec
                self.prev_target_tvec = target_tvec
                

            self.ref_tvel = (ref_tvec - self.prev_ref_tvec) / self.dt 
            self.ref_rvel = (ref_rvec - self.prev_ref_rvec) / self.dt

            self.target_tvel = (target_tvec - self.prev_target_tvec) / self.dt
            self.target_rvel = (target_rvec - self.prev_target_rvec) / self.dt

            self.prev_ref_rvec = ref_rvec
            self.prev_ref_tvec = ref_tvec
            self.prev_target_rvec = target_rvec
            self.prev_target_tvec = target_tvec

            if target_val != 0 and ref_val != 0:
                # Relative Translation Calculation
                # Go from object coordinates to world coordinates
                # T_{t//r} = T_{c//r} + (-T_{c//t})
                rel_trans = ref_tvec - target_tvec
                
                # Reative orientation calculation
                # Put into matrix form, return rotation matrix and jacobian of rotation matrix
                target_rot_mat, _ = cv2.Rodrigues(target_rvec)
                ref_rot_mat, _= cv2.Rodrigues(ref_rvec)
                target_rot_mat = np.array(target_rot_mat)
                ref_rot_mat = np.array(ref_rot_mat)

                # solvePnp returns rotation of camera relative to marker !!
                # R_{t//r} = R_{t//c} @ R_{c//r}
                # As Matrix
                rel_rot_matrix = target_rot_mat.T @ ref_rot_mat

                # As roll-pitch-yaw (rpy) vector 
                rel_rot_rpy = R.from_matrix(rel_rot_matrix).as_euler('xyz',degrees=True)

                _ , tar_jacobian = cv2.projectPoints(target_obj_pts,target_rvec,target_tvec,self.camera_matrix,self.dist_coeffs)

                sigma = np.linalg.inv(np.dot(tar_jacobian.T,tar_jacobian)[0:6,0:6])
                std_dev = np.sqrt(np.diag(np.abs(sigma)))
                
                
            return rel_trans, ref_rot_mat, std_dev

        
        
    def estimate_pose_marker(self,corners,ids, target_id: int):
        """
        Arguments:
        -----------
        corners: cv2.MatLike
            list of corners return from aruco detector
        ids: cv2.Matlike
            list of ids returned from aruco detector
        target_id: cv2.Matlike
            target/tool id number of marker wanting to track 
        
        Returns:
        -----------
        rel_trans: Relative translation between target and reference marker in world frame
                    stored as numpy array
        rel_rot_matrix: Relative orientation between target and reference marker in world frame
                        stored as numpy matrix representing the rotation of target relative to reference.
        """
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners,self.marker_length,
                                                              self.camera_matrix,
                                                              self.dist_coeffs)
        # If found less/more than 2 markers, return None
        if len(ids) == 2:
            # Target/Tool marker found first
            if (ids[0][0] == target_id):
                target_tvec = tvecs[0]
                target_rvec = rvecs[0]
                ref_tvec = tvecs[1]
                ref_rvec = rvecs[1]
            else:
                target_tvec = tvecs[1]
                target_rvec = rvecs[1]
                ref_tvec = tvecs[0]
                ref_rvec = rvecs[0]
                
            # Relative Translation Calculation
            # Go from object coordinates to world coordinates
            # T_{t//r} = T_{c//r} + (-T_{c//t})
            rel_trans = ref_tvec - target_tvec
            
            # Relative orientation calculation
            # Put into matrix form, return rotation matrix and jacobian of rotation matrix
            target_rot_mat, _ = cv2.Rodrigues(target_rvec)
            ref_rot_mat, _= cv2.Rodrigues(ref_rvec)
            target_rot_mat = np.array(target_rot_mat)
            ref_rot_mat = np.array(ref_rot_mat)

            # solvePnp returns rotation of camera relative to marker !!
            # R_{t//r} = R_{t//c} @ R_{c//r}
            # As Matrix
            rel_rot_matrix = target_rot_mat.T @ ref_rot_mat
            
            return rel_trans, rel_rot_matrix
            
        else:
            return None
                
                
    