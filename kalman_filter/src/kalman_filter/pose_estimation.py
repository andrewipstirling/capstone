import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import time


class pose_estimation:
    
    def __init__(self, framerate=75, plotting=False):
        # OpenCV Stuff
        self.camera_matrix = np.array([[1.56842921e+03, 0, 2.89275503e+02], 
                                       [0, 1.57214434e+03, 2.21092150e+02], 
                                       [0, 0, 1]])
        self.cv_cam_mat = cv2.Mat(self.camera_matrix)
        self.dist_coeffs = np.array([[ 2.28769970e-02, -4.54632281e+00, -3.04424079e-03, -2.06207084e-03, 9.30400565e+01]])
        self.cv_dist_coeffs = cv2.Mat(self.dist_coeffs)
        self.image = None
        # self.marker_length = 0.04 # Side length of marker (currently set at 4cm)
        
        # Board Information
        # self.m = self.marker_length/2 # half of marker length
        # self.c = 0.05/2 # half of cube length
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
        # self.objectPoints = np.array([[-self.m, self.m, 0], [self.m, self.m, 0], [self.m, -self.m, 0], [-self.m, -self.m, 0]])
        
        # Plotting
        self.plotting = plotting
        self.total_distance = []
        self.total_stddev = []
        self.total_rot = []
        
        # Pose Estimation Variables
        # Check this !!
        self.frame_rate = framerate
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
        pose: numpy.ndarray
        Relative Pose (x,y,z,yaw,pitch,roll)
        cov_matrix: numpy.ndarray
        Covariance matrix (6,6) of estimated pose
        """
        if ids is not None and len(ids) > 0:
            ref_obj_pts, ref_img_pts = reference_board.matchImagePoints(corners,ids)
            target_obj_pts, target_img_pts = target_board.matchImagePoints(corners,ids)

            if (target_img_pts is None or ref_img_pts is None):
                print("Couldn't match object points...")
                return None, None
            
            if (len(target_img_pts) < 3 or len(ref_img_pts) < 3):
                print("Not enough object points for SolvePnP")
                return None, None
            
            # Set Solving Method
            solve_flag = cv2.SOLVEPNP_ITERATIVE #cv2.SOLVEPNP_EPNP
            if self.prev_target_tvec is not None:
                # Estimate pose of Reference board
                ref_val, ref_rvec, ref_tvec = cv2.solvePnP(ref_obj_pts,ref_img_pts,self.cv_cam_mat,self.cv_dist_coeffs,
                                            rvec=self.prev_ref_rvec,tvec=self.prev_ref_tvec,useExtrinsicGuess=True,flags=solve_flag)
                
                # Estimate Pose of Target Board
                target_val, target_rvec, target_tvec = cv2.solvePnP(target_obj_pts,target_img_pts,self.cv_cam_mat,self.cv_dist_coeffs,
                                            rvec=self.prev_target_rvec,tvec=self.prev_target_tvec,useExtrinsicGuess=True,flags=solve_flag)
                   
            else:    
                # Pose of Reference Board
                ref_val, ref_rvec, ref_tvec = cv2.solvePnP(ref_obj_pts,ref_img_pts,self.cv_cam_mat,self.cv_dist_coeffs,
                                                           rvec=None,tvec=None,useExtrinsicGuess=False,flags=solve_flag)
                # Pose of Target Board
                target_val, target_rvec, target_tvec = cv2.solvePnP(target_obj_pts,target_img_pts,self.cv_cam_mat,self.cv_dist_coeffs,
                                                                    rvec=None,tvec=None,useExtrinsicGuess=False,flags=solve_flag)
                self.prev_ref_rvec = ref_rvec
                self.prev_ref_tvec = ref_tvec
                self.prev_target_rvec = target_rvec
                self.prev_target_tvec = target_tvec

            # Pose Refinement added
            # https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
            # Not sure if this helps
            ref_rvec, ref_tvec = cv2.solvePnPRefineLM(ref_obj_pts,ref_img_pts,self.cv_cam_mat,self.cv_dist_coeffs,
                                                      ref_rvec,ref_tvec)
            
            target_rvec, target_tvec = cv2.solvePnPRefineLM(target_obj_pts,target_img_pts,self.cv_cam_mat,self.cv_dist_coeffs,
                                                            target_rvec,target_tvec)
            
            self.ref_tvel = (ref_tvec - self.prev_ref_tvec) / self.dt 
            self.ref_rvel = (ref_rvec - self.prev_ref_rvec) / self.dt

            self.target_tvel = (target_tvec - self.prev_target_tvec) / self.dt
            self.target_rvel = (target_rvec - self.prev_target_rvec) / self.dt

            self.prev_ref_rvec = ref_rvec
            self.prev_ref_tvec = ref_tvec
            self.prev_target_rvec = target_rvec
            self.prev_target_tvec = target_tvec

            if target_val != 0 and ref_val != 0:
                # Reative orientation calculation
                # Put into matrix form, return rotation matrix and jacobian of rotation matrix
                target_rot_mat, _ = cv2.Rodrigues(target_rvec)
                ref_rot_mat, _= cv2.Rodrigues(ref_rvec)
                target_rot_mat = np.array(target_rot_mat)
                ref_rot_mat = np.array(ref_rot_mat)
                
                # Relative Translation Calculation
                # Go from object coordinates to world coordinates
                # T_{t//r} = T_{c//r} + (-T_{c//t})
                rel_trans = ref_tvec - target_tvec  # Camera's reference frame
                rel_trans = ref_rot_mat.T @ rel_trans  # Reference's reference frame
                rel_trans = np.array(rel_trans).reshape((3,1))
                
                # solvePnp returns rotation of camera relative to marker !!
                # R_{t//r} = R_{t//c} @ R_{c//r}
                # As Matrix
                rel_rot_matrix = target_rot_mat.T @ ref_rot_mat
                
                # As Yaw-Pitch-Roll Vector
                rel_rot_ypr = R.from_matrix(rel_rot_matrix).as_euler('ZYX',degrees=True)
                rel_rot_ypr = rel_rot_ypr.reshape((3,1))

                _ , tar_jacobian = cv2.projectPoints(target_obj_pts,target_rvec,target_tvec,self.camera_matrix,self.dist_coeffs)

                sigma = np.linalg.inv(np.dot(tar_jacobian.T,tar_jacobian)[0:6,0:6])
                std_dev = np.sqrt(np.diag(np.abs(sigma)))
                covariance = np.zeros((6,6))
                np.fill_diagonal(covariance,std_dev)
                
                if self.plotting:
                    self.total_distance.append(rel_trans)
                    self.total_stddev.append(std_dev)
                    self.total_rot.append(rel_rot_ypr)
                
            pose = np.vstack((rel_trans, rel_rot_ypr))
            return pose, covariance
        
        else: 
            return None, None
    
    def plot(self, trueTrans=None, trueRot=None):
        total_distance = np.array(self.total_distance)
        total_rot = np.array(self.total_rot)
        x = total_distance[:,1].flatten()
        y = total_distance[:,0].flatten()
        z = total_distance[:,2].flatten()
        if trueTrans is not None: trueTrans = np.array(trueTrans)
        if trueRot is not None: trueRot = np.array(trueRot)
        
        plt.tight_layout()
        plt.subplot(2,1,1)
        plt.plot(x,color='red',linestyle='--',label='ref x')
        plt.plot(y,color='blue',linestyle='--',label='ref y')
        plt.plot(z,color='green',linestyle='--',label='ref z')
        plt.title('Marker Position in Reference Frame')
        plt.ylabel('Position [mm]')
        plt.xlabel('Frames')
        plt.legend()
        
        if trueTrans is not None:
            true_x = trueTrans[:,0]
            true_y = trueTrans[:,1]
            true_z = trueTrans[:,2]  # this had 0.025 subtracted before and I don't know why
            # Need 0.025 in gazebo, as true state of block is relative to gazebo origin 
            # Aah got it, thanks
            plt.plot(true_x,color='red',linestyle='-',label='true x')
            plt.plot(true_y,color='blue',linestyle='-',label='true y')
            plt.plot(true_z,color='green',linestyle='-',label='true z')

        plt.subplot(2,1,2)
        plt.plot(total_rot[:,0],color='red',linestyle='--',label='yaw')
        
        plt.plot(total_rot[:,1],color='blue',linestyle='--',label='pitch')
        
        plt.plot(total_rot[:,2],color='green',linestyle='--',label='roll')
        
        plt.title('Marker Rotation in Reference Frame')
        plt.ylabel('Angular Displacement [deg]')
        plt.xlabel('Frames')
        plt.tight_layout()
        plt.legend()
        
        if trueRot is not None:
            plt.plot(trueRot[:,0],color='red',linestyle='-')
            plt.plot(trueRot[:,1],color='blue',linestyle='-')
            plt.plot(trueRot[:,2],color='green',linestyle='-')
        
        plt.savefig(f'plots/marker_pos {time.strftime("%Y-%m-%d %H-%M-%S")}.pdf')
        plt.show()
        
        if trueTrans is not None:
            x_err = np.abs(x-true_x)
            y_err = np.abs(y-true_y)
            z_err = (z-true_z)
            x_err_rel = x_err / np.abs(true_x)
            y_err_rel = y_err / np.abs(true_y)
            plt.subplot(2,1,1)
            plt.tight_layout()
            plt.plot(x_err,color='red',linestyle='-',label='x error')
            plt.plot(y_err,color='blue',linestyle='-',label='y error')
            plt.plot(z_err,color='green',linestyle='-',label='z error')
            plt.title('Absolute Error')
            plt.ylabel('[mm]')
            plt.xlabel('Frames')
            plt.legend(loc='upper right')

        # plt.subplot(3,1,2)
        # plt.plot(x_err_rel,color='red',linestyle='-',label='x error')
        # plt.plot(y_err_rel,color='blue',linestyle='-',label='y error')
        # plt.title("Relative Error")
        # plt.ylabel('[%]')
        # plt.xlabel('Frames')
        # plt.legend(loc='upper right')
        
        std_dev = np.array(self.total_stddev)
        plt.subplot(2,1,2)
        plt.plot(std_dev[:,0],color='red',linestyle='-',label='x std dev')
        plt.plot(std_dev[:,1],color='blue',linestyle='-',label='y std dev')
        plt.plot(std_dev[:,2],color='green',linestyle='-',label='z std dev')
        plt.title('Standard Deviation')
        plt.ylabel('[pixels]')
        plt.xlabel('Frames')
        plt.legend(loc='upper right')

        
        plt.savefig(f'plots/marker_error {time.strftime("%Y-%m-%d %H-%M-%S")}.pdf')
        plt.show()
        
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
        # rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners,self.marker_length,
        #                                                       self.camera_matrix,
        #                                                       self.dist_coeffs)
        
        if ids is None: return None, None
        
        N_markers = len(ids)
        rvecs = np.zeros((N_markers, 3, 1))
        tvecs = np.zeros((N_markers, 3, 1))
        
        for i in range(N_markers):
            imagePoints = np.ascontiguousarray(corners[i]).reshape((4,1,2))
            retval, rvecs[i], tvecs[i] = cv2.solvePnP(self.objectPoints, imagePoints, self.cv_cam_mat, self.cv_dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        
        # retval, rvecs, tvecs = cv2.solvePnP(self.objectPoints, np.ascontiguousarray(corners).reshape((-1,4,1,2)), self.cv_cam_mat, self.cv_dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        
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
            return None, None
                
                
    