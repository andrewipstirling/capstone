import numpy as np

class KalmanFilterCV():
    """Constant Velocity Kalman Filter."""
    def __init__(self, freq=60, process_noise=0.001):
        """
        Initializes the Kalman Filter with given parameters.

        Args:
            freq (float): Frequency of measurements in Hz (default 60).
            process_noise (float): Noise of dynamics (process) model (default = 1mm)
        """
        # States = x, y, z, yaw, pitch, roll, dx, dy, dz, dyaw, dpitch, droll
        # []
        dt = 1 / freq

        self.A = np.array([[1, 0, 0, 0, 0, 0, dt, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, dt, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, dt, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, dt, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, dt, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, dt],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        
        self.B = 0
        # Measurement is only x, not dx
        self.C = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
        # Process Noise
        self.Q = np.eye(12,12) * process_noise
        self.R = np.diag([0.008,0.008,0.01,0.0008,0.0008,0.0008])
        self.P_k = np.zeros((12,12))
        self.K_k = None
        # State
        self.x = None
        # Measurements
        self.u_acc = None
        self.y_k = None

    def initiate_state(self, x0):
        """
        Initializes the state of the Kalman filter.

        Args:
            x0 (np.ndarray): Initial state vector. Shape must be (6,1).
        """
        # State
        self.x = np.vstack((x0,np.zeros((6,1))))

    def set_measurement(self, y_k):
        """
        Sets the measurement for the Kalman filter.

        Args:
            y_k (np.ndarray): Measurement vector. Measurement must be (6,1).
        """
        self.u_acc = 0
        self.y_k = y_k

    def set_dt(self,dt):
        """
        Sets the time interval between measurements.

        Args:
            dt (float): Time interval between measurements.
        """
        self.A = np.array([[1, 0, 0, 0, 0, 0, dt, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, dt, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, dt, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, dt, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, dt, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, dt],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        
    def predict(self) -> np.ndarray:
        """
        Prediction step of the Kalman filter.

        Returns:
            np.ndarray: Predicted state vector, of shape (12,1). First 6 contain pose.
        """
        # Prediction step
        self.x = self.A @ self.x 
        self.P_k = self.A @ self.P_k @ self.A.T + self.Q
        return self.x
    
    def correct(self) -> None:
        """
        Correction step of the Kalman filter.
        """
        # Correction Step
        tmp = (self.C @ self.P_k @ self.C.T) + self.R
        self.K_k = self.P_k @ self.C.T @ np.linalg.solve(tmp,np.eye(tmp.shape[0],tmp.shape[1]))
        self.x = self.x + self.K_k @ (self.y_k - (self.C @ self.x))
        self.P_k = (np.eye(self.K_k.shape[0],self.C.shape[1]) - 
                    (self.K_k @ self.C)) @ self.P_k
        return
