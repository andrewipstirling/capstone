import numpy as np

class KalmanFilter():
    def __init__(self,A:np.ndarray,B:np.ndarray,C:np.ndarray,
                 Q:np.ndarray,R:np.ndarray,x0:np.ndarray, P0:np.ndarray):
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.P_k = P0
        self.K_k = None
        # State
        self.x = x0
        # Measurements
        self.u_acc = None
        self.y_k = None

    def get_measurement(self, u_acc, y_k):
        self.u_acc = np.array([[u_acc]])
        self.y_k = np.array([[y_k]])

    def predict(self) -> np.ndarray:
        # Insert Prediction step
        self.x = self.A @ self.x + self.B @ self.u_acc
        self.P_k = self.A @ self.P_k @ self.A.T + self.Q
        return self.x
    
    def correct(self) -> None:
        # Insert Correction Step
        tmp = (self.C @ self.P_k @ self.C.T) + self.R
        self.K_k = self.P_k @ self.C.T @ np.linalg.inv(tmp)
        self.x = self.x + self.K_k @ (self.y_k - (self.C @ self.x))
        self.P_k = (np.eye(self.K_k.shape[0],self.C.shape[1]) - 
                    (self.K_k @ self.C)) @ self.P_k
        return