from collections import deque
import numpy as np
import time


class FilterNone:
    def __init__(self):
        pass

    def filter(self, point):
        return point


class FilterMedian:
    BUFFER_SIZE = 9

    def __init__(self):
        self.buffer = deque([], self.BUFFER_SIZE)

    def filter(self, point):
        self.buffer.append(np.array(point))
        if len(self.buffer) == self.BUFFER_SIZE:
            vals = np.array(list(self.buffer))
            return np.median(vals, axis=0)
        else:
            return None


class FilterKalman:
    '''
    Kalman filter with constant Q and R
    '''

    def __init__(self, freq = 65):
        self.prev_time = 0
        self.prev_point = np.array([0, 0, 0], dtype=float)

        # States
        self.x = np.matrix(np.zeros(6)).T  # state prediction [x, y, z, vx, vy, vz]
        self.P = np.matrix(np.zeros((6, 6)), dtype=float)  # state err matrix

        self.q = 0.2  # Action uncertainty (process variance)
        self.r = 0.1  # Sensor noise

        # varies with dt
        self.A = np.matrix([])  # State transition
        # constants
        self.H = np.matrix([])  # Measurement. influences Kalman Gain K
        self.Q = np.matrix([])  # Action uncertainty (process variance)
        self.R = np.matrix([])  # Sensor noise
        
        self.dt = 1/freq

        self.init_matrix_except_depending_on_dt()
        self.init_matrix_depending_on_dt(dt=self.dt)

    def filter(self, measured_point):
        current_time = time.time()  # TODO use time of img capture
        self.init_matrix_depending_on_dt(current_time - self.prev_time)
        x = self.x
        P = self.P
        I = np.matrix((np.eye(6, 6)))
        A = self.A
        H = self.H
        Q = self.Q
        R = self.R

        # measurement vector
        measured_point, m = self.measurement(measured_point)

        # prediction step
        x = A * x
        P = Q + (A * P * A.T)

        # correction step
        S = R + (H * P * H.T)
        K = P * H.T * S.I
        y = m - (H * x)
        self.x = x + (K * y)
        self.P = (I - (K * H)) * P

        new_point = np.array(self.x[0:3].T)[0, :]
        self.prev_point = new_point
        self.prev_time = current_time
        return np.append(new_point, [1])  # cart 2 hom

    def init_matrix_depending_on_dt(self, dt=0.1):
        self.A = np.matrix([[1, 0, 0, dt, 0, 0],
                            [0, 1, 0, 0, dt, 0],
                            [0, 0, 1, 0, 0, dt],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]], dtype=float)

    def init_matrix_except_depending_on_dt(self):
        self.H = np.matrix([[1, 0, 0, 1, 0, 0],
                            [0, 1, 0, 0, 1, 0],
                            [0, 0, 1, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0]], dtype=float)
        self.Q = np.matrix([[0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]], dtype=float) * self.q
        self.R = np.matrix(np.eye(6, 6), dtype=float) * self.r

    def measurement(self, point):
        point = np.array(point[0:3])
        dPoint = point - self.prev_point
        return point, np.matrix(np.concatenate([point, dPoint], axis=None)).T

class KalmanFilterCV():
    def __init__(self, freq=60):
        # States = x, y, z, yaw, pitch, roll, dx, dy, dz, dyaw, dpitch, droll
        # []
        dt = 1 / freq
        
        process_noise = 0.0001
        

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
        # State
        self.x = np.vstack((x0,np.zeros((6,1))))

    def get_measurement(self, y_k):
        self.u_acc = 0
        self.y_k = y_k

    def predict(self) -> np.ndarray:
        # Prediction step
        self.x = self.A @ self.x 
        self.P_k = self.A @ self.P_k @ self.A.T + self.Q
        return self.x
    
    def correct(self) -> None:
        # Correction Step
        tmp = (self.C @ self.P_k @ self.C.T) + self.R
        self.K_k = self.P_k @ self.C.T @ np.linalg.solve(tmp,np.eye(tmp.shape[0],tmp.shape[1]))
        self.x = self.x + self.K_k @ (self.y_k - (self.C @ self.x))
        self.P_k = (np.eye(self.K_k.shape[0],self.C.shape[1]) - 
                    (self.K_k @ self.C)) @ self.P_k
        return
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    kalman_filter = FilterKalman()
    # kalman_filter = FilterMedian()

    n_iter = 50
    sz = (n_iter,)  # size of array

    # allocate space for arrays
    xhat = np.zeros(sz)  # a posteri estimate of x
    yhat = np.zeros(sz)

    start_x = 20  # TODO try 20
    start_y = 20  # TODO try 20

    xtrue = np.add.accumulate(np.linspace(start_x, 100, n_iter))
    ytrue = np.linspace(start_y, 125, n_iter)
    xsensor = xtrue + np.random.normal(0, 0.8, size=sz)
    ysensor = ytrue + np.random.normal(0, 0.8, size=sz)

    for k in range(1, n_iter):
        time.sleep(0.05)
        point = np.array([xsensor[k], ysensor[k], 0])
        new_point = kalman_filter.filter(point)
        if new_point is None:
            new_point = (0, 0, 0)
        xhat[k], yhat[k], _ = new_point

    plt.figure()
    plt.plot(xsensor, 'k+', label='measurement (kalman input)')
    plt.plot(xhat, 'b-', label='kalman position estimate')
    plt.plot(xtrue, color='g', label='truth')
    plt.plot(ysensor, 'k+', label='measurement (kalman input)')
    plt.plot(yhat, 'b-', label='kalman position estimate')
    plt.plot(ytrue, color='g', label='truth')
    plt.legend()
    plt.title('Estimate vs. iteration step', fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('Voltage')
    plt.show()
