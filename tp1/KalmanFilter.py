import numpy as np

class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        self.dt = dt
        
        self.u = np.array([[u_x], [u_y]])
        
        self.x = np.array([[0], [0], [0], [0]])
        
        self.A = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        self.B = np.array([
            [(self.dt ** 2) / 2, 0],
            [0, (self.dt ** 2) / 2],
            [self.dt, 0],
            [0, self.dt]
        ])
        
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        self.Q = std_acc ** 2 * np.array([
            [(self.dt ** 4) / 4, 0, (self.dt ** 3) / 2, 0],
            [0, (self.dt ** 4) / 4, 0, (self.dt ** 3) / 2],
            [(self.dt ** 3) / 2, 0, self.dt ** 2, 0],
            [0, (self.dt ** 3) / 2, 0, self.dt ** 2]
        ])
        
        self.R = np.array([
            [x_std_meas ** 2, 0],
            [0, y_std_meas ** 2]
        ])
        
        self.P = np.eye(self.A.shape[1])

    def predict(self):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        
        return self.x[:2]

    def update(self, z):
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)

        I = np.eye(self.H.shape[1])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

        return self.x[:2]
