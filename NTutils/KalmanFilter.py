import numpy as np
import copy
class KalmanFilter(object):
    def __init__(self, dt, u_x,u_y, std_acc, x_std_meas, y_std_meas,w_max=-1,h_max=-1):
        """
        :param dt: sampling time (time for 1 cycle)
        :param u_x: acceleration in x-direction
        :param u_y: acceleration in y-direction
        :param std_acc: process noise magnitude
        :param x_std_meas: standard deviation of the measurement in x-direction
        :param y_std_meas: standard deviation of the measurement in y-direction
        https://machinelearningspace.com/2d-object-tracking-using-kalman-filter/
        """
        self.w_max=float(w_max)
        self.h_max=float(h_max)
        # Define sampling time
        self.dt = dt
        # Define the  control input variables
        self.u = np.matrix([[u_x],[u_y]])
        # Intial State
        self.x = np.matrix([[0], [0], [0], [0]])
        # Define the State Transition Matrix A
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        # Define the Control Input Matrix B
        self.B = np.matrix([[(self.dt**2)/2, 0],
                            [0, (self.dt**2)/2],
                            [self.dt,0],
                            [0,self.dt]])
        # Define Measurement Mapping Matrix
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        #Initial Process Noise Covariance
        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2
        #Initial Measurement Noise Covariance
        self.R = np.matrix([[x_std_meas**2,0],
                           [0, y_std_meas**2]])
        #Initial Covariance Matrix
        self.P = np.eye(self.A.shape[1])
        self.flag_stable = 0
        
    def predict(self):
        # Refer to :Eq.(9) and Eq.(10)  in https://machinelearningspace.com/object-tracking-simple-implementation-of-kalman-filter-in-python/?preview_id=1364&preview_nonce=52f6f1262e&preview=true&_thumbnail_id=1795
        # Update time state
        #x_k =Ax_(k-1) + Bu_(k-1)     Eq.(9)
        if self.flag_stable:
            return self.stable_pos
        self.x_old = copy.deepcopy(self.x)
        
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        # Calculate error covariance
        # P= A*P*A' + Q               Eq.(10)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        

        if self.w_max>-1 and self.h_max>-1:
            (kf_pred_x,kf_pred_y) = copy.deepcopy(self.x[0:2])
            kf_pred_pos = np.squeeze(np.asarray([kf_pred_x,kf_pred_y]))[0]
            
            if kf_pred_pos[0]>self.w_max or kf_pred_pos[0] <=0 :
                self.flag_stable = 1
                
            if kf_pred_pos[1]>self.h_max or kf_pred_pos[1] <=0 :
                self.flag_stable = 1
                
            if self.flag_stable:
                (kf_pred_x,kf_pred_y) = copy.deepcopy(self.x_old[0:2])
                kf_pred_pos = np.squeeze(np.asarray([kf_pred_x,kf_pred_y]))[0]
                self.stable_pos = copy.deepcopy(kf_pred_pos)
            return kf_pred_pos
        
        return self.x[0:2]
    
    def update(self, z):
        # Refer to :Eq.(11), Eq.(12) and Eq.(13)  in https://machinelearningspace.com/object-tracking-simple-implementation-of-kalman-filter-in-python/?preview_id=1364&preview_nonce=52f6f1262e&preview=true&_thumbnail_id=1795
        # S = H*P*H'+R
        self.flag_stable = 0
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  #Eq.(11)
        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))   #Eq.(12)
        I = np.eye(self.H.shape[1])
        # Update error covariance matrix
        self.P = (I - (K * self.H)) * self.P   #Eq.(13)
        return self.x[0:2]