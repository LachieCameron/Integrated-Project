import numpy as np

class RobotEKF:
    def __init__(self, wheel_base=0.287):
        self.L = wheel_base
        self.state = np.zeros(3)  # [x, y, theta]
        self.P = np.eye(3) * 0.01  # Covariance matrix

    def update(self, d_left, d_right, gyro_z, dt):
        """Kinematics prediction optimized with gyro yaw measurement injection"""
        d_center = (d_left + d_right) / 2.0
        
        # Use gyro directly for heading delta if moving, else fallback to standard wheel kinematics
        if abs(gyro_z) > 0.0001:
            d_theta = gyro_z * dt
        else:
            d_theta = (d_right - d_left) / self.L
            
        # Standard Dead-Reckoning equations
        self.state[0] += d_center * np.cos(self.state[2])
        self.state[1] += d_center * np.sin(self.state[2])
        self.state[2] = (self.state[2] + d_theta) % (2 * np.pi)
        
        return self.state