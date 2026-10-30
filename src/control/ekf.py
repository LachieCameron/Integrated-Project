import math
import numpy as np


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class PoseEKF:
    """Differential drive encoder prediction and known-landmark range/bearing update."""
    def __init__(self, initial_pose=(0., 0., 0.), wheel_radius=.033, wheel_base=.287):
        self.x = np.array(initial_pose, dtype=float)
        self.P = np.diag([.15**2, .15**2, math.radians(15)**2])
        self.r, self.L = wheel_radius, wheel_base
        self.Q_per_meter = np.diag([.025**2, .025**2, math.radians(3)**2])
        self.R = np.diag([.08**2, math.radians(5)**2])

    def predict(self, delta_left, delta_right):
        dl, dr = self.r * delta_left, self.r * delta_right
        ds = (dr + dl) / 2
        dtheta = (dr - dl) / self.L
        mid = self.x[2] + dtheta / 2
        F = np.eye(3)
        F[0, 2] = -ds * math.sin(mid)
        F[1, 2] = ds * math.cos(mid)
        self.x += [ds * math.cos(mid), ds * math.sin(mid), dtheta]
        self.x[2] = wrap(self.x[2])
        self.P = F @ self.P @ F.T + self.Q_per_meter * max(abs(ds), .001)

    def update_marker(self, distance, bearing, marker_xy):
        dx, dy = np.asarray(marker_xy) - self.x[:2]
        q = dx*dx + dy*dy
        if q < 1e-8 or not np.isfinite(distance):
            return False
        rho = math.sqrt(q)
        h = np.array([rho, wrap(math.atan2(dy, dx) - self.x[2])])
        H = np.array([[-dx/rho, -dy/rho, 0], [dy/q, -dx/q, -1]])
        innovation = np.array([distance - h[0], wrap(bearing - h[1])])
        S = H @ self.P @ H.T + self.R
        if innovation @ np.linalg.solve(S, innovation) > 9.21:
            return False
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ innovation
        self.x[2] = wrap(self.x[2])
        I = np.eye(3)
        self.P = (I-K@H) @ self.P @ (I-K@H).T + K@self.R@K.T
        return True
