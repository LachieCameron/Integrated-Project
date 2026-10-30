# src/control/Tag_follow.py
import numpy as np

class DiffDriveController:
    def __init__(self, wheel_base=0.287, wheel_radius=0.033):
        self.L = wheel_base
        self.R = wheel_radius
        self.max_limit = 7.88

    def twist_to_wheel_velocities(self, linear_v, angular_w):
        left_vel = (linear_v - (angular_w * self.L / 2.0)) / self.R
        right_vel = (linear_v + (angular_w * self.L / 2.0)) / self.R
        left_vel = np.clip(left_vel, -self.max_limit, self.max_limit)
        right_vel = np.clip(right_vel, -self.max_limit, self.max_limit)
        return left_vel, right_vel

class PoseTracker:
    """State-feedback controller using estimated robot pose."""

    def __init__(self, k_rho=0.8, k_alpha=1.8,
                 k_beta=-0.5, max_v=0.15, max_w=0.8):
        self.k_rho = k_rho
        self.k_alpha = k_alpha
        self.k_beta = k_beta
        self.max_v = max_v
        self.max_w = max_w

    def command(self, estimated_pose, target_pose, final=False):
        """
        State-feedback navigation using EKF estimated pose.

        Returns:
            v_cmd: linear velocity (m/s)
            w_cmd: angular velocity (rad/s)
            reached: whether the target pose has been reached
        """

        x, y, theta = estimated_pose
        gx, gy, gtheta = target_pose

        distance = np.hypot(gx - x, gy - y)

        heading_error = np.arctan2(
            np.sin(gtheta - theta),
            np.cos(gtheta - theta)
        )

        # Position and orientation tolerances
        position_tolerance = 0.05
        heading_tolerance = 0.08

        # Check whether target position has been reached
        if distance < position_tolerance:

            if not final:
                return 0.0, 0.0, True

            # Final docking requires correct orientation
            if abs(heading_error) < heading_tolerance:
                return 0.0, 0.0, True

            # Rotate to final docking orientation
            w_cmd = np.clip(
                1.5 * heading_error,
                -self.max_w,
                self.max_w
            )

            return 0.0, float(w_cmd), False

        # Use existing state-feedback controller
        v_cmd, w_cmd = self.compute_commands(
            estimated_pose,
            target_pose
        )

        return v_cmd, w_cmd, False
    def compute_commands(self, estimated_pose, target_pose):
        x, y, theta = estimated_pose
        gx, gy, gtheta = target_pose

        dx = gx - x
        dy = gy - y

        rho = np.hypot(dx, dy)

        if rho < 0.05:
            angle_error = np.arctan2(
                np.sin(gtheta - theta),
                np.cos(gtheta - theta)
            )
            if abs(angle_error) < 0.08:
                return 0.0, 0.0

            return 0.0, float(np.clip(
                1.5 * angle_error, -self.max_w, self.max_w
            ))

        alpha = np.arctan2(
            np.sin(np.arctan2(dy, dx) - theta),
            np.cos(np.arctan2(dy, dx) - theta)
        )

        beta = np.arctan2(
            np.sin(gtheta - theta - alpha),
            np.cos(gtheta - theta - alpha)
        )

        linear_v = self.k_rho * rho
        angular_w = self.k_alpha * alpha + self.k_beta * beta

        # Rotate toward the target before moving forward.
        if abs(alpha) > 0.6:
            linear_v = 0.0

        return (
            float(np.clip(linear_v, 0.0, self.max_v)),
            float(np.clip(angular_w, -self.max_w, self.max_w))
        )

class TagFollower:
    def __init__(self, target_id=0, image_width=640, Kp_linear=0.6, Kp_angular=1.8, scan_speed=0.4):
        self.target_id = target_id
        self.center_x = image_width / 2.0
        self.Kp_linear = Kp_linear
        self.Kp_angular = Kp_angular
        self.scan_speed = scan_speed
        
        # State Machine Flags
        self.state = "INITIAL_360_SCAN"  # States: INITIAL_360_SCAN, TRACKING, LOST_STOPPED
        self.has_completed_360 = False
        self.initial_heading = None
        self.accumulated_turn = 0.0
        self.last_heading = None
        
        # New Flag: Remember if we spotted the target at any point during our 360-degree rotation
        self.spotted_target_during_scan = False

    def compute_commands(self, corners, ids, current_heading):
        """
        Computes target [v, w] vectors while tracking a complete 360-degree rotation.
        """
        # Initialize heading references on the very first loop step
        if self.initial_heading is None:
            self.initial_heading = current_heading
            self.last_heading = current_heading
            return 0.0, self.scan_speed

        # Core Check: If we see our target tag during the initial scan phase, log it!
        if ids is not None and self.target_id in ids:
            self.spotted_target_during_scan = True

        # --- STATE 1: FORCED INITIAL 360 SCAN ---
        if self.state == "INITIAL_360_SCAN":
            # Track shortest angular difference across wraparounds (-pi to +pi)
            delta_heading = np.arctan2(np.sin(current_heading - self.last_heading), 
                                      np.cos(current_heading - self.last_heading))
            self.accumulated_turn += abs(delta_heading)
            self.last_heading = current_heading

            # Check if we have made a complete revolution (360 degrees)
            if self.accumulated_turn >= (2 * np.pi - 0.05):
                print(f"[SCAN COMPLETED] Full 360-degree spin executed.")
                self.has_completed_360 = True
                
                # Check memory flags to select the next state routing
                if self.spotted_target_during_scan or (ids is not None and self.target_id in ids):
                    self.state = "TRACKING"
                else:
                    self.state = "LOST_STOPPED"

            return 0.0, self.scan_speed

        # --- STATE 2: ACTIVE TARGET TRACKING ---
        elif self.state == "TRACKING":
            if ids is None or self.target_id not in ids:
                # If we lose the tag while steering, spin slowly in place to re-acquire it
                return 0.0, self.scan_speed * 0.5

            # Find the tag's index array location and flatten it safely
            idx = np.where(ids.flatten() == self.target_id)[0][0]
            # Reshape from OpenCV's native (1, 4, 2) layout down to a clean (4, 2) array layout
            marker_corners = corners[idx].reshape(4, 2)  # Corners order: TL, TR, BR, BL

            # Horizontal alignment calculations: Average across all X points
            marker_center_x = np.mean(marker_corners[:, 0])
            error_angular = (marker_center_x - self.center_x) / self.center_x

            # Proximity metric calculations: Compute Euclidean distance between top-left [0] and top-right [1] vertices
            marker_width_px = np.linalg.norm(marker_corners[0] - marker_corners[1])
            target_width_px = 250.0
            error_linear = (target_width_px - marker_width_px) / target_width_px

            # Control Law calculations
            angular_w = -self.Kp_angular * error_angular
            linear_v = self.Kp_linear * error_linear

            # If we arrive at our stop threshold boundary or overshoot, press the brakes
            if error_linear <= 0.0:
                linear_v = 0.0
                angular_w = 0.0  # Lock steering completely on arrival

            return linear_v, angular_w

        # --- STATE 3: SAFETY HALT ---
        else:
            return 0.0, 0.0
