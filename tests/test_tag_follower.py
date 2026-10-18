import sys
import os
import time
import math
import numpy as np
import cv2
import mujoco
import mujoco.viewer

# Append path maps to resolve workspace package discovery
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.sensing.cam import RaspiCamera
from src.sensing.IMU import IMUSensor
from src.control.Tag_follow import DiffDriveController, TagFollower

# Global tracking variables to preserve state across fast callback threads
current_v, current_w = 0.0, 0.0
last_cam_time = 0.0
cam_interval = 1.0 / 30.0  # Throttle vision tasks to 30 Hz

# --- IMU Heading Integrator Variables ---
integrated_heading = 0.0   # Tracks current heading orientation in radians
last_imu_time = 0.0

def main():
    global integrated_heading, last_imu_time
    
    xml_path = os.path.join(project_root, "assets", "scene_turtlebot3_waffle_pi.xml")
    if not os.path.exists(xml_path):
        print(f"Error: Target XML layout file missing at: {xml_path}")
        return

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Initialize your modular software stack workers
    camera = RaspiCamera(model)
    imu = IMUSensor(model)
    diff_drive = DiffDriveController()
    
    # Initialize the 5x5 tag follower (passing image resolution width)
    follower = TagFollower(target_id=0, image_width=camera.width)

    print("\n" + "="*60)
    print("  AUTONOMOUS INTEGRATED SENSING & CONTROL PIPELINE BOOTED  ")
    print("  Tracking Mode: Forced 360 Scan -> Seek Tag 0 (5x5 Matrix) ")
    print("="*60 + "\n")

    def control_loop(model, data):
        global integrated_heading, last_imu_time, current_v, current_w, last_cam_time
        
        dt = model.opt.timestep

        # 1. READ IMU AND INTEGRATE YAW RATE (Executed at full 100Hz physics speed)
        # accel vector = [ax, ay, az], gyro vector = [wx, wy, wz]
        accel, gyro = imu.read(data)
        gyro_z = gyro[2]  # Pull the vertical Z-axis turning speed (rad/s)

        # Integrate the turning velocity to compute current relative angle orientation
        if data.time > 0:
            # Apply a minor deadband threshold to filter out static sensor noise drift
            if abs(gyro_z) > 0.001:
                integrated_heading += gyro_z * dt
                
            # Bound heading angle values strictly between -pi and +pi radians (standard ROS convention)
            integrated_heading = math.atan2(math.sin(integrated_heading), math.cos(integrated_heading))

        # 2. VISION PROCESSING LOOP (Executed at throttled 30 FPS slots)
        if data.time - last_cam_time >= cam_interval:
            last_cam_time = data.time
            
            # Capture frame matrix and run 5x5 ArUco marker tracker arrays
            frame = camera.capture_cv_frame(data)
            corners, ids, annotated_frame = camera.detect_markers(frame)
            
            # --- FIXED: Pass the integrated IMU heading as the 3rd argument ---
            current_v, current_w = follower.compute_commands(corners, ids, integrated_heading)
            
            # Telemetry printing management
            if follower.state == "INITIAL_360_SCAN":
                print(f"[SCANNING] Time: {data.time:.2f}s | Turn Progress: {math.degrees(follower.accumulated_turn):.1f}° / 360°")
            elif follower.state == "TRACKING":
                print(f"[TRACKING] Time: {data.time:.2f}s | Target Locked! Driving with V={current_v:.2f}m/s, W={current_w:.2f}rad/s")
            else:
                print(f"[HALTED] Time: {data.time:.2f}s | Scan complete. Room empty. Safe brakes engaged.")

            # Overlay real-time states and metrics directly onto the OpenCV stream frame
            status_text = f"Mode: {follower.state}"
            angle_text = f"IMU Yaw: {math.degrees(integrated_heading):.1f} deg"
            cv2.putText(annotated_frame, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated_frame, angle_text, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Display live camera frame point-of-view layout
            cv2.imshow("Waffle Pi - Headless Tracking Stream Monitor", annotated_frame)
            cv2.waitKey(1)

        # 3. ACTUATOR DESPATCH (Converts v, w target maps -> wheel velocity commands)
        left_vel, right_vel = diff_drive.twist_to_wheel_velocities(current_v, current_w)
        data.ctrl[0] = left_vel
        data.ctrl[1] = right_vel

    try:
        # Register the completed integration loop directly with the solver hook
        mujoco.set_mjcb_control(control_loop)
        
        # Fire up active window bindings natively on the main process thread
        mujoco.viewer.launch(model, data)
        
    except KeyboardInterrupt:
        print("\nSimulation session terminated by terminal context break.")
    finally:
        # Erase lingering handles and tear down UI windows cleanly
        mujoco.set_mjcb_control(None)
        cv2.destroyAllWindows()
        print("Sensing and tracking integration suite exited cleanly.")

if __name__ == "__main__":
    main()
