# tests/test_camera_stream.py
import sys
import os
import time
import math
import numpy as np
import cv2
import mujoco

# Force workspace package discovery
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.sensing.cam import RaspiCamera

def main():
    # 1. Load the scene layout model directly
    xml_path = os.path.join(project_root, "assets", "scene_turtlebot3_waffle_pi.xml")
    if not os.path.exists(xml_path):
        print(f"Error: Target XML layout file not found at: {xml_path}")
        return

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # 2. Instantiate the OpenCV camera worker profile
    camera = RaspiCamera(model)
    
    print("\n" + "="*50)
    print("  HEADLESS OPENCV CAMERA TEST PIPELINE BOOTED  ")
    print("  Press 'q' inside the video window to exit.   ")
    print("="*50 + "\n")
    
    dt = model.opt.timestep
    last_vision_update = 0.0
    vision_interval = 1.0 / 30.0  # Force image updates to evaluate at a steady 30 Hz

    try:
        while data.time < 10.0:  # Run test loop for up to 10 simulated seconds
            start_wall_time = time.time()
            
            # Spin the motors slightly so the robot circles around to seek markers
            data.ctrl[0] = -1.5  # Left motor wheel spin speed
            data.ctrl[1] = 1.5   # Right motor wheel spin speed
            
            # Progress physics step forward
            mujoco.mj_step(model, data)
            
            # Throttle offscreen rendering frames to 30 FPS to optimize performance
            if data.time - last_vision_update >= vision_interval:
                last_vision_update = data.time
                
                # Extract image matrices straight out of the GPU rendering pipeline
                frame = camera.capture_cv_frame(data)
                corners, ids, annotated_frame = camera.detect_markers(frame)
                
                if ids is not None:
                    print(f"[FOUND] Sim Time: {data.time:.2f}s | Detected ArUco ID(s): {ids.flatten()}")
                else:
                    if int(data.time * 2) % 2 == 0 and (data.time % vision_interval < dt):
                        print(f"[SCANNING] Sim Time: {data.time:.2f}s | No markers in view...")

                # Display frame window output using OpenCV's native high-speed engine
                cv2.imshow("Waffle Pi - Headless RasPi Cam POV", annotated_frame)
                
                # Check keyboard inputs inside the OpenCV window stream container
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting pipeline on user keyboard request.")
                    break
                    
            # Keep execution speed synchronized with real-world wall clock limits
            elapsed = time.time() - start_wall_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
                
    except KeyboardInterrupt:
        print("\nTest pipeline execution stopped by terminal control interrupt.")
        
    finally:
        # Properly destroy graphic pointers when exiting execution scope
        cv2.destroyAllWindows()
        print("Camera testing suite closed cleanly.")

if __name__ == "__main__":
    main()
