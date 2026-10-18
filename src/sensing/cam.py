# src/sensing/cam.py
import mujoco
import cv2
import numpy as np

class RaspiCamera:
    def __init__(self, model, camera_name="raspi_cam_v2.1"):
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.cam_id == -1:
            raise RuntimeError(f"Camera '{camera_name}' not found in XML model.")
            
        self.width = int(model.cam_resolution[self.cam_id][0])
        self.height = int(model.cam_resolution[self.cam_id][1])
        
        self.renderer = mujoco.Renderer(model, self.height, self.width)
        
        # Backward-compatible ArUco Initialization Structure
        # Handles both older and modern OpenCV environments seamlessly
        if hasattr(cv2.aruco, 'ArucoDetector'):
            # OpenCV 4.7.0+ API
            self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
            self.parameters = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
            self.legacy_mode = False
        else:
            # Legacy OpenCV (< 4.7.0) API
            self.dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
            self.parameters = cv2.aruco.DetectorParameters_create()
            self.legacy_mode = True

    def capture_cv_frame(self, data):
        """Extracts the image and converts it into a standard OpenCV BGR image matrix"""
        self.renderer.update_scene(data, camera=self.cam_id)
        rgb_image = self.renderer.render()
        
        # Transform: Flip vertically (MuJoCo bottom-left to OpenCV top-left) and map RGB->BGR
        cv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        return cv_image

    def detect_markers(self, cv_image):
        """Detects 5x5 ArUco markers within the camera frame"""
        if not self.legacy_mode:
            # Modern detection
            corners, ids, rejected = self.detector.detectMarkers(cv_image)
        else:
            # Legacy detection fallback
            corners, ids, rejected = cv2.aruco.detectMarkers(cv_image, self.dictionary, parameters=self.parameters)
            
        annotated_image = cv_image.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(annotated_image, corners, ids)
        return corners, ids, annotated_image
