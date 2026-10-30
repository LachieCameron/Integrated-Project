"""MuJoCo camera capture and marker detection."""

from pathlib import Path

import mujoco
import cv2
import numpy as np

from src.sensing.camera_calibration import load_calibration


class RaspiCamera:
    def __init__(
        self,
        model,
        camera_name="raspi_cam_v2.1",
        calibration_path: str | Path | None = None,
    ):
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.cam_id == -1:
            raise RuntimeError(f"Camera '{camera_name}' not found in XML model.")
            
        self.width = int(model.cam_resolution[self.cam_id][0])
        self.height = int(model.cam_resolution[self.cam_id][1])
        
        self.renderer = mujoco.Renderer(model, self.height, self.width)
        self.camera_matrix = None
        self.distortion = None
        if calibration_path is not None:
            self.camera_matrix, self.distortion = load_calibration(calibration_path)
            if self.camera_matrix.shape != (3, 3):
                raise ValueError("camera calibration matrix must have shape (3, 3)")
            if self.distortion.ndim not in (1, 2):
                raise ValueError("camera distortion must be a one- or two-dimensional array")
        
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

    @property
    def calibrated(self):
        return self.camera_matrix is not None

    def capture_cv_frame(self, data, undistort=None):
        """Render a BGR frame, optionally applying the saved calibration."""
        self.renderer.update_scene(data, camera=self.cam_id)
        rgb_image = self.renderer.render()
        
        # Map MuJoCo RGB output to OpenCV BGR.
        cv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        if undistort is None:
            undistort = self.calibrated
        if undistort:
            if not self.calibrated:
                raise ValueError("undistort=True requires calibration_path")
            cv_image = cv2.undistort(
                cv_image,
                self.camera_matrix,
                self.distortion,
            )
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
