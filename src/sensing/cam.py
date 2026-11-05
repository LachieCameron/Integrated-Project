"""MuJoCo camera capture and marker detection."""

from pathlib import Path
from dataclasses import dataclass

import mujoco
import cv2
import numpy as np

from src.sensing.camera_calibration import load_calibration


@dataclass(frozen=True)
class MarkerPose:
    """Pose of a marker in the OpenCV camera frame."""

    rotation: np.ndarray
    translation: np.ndarray
    reprojection_error_px: float


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
        self.focal_length_px = (
            (self.height / 2.0)
            / np.tan(np.deg2rad(float(model.cam_fovy[self.cam_id])) / 2.0)
        )
        
        self.renderer = mujoco.Renderer(model, self.height, self.width)
        # Keep rangefinder measurements active, but do not draw their debug rays
        # into the image used by the ArUco detector.
        self.render_options = mujoco.MjvOption()
        self.render_options.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0

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
            self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
            self.parameters = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
            self.legacy_mode = False
        else:
            # Legacy OpenCV (< 4.7.0) API
            self.dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
            self.parameters = cv2.aruco.DetectorParameters_create()
            self.legacy_mode = True

    @property
    def calibrated(self):
        return self.camera_matrix is not None

    def capture_cv_frame(self, data, undistort=None):
        """Render a BGR frame, optionally applying the saved calibration."""
        self.renderer.update_scene(
            data,
            camera=self.cam_id,
            scene_option=self.render_options,
        )
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

    def estimate_marker_pose(
        self,
        corners: np.ndarray,
        marker_size_m: float,
        camera_matrix: np.ndarray | None = None,
    ) -> MarkerPose:
        """Estimate one square marker pose relative to the camera.

        ``corners`` must be the four points returned by OpenCV in its
        top-left, top-right, bottom-right, bottom-left order. The returned
        transform maps marker-frame points into the OpenCV camera frame.
        """
        if marker_size_m <= 0.0:
            raise ValueError("marker_size_m must be positive")
        image_points = np.asarray(corners, dtype=float).reshape(4, 2)
        if not np.isfinite(image_points).all():
            raise ValueError("corners must be finite")

        matrix = self.camera_matrix if camera_matrix is None else np.asarray(camera_matrix)
        if matrix is None:
            matrix = np.array(
                [
                    [self.focal_length_px, 0.0, self.width / 2.0],
                    [0.0, self.focal_length_px, self.height / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("camera_matrix must be a finite 3x3 matrix")
        distortion = (
            np.zeros((5, 1), dtype=float)
            if self.distortion is None
            else self.distortion
        )
        half = marker_size_m / 2.0
        object_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float64,
        )
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success or float(tvec[2, 0]) <= 0.0:
            raise ValueError("could not estimate a marker pose in front of the camera")
        rotation, _ = cv2.Rodrigues(rvec)
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, matrix, distortion
        )
        error = np.linalg.norm(projected.reshape(4, 2) - image_points, axis=1)
        return MarkerPose(
            rotation=rotation,
            translation=tvec.reshape(3),
            reprojection_error_px=float(np.mean(error)),
        )

    @staticmethod
    def camera_pose_from_marker(
        marker_position: np.ndarray,
        marker_rotation: np.ndarray,
        marker_pose: MarkerPose,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Invert a marker observation into camera pose in the world frame.

        ``marker_rotation`` maps marker-frame coordinates into world-frame
        coordinates. The returned tuple is ``(camera_position, camera_rotation)``.
        """
        marker_position = np.asarray(marker_position, dtype=float)
        marker_rotation = np.asarray(marker_rotation, dtype=float)
        if marker_position.shape != (3,):
            raise ValueError("marker_position must have shape (3,)")
        if marker_rotation.shape != (3, 3):
            raise ValueError("marker_rotation must have shape (3, 3)")
        camera_rotation = marker_rotation @ marker_pose.rotation.T
        camera_position = marker_position - camera_rotation @ marker_pose.translation
        return camera_position, camera_rotation
