"""Checkerboard-based camera calibration utilities."""

from pathlib import Path

import cv2
import numpy as np


def find_checkerboard(
    image: np.ndarray,
    pattern_size: tuple[int, int] = (6, 5),
) -> tuple[bool, np.ndarray | None]:
    """Find and sub-pixel refine checkerboard inner corners."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("image must be grayscale or BGR")

    found, corners = cv2.findChessboardCornersSB(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
    )
    if not found or corners is None:
        return False, None

    refined = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(5, 5),
        zeroZone=(-1, -1),
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        ),
    )
    return True, refined.reshape(-1, 1, 2)


def calibrate_from_images(
    image_paths: list[str | Path],
    pattern_size: tuple[int, int] = (6, 5),
    square_size: float = 1.0,
) -> dict[str, np.ndarray | float | int]:
    """Calibrate a pinhole camera from checkerboard images.

    ``square_size`` is in any consistent unit. If the physical board square
    is 25 mm, pass ``square_size=0.025`` to obtain translation vectors in m.
    """
    if not image_paths:
        raise ValueError("at least one calibration image is required")
    if square_size <= 0.0:
        raise ValueError("square_size must be positive")

    object_points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    object_points[:, :2] = (
        np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]]
        .T.reshape(-1, 2)
        * square_size
    )

    all_object_points: list[np.ndarray] = []
    all_image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"could not read calibration image: {path}")
        found, corners = find_checkerboard(image, pattern_size)
        if not found or corners is None:
            continue
        all_object_points.append(object_points.copy())
        all_image_points.append(corners)
        height, width = image.shape[:2]
        current_size = (width, height)
        if image_size is None:
            image_size = current_size
        elif image_size != current_size:
            raise ValueError("all calibration images must have the same size")

    if image_size is None or len(all_image_points) < 3:
        raise ValueError(
            "at least three images with detected checkerboards are required"
        )

    rms, camera_matrix, distortion, rotations, translations = cv2.calibrateCamera(
        all_object_points,
        all_image_points,
        image_size,
        None,
        None,
    )
    return {
        "camera_matrix": camera_matrix,
        "distortion": distortion,
        "rms": float(rms),
        "image_width": image_size[0],
        "image_height": image_size[1],
        "views_used": len(all_image_points),
        "rotations": rotations,
        "translations": translations,
    }


def save_calibration(
    calibration: dict[str, np.ndarray | float | int],
    output_path: str | Path,
) -> None:
    """Save the camera matrix and distortion coefficients as an NPZ file."""
    np.savez(
        output_path,
        camera_matrix=calibration["camera_matrix"],
        distortion=calibration["distortion"],
        rms=calibration["rms"],
        image_width=calibration["image_width"],
        image_height=calibration["image_height"],
        views_used=calibration["views_used"],
    )


def load_calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``camera_matrix`` and ``distortion`` from a saved calibration."""
    with np.load(path) as data:
        return data["camera_matrix"], data["distortion"]
