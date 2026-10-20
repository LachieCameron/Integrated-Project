"""Calibrate the camera from checkerboard image files.

Example:
    python scripts/calibrate_camera.py captures/*.png --square-size 0.025
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.sensing.camera_calibration import calibrate_from_images, save_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("camera_calibration.npz"))
    parser.add_argument("--square-size", type=float, default=1.0)
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()

    calibration = calibrate_from_images(
        args.images,
        pattern_size=(args.columns, args.rows),
        square_size=args.square_size,
    )
    save_calibration(calibration, args.output)
    print(
        f"Used {calibration['views_used']} views; "
        f"RMS reprojection error: {calibration['rms']:.4f}"
    )
    print(f"Saved calibration to {args.output}")


if __name__ == "__main__":
    main()
