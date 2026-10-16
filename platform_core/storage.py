import os
from datetime import datetime


class ImageStorage:
    """Optional image persistence capability; OpenCV is imported only on use."""

    def __init__(self, base_save_dir: str):
        self.base_save_dir = base_save_dir

    def save_cycle_image(self, frame, cycle_id: str, camera_id: str) -> str:
        import cv2 as cv

        save_dir = os.path.join(
            self.base_save_dir,
            "{}_{}".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), cycle_id),
        )
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, "{}.jpg".format(camera_id))
        if not cv.imwrite(file_path, frame):
            raise IOError("Failed to write image to {}".format(file_path))
        return file_path
