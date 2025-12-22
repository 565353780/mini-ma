import cv2
import numpy as np
from typing import Optional


def loadImage(
    image_file_path: str,
    is_gray: bool=False,
) -> Optional[np.ndarray]:
    imread_flag = cv2.IMREAD_GRAYSCALE if is_gray else cv2.IMREAD_COLOR

    image_data = cv2.imread(image_file_path, imread_flag)

    if image_data is None:
        print('[ERROR][io::loadImage]')
        print('\t imread failed!')
        print('\t image_file_path:', image_file_path)
        return None

    return image_data
