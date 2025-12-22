import os
from typing import Optional, Union

from mini_ma.Method.path import createFileFolder


class Detector(object):
    def __init__(
        self,
        model_file_path: Optional[str]=None,
    ) -> None:
        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(
        self,
        model_file_path: str,
    ) -> bool:
        if not os.path.exists(model_file_path):
            print('[ERROR][Detector::loadModel]')
            print('\t model file not exist!')
            print('\t model_file_path:', model_file_path)
            return False

        return True

    def detect(
        self,
    ) -> dict:
        return {}

    def detectImageFilePair(
        self,
        image1_file_path: str,
        image2_file_path: str,
    ) -> Union[dict, None]:
        if not os.path.exists(image1_file_path):
            print('[ERROR][Detector::detectImageFilePair]')
            print('\t image 1 file not exist!')
            print('\t image1_file_path:', image1_file_path)
            return None

        if not os.path.exists(image2_file_path):
            print('[ERROR][Detector::detectImageFilePair]')
            print('\t image 2 file not exist!')
            print('\t image2_file_path:', image2_file_path)
            return None

        return
