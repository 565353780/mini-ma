import os
import torch

from mini_ma.Module.detector import Detector


def demo():
    home = os.environ['HOME']
    image1_file_path = home + '/chLi/Dataset/MM/Match/people_1/input.PNG'
    image2_file_path = home + '/chLi/Dataset/MM/Match/people_1/gen.PNG'

    detector = Detector()

    match_results = detector.detectImageFilePair(image1_file_path, image2_file_path)

    for key, value in match_results.items():
        try:
            print(key, value.shape)
        except:
            print(key, value)
    return True
