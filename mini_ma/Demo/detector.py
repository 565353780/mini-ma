from mini_ma.Module.detector import Detector


def demo():
    model_file_path = "./weights/minima_lightglue.pth"

    detector = Detector(model_file_path)

    match_results = detector.detectImageFilePair(image1_file_path, image2_file_path)
    return True
