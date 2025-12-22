import os
import cv2

from mini_ma.Method.path import createFileFolder
from mini_ma.Module.detector import Detector


def demo():
    home = os.environ['HOME']
    model_file_path = home + '/chLi/Model/MINIMA/minima_lightglue.pth'
    image1_file_path = home + '/chLi/Dataset/MM/Match/inputimage/c6c113443a8ebb331ed307f33b1385c31a7d0c2fa8ed97b511511048e9e1a4af.jpg'
    image2_file_path = home + '/chLi/Dataset/MM/Match/people_1/gen.PNG'
    save_match_result_folder_path = home + '/chLi/Dataset/MM/Match/people_1/minima_sp_lg/'

    detector = Detector(
        method='sp_lg',
        model_file_path=model_file_path,
    )

    match_result = detector.detectImageFilePair(image1_file_path, image2_file_path)

    if match_result is None:
        print('detectImageFilePair failed!')
        return False

    for key, value in match_result.items():
        try:
            print(key, value.shape)
        except:
            print(key, value)

    img_vis = detector.renderMatchResult(
        match_result,
        image1_file_path,
        image2_file_path,
    )
    save_path=save_match_result_folder_path + "matches_all.jpg"
    createFileFolder(save_path)
    cv2.imwrite(save_path, img_vis)

    img_vis_inliers = detector.renderMatchResult(
        match_result,
        image1_file_path,
        image2_file_path,
        show_inliers_only=True,
    )
    save_path=save_match_result_folder_path + "matches_inliers.jpg"
    createFileFolder(save_path)
    cv2.imwrite(save_path, img_vis_inliers)
    return True
