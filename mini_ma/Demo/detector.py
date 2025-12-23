import os
import cv2

from mini_ma.Method.path import createFileFolder
from mini_ma.Module.detector import Detector


home = os.environ['HOME']
image_pairs_dict = {
    'people_1': [
        home + '/chLi/Dataset/MM/Match/inputimage/c6c113443a8ebb331ed307f33b1385c31a7d0c2fa8ed97b511511048e9e1a4af.jpg',
        home + '/chLi/Dataset/MM/Match/people_1/gen.PNG',
    ],
    'people_2': [
        home + '/chLi/Dataset/MM/Match/inputimage/download_fullbody_man.png',
        home + '/chLi/Dataset/MM/Match/people_2/gen.PNG',
    ],
    'people_1_stage1gt': [
        home + '/chLi/Dataset/MM/Match/inputimage/c6c113443a8ebb331ed307f33b1385c31a7d0c2fa8ed97b511511048e9e1a4af.jpg',
        home + '/chLi/Dataset/MM/Match/people_1/gen_stage1gt.PNG',
    ],
    'people_2_stage1gt': [
        home + '/chLi/Dataset/MM/Match/inputimage/download_fullbody_man.png',
        home + '/chLi/Dataset/MM/Match/people_2/gen_stage1gt.PNG',
    ],
}


def demo_folders():
    model_file_path = home + '/chLi/Model/MINIMA/minima_lightglue.pth'

    detector = Detector(
        method='sp_lg',
        model_file_path=model_file_path,
    )

    for image_pair_id in image_pairs_dict.keys():
        model_file_path = home + '/chLi/Model/MINIMA/minima_lightglue.pth'
        image1_file_path, image2_file_path = image_pairs_dict[image_pair_id]
        save_match_result_folder_path = home + '/chLi/Dataset/MM/Match/' + image_pair_id + '/minima_sp_lg/'

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

def demo():
    image_pair_id = 'people_1'

    model_file_path = home + '/chLi/Model/MINIMA/minima_lightglue.pth'
    image1_file_path, image2_file_path = image_pairs_dict[image_pair_id]
    save_match_result_folder_path = home + '/chLi/Dataset/MM/Match/' + image_pair_id + '/minima_sp_lg/'

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
