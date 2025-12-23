import sys
sys.path.append('../camera-control')

import os
import cv2
import numpy as np

from mini_ma.Method.path import createFileFolder
from mini_ma.Module.mesh_matcher import MeshMatcher


def demo():
    home = os.environ['HOME']
    model_file_path = home + '/chLi/Model/MINIMA/minima_lightglue.pth'
    image_file_path = home + '/chLi/Dataset/MM/Match/inputimage/c6c113443a8ebb331ed307f33b1385c31a7d0c2fa8ed97b511511048e9e1a4af.jpg'
    mesh_file_path = home + '/chLi/Dataset/MM/Match/1024result/c6c113443a8ebb331ed307f33b1385c31a7d0c2fa8ed97b511511048e9e1a4afv1_5_-1_stagetwo_1024.glb'
    save_match_result_folder_path = home + '/chLi/Dataset/MM/Match/people_1/minima_sp_lg_mesh/'

    mesh_matcher = MeshMatcher(
        method='sp_lg',
        model_file_path=model_file_path,
    )

    mesh = mesh_matcher.loadMeshFile(mesh_file_path)

    # 打印边界框信息
    min_bound = np.min(mesh.vertices, axis=0)
    max_bound = np.max(mesh.vertices, axis=0)
    center = (min_bound + max_bound) / 2.0

    camera_info = mesh_matcher.queryCamera(
        mesh,
        width=2560,
        height=1440,
        cx=1280,
        cy=720,
        pos=[
            center + [0, 0, -1.0],
            center + [0, 0, 1.0],
        ],
        save_debug_image_path=save_match_result_folder_path + 'debug.png'
    )

    print(mesh)
    for key, value in camera_info.items():
        try:
            print(key, value.shape)
        except:
            pass
    exit()

    match_result = mesh_matcher.matchMeshFileToImageFile(image_file_path, mesh_file_path)

    if match_result is None:
        print('detectImageFilePair failed!')
        return False

    for key, value in match_result.items():
        try:
            print(key, value.shape)
        except:
            print(key, value)

    img_vis = mesh_matcher.renderMatchResult(
        match_result,
        image1_file_path,
        image2_file_path,
    )
    save_path=save_match_result_folder_path + "matches_all.jpg"
    createFileFolder(save_path)
    cv2.imwrite(save_path, img_vis)
    return True
