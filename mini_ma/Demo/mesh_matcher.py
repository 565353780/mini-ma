import sys
sys.path.append('../camera-control')

import os

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

    render_dict, match_result = mesh_matcher.matchMeshFileToImageFile(
        image_file_path,
        mesh_file_path,
        save_match_result_folder_path,
    )

    if render_dict is None:
        print('render failed!')
        return False

    if match_result is None:
        print('match failed!')
        return False

    for key, value in render_dict.items():
        try:
            print(key, value.shape)
        except:
            pass

    for key, value in match_result.items():
        try:
            print(key, value.shape)
        except:
            print(key, value)
    return True
