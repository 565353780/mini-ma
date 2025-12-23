import sys
sys.path.append('../camera-control')

import os

from mini_ma.Module.mesh_matcher import MeshMatcher


def demo():
    home = os.environ['HOME']
    mesh_file_path = home + '/chLi/Dataset/MM/Match/1024result/c6c113443a8ebb331ed307f33b1385c31a7d0c2fa8ed97b511511048e9e1a4afv1_5_-1_stagetwo_1024.glb'
    model_file_path = home + '/chLi/Model/MINIMA/minima_lightglue.pth'
    image_file_path = home + '/chLi/Dataset/MM/Match/inputimage/c6c113443a8ebb331ed307f33b1385c31a7d0c2fa8ed97b511511048e9e1a4af.jpg'
    save_match_result_folder_path = home + '/chLi/Dataset/MM/Match/people_1/minima_sp_lg_mesh/'
    device = 'cuda:7'
    color = [178, 178, 178]

    mesh_matcher = MeshMatcher(
        mesh_file_path,
        method='sp_lg',
        model_file_path=model_file_path,
        device=device,
        color=color,
    )

    render_dict, match_result = mesh_matcher.matchMeshToImageFile(
        image_file_path,
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
