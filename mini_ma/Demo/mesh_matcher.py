import sys
sys.path.append('../camera-control')

import os

from mini_ma.Module.mesh_matcher import MeshMatcher


def demo():
    home = os.environ['HOME']
    mesh_file_path = home + '/chLi/Dataset/MM/Match/1024result/c6c113443a8ebb331ed307f33b1385c31a7d0c2fa8ed97b511511048e9e1a4afv1_5_-1_stagetwo_1024.glb'
    mesh_file_path = home + '/chLi/Dataset/MM/Match/GTstageone/c6c113443a8ebb331ed307f33b1385c31a7d0c2fa8ed97b511511048e9e1a4af_decoded.ply'
    model_file_path = home + '/chLi/Model/MINIMA/minima_lightglue.pth'
    image_file_path = home + '/chLi/Dataset/MM/Match/inputimage/c6c113443a8ebb331ed307f33b1385c31a7d0c2fa8ed97b511511048e9e1a4af.jpg'
    device = 'cuda:7'
    mesh_color = [178, 178, 178]
    save_match_result_folder_path = home + '/chLi/Dataset/MM/Match/people_1/minima_sp_lg_mesh/'

    mesh_matcher = MeshMatcher(
        mesh_file_path,
        method='sp_lg',
        model_file_path=model_file_path,
        device=device,
        color=mesh_color,
    )

    camera = mesh_matcher.matchMeshToImageFile(
        image_file_path,
        save_match_result_folder_path,
    )

    camera.outputInfo()
    return True
