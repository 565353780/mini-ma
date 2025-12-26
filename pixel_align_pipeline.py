import sys

sys.path.append('../camera-control')
sys.path.append('../non-rigid-icp')
sys.path.append('../cage-deform')

import os
import cv2
import trimesh
import numpy as np

from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer

from mini_ma.Method.io import loadImage, loadMeshFile
from mini_ma.Module.detector import Detector
from mini_ma.Module.camera_matcher import CameraMatcher
from mini_ma.Module.mesh_deformer import MeshDeformer


if __name__ == '__main__':
    home = os.environ['HOME']
    mesh_file_path = home + '/chLi/Dataset/MM/Match/1024result/c6c113443a8ebb331ed307f33b1385c31a7d0c2fa8ed97b511511048e9e1a4afv1_5_-1_stagetwo_1024.glb'
    model_file_path = home + '/chLi/Model/MINIMA/minima_roma.pth'
    image_file_path = home + '/chLi/Dataset/MM/Match/inputimage/c6c113443a8ebb331ed307f33b1385c31a7d0c2fa8ed97b511511048e9e1a4af.jpg'
    device = 'cuda:7'
    mesh_color = [178, 178, 178]
    max_deform_ratio = 0.05
    save_match_result_folder_path = home + '/chLi/Dataset/MM/Match/people_1/minima_mesh/'

    detector = Detector(
        method='roma',
        model_file_path=model_file_path,
        device=device,
    )

    # HxWx3
    image = loadImage(image_file_path)
    assert image

    mesh = loadMeshFile(mesh_file_path)
    assert mesh

    camera, render_dict, match_result = CameraMatcher.matchCameraToMeshImage(
        image,
        mesh,
        detector,
        save_match_result_folder_path,
        iter_num=1,
    )
    assert camera
    assert render_dict
    assert match_result

    matched_uv, matched_triangle_idxs = CameraMatcher.extractMatchedUVTriangle(
        render_dict, match_result, device)

    source_vertex_idxs, target_vertices = MeshDeformer.searchDeformPairs(
        mesh, camera, render_dict, match_result)

    filtered_source_vertex_idxs, filtered_target_vertices = MeshDeformer.filterDeformPairs(
        mesh, source_vertex_idxs, target_vertices, max_deform_ratio)

    deformed_mesh = MeshDeformer.deformMeshByCage(
        mesh, filtered_source_vertex_idxs, filtered_target_vertices, device)

    # 保存为PLY点云文件
    if save_match_result_folder_path is not None:
        os.makedirs(save_match_result_folder_path, exist_ok=True)

        source_pcd_path = save_match_result_folder_path + 'matched_source_vertices.ply'
        target_pcd_path = save_match_result_folder_path + 'matched_target_vertices.ply'
        deformed_mesh_path = save_match_result_folder_path + 'deformed_mesh.ply'
        deformed_mesh_iou_image_path = save_match_result_folder_path + 'deformed_iou.png'

        source_pcd = trimesh.PointCloud(vertices=mesh.vertices[filtered_source_vertex_idxs])
        target_pcd = trimesh.PointCloud(vertices=filtered_target_vertices)

        source_pcd.export(source_pcd_path)
        target_pcd.export(target_pcd_path)

        deformed_mesh.export(deformed_mesh_path)

        nvdiffrast_renderer = NVDiffRastRenderer(deformed_mesh_path, [178, 178, 178])

        render_dict = nvdiffrast_renderer.renderImage(
            camera,
            light_direction=[1, 1, 1],
        )

        iou_vis = CameraMatcher.renderIoU(image, render_dict)
        cv2.imwrite(deformed_mesh_iou_image_path, iou_vis)

        # HxWx4, u right, v down
        rasterize_output = render_dict['rasterize_output']

        # 计算mesh渲染出来的像素mask
        mesh_mask = rasterize_output[..., 3] > 0  # [H, W] bool

        # RGB阈值
        white_thr = int(0.93 * 255)   # 可以调
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            # 灰度图情况
            white_mask = image < white_thr
        elif image.ndim == 3 and image.shape[2] >= 3:
            # 彩色或有alpha通道的情况
            white_mask = np.all(image[..., :3] < white_thr, axis=-1)
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        mesh_mask_np = mesh_mask.cpu().numpy()

        # 保存mask为图片
        # HxW单通道mask保存为png需转换为HxWx1, 否则有些cv2读取为三通道错误
        mesh_mask_img = (mesh_mask_np.astype(np.uint8) * 255)
        mesh_mask_img = mesh_mask_img[..., None]  # [H,W,1]
        white_mask_img = (white_mask.astype(np.uint8) * 255)
        white_mask_img = white_mask_img[..., None]  # [H,W,1]

        mesh_mask_path = save_match_result_folder_path + 'mesh_mask.png'
        white_mask_path = save_match_result_folder_path + 'white_mask.png'

        cv2.imwrite(mesh_mask_path, mesh_mask_img)
        cv2.imwrite(white_mask_path, white_mask_img)
