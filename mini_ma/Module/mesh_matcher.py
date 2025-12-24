import os
import cv2
import torch
import trimesh
import numpy as np
from typing import Optional, Tuple

from camera_control.Module.camera import Camera
from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer

from mini_ma.Method.io import loadImage
from mini_ma.Method.data import toTensor
from mini_ma.Method.path import createFileFolder
from mini_ma.Module.detector import Detector


class MeshMatcher(object):
    def __init__(
        self,
        mesh_file_path: str,
        method: str = "sp_lg",
        model_file_path: Optional[str] = None,
        color: list=[178, 178, 178],
        device: str = 'cuda:0',
    ) -> None:
        self.device = device

        self.detector = Detector(
            method=method,
            model_file_path=model_file_path,
        )

        self.nvdiffrast_renderer = NVDiffRastRenderer(
            mesh_file_path,
            color,
        )
        return

    @property
    def mesh(self) -> trimesh.Trimesh:
        return self.nvdiffrast_renderer.mesh

    def matchMeshToImage(
        self,
        image: np.ndarray,
        camera: Camera,
    ) -> Tuple[dict, dict]:
        render_dict = self.nvdiffrast_renderer.renderImage(
            camera,
            light_direction=[1, 1, 1],
        )

        match_result = self.detector.detect(image, render_dict['image'])

        if match_result is None:
            print('[ERROR][MeshMatcher::matchMeshToImage]')
            print('\t matching pairs detect failed!')
            return render_dict, {}

        return render_dict, match_result

    def matchMeshToImageFile(
        self,
        image_file_path: str,
        save_match_result_folder_path: str,
    ) -> Tuple[dict, dict]:
        if not os.path.exists(image_file_path):
            print('[ERROR][MeshMatcher::matchMeshToImageFile]')
            print('\t image file not exist!')
            print('\t image_file_path:', image_file_path)
            return {}, {}

        min_bound = np.min(self.mesh.vertices, axis=0)
        max_bound = np.max(self.mesh.vertices, axis=0)
        center = (min_bound + max_bound) / 2.0

        # HxWx3
        image = loadImage(image_file_path, is_gray=True)

        init_camera = Camera(
            width=image.shape[1],
            height=image.shape[0],
            pos=center + [0, 0, 1],
            look_at=center,
            up=[0, 1, 0],
            device=self.device,
        )

        render_dict, match_result = self.matchMeshToImage(image, init_camera)

        render_image_file_path = save_match_result_folder_path + 'debug.png'
        createFileFolder(render_image_file_path)
        cv2.imwrite(render_image_file_path, render_dict['image'])

        img_vis = self.detector.renderMatchResult(
            match_result,
            image_file_path,
            render_image_file_path,
        )
        save_path=save_match_result_folder_path + "render_matches_all.jpg"
        createFileFolder(save_path)
        cv2.imwrite(save_path, img_vis)

        # [W, H]
        render_image_pts = torch.from_numpy(np.round(match_result['mkpts1'])).to(torch.int32)

        # HxWx4, u right, v down
        rasterize_output = render_dict['rasterize_output'].detach().cpu()

        matched_mesh_data = rasterize_output[render_image_pts[:, 1], render_image_pts[:, 0]]
        on_mesh_idxs = (matched_mesh_data[:, 3] > 0).nonzero(as_tuple=False).flatten()

        matched_triangle_idxs = matched_mesh_data[on_mesh_idxs, 3].to(torch.int32)
        triangle_vertices = self.mesh.vertices[self.mesh.faces[matched_triangle_idxs]]
        matched_triangle_centers = triangle_vertices.mean(axis=1)

        '''
        mesh_uv = init_camera.project_points_to_uv(matched_triangle_centers)
        mesh_uv[:, 1] = 1.0 - mesh_uv[:, 1]
        mesh_pixel = mesh_uv * torch.tensor([image.shape[1], image.shape[0]], dtype=torch.int32, device=init_camera.device)
        mesh_pixel = mesh_pixel.to(torch.int32)

        img_points_vis = render_dict['image'].copy()
        if len(img_points_vis.shape) == 2:  # grayscale, expand to 3 channels
            img_points_vis = cv2.cvtColor(img_points_vis, cv2.COLOR_GRAY2BGR)

        for pt in mesh_pixel.cpu().numpy():
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(img_points_vis, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

        save_vis_points_path = save_match_result_folder_path + 'mesh_and_image_points.png'
        createFileFolder(save_vis_points_path)
        cv2.imwrite(save_vis_points_path, img_points_vis)

        # 查询rasterize_output[:, 3] >= 0的所有uv，将其颜色设为白色，保存为图片
        mask_valid = (rasterize_output[:, :, 3] > 0)

        # 创建副本以防止修改原图
        green_mask_vis = render_dict['image'].copy()
        if len(green_mask_vis.shape) == 2:  # 如果是灰度，转换为3通道
            green_mask_vis = cv2.cvtColor(green_mask_vis, cv2.COLOR_GRAY2BGR)
        # 将mask区域设为白色
        green_mask_vis[mask_valid] = [0, 255, 0]

        save_green_mask_path = save_match_result_folder_path + 'green_mask_pixels.png'
        createFileFolder(save_green_mask_path)
        cv2.imwrite(save_green_mask_path, green_mask_vis)
        exit()
        '''

        image_uv = toTensor(match_result['mkpts0'] / (image.shape[1], image.shape[0]), device=init_camera.device)
        matched_uv = image_uv[on_mesh_idxs]

        matched_uv[:, 1] = 1.0 - matched_uv[:, 1]

        estimated_camera = Camera.fromUVPoints(
            matched_triangle_centers,
            matched_uv,
            width=init_camera.width,
            height=init_camera.height,
            device=init_camera.device,
        )

        init_camera.outputInfo()
        estimated_camera.outputInfo()

        render_dict, match_result = self.matchMeshToImage(image, estimated_camera)

        render_image_file_path = save_match_result_folder_path + 'debug_fitting.png'
        createFileFolder(render_image_file_path)
        cv2.imwrite(render_image_file_path, render_dict['image'])

        img_vis = self.detector.renderMatchResult(
            match_result,
            image_file_path,
            render_image_file_path,
        )
        save_path=save_match_result_folder_path + "render_matches_all_fitting.jpg"
        createFileFolder(save_path)
        cv2.imwrite(save_path, img_vis)
        return render_dict, match_result
