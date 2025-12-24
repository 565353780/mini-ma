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
        camera: Optional[Camera]=None,
    ) -> Tuple[dict, dict, Camera]:
        min_bound = np.min(self.mesh.vertices, axis=0)
        max_bound = np.max(self.mesh.vertices, axis=0)
        center = (min_bound + max_bound) / 2.0

        if camera is None:
            camera = Camera(
                width=image.shape[1],
                height=image.shape[0],
                pos=center + [0, 0, 1],
                look_at=center,
                up=[0, 1, 0],
                device=self.device,
            )

        render_dict = self.nvdiffrast_renderer.renderImage(
            camera,
            light_direction=[1, 1, 1],
        )

        match_result = self.detector.detect(image, render_dict['image'])

        if match_result is None:
            print('[ERROR][MeshMatcher::matchMeshToImage]')
            print('\t matching pairs detect failed!')
            return render_dict, {}, camera

        return render_dict, match_result, camera

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

        image = loadImage(image_file_path, is_gray=True)

        render_dict, match_result, init_camera = self.matchMeshToImage(image)

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

        render_image_pts = torch.from_numpy(match_result['mkpts1']).to(torch.int32)

        rasterize_output = render_dict['rasterize_output'].detach().cpu()

        matched_mesh_data = rasterize_output[render_image_pts[:, 1], render_image_pts[:, 0]]

        on_mesh_idxs = (matched_mesh_data != 0).any(dim=-1).nonzero(as_tuple=False).flatten()

        matched_triangle_idxs = matched_mesh_data[on_mesh_idxs, 3].to(torch.int32)
        triangle_vertices = self.mesh.vertices[self.mesh.faces[matched_triangle_idxs]]
        matched_triangle_centers = triangle_vertices.mean(axis=1)

        image_uv = toTensor(match_result['mkpts0'] / (image.shape[1], image.shape[0]), device=init_camera.device)
        matched_uv = image_uv[on_mesh_idxs]

        estimated_camera = Camera.fromUVPoints(
            matched_triangle_centers,
            matched_uv,
            width=init_camera.width,
            height=init_camera.height,
            device=init_camera.device,
        )

        render_dict, match_result, _ = self.matchMeshToImage(image, estimated_camera)

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
