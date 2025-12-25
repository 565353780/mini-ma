import os
import cv2
import torch
import trimesh
import numpy as np
from typing import Optional, Tuple

from camera_control.Module.camera import Camera
from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer

from mini_ma.Method.io import loadImage
from mini_ma.Method.data import toNumpy, toTensor
from mini_ma.Method.path import createFileFolder
from mini_ma.Module.detector import Detector


class CameraMatcher(object):
    def __init__(
        self,
        mesh_file_path: str,
        method: str = "roma",
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

    def estimateCamera(
        self,
        render_dict: dict,
        match_result: dict,
    ) -> Camera:
        # [W, H]
        render_image_pts = torch.from_numpy(
            np.round(match_result['mkpts1'])).to(dtype=torch.int32, device=self.device)

        # HxWx4, u right, v down
        rasterize_output = render_dict['rasterize_output']

        height, width = rasterize_output.shape[:2]

        # 保证索引不越界：只保留在[0, H)和[0, W)内的点
        valid_mask = (render_image_pts[:, 0] >= 0) & (render_image_pts[:, 0] < width) & \
                     (render_image_pts[:, 1] >= 0) & (render_image_pts[:, 1] < height)
        safe_pts = render_image_pts[valid_mask]

        matched_mesh_data = rasterize_output[safe_pts[:, 1], safe_pts[:, 0]]
        on_mesh_idxs = (matched_mesh_data[:, 3] > 0).nonzero(as_tuple=False).flatten()

        matched_triangle_idxs = toNumpy(matched_mesh_data[on_mesh_idxs, 3].to(torch.int32))
        triangle_vertices = self.mesh.vertices[self.mesh.faces[matched_triangle_idxs]]
        matched_triangle_centers = triangle_vertices.mean(axis=1)

        image_uv = toTensor(match_result['mkpts0'], torch.float32, self.device) / torch.tensor(
            [width, height], dtype=torch.float32, device=self.device)
        matched_uv = image_uv[on_mesh_idxs]

        matched_uv[:, 1] = 1.0 - matched_uv[:, 1]

        estimated_camera = Camera.fromUVPoints(
            matched_triangle_centers,
            matched_uv,
            width=width,
            height=height,
            device=self.device,
        )

        return estimated_camera

    def getIoU(
        self,
        image: np.ndarray,
        render_dict: dict,
    ) -> float:
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

        # 计算IoU
        intersection = np.logical_and(mesh_mask.cpu().numpy(), white_mask)
        union = np.logical_or(mesh_mask.cpu().numpy(), white_mask)
        iou = intersection.sum() / (union.sum() + 1e-5)

        return iou

    def matchCameraToMeshImageFile(
        self,
        image_file_path: str,
        save_match_result_folder_path: Optional[str],
        iter_num: int = 1,
    ) -> Optional[Camera]:
        render = save_match_result_folder_path is not None

        if not os.path.exists(image_file_path):
            print('[ERROR][CameraMatcher::matchCameraToMeshImageFile]')
            print('\t image file not exist!')
            print('\t image_file_path:', image_file_path)
            return {}, {}

        # HxWx3
        image = loadImage(image_file_path, is_gray=True)

        init_camera = Camera(
            width=image.shape[1],
            height=image.shape[0],
            pos=[0, 0, 1],
            look_at=[0, 0, 0],
            up=[0, 1, 0],
            device=self.device,
        )
        init_camera.focusOnPoints(self.mesh.vertices)
        light_direction = [1, 1, 1]

        render_dict = self.nvdiffrast_renderer.renderImage(
            init_camera,
            light_direction=light_direction,
        )

        is_match_updated = False
        if render:
            match_result = self.detector.detect(image, render_dict['image'])
            is_match_updated = True

            if match_result is None:
                print('[ERROR][CameraMatcher::matchMeshImagePairs]')
                print('\t matching pairs detect failed!')
                return None

            concat_vis = self.renderMatchResult(
                image_file_path,
                render_dict,
                match_result,
            )

            save_path = save_match_result_folder_path + "matches_0.jpg"
            createFileFolder(save_path)
            cv2.imwrite(save_path, concat_vis)

        best_iou = 0
        best_camera = init_camera.clone()

        for i in range(1, 1 + iter_num):
            if not is_match_updated:
                match_result = self.detector.detect(image, render_dict['image'])
                is_match_updated = True

            estimated_camera = self.estimateCamera(render_dict, match_result)

            render_dict = self.nvdiffrast_renderer.renderImage(
                estimated_camera,
                light_direction=light_direction,
            )

            iou = self.getIoU(image, render_dict)

            if iou > best_iou:
                best_iou = iou
                best_camera = estimated_camera.clone()
                print('[INFO][CameraMatcher::matchCameraToMeshImageFile]')
                print('\t best_iou:', best_iou)
                print('\t best_match_idx:', i)

            is_match_updated = False
            if render:
                match_result = self.detector.detect(image, render_dict['image'])
                is_match_updated = True

                if match_result is None:
                    print('[ERROR][CameraMatcher::matchMeshImagePairs]')
                    print('\t matching pairs detect failed!')
                    return None

                concat_vis = self.renderMatchResult(
                    image_file_path,
                    render_dict,
                    match_result,
                )
                save_path = save_match_result_folder_path + "matches_" + str(i) + ".jpg"
                createFileFolder(save_path)
                cv2.imwrite(save_path, concat_vis)

        return best_camera

    def renderIoU(
        self,
        image: np.ndarray,
        render_dict: dict,
    ) -> np.ndarray:
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

        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            # 灰度图转三通道
            image_vis = np.stack([image.squeeze()] * 3, axis=-1)
        else:
            image_vis = image.copy()

        # 计算IoU
        intersection = np.logical_and(mesh_mask.cpu().numpy(), white_mask)
        union = np.logical_or(mesh_mask.cpu().numpy(), white_mask)
        iou = intersection.sum() / (union.sum() + 1e-5)

        iou_vis = np.zeros_like(image_vis)
        mesh_mask_np = mesh_mask.cpu().numpy()
        intersection_mask = np.logical_and(mesh_mask_np, white_mask)
        mesh_only_mask = np.logical_and(mesh_mask_np, ~white_mask)
        white_only_mask = np.logical_and(~mesh_mask_np, white_mask)
        iou_vis[mesh_only_mask] = [0,0,255]
        iou_vis[white_only_mask] = [0,255,0]
        iou_vis[intersection_mask] = [255,0,0]

        # 在iou_vis的左上角写上IoU数值（白色字体）
        iou_text = f"IoU: {iou:.3f}" if isinstance(iou, float) or isinstance(iou, np.floating) else f"IoU: {float(iou):.3f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (255, 255, 255) # 白色字体
        thickness = 2
        org = (10, 30)  # 左上角位置
        cv2.putText(iou_vis, iou_text, org, font, font_scale, color, thickness, cv2.LINE_AA)
        return iou_vis

    def renderMatchResult(
        self,
        image_file_path: str,
        render_dict: dict,
        match_result: dict,
    ) -> np.ndarray:
        img_vis = self.detector.renderMatchResult(
            match_result,
            image_file_path,
            render_dict['image'],
        )

        image = loadImage(image_file_path, is_gray=False)

        iou_vis = self.renderIoU(image, render_dict)

        # 拼接iou_vis和img_vis（H相同，W可能不同），沿第1维（水平方向）拼接
        iou_vis_uint8 = iou_vis.astype(np.uint8) if iou_vis.dtype != np.uint8 else iou_vis
        img_vis_uint8 = img_vis.astype(np.uint8) if img_vis.dtype != np.uint8 else img_vis

        # 如果H不同则取最小H
        min_h = min(iou_vis_uint8.shape[0], img_vis_uint8.shape[0])
        iou_vis_slice = iou_vis_uint8[:min_h]
        img_vis_slice = img_vis_uint8[:min_h]

        concat_vis = np.concatenate([img_vis_slice, iou_vis_slice], axis=1)

        return concat_vis
