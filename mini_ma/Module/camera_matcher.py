import os
import cv2
import torch
import trimesh
import numpy as np
from typing import Tuple, Optional

from camera_control.Module.camera import Camera
from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer

from mini_ma.Method.data import toNumpy, toTensor, toGPU
from mini_ma.Method.path import createFileFolder
from mini_ma.Module.detector import Detector


class CameraMatcher(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def extractMatchedUVTriangle(
        render_dict: dict,
        match_result: dict,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        dtype = render_dict['rasterize_output'].dtype
        device = render_dict['rasterize_output'].device

        # HxWx4, u right, v down
        rasterize_output = render_dict['rasterize_output']

        # [W, H]
        render_image_pts = torch.from_numpy(
            np.round(match_result['mkpts1'])).to(dtype=torch.int32, device=device)

        height, width = rasterize_output.shape[:2]

        # 保证索引不越界：只保留在[0, H)和[0, W)内的点
        valid_mask = (render_image_pts[:, 0] >= 0) & (render_image_pts[:, 0] < width) & \
                     (render_image_pts[:, 1] >= 0) & (render_image_pts[:, 1] < height)
        safe_pts = render_image_pts[valid_mask]

        matched_mesh_data = rasterize_output[safe_pts[:, 1], safe_pts[:, 0]]
        on_mesh_idxs = (matched_mesh_data[:, 3] > 0).nonzero(as_tuple=False).flatten()

        matched_triangle_idxs = toNumpy(matched_mesh_data[on_mesh_idxs, 3], np.int32)

        image_uv = toTensor(match_result['mkpts0'], dtype, device) / torch.tensor(
            [width, height], dtype=dtype, device=device)
        matched_uv = image_uv[on_mesh_idxs]

        matched_uv[:, 1] = 1.0 - matched_uv[:, 1]

        return matched_uv, matched_triangle_idxs

    @staticmethod
    def estimateCamera(
        mesh: trimesh.Trimesh,
        render_dict: dict,
        match_result: dict,
    ) -> Camera:
        device = render_dict['rasterize_output'].device

        height, width = render_dict['image'].shape[:2]

        matched_uv, matched_triangle_idxs = CameraMatcher.extractMatchedUVTriangle(
            render_dict, match_result)

        triangle_vertices = mesh.vertices[mesh.faces[matched_triangle_idxs]]
        matched_triangle_centers = triangle_vertices.mean(axis=1)

        estimated_camera = Camera.fromUVPoints(
            matched_triangle_centers,
            matched_uv,
            width=width,
            height=height,
            device=device,
        )

        return estimated_camera

    @staticmethod
    def getIoU(
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

    @staticmethod
    def matchCameraToMeshImage(
        image: np.ndarray,
        mesh: trimesh.Trimesh,
        detector: Detector,
        save_match_result_folder_path: Optional[str],
        iter_num: int = 1,
    ) -> Tuple[Optional[Camera], Optional[dict], Optional[dict]]:
        render = save_match_result_folder_path is not None

        init_camera = Camera(
            width=image.shape[1],
            height=image.shape[0],
            pos=[0, 0, 1],
            look_at=[0, 0, 0],
            up=[0, 1, 0],
            device=detector.device,
        )
        init_camera.focusOnPoints(mesh.vertices)
        light_direction = [1, 1, 1]

        render_dict = NVDiffRastRenderer.renderImage(
            mesh,
            init_camera,
            light_direction=light_direction,
        )

        match_result = detector.detect(image, render_dict['image'])

        if match_result is None:
            print('[ERROR][CameraMatcher::matchCameraToMeshImage]')
            print('\t matching pairs detect failed!')
            return None, render_dict, None

        if render:
            concat_vis = CameraMatcher.renderMatchResult(
                image,
                detector,
                render_dict,
                match_result,
            )

            save_path = save_match_result_folder_path + "matches_0.jpg"
            createFileFolder(save_path)
            cv2.imwrite(save_path, concat_vis)

        best_iou = 0
        best_camera = init_camera.clone()

        for i in range(1, 1 + iter_num):
            estimated_camera = CameraMatcher.estimateCamera(mesh, render_dict, match_result)

            render_dict = NVDiffRastRenderer.renderImage(
                mesh,
                estimated_camera,
                light_direction=light_direction,
            )

            match_result = detector.detect(image, render_dict['image'])

            if match_result is None:
                print('[ERROR][CameraMatcher::matchMeshImagePairs]')
                print('\t matching pairs detect failed!')
                return best_camera, render_dict, None

            iou = CameraMatcher.getIoU(image, render_dict)

            if iou > best_iou:
                best_iou = iou
                best_camera = estimated_camera.clone()
                print('[INFO][CameraMatcher::matchCameraToMeshImageFile]')
                print('\t best_iou:', best_iou)
                print('\t best_match_idx:', i)

            if render:
                concat_vis = CameraMatcher.renderMatchResult(
                    image,
                    detector,
                    render_dict,
                    match_result,
                )
                save_path = save_match_result_folder_path + "matches_" + str(i) + ".jpg"
                createFileFolder(save_path)
                cv2.imwrite(save_path, concat_vis)

        return best_camera, render_dict, match_result

    @staticmethod
    def matchCameraToMeshImageWithCache(
        image: np.ndarray,
        mesh: trimesh.Trimesh,
        detector: Detector,
        save_match_result_folder_path: Optional[str],
        iter_num: int = 1,
        cache_id: str = 'replace',
    ) -> Tuple[Optional[Camera], Optional[dict], Optional[dict]]:
        import pickle

        # 定义保存路径
        output_tmp_folder = "./output/tmp/"
        os.makedirs(output_tmp_folder, exist_ok=True)
        camera_file = output_tmp_folder + cache_id + f"_camera.pkl"
        render_dict_file = output_tmp_folder + cache_id + "_render_dict.pkl"
        match_result_file = output_tmp_folder + cache_id + "_match_result.pkl"

        # 检查文件是否已存在并可用
        if os.path.exists(camera_file) and os.path.exists(render_dict_file) and os.path.exists(match_result_file):
            try:
                with open(camera_file, 'rb') as f:
                    camera = pickle.load(f)
                with open(render_dict_file, 'rb') as f:
                    render_dict = pickle.load(f)
                with open(match_result_file, 'rb') as f:
                    match_result = pickle.load(f)

                render_dict = toGPU(render_dict, camera.device)
                match_result = toGPU(match_result, camera.device)
                print("[INFO][MeshDeformer::matchMeshToImageFile] 读取已缓存的匹配结果。")

                return camera, render_dict, match_result
            except:
                pass

        camera, render_dict, match_result = CameraMatcher.matchCameraToMeshImage(
            image,
            mesh,
            detector,
            save_match_result_folder_path,
            iter_num,
        )
        with open(camera_file, 'wb') as f:
            pickle.dump(camera, f)
        def tensor_dict_to_cpu(d):
            return {k: (v.cpu() if hasattr(v,"cpu") else v) for k, v in d.items()}
        with open(render_dict_file, 'wb') as f:
            pickle.dump(tensor_dict_to_cpu(render_dict), f)
        with open(match_result_file, 'wb') as f:
            pickle.dump(tensor_dict_to_cpu(match_result), f)

        return camera, render_dict, match_result

    @staticmethod
    def renderIoU(
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

    @staticmethod
    def renderMatchResult(
        image: np.ndarray,
        detector: Detector,
        render_dict: dict,
        match_result: dict,
    ) -> np.ndarray:
        img_vis = detector.renderMatchResult(
            match_result,
            image,
            render_dict['image'],
        )

        iou_vis = CameraMatcher.renderIoU(image, render_dict)

        # 拼接iou_vis和img_vis（H相同，W可能不同），沿第1维（水平方向）拼接
        iou_vis_uint8 = iou_vis.astype(np.uint8) if iou_vis.dtype != np.uint8 else iou_vis
        img_vis_uint8 = img_vis.astype(np.uint8) if img_vis.dtype != np.uint8 else img_vis

        # 如果H不同则取最小H
        min_h = min(iou_vis_uint8.shape[0], img_vis_uint8.shape[0])
        iou_vis_slice = iou_vis_uint8[:min_h]
        img_vis_slice = img_vis_uint8[:min_h]

        concat_vis = np.concatenate([img_vis_slice, iou_vis_slice], axis=1)

        return concat_vis
