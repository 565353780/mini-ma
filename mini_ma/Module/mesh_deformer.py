import os
import cv2
import torch
import trimesh
import numpy as np
from typing import Tuple, Optional

from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer

from non_rigid_icp.Data.mesh import Mesh
from non_rigid_icp.Module.optimal_mapper import OptimalMapper

from cage_deform.Module.cage_deformer import CageDeformer

from camera_control.Module.camera import Camera
from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer

from mini_ma.Method.io import loadImage
from mini_ma.Method.data import toNumpy, toTensor
from mini_ma.Module.detector import Detector
from mini_ma.Module.camera_matcher import CameraMatcher


def toGPU(data_dict: dict, device: str = 'cuda:0') -> dict:
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            data_dict[key] = data_dict[key].to(device=device)
    return data_dict

class MeshDeformer(object):
    def __init__(
        self,
        mesh_file_path: str,
        method: str = "roma",
        model_file_path: Optional[str] = None,
        color: list=[178, 178, 178],
        device: str = 'cuda:0',
    ) -> None:
        self.device = device

        self.camera_matcher = CameraMatcher(
            mesh_file_path,
            method=method,
            model_file_path=model_file_path,
            color=color,
            device=device,
        )
        return

    @property
    def detector(self) -> Detector:
        return self.camera_matcher.detector

    @property
    def nvdiffrast_renderer(self) -> NVDiffRastRenderer:
        return self.camera_matcher.nvdiffrast_renderer

    @property
    def mesh(self) -> trimesh.Trimesh:
        return self.nvdiffrast_renderer.mesh

    def searchDeformPairs(
        self,
        camera: Camera,
        render_dict: dict,
        match_result: dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回source mesh顶点去重后的索引 unique_vertex_idxs，以及每个source vertex对应的target空间坐标。
        """
        matched_uv, matched_triangle_idxs = self.camera_matcher.extractMatchedUVTriangle(render_dict, match_result)

        # 获取所有匹配三角面的顶点索引 (N, 3)
        matched_face_vertex_idxs = self.mesh.faces[matched_triangle_idxs]  # (N, 3)
        all_vertex_idxs = matched_face_vertex_idxs.reshape(-1)  # (N*3,)

        # 对顶点索引去重，得到唯一的顶点索引及每个all_vertex_idxs对应的去重后索引
        unique_vertex_idxs, inverse_indices = np.unique(all_vertex_idxs, return_inverse=True)  # unique_vertex_idxs: (M,), inverse_indices: (N*3,)

        # 取原始source mesh上unique的顶点位置
        matched_source_vertices_np = self.mesh.vertices[unique_vertex_idxs]  # (M, 3)

        # === 计算每个face center的target点偏移 ===
        # (1) 取三角形顶点的坐标，计算质心
        triangle_vertices = self.mesh.vertices[matched_face_vertex_idxs]  # (N, 3, 3)
        matched_triangle_centers = triangle_vertices.mean(axis=1)  # (N, 3)

        # (2) 质心到相机坐标系
        matched_triangle_centers_tensor = toTensor(matched_triangle_centers, camera.dtype, camera.device)
        matched_triangle_centers_homo = torch.cat([
            matched_triangle_centers_tensor,
            torch.ones((len(matched_triangle_centers_tensor), 1), dtype=camera.dtype, device=camera.device)
        ], dim=1)
        matched_triangle_centers_camera_homo = torch.matmul(matched_triangle_centers_homo, camera.world2camera.T)
        matched_triangle_centers_camera = matched_triangle_centers_camera_homo[:, :3]

        # (3) -z 作为深度
        depth = -matched_triangle_centers_camera[:, 2]

        # (4) uv+depth反投影空间点 (N, 3)
        matched_target_points = camera.projectUV2Points(matched_uv, depth)  # (N, 3)

        # (5) 得到每个三角形的平移
        translation_vectors = matched_target_points - matched_triangle_centers_tensor  # (N, 3)
        translation_vectors_expanded = translation_vectors.unsqueeze(1)  # (N, 1, 3)

        triangle_vertices_tensor = toTensor(triangle_vertices, camera.dtype, camera.device)  # (N,3,3)
        matched_target_vertices_tensor = triangle_vertices_tensor + translation_vectors_expanded  # (N,3,3)
        all_target_vertices = matched_target_vertices_tensor.cpu().numpy().reshape(-1, 3)  # (N*3, 3)

        # === 重新排序/唯一化target，和unique_vertex_idxs一一对应 ===
        # 保证同一source vertex的target值一致，优先按第一次出现的target
        matched_target_vertices_np = np.zeros_like(matched_source_vertices_np)  # (M, 3)
        for idx, unique_idx in enumerate(unique_vertex_idxs):
            positions = np.where(all_vertex_idxs == unique_idx)[0]
            first_pos = positions[0]
            matched_target_vertices_np[idx] = all_target_vertices[first_pos]

        # --- 这里返回 unique_vertex_idxs, matched_target_vertices_np ---
        return unique_vertex_idxs, matched_target_vertices_np

    def filterDeformPairs(
        self,
        source_vertex_idxs: np.ndarray,
        target_vertices: np.ndarray,
        max_deform_ratio: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        source_vertices = self.mesh.vertices[source_vertex_idxs]

        deform_vectors = target_vertices - source_vertices
        deform_lengths = np.linalg.norm(deform_vectors, axis=1)

        # 计算source pcd的bbox最大边长
        bbox_min = source_vertices.min(axis=0)
        bbox_max = source_vertices.max(axis=0)
        bbox_size = bbox_max - bbox_min
        max_bbox_len = np.max(bbox_size)

        # 超过阈值的过滤掉
        deform_threshold = max_bbox_len * max_deform_ratio
        valid_mask = deform_lengths <= deform_threshold

        # 只保留符合要求的点
        filtered_source_vertex_idxs = source_vertex_idxs[valid_mask]
        filtered_target_vertices = target_vertices[valid_mask]

        return filtered_source_vertex_idxs, filtered_target_vertices

    def deformMeshByNICP(
        self,
        source_vertex_idxs: np.ndarray,
        target_vertices: np.ndarray,
    ) -> trimesh.Trimesh:
        inner_iter = 50
        outer_iter = 200
        milestones = np.arange(10, outer_iter, 4)
        masked_dist_thresh = 0.04
        masked_dist_thresh = float("inf")
        masked_dist_weight = 1.0
        stiffness_weights = 64 * 0.8 ** np.arange(milestones.shape[0] + 1)
        laplacian_weight = 1.0
        target_vertices_weight = 1.0
        save_result_folder_path = "auto"
        save_log_folder_path = "auto"
        render = False

        print("milestones:", milestones)
        print("stiffness_weights:", stiffness_weights)

        optimal_mapper = OptimalMapper(
            inner_iter,
            outer_iter,
            milestones,
            masked_dist_thresh,
            masked_dist_weight,
            stiffness_weights,
            laplacian_weight,
            target_vertices_weight,
            self.device,
            save_result_folder_path,
            save_log_folder_path,
            render,
        )

        source_mesh = Mesh()
        source_mesh.vertices = self.mesh.vertices
        source_mesh.normalize()
        optimal_mapper.loadTemplateMesh(source_mesh.vertices, self.mesh.faces)

        target_mesh = Mesh()
        target_mesh.vertices = target_vertices
        target_mesh.transform(source_mesh.norm_center, source_mesh.norm_scale, is_inverse=False)
        optimal_mapper.addTargetVerticesConstraint(source_vertex_idxs, target_mesh.vertices)

        optimal_mapper.map()

        deformed_mesh = optimal_mapper.toDeformedTemplateMesh()

        deformed_mesh.transform(
            source_mesh.norm_center, source_mesh.norm_scale, is_inverse=True
        )

        deformed_trimesh = trimesh.Trimesh(
            vertices=deformed_mesh.vertices,
            faces=deformed_mesh.triangles,
        )

        return deformed_trimesh

    def deformMeshByCage(
        self,
        source_vertex_idxs: np.ndarray,
        target_vertices: np.ndarray,
    ) -> trimesh.Trimesh:
        voxel_size = 1.0 / 64
        padding = 0.1
        lr = 1e-2
        lambda_reg = 1e4
        steps = 1000
        dtype = torch.float32

        vertices = toTensor(self.mesh.vertices, dtype, self.device)

        cage_deformer = CageDeformer(dtype, self.device)

        deformed_vertices = cage_deformer.deformPoints(
            vertices, source_vertex_idxs, target_vertices,
            voxel_size, padding, lr, lambda_reg, steps,
        )

        deformed_trimesh = trimesh.Trimesh(
            vertices=toNumpy(deformed_vertices, np.float64),
            faces=self.mesh.faces,
        )

        return deformed_trimesh

    def matchMeshToImageFile(
        self,
        image_file_path: str,
        save_match_result_folder_path: str,
        max_deform_ratio: float = 0.05,
    ) -> Optional[trimesh.Trimesh]:
        camera, render_dict, match_result = self.matchCameraToMeshImageFile(
            image_file_path,
            save_match_result_folder_path,
            iter_num=1,
        )

        if camera is None or render_dict is None or match_result is None:
            print('[ERROR][MeshDeformer::matchMeshToImageFile]')
            print('\t matchCamera failed!')
            return None

        source_vertex_idxs, target_vertices = self.searchDeformPairs(camera, render_dict, match_result)

        filtered_source_vertex_idxs, filtered_target_vertices = self.filterDeformPairs(
            source_vertex_idxs, target_vertices, max_deform_ratio)

        deformed_mesh = self.deformMeshByCage(filtered_source_vertex_idxs, filtered_target_vertices)

        # 保存为PLY点云文件
        # 确保保存文件夹存在
        if save_match_result_folder_path is not None:
            os.makedirs(save_match_result_folder_path, exist_ok=True)

            source_pcd_path = save_match_result_folder_path + 'matched_source_vertices.ply'
            target_pcd_path = save_match_result_folder_path + 'matched_target_vertices.ply'
            deformed_mesh_path = save_match_result_folder_path + 'deformed_mesh.ply'
            deformed_mesh_iou_image_path = save_match_result_folder_path + 'deformed_iou.png'

            source_pcd = trimesh.PointCloud(vertices=self.mesh.vertices[filtered_source_vertex_idxs])
            target_pcd = trimesh.PointCloud(vertices=filtered_target_vertices)

            source_pcd.export(source_pcd_path)
            target_pcd.export(target_pcd_path)

            deformed_mesh.export(deformed_mesh_path)

            nvdiffrast_renderer = NVDiffRastRenderer(deformed_mesh_path, [178, 178, 178])

            render_dict = nvdiffrast_renderer.renderImage(
                camera,
                light_direction=[1, 1, 1],
            )

            image = loadImage(image_file_path)
            iou_vis = self.camera_matcher.renderIoU(image, render_dict)
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

        return deformed_mesh

    def matchCameraToMeshImageFile(
        self,
        image_file_path: str,
        save_match_result_folder_path: Optional[str],
        iter_num: int = 1,
    ) -> Tuple[Optional[Camera], Optional[dict], Optional[dict]]:
        import pickle

        # 定义保存路径
        output_tmp_folder = "./output/tmp"
        os.makedirs(output_tmp_folder, exist_ok=True)
        import hashlib
        def get_hash_id(*args):
            s = ",".join([str(x) for x in args])
            return hashlib.md5(s.encode()).hexdigest()
        # 用于唯一标识保存的文件名
        base_id = get_hash_id(image_file_path, save_match_result_folder_path)
        camera_file = os.path.join(output_tmp_folder, f"{base_id}_camera.pkl")
        render_dict_file = os.path.join(output_tmp_folder, f"{base_id}_render_dict.pkl")
        match_result_file = os.path.join(output_tmp_folder, f"{base_id}_match_result.pkl")

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

        camera, render_dict, match_result = self.camera_matcher.matchCameraToMeshImageFile(
            image_file_path,
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
