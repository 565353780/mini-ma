import torch
import trimesh
import numpy as np
from typing import Tuple

from cage_deform.Module.cage_deformer import CageDeformer

from camera_control.Module.camera import Camera

from mini_ma.Method.data import toNumpy, toTensor
from mini_ma.Module.camera_matcher import CameraMatcher


class MeshDeformer(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def searchDeformPairs(
        mesh: trimesh.Trimesh,
        camera: Camera,
        render_dict: dict,
        match_result: dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回source mesh顶点去重后的索引 unique_vertex_idxs，以及每个source vertex对应的target空间坐标。
        """
        matched_uv, matched_triangle_idxs = CameraMatcher.extractMatchedUVTriangle(
            render_dict, match_result)

        # 获取所有匹配三角面的顶点索引 (N, 3)
        matched_face_vertex_idxs = mesh.faces[matched_triangle_idxs]  # (N, 3)
        all_vertex_idxs = matched_face_vertex_idxs.reshape(-1)  # (N*3,)

        # 对顶点索引去重，得到唯一的顶点索引及每个all_vertex_idxs对应的去重后索引
        unique_vertex_idxs, inverse_indices = np.unique(all_vertex_idxs, return_inverse=True)  # unique_vertex_idxs: (M,), inverse_indices: (N*3,)

        # 取原始source mesh上unique的顶点位置
        matched_source_vertices_np = mesh.vertices[unique_vertex_idxs]  # (M, 3)

        # === 计算每个face center的target点偏移 ===
        # (1) 取三角形顶点的坐标，计算质心
        triangle_vertices = mesh.vertices[matched_face_vertex_idxs]  # (N, 3, 3)
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

    @staticmethod
    def filterDeformPairs(
        mesh: trimesh.Trimesh,
        source_vertex_idxs: np.ndarray,
        target_vertices: np.ndarray,
        max_deform_ratio: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        source_vertices = mesh.vertices[source_vertex_idxs]

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

    @staticmethod
    def deformMeshByCage(
        mesh: trimesh.Trimesh,
        source_vertex_idxs: np.ndarray,
        target_vertices: np.ndarray,
        voxel_size = 1.0 / 64,
        padding = 0.1,
        lr = 1e-2,
        lambda_reg = 1e4,
        steps = 1000,
        dtype = torch.float32,
        device: str = 'cpu',
    ) -> trimesh.Trimesh:
        vertices = toTensor(mesh.vertices, dtype, device)

        cage_deformer = CageDeformer(dtype, device)

        deformed_vertices = cage_deformer.deformPoints(
            vertices, source_vertex_idxs, target_vertices,
            voxel_size, padding, lr, lambda_reg, steps,
        )

        deformed_trimesh = trimesh.Trimesh(
            vertices=toNumpy(deformed_vertices, np.float64),
            faces=mesh.faces,
        )

        return deformed_trimesh
