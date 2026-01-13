import torch
import trimesh
import numpy as np
from typing import Tuple
from copy import deepcopy

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回source mesh顶点去重后的索引 unique_vertex_idxs，以及每个source vertex对应的target空间坐标。
        纯tensor计算，并保证与camera.device一致。
        """
        matched_uv, matched_triangle_idxs = CameraMatcher.extractMatchedUVTriangle(
            render_dict, match_result)

        device = camera.device
        dtype = camera.dtype

        # === 将face, vertex, triangle等全部转tensor并对齐设备 ===
        mesh_faces_tensor = torch.from_numpy(np.ascontiguousarray(mesh.faces)).to(device=device)
        mesh_vertices_tensor = torch.from_numpy(np.ascontiguousarray(mesh.vertices)).to(dtype=dtype, device=device)

        # (N,3) 每个匹配三角面的顶点索引
        matched_face_vertex_idxs = mesh_faces_tensor[matched_triangle_idxs]  # (N, 3)
        all_vertex_idxs = matched_face_vertex_idxs.reshape(-1)  # (N*3,)

        # 对顶点索引去重
        unique_vertex_idxs, inverse_indices = torch.unique(all_vertex_idxs, return_inverse=True)

        # (M,3) unique mesh顶点位置
        matched_source_vertices_tensor = mesh_vertices_tensor[unique_vertex_idxs]  # (M, 3)

        # === 计算每个face center的target点偏移 ===
        triangle_vertices_tensor = mesh_vertices_tensor[matched_face_vertex_idxs]  # (N, 3, 3)
        matched_triangle_centers_tensor = triangle_vertices_tensor.mean(dim=1)  # (N, 3)

        # 质心到相机坐标
        ones = torch.ones((matched_triangle_centers_tensor.size(0), 1), dtype=dtype, device=device)
        matched_triangle_centers_homo = torch.cat([matched_triangle_centers_tensor, ones], dim=1)  # (N,4)
        matched_triangle_centers_camera_homo = torch.matmul(matched_triangle_centers_homo, camera.world2camera.T)
        matched_triangle_centers_camera = matched_triangle_centers_camera_homo[:, :3]

        # -z 作为深度
        depth = -matched_triangle_centers_camera[:, 2]  # (N,)

        # matched_uv to tensor
        if isinstance(matched_uv, np.ndarray):
            matched_uv_tensor = torch.from_numpy(matched_uv).to(dtype=dtype, device=device)
        else:
            matched_uv_tensor = matched_uv.to(dtype=dtype, device=device)

        # uv+depth反投影 (N,3)
        matched_target_points_tensor = camera.projectUV2Points(matched_uv_tensor, depth)  # (N, 3)

        # 平移向量
        translation_vectors = matched_target_points_tensor - matched_triangle_centers_tensor  # (N, 3)
        translation_vectors_expanded = translation_vectors.unsqueeze(1)  # (N, 1, 3)

        # 滑动平移三角形每个顶点
        matched_target_vertices_tensor = triangle_vertices_tensor + translation_vectors_expanded  # (N,3,3)
        all_target_vertices_tensor = matched_target_vertices_tensor.reshape(-1, 3)  # (N*3,3)

        # === 配unique_vertex_idxs分配target ===
        # 按unique_vertex_idxs第一次出现分配target，和Numpy逻辑保持一致
        matched_target_vertices_tensor_final = torch.zeros_like(matched_source_vertices_tensor)  # (M,3)
        # positions = torch.where(all_vertex_idxs[None, :] == unique_vertex_idxs[:, None]) # M x N*3 -> 直接find first
        for idx, unique_idx in enumerate(unique_vertex_idxs):
            positions = (all_vertex_idxs == unique_idx).nonzero(as_tuple=True)[0]
            first_pos = positions[0]
            matched_target_vertices_tensor_final[idx] = all_target_vertices_tensor[first_pos]

        # 返回 numpy
        return unique_vertex_idxs, matched_target_vertices_tensor_final

    @staticmethod
    def filterDeformPairs(
        mesh: trimesh.Trimesh,
        source_vertex_idxs: torch.Tensor,
        target_vertices: torch.Tensor,
        max_deform_ratio: float = 0.05,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = target_vertices.device
        dtype = target_vertices.dtype

        # === tensor化顶点数据，确保在同一设备与类型 ===
        mesh_vertices = torch.from_numpy(np.ascontiguousarray(mesh.vertices)).to(dtype=dtype, device=device)
        source_vertices = mesh_vertices[source_vertex_idxs]  # (N, 3)

        # 形变量向量 (N,3)
        deform_vectors = target_vertices - source_vertices  # (N,3)
        deform_lengths = torch.norm(deform_vectors, dim=1)  # (N,)

        # 计算source pcd的bbox最大边长（全用torch）
        bbox_min = torch.min(source_vertices, dim=0).values
        bbox_max = torch.max(source_vertices, dim=0).values
        bbox_size = bbox_max - bbox_min
        max_bbox_len = torch.max(bbox_size)

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

        deformed_trimesh = deepcopy(mesh)
        deformed_trimesh.vertices = toNumpy(deformed_vertices, np.float32)

        return deformed_trimesh
