import torch
import trimesh
import numpy as np
from typing import Tuple
from copy import deepcopy

from cage_deform.Module.bspline_deformer import BSplineDeformer

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
        直接计算匹配三角面片质心source point到对应target point的映射，返回source_points和target_points (Nx3 tensors)
        纯tensor计算，并保证与camera.device一致。
        """
        matched_uv, matched_triangle_idxs = CameraMatcher.extractMatchedUVTriangle(
            render_dict, match_result)

        device = camera.device
        dtype = camera.dtype

        # 将 mesh 数据转为 tensor
        mesh_faces_tensor = torch.from_numpy(np.ascontiguousarray(mesh.faces)).to(device=device)
        mesh_vertices_tensor = torch.from_numpy(np.ascontiguousarray(mesh.vertices)).to(dtype=dtype, device=device)

        # 取出所有匹配三角形顶点的位置
        matched_face_vertex_idxs = mesh_faces_tensor[matched_triangle_idxs]  # (N, 3)
        triangle_vertices_tensor = mesh_vertices_tensor[matched_face_vertex_idxs]  # (N, 3, 3)

        # 计算每个三角面的质心作为源点
        source_points = triangle_vertices_tensor.mean(dim=1)  # (N, 3)

        # 计算质心到相机坐标
        ones = torch.ones((source_points.size(0), 1), dtype=dtype, device=device)
        source_points_homo = torch.cat([source_points, ones], dim=1)  # (N,4)
        source_points_camera_homo = torch.matmul(source_points_homo, camera.world2camera.T)
        source_points_camera = source_points_camera_homo[:, :3]

        # 用-z得到深度
        depth = -source_points_camera[:, 2]  # (N,)

        # 将 matched_uv 转为 tensor
        if isinstance(matched_uv, np.ndarray):
            matched_uv_tensor = torch.from_numpy(matched_uv).to(dtype=dtype, device=device)
        else:
            matched_uv_tensor = matched_uv.to(dtype=dtype, device=device)

        # 反投影生成每个target point
        target_points = camera.projectUV2Points(matched_uv_tensor, depth)  # (N, 3)

        return source_points, target_points

    @staticmethod
    def filterDeformPairs(
        source_points: torch.Tensor,
        target_points: torch.Tensor,
        max_deform_ratio: float = 0.05,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 形变量向量 (N,3)
        deform_vectors = target_points - source_points  # (N,3)
        deform_lengths = torch.norm(deform_vectors, dim=1)  # (N,)

        # 计算source pcd的bbox最大边长（全用torch）
        bbox_min = torch.min(source_points, dim=0).values
        bbox_max = torch.max(source_points, dim=0).values
        bbox_size = bbox_max - bbox_min
        max_bbox_len = torch.max(bbox_size)

        # 超过阈值的过滤掉
        deform_threshold = max_bbox_len * max_deform_ratio
        valid_mask = deform_lengths <= deform_threshold

        # 只保留符合要求的点
        filtered_source_points = source_points[valid_mask]
        filtered_target_points = target_points[valid_mask]

        return filtered_source_points, filtered_target_points

    @staticmethod
    def deformMeshByCage(
        mesh: trimesh.Trimesh,
        source_points: torch.Tensor,
        target_points: torch.Tensor,
        voxel_size = 1.0 / 64,
        padding = 0.1,
        lr = 1e-2,
        lambda_smooth: float = 1e3,
        lambda_magnitude: float = 1.0,
        steps = 1000,
        dtype = torch.float32,
        device: str = 'cpu',
    ) -> trimesh.Trimesh:
        vertices = toTensor(mesh.vertices, dtype, device)

        bspline_deformer = BSplineDeformer(dtype, device)

        bspline_deformer.loadPoints(mesh.vertices, voxel_size, padding)

        deformed_points = bspline_deformer.deformPoints(
            source_points, target_points,
            lr, lambda_smooth, lambda_magnitude, steps,
        )

        deformed_vertices = bspline_deformer.queryPoints(vertices)

        deformed_trimesh = deepcopy(mesh)
        deformed_trimesh.vertices = toNumpy(deformed_vertices, np.float32)

        return deformed_trimesh
