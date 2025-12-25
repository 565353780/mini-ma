import os
import torch
import trimesh
import numpy as np
from typing import Optional

from camera_control.Module.camera import Camera
from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer

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

    def matchMeshToImageFile(
        self,
        image_file_path: str,
        save_match_result_folder_path: str,
    ) -> Optional[trimesh.Trimesh]:

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
                print("[INFO][MeshDeformer::matchMeshToImageFile] 读取已缓存的匹配结果。")
            except Exception as e:
                print(f"[WARNING][MeshDeformer::matchMeshToImageFile] 读取缓存失败，将重新计算。错误: {e}")
                camera, render_dict, match_result = self.camera_matcher.matchCameraToMeshImageFile(
                    image_file_path,
                    save_match_result_folder_path,
                )
                # 保存到文件
                with open(camera_file, 'wb') as f:
                    pickle.dump(camera, f)
                # 转为 cpu，防止gpu环境不兼容
                def tensor_dict_to_cpu(d):
                    return {k: (v.cpu() if hasattr(v,"cpu") else v) for k, v in d.items()}
                with open(render_dict_file, 'wb') as f:
                    pickle.dump(tensor_dict_to_cpu(render_dict), f)
                with open(match_result_file, 'wb') as f:
                    pickle.dump(tensor_dict_to_cpu(match_result), f)
        else:
            camera, render_dict, match_result = self.camera_matcher.matchCameraToMeshImageFile(
                image_file_path,
                save_match_result_folder_path,
            )
            with open(camera_file, 'wb') as f:
                pickle.dump(camera, f)
            def tensor_dict_to_cpu(d):
                return {k: (v.cpu() if hasattr(v,"cpu") else v) for k, v in d.items()}
            with open(render_dict_file, 'wb') as f:
                pickle.dump(tensor_dict_to_cpu(render_dict), f)
            with open(match_result_file, 'wb') as f:
                pickle.dump(tensor_dict_to_cpu(match_result), f)

        if camera is None or render_dict is None or match_result is None:
            print('[ERROR][MeshDeformer::matchMeshToImageFile]')
            print('\t matchCamera failed!')
            return None

        render_dict = toGPU(render_dict, camera.device)
        match_result = toGPU(match_result, camera.device)

        matched_uv, matched_triangle_idxs = self.camera_matcher.extractMatchedUVTriangle(render_dict, match_result)
        # 获取所有匹配的三角面片的顶点索引 (N, 3)
        matched_face_vertex_idxs = self.mesh.faces[matched_triangle_idxs]  # (N, 3)
        
        # 展平成一维，收集全部索引
        all_vertex_idxs = matched_face_vertex_idxs.reshape(-1)
        
        # 按index去重，获取唯一的顶点索引
        unique_vertex_idxs, inverse_indices = np.unique(all_vertex_idxs, return_inverse=True)
        
        # 取原始source mesh上的唯一顶点
        matched_source_vertices_np = self.mesh.vertices[unique_vertex_idxs]
        
        # 下面需要将每个三角形的3个顶点，映射到唯一点集的顺序——记录每个(N, 3)的三角面用唯一点的索引
        triangle_vertices = self.mesh.vertices[matched_face_vertex_idxs]   # (N, 3, 3)
        matched_triangle_centers = triangle_vertices.mean(axis=1)

        # 将三角形中心点转换为tensor并变换到相机坐标系
        matched_triangle_centers_tensor = toTensor(matched_triangle_centers, camera.dtype, camera.device)
        
        # 将三角形中心点从世界坐标系变换到相机坐标系
        matched_triangle_centers_homo = torch.cat([
            matched_triangle_centers_tensor,
            torch.ones((len(matched_triangle_centers_tensor), 1), dtype=camera.dtype, device=camera.device)
        ], dim=1)
        matched_triangle_centers_camera_homo = torch.matmul(matched_triangle_centers_homo, camera.world2camera.T)
        matched_triangle_centers_camera = matched_triangle_centers_camera_homo[:, :3]
        
        # 提取深度值：使用 -z 作为深度（因为相机看向 -Z 方向，Z 轴向后）
        # 在相机坐标系中，可见点的 Z < 0，所以深度 = -Z > 0
        depth = -matched_triangle_centers_camera[:, 2]
        
        # 使用 projectUV2Points 将 UV 和深度反投影回世界坐标系
        matched_target_points = camera.projectUV2Points(matched_uv, depth)
        
        # 计算每个三角形的平移向量
        # matched_target_points 是目标位置，matched_triangle_centers_tensor 是原始位置
        translation_vectors = matched_target_points - matched_triangle_centers_tensor
        
        # 将平移应用到每个三角形的三个顶点
        translation_vectors_expanded = translation_vectors.unsqueeze(1)  # (N, 1, 3)
        
        triangle_vertices_tensor = toTensor(triangle_vertices, camera.dtype, camera.device)  # (N,3,3)
        matched_target_vertices_tensor = triangle_vertices_tensor + translation_vectors_expanded  # (N,3,3)
        
        # 合成所有目标点，形状 (N*3, 3)
        all_target_vertices = matched_target_vertices_tensor.cpu().numpy().reshape(-1, 3)
        
        # 采用相同顺序、同样的唯一化方案，保证target和source是完全一一对应的
        # 重新排列，得到target的唯一坐标，与unique_vertex_idxs顺序一致
        matched_target_vertices_np = np.zeros_like(matched_source_vertices_np)
        # inverse_indices 的长度就是N*3
        for idx, unique_idx in enumerate(unique_vertex_idxs):
            # 找到all_vertex_idxs中等于unique_idx的所有位置（可能有多个，在flatten之后）
            positions = np.where(all_vertex_idxs == unique_idx)[0]
            # 取第一个出现的位置即可（因为这类点的target都会被平移到同一位置，只需一致）
            first_pos = positions[0]
            matched_target_vertices_np[idx] = all_target_vertices[first_pos]

        # 保存为PLY点云文件
        # 确保保存文件夹存在
        if os.path.isfile(save_match_result_folder_path):
            save_folder = os.path.dirname(save_match_result_folder_path)
        else:
            save_folder = save_match_result_folder_path
        os.makedirs(save_folder, exist_ok=True)
        
        source_pcd_path = os.path.join(save_folder, 'matched_source_vertices.ply')
        target_pcd_path = os.path.join(save_folder, 'matched_target_vertices.ply')
        
        # 使用trimesh创建点云并保存
        source_pcd = trimesh.PointCloud(vertices=matched_source_vertices_np)
        target_pcd = trimesh.PointCloud(vertices=matched_target_vertices_np)
        
        source_pcd.export(source_pcd_path)
        target_pcd.export(target_pcd_path)
        
        print(f'[INFO][MeshDeformer::matchMeshToImageFile]')
        print(f'\t 保存源顶点点云: {source_pcd_path}')
        print(f'\t 保存目标顶点点云: {target_pcd_path}')
        print(f'\t 点数: {len(matched_source_vertices_np)}')

        return None
