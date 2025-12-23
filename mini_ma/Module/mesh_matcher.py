import os
import cv2
import torch
import trimesh
import numpy as np
import nvdiffrast.torch as dr
from typing import Optional, Union, Tuple

from camera_control.Method.data import toTensor
from camera_control.Module.camera import Camera

from mini_ma.Method.io import loadImage
from mini_ma.Method.path import createFileFolder
from mini_ma.Module.detector import Detector


class MeshMatcher(object):
    def __init__(
        self,
        method: str = "sp_lg",
        model_file_path: Optional[str] = None,
    ) -> None:
        self.detector = Detector(
            method=method,
            model_file_path=model_file_path,
        )
        self.device = self.detector.device

        # self.device = torch.device('cuda:7')
        return

    def loadMeshFile(
        self,
        mesh_file_path: str,
        color: list=[178, 178, 178],
    ) -> Optional[trimesh.Trimesh]:
        """
        使用trimesh读取三角网格文件，支持多种格式（.obj, .ply, .glb, .stl, .off等）

        Args:
            mesh_file_path: 网格文件路径（trimesh支持的所有格式）
            color: 顶点颜色，默认[0.7, 0.7, 0.7]（灰色）

        Returns:
            trimesh.Trimesh对象
        """
        if not os.path.exists(mesh_file_path):
            print('[ERROR][MeshMatcher::loadMeshFile]')
            print('\t mesh file not exist!')
            print('\t mesh_file_path:', mesh_file_path)
            return None

        mesh_trimesh = trimesh.load(mesh_file_path)

        if isinstance(mesh_trimesh, trimesh.Scene):
            mesh_trimesh = trimesh.util.concatenate(
                [g for g in mesh_trimesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            )

        if not isinstance(mesh_trimesh, trimesh.Trimesh):
            raise ValueError(f"无法从文件 {mesh_file_path} 中提取三角网格数据")

        # 如果mesh没有顶点颜色，添加默认颜色
        if not hasattr(mesh_trimesh.visual, 'vertex_colors') or mesh_trimesh.visual.vertex_colors is None:
            # 创建顶点颜色数组
            num_verts = len(mesh_trimesh.vertices)
            vertex_colors = np.tile(np.array(color), (num_verts, 1))
            mesh_trimesh.visual.vertex_colors = vertex_colors
        return mesh_trimesh

    def queryCamera(
        self,
        mesh: trimesh.Trimesh,
        camera: Camera,
        light_direction: Union[torch.Tensor, np.ndarray, list] = [1, 1, 1],
    ) -> dict:
        """
        使用nvdiffrast渲染三角网格，并获取渲染图中每个像素对应的
        三角网格表面的顶点插值信息

        Args:
            mesh: trimesh.Trimesh对象，要渲染的三角网格
            camera: Camera对象，包含相机的所有参数（位置、旋转、内参等）
            light_direction: 光照方向（世界坐标系），默认为[1, 1, 1]

        Returns:
            dict包含:
                - image: [H, W, 3] 渲染的图像 (RGB)
                - rasterize_output: [H, W, 4] rasterize主输出 (u, v, z/w, triangle_id)
                - bary_derivs: [H, W, 4] 重心坐标的图像空间导数 (du/dX, du/dY, dv/dX, dv/dY)
        """
        # 1. 从Camera对象获取参数
        width = camera.width
        height = camera.height
        fx = camera.fx
        fy = camera.fy
        cx = camera.cx
        cy = camera.cy

        pos = camera.pos.float().to(self.device)  # [3]
        rot = camera.rot.float().to(self.device)  # [3, 3]

        # Camera类的坐标系: X=right, Y=down, Z=forward
        # OpenGL/nvdiffrast坐标系: X=right, Y=up, Z=-forward
        # 需要坐标系转换矩阵 C，将Camera类坐标系转换到OpenGL坐标系
        # C = [1,  0,  0]    (right -> right)
        #     [0, -1,  0]    (down -> up)
        #     [0,  0, -1]    (forward -> -forward)
        coord_conversion = torch.tensor([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0]
        ], dtype=torch.float32, device=self.device)

        glctx = dr.RasterizeCudaContext(device=self.device)

        vertices = torch.from_numpy(mesh.vertices).float().to(self.device)  # [V, 3]
        faces = torch.from_numpy(mesh.faces).int().to(self.device)  # [F, 3]

        # 确保mesh有法向量，如果没有则计算
        if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
            mesh.compute_vertex_normals()

        vertex_normals = torch.from_numpy(mesh.vertex_normals).float().to(self.device)  # [V, 3]

        # 设置光照方向（世界坐标系）
        light_direction = toTensor(light_direction, device=self.device)
        light_direction = light_direction / (torch.norm(light_direction) + 1e-8)

        # 4. 构建投影矩阵
        bbox_size = np.linalg.norm(np.max(mesh.vertices, axis=0) - np.min(mesh.vertices, axis=0))

        near = bbox_size * 0.1
        far = bbox_size * 10.0

        def perspective_projection(fovy_radians, aspect, n, f):
            """构建OpenGL透视投影矩阵"""
            y = np.tan(fovy_radians / 2)
            return np.array([
                [1/(y*aspect),    0,            0,              0],
                [           0, 1/-y,            0,              0],
                [           0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                [           0,    0,           -1,              0]
            ], dtype=np.float32)

        fovy = 2 * np.arctan(height / (2 * fy))
        aspect = width / height
        proj_mtx = torch.from_numpy(perspective_projection(fovy, aspect, near, far)).to(self.device)

        # 5. 构建视图矩阵
        # Camera类的rot是从Camera坐标系到世界坐标系的变换
        # rot.T 是从世界坐标系到Camera坐标系的变换
        # 然后通过coord_conversion转换到OpenGL坐标系
        # R_view = C @ rot.T，其中C是坐标系转换矩阵
        R_world_to_camera = rot.T  # [3, 3] 世界坐标系 -> Camera类坐标系
        R_view = coord_conversion @ R_world_to_camera  # [3, 3] 世界坐标系 -> OpenGL坐标系

        view_mtx = torch.eye(4, dtype=torch.float32, device=self.device)
        view_mtx[:3, :3] = R_view
        view_mtx[:3, 3] = -R_view @ pos  # 平移部分

        mvp = proj_mtx @ view_mtx  # [4, 4]

        # 顶点变换到裁剪空间
        vertices_homo = torch.cat([
            vertices,
            torch.ones((vertices.shape[0], 1), dtype=torch.float32, device=self.device)
        ], dim=1)  # [V, 4]

        vertices_clip = torch.matmul(vertices_homo, mvp.t())  # [V, 4]
        vertices_clip_batch = vertices_clip.unsqueeze(0).contiguous()

        # 光栅化
        rast_out, rast_out_db = dr.rasterize(
            glctx,
            vertices_clip_batch,  # [1, V, 4]
            faces,
            resolution=[height, width]
        )

        # 法向着色：插值顶点法向量
        normals_interp, _ = dr.interpolate(
            vertex_normals.unsqueeze(0),  # [1, V, 3]
            rast_out,
            faces
        )

        # 归一化插值后的法向量
        normals_interp = normals_interp / (torch.norm(normals_interp, dim=-1, keepdim=True) + 1e-8)

        # 将法向量从世界坐标系转换到OpenGL相机坐标系
        # 注意：法向量的变换使用R_view.T（而不是view矩阵的逆）
        normals_cam = torch.matmul(normals_interp[0], R_view.T)  # [H, W, 3]

        # 将光照方向也转换到OpenGL相机坐标系
        light_dir_cam = torch.matmul(light_direction, R_view.T)  # [3]
        light_dir_cam = light_dir_cam / (torch.norm(light_dir_cam) + 1e-8)

        # 计算Lambert着色
        diffuse = torch.sum(normals_cam * light_dir_cam, dim=-1, keepdim=False)  # [H, W]
        diffuse = torch.clamp(diffuse, min=0.0, max=1.0)

        # 添加环境光
        ambient = 0.3
        image = ambient + (1.0 - ambient) * diffuse  # [H, W]

        # 处理背景
        mask = rast_out[0, :, :, 3] > 0  # [H, W]
        background = torch.ones_like(image)  # 白色背景
        image = torch.where(mask, image, background)
        render_image = image.detach().cpu().numpy()
        render_image_np = np.clip(np.rint(render_image * 255), 0, 255).astype(np.uint8)

        result = {
            'image': render_image_np,  # [H, W]
            'rasterize_output': rast_out[0],  # [H, W, 4]
            'bary_derivs': rast_out_db[0] if rast_out_db is not None else torch.zeros_like(rast_out[0]),  # [H, W, 4]
        }

        return result

    def matchMeshToImage(
        self,
        image: np.ndarray,
        mesh: trimesh.Trimesh,
        camera: Optional[Camera]=None,
    ) -> Tuple[dict, dict, Camera]:
        min_bound = np.min(mesh.vertices, axis=0)
        max_bound = np.max(mesh.vertices, axis=0)
        center = (min_bound + max_bound) / 2.0

        if camera is None:
            camera = Camera(
                width=2560,
                height=1440,
                cx=1280,
                cy=720,
                pos=center + [0, 0, 1],
                look_at=center,
                up=[0, 1, 0],
            )

        render_dict = self.queryCamera(
            mesh=mesh,
            camera=camera,
            light_direction=[1, 1, 1],
        )

        match_result = self.detector.detect(image, render_dict['image'])

        if match_result is None:
            print('[ERROR][MeshMatcher::matchMeshToImage]')
            print('\t matching pairs detect failed!')
            return render_dict, {}, camera

        return render_dict, match_result, camera

    def matchMeshFileToImageFile(
        self,
        image_file_path: str,
        mesh_file_path: str,
        save_match_result_folder_path: str,
    ) -> Tuple[dict, dict]:
        if not os.path.exists(image_file_path):
            print('[ERROR][MeshMatcher::matchMeshFileToImageFile]')
            print('\t image file not exist!')
            print('\t image_file_path:', image_file_path)
            return {}, {}

        if not os.path.exists(mesh_file_path):
            print('[ERROR][MeshMatcher::matchMeshFileToImageFile]')
            print('\t mesh file not exist!')
            print('\t mesh_file_path:', mesh_file_path)
            return {}, {}

        image = loadImage(image_file_path, is_gray=True)

        mesh = self.loadMeshFile(mesh_file_path)

        render_dict, match_result, init_camera = self.matchMeshToImage(image, mesh)

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
        triangle_vertices = mesh.vertices[mesh.faces[matched_triangle_idxs]]
        matched_triangle_centers = triangle_vertices.mean(axis=1)

        image_uv = torch.from_numpy(match_result['mkpts0'] / [image.shape[1], image.shape[0]]) - 0.5
        matched_uv = image_uv[on_mesh_idxs]

        estimated_camera = Camera.fromUVPoints(
            matched_triangle_centers,
            matched_uv,
            width=init_camera.width,
            height=init_camera.height,
        )

        render_dict, match_result, camera = self.matchMeshToImage(image, mesh, estimated_camera)

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
