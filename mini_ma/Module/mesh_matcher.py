import os
import cv2
from mini_ma.Method.path import createFileFolder
import torch
import trimesh
import numpy as np
import nvdiffrast.torch as dr
from typing import Optional, Union

from camera_control.Method.render import create_line_set
from camera_control.Module.camera import Camera

from mini_ma.Method.io import loadImage
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
        width: int = 640,
        height: int = 480,
        fx: float = 500.0,
        fy: float = 500.0,
        cx: float = 320.0,
        cy: float = 240.0,
        pos: Union[torch.Tensor, np.ndarray, list] = None,
        rot: Union[torch.Tensor, np.ndarray, list] = None,
        save_debug_image_path: Optional[Union[str, list]] = None,
        use_normal_shading: bool = False,
        light_direction: Union[torch.Tensor, np.ndarray, list] = None,
    ) -> dict:
        """
        使用nvdiffrast渲染三角网格，并获取渲染图中每个像素对应的
        三角网格表面的顶点插值信息

        Args:
            mesh: trimesh.Trimesh对象，要渲染的三角网格
            width: 图像宽度
            height: 图像高度
            fx: 相机焦距x
            fy: 相机焦距y
            cx: 主点x坐标
            cy: 主点y坐标
            pos: 相机在世界坐标系中的位置，支持单个[N, 3]或批量[N, 3]的tensor/array/list
            rot: 相机旋转矩阵，支持单个[3, 3]或批量[N, 3, 3]的tensor/array/list，从相机坐标系到世界坐标系的旋转
            save_debug_image_path: 可选，如果提供路径字符串，将保存所有视角的渲染图像（按idx命名）
                                  如果提供路径列表，将按列表顺序保存每张图片
            use_normal_shading: 是否使用法向着色（白模效果），默认False使用顶点颜色
            light_direction: 光照方向（世界坐标系），默认为[0, 0, -1]（从相机方向照射）

        Returns:
            dict包含:
                - images: [N, H, W, 3] 渲染的图像 (RGB)
                - rasterize_output: [N, H, W, 4] rasterize主输出 (u, v, z/w, triangle_id)
                - bary_derivs: [N, H, W, 4] 重心坐标的图像空间导数 (du/dX, du/dY, dv/dX, dv/dY)
        """
        # 1. 处理输入参数，转换为tensor
        if pos is None:
            # 默认相机位置：在mesh中心前方
            bbox_center = (np.min(mesh.vertices, axis=0) + np.max(mesh.vertices, axis=0)) / 2
            bbox_size = np.linalg.norm(np.max(mesh.vertices, axis=0) - np.min(mesh.vertices, axis=0))
            pos = [[bbox_center[0], bbox_center[1], bbox_center[2] - bbox_size * 2.0]]
        
        # 转换pos为tensor [N, 3]
        if isinstance(pos, list):
            pos = torch.tensor(pos, dtype=torch.float32, device=self.device)
        elif isinstance(pos, np.ndarray):
            pos = torch.from_numpy(pos).float().to(self.device)
        elif isinstance(pos, torch.Tensor):
            pos = pos.float().to(self.device)
        
        # 确保pos是2D张量 [N, 3]
        if pos.dim() == 1:
            pos = pos.unsqueeze(0)
        
        N = pos.shape[0]  # 相机数量
        
        # 计算mesh中心（用于默认相机朝向）
        bbox_center = (np.min(mesh.vertices, axis=0) + np.max(mesh.vertices, axis=0)) / 2
        bbox_center_tensor = torch.from_numpy(bbox_center).float().to(self.device)
        
        # 转换rot为tensor [N, 3, 3]
        if rot is None:
            # 默认旋转：让相机朝向mesh中心
            rot = []
            for i in range(N):
                # 计算从相机到mesh中心的方向（相机朝向）
                forward = bbox_center_tensor - pos[i]  # [3]
                forward = forward / (torch.norm(forward) + 1e-8)  # 归一化
                
                # 在OpenGL坐标系中，forward应该是-z方向
                # 选择up向量为世界坐标系的y轴
                world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device)
                
                # 计算right = forward × up
                right = torch.cross(forward, world_up)
                right = right / (torch.norm(right) + 1e-8)
                
                # 重新计算up = right × forward
                up = torch.cross(right, forward)
                up = up / (torch.norm(up) + 1e-8)
                
                # 构建旋转矩阵（从相机到世界）
                # 在OpenGL坐标系中：x=right, y=up, z=-forward
                R_cam_to_world = torch.stack([right, up, -forward], dim=1)  # [3, 3]
                rot.append(R_cam_to_world)
            
            rot = torch.stack(rot, dim=0)  # [N, 3, 3]
        else:
            if isinstance(rot, list):
                rot = torch.tensor(rot, dtype=torch.float32, device=self.device)
            elif isinstance(rot, np.ndarray):
                rot = torch.from_numpy(rot).float().to(self.device)
            elif isinstance(rot, torch.Tensor):
                rot = rot.float().to(self.device)
            
            # 确保rot是3D张量 [N, 3, 3]
            if rot.dim() == 2:
                rot = rot.unsqueeze(0).repeat(N, 1, 1)
        
        # 2. 创建nvdiffrast上下文
        glctx = dr.RasterizeCudaContext(device=self.device)
        
        # 3. 准备mesh数据
        vertices = torch.from_numpy(mesh.vertices).float().to(self.device)  # [V, 3]
        faces = torch.from_numpy(mesh.faces).int().to(self.device)  # [F, 3]
        
        # 计算或获取顶点法向量
        if use_normal_shading:
            # 确保mesh有法向量，如果没有则计算
            if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
                mesh.compute_vertex_normals()
            
            vertex_normals = torch.from_numpy(mesh.vertex_normals).float().to(self.device)  # [V, 3]
            
            # 设置光照方向（世界坐标系）
            if light_direction is None:
                # 默认使用相机方向的平均值作为光照方向
                # 这样可以产生类似前向照明的效果
                light_direction = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)
            else:
                if isinstance(light_direction, list):
                    light_direction = torch.tensor(light_direction, dtype=torch.float32, device=self.device)
                elif isinstance(light_direction, np.ndarray):
                    light_direction = torch.from_numpy(light_direction).float().to(self.device)
                elif isinstance(light_direction, torch.Tensor):
                    light_direction = light_direction.float().to(self.device)
            
            # 归一化光照方向
            light_direction = light_direction / (torch.norm(light_direction) + 1e-8)
        else:
            # 获取顶点颜色
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                vertex_colors = torch.from_numpy(mesh.visual.vertex_colors[:, :3]).float().to(self.device) / 255.0  # [V, 3]
            else:
                vertex_colors = torch.ones_like(vertices) * 0.7  # 默认灰色

        # 4. 构建投影矩阵（参考nvdiffrast官方示例）
        # 根据mesh大小动态设置near和far
        bbox_size = np.linalg.norm(np.max(mesh.vertices, axis=0) - np.min(mesh.vertices, axis=0))
        near = bbox_size * 0.1
        far = bbox_size * 10.0

        # 构建标准OpenGL透视投影矩阵
        # 参考: https://github.com/NVlabs/nvdiffrast/blob/main/samples/torch/util.py
        def perspective_projection(fovy_radians, aspect, n, f):
            """构建OpenGL透视投影矩阵"""
            y = np.tan(fovy_radians / 2)
            return np.array([
                [1/(y*aspect),    0,            0,              0],
                [           0, 1/-y,            0,              0],
                [           0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                [           0,    0,           -1,              0]
            ], dtype=np.float32)
        
        # 从fx, fy计算fov
        fovy = 2 * np.arctan(height / (2 * fy))
        aspect = width / height
        proj_mtx = torch.from_numpy(perspective_projection(fovy, aspect, near, far)).to(self.device)
        
        # 5. 批量渲染
        all_images = []
        all_rast_out = []
        all_bary_derivs = []
        
        for i in range(N):
            # 构建视图矩阵 (world to camera)
            # 参考nvdiffrast官方示例的transform_pos函数
            R = rot[i].T  # [3, 3] 从世界到相机的旋转
            t = pos[i]    # [3] 相机在世界坐标系中的位置
            
            # 构建4x4视图矩阵
            view_mtx = torch.eye(4, dtype=torch.float32, device=self.device)
            view_mtx[:3, :3] = R
            view_mtx[:3, 3] = -R @ t  # 平移部分
            
            # 组合MVP矩阵 (projection * view)
            mvp = proj_mtx @ view_mtx  # [4, 4]
            
            # 顶点变换到裁剪空间 (参考官方示例的transform_pos)
            # (x,y,z) -> (x,y,z,1)
            vertices_homo = torch.cat([
                vertices,
                torch.ones((vertices.shape[0], 1), dtype=torch.float32, device=self.device)
            ], dim=1)  # [V, 4]
            
            # 应用MVP变换: posw @ mvp.t()
            # 注意：官方示例使用 matmul(posw, mvp.t())
            vertices_clip = torch.matmul(vertices_homo, mvp.t())  # [V, 4]
            
            # 添加batch维度 [1, V, 4] (参考官方示例返回 [None, ...])
            vertices_clip_batch = vertices_clip.unsqueeze(0).contiguous()
            
            # 光栅化
            rast_out, rast_out_db = dr.rasterize(
                glctx,
                vertices_clip_batch,  # [1, V, 4]
                faces,
                resolution=[height, width]
            )

            if use_normal_shading:
                # 法向着色：插值顶点法向量
                normals_interp, _ = dr.interpolate(
                    vertex_normals.unsqueeze(0),  # [1, V, 3]
                    rast_out,
                    faces
                )
                
                # 归一化插值后的法向量
                normals_interp = normals_interp / (torch.norm(normals_interp, dim=-1, keepdim=True) + 1e-8)
                
                # 将法向量从世界坐标系转换到相机坐标系
                R = rot[i].T  # [3, 3] 世界到相机的旋转
                normals_cam = torch.matmul(normals_interp[0], R.T)  # [H, W, 3]
                
                # 将光照方向也转换到相机坐标系
                light_dir_cam = torch.matmul(light_direction, R.T)  # [3]
                light_dir_cam = light_dir_cam / (torch.norm(light_dir_cam) + 1e-8)
                
                # 计算Lambert着色：max(0, dot(normal, light))
                # 点积：[H, W, 3] * [3] -> [H, W]
                diffuse = torch.sum(normals_cam * light_dir_cam, dim=-1, keepdim=True)  # [H, W, 1]
                diffuse = torch.clamp(diffuse, min=0.0, max=1.0)
                
                # 添加环境光，避免完全黑暗的区域（类似MeshLab的效果）
                ambient = 0.3
                shading = ambient + (1.0 - ambient) * diffuse  # [H, W, 1]
                
                # 转换为RGB（白模：所有通道相同的灰度值）
                # 背景设为白色，模型表面根据法向着色
                image = shading.repeat(1, 1, 3)  # [H, W, 3]
                
                # 处理背景（triangle_id < 0 的像素）
                mask = rast_out[0, :, :, 3] > 0  # [H, W]
                background = torch.ones_like(image)  # 白色背景
                image = torch.where(mask.unsqueeze(-1), image, background)
            else:
                # 插值顶点颜色
                colors_interp, _ = dr.interpolate(
                    vertex_colors.unsqueeze(0),  # [1, V, 3]
                    rast_out,
                    faces
                )

                # 获取RGB图像 [H, W, 3]
                image = colors_interp[0]

            # 保存调试图像
            if save_debug_image_path is not None:
                if isinstance(save_debug_image_path, list):
                    if i < len(save_debug_image_path):
                        save_path = save_debug_image_path[i]
                    else:
                        save_path = None
                else:
                    # 如果是字符串，添加索引
                    base_path = os.path.splitext(save_debug_image_path)[0]
                    ext = os.path.splitext(save_debug_image_path)[1]
                    save_path = f"{base_path}_{i}{ext}"

                if save_path is not None:
                    save_dir = os.path.dirname(save_path)
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                    img_np = image.detach().cpu().numpy()
                    img_np = np.clip(np.rint(img_np * 255), 0, 255).astype(np.uint8)
                    cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                    print(f"Saved render image {i} to: {save_path}")

            all_images.append(image)
            all_rast_out.append(rast_out[0])
            all_bary_derivs.append(rast_out_db[0] if rast_out_db is not None else torch.zeros_like(rast_out[0]))

        # 6. 组合结果
        result = {
            'images': torch.stack(all_images, dim=0),  # [N, H, W, 3]
            'rasterize_output': torch.stack(all_rast_out, dim=0),  # [N, H, W, 4]
            'bary_derivs': torch.stack(all_bary_derivs, dim=0),  # [N, H, W, 4]
        }

        return result

    def matchMeshFileToImageFile(
        self,
        image_file_path: str,
        mesh_file_path: str,
        save_match_result_folder_path: str,
    ) -> dict:
        if self.detector.method == "roma":
            # RoMa 使用彩色图片
            is_gray = False
        else:
            # LoFTR, sp_lg, xoftr 使用灰度图片
            is_gray = True

        # image_data = loadImage(image_file_path, is_gray)

        mesh = self.loadMeshFile(mesh_file_path)

        min_bound = np.min(mesh.vertices, axis=0)
        max_bound = np.max(mesh.vertices, axis=0)
        center = (min_bound + max_bound) / 2.0

        render_dict = self.queryCamera(
            mesh=mesh,
            width=2560,
            height=1440,
            cx=1280,
            cy=720,
            pos=[
                center + [0, 0, 1],
            ],
            save_debug_image_path=save_match_result_folder_path + 'debug.png',
            use_normal_shading=True,
            light_direction=[1, 1, 1],
        )

        render

        render_image_file_path = save_match_result_folder_path + 'debug_0.png'
        # render_image_data = loadImage(render_image_file_path, is_gray)

        match_result = self.detector.detectImageFilePair(image_file_path, render_image_file_path)

        if match_result is None:
            print('[ERROR][MeshMatcher::matchMeshFileToImageFile]')
            print('\t detectImageFilePair failed!')
            return {}

        for key, value in match_result.items():
            try:
                print(key, value.shape)
            except:
                print(key, value)

        img_vis = self.detector.renderMatchResult(
            match_result,
            image_file_path,
            render_image_file_path,
        )
        save_path=save_match_result_folder_path + "render_matches_all.jpg"
        createFileFolder(save_path)
        cv2.imwrite(save_path, img_vis)
        return {}
