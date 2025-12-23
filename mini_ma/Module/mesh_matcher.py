import os
import torch
import trimesh
import tempfile
import numpy as np
import open3d as o3d
from typing import Optional, Union
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
    PerspectiveCameras,
)

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
        return

    def loadMeshFile(
        self,
        mesh_file_path: str,
    ):
        """
        使用pytorch3d读取三角网格文件到指定device

        Args:
            mesh_file_path: 网格文件路径（支持.obj、.ply和.glb格式）

        Returns:
            Meshes对象，已移动到指定device
        """
        device = self.detector.device

        # 获取文件扩展名
        file_ext = os.path.splitext(mesh_file_path)[1].lower()

        if file_ext == '.obj':
            # 加载.obj文件
            verts, faces, aux = load_obj(mesh_file_path)
            faces_idx = faces.verts_idx
        elif file_ext == '.ply':
            # 使用open3d加载.ply文件，然后转换为pytorch3d格式
            mesh_o3d = o3d.io.read_triangle_mesh(mesh_file_path)
            if not mesh_o3d.has_vertices():
                raise ValueError(f"无法从文件 {mesh_file_path} 中读取顶点数据")

            # 转换为numpy数组，然后转为torch tensor
            verts = torch.from_numpy(np.asarray(mesh_o3d.vertices)).float()

            # 获取三角形面
            if mesh_o3d.has_triangles():
                faces_idx = torch.from_numpy(np.asarray(mesh_o3d.triangles)).long()
            else:
                raise ValueError(f"网格文件 {mesh_file_path} 不包含三角形面数据")
        elif file_ext == '.glb':
            mesh_trimesh = trimesh.load(mesh_file_path)

            if isinstance(mesh_trimesh, trimesh.Scene):
                mesh_trimesh = trimesh.util.concatenate(
                    [g for g in mesh_trimesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                )

            if not isinstance(mesh_trimesh, trimesh.Trimesh):
                raise ValueError(f"无法从.glb文件中提取三角网格数据")

            try:
                verts = torch.from_numpy(mesh_trimesh.vertices).float()
                faces_idx = torch.from_numpy(mesh_trimesh.faces).long()
            except Exception as e:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as tmp_file:
                    tmp_obj_path = tmp_file.name

                mesh_trimesh.export(tmp_obj_path)

                # 使用pytorch3d加载临时.obj文件
                verts, faces, aux = load_obj(tmp_obj_path)
                faces_idx = faces.verts_idx

                if os.path.exists(tmp_obj_path):
                    os.remove(tmp_obj_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}，仅支持.obj、.ply和.glb格式")

        # 将数据移动到指定device
        verts = verts.to(device)
        faces_idx = faces_idx.to(device)

        # 创建Meshes对象
        mesh = Meshes(verts=[verts], faces=[faces_idx])

        return mesh

    def queryCamera(
        self,
        mesh: Meshes,
        width: int = 640,
        height: int = 480,
        fx: float = 500.0,
        fy: float = 500.0,
        cx: float = 320.0,
        cy: float = 240.0,
        pos: Union[torch.Tensor, np.ndarray, list] = [0, 0, 0],
        rot: Union[torch.Tensor, np.ndarray, list] = np.eye(3),
    ) -> dict:
        """
        创建pytorch3d中的相机，渲染三角网格，并获取渲染图中每个像素对应的
        三角网格表面的顶点插值信息
        
        Args:
            mesh: Meshes对象，要渲染的三角网格
            width: 图像宽度
            height: 图像高度
            fx: 相机焦距x
            fy: 相机焦距y
            cx: 主点x坐标
            cy: 主点y坐标
            pos: 相机在世界坐标系中的位置 [x, y, z]
            rot: 相机旋转矩阵（3x3），从相机坐标系到世界坐标系的旋转
            
        Returns:
            dict包含:
                - pix_to_face: [1, H, W, 1] 每个像素对应的面索引，-1表示背景
                - bary_coords: [1, H, W, 1, 3] 每个像素对应的重心坐标
                - fragments: 完整的fragments对象
        """
        device = self.detector.device
        
        # 转换pos和rot为torch tensor
        if isinstance(pos, (list, np.ndarray)):
            pos = torch.tensor(pos, dtype=torch.float32, device=device)
        else:
            pos = pos.to(device)
            
        if isinstance(rot, np.ndarray):
            rot = torch.tensor(rot, dtype=torch.float32, device=device)
        elif isinstance(rot, list):
            rot = torch.tensor(rot, dtype=torch.float32, device=device)
        else:
            rot = rot.to(device)
        
        # 确保rot是3x3矩阵
        if rot.dim() == 2:
            rot = rot.unsqueeze(0)  # [1, 3, 3]
        if pos.dim() == 1:
            pos = pos.unsqueeze(0)  # [1, 3]
        
        # pytorch3d的PerspectiveCameras使用R和T表示从世界坐标系到相机坐标系的变换
        # 如果rot是从相机到世界的旋转，需要转置得到从世界到相机的旋转
        R = rot.transpose(-2, -1)  # [1, 3, 3]
        
        # T是相机在世界坐标系中的位置，但pytorch3d期望的是从世界到相机的平移
        # T = -R @ pos
        T = -torch.bmm(R, pos.unsqueeze(-1)).squeeze(-1)  # [1, 3]
        
        # 创建相机
        cameras = PerspectiveCameras(
            focal_length=((fx, fy),),
            principal_point=((cx, cy),),
            image_size=((height, width),),
            R=R,
            T=T,
            device=device
        )
        
        # 设置光栅化参数
        raster_settings = RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1
        )
        
        # 创建光栅化器
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )
        
        # 渲染网格，获取fragments
        fragments = rasterizer(mesh)
        
        # 提取每个像素对应的面索引和重心坐标
        pix_to_face = fragments.pix_to_face  # [1, H, W, 1]
        bary_coords = fragments.bary_coords  # [1, H, W, 1, 3]
        
        return {
            'pix_to_face': pix_to_face,
            'bary_coords': bary_coords,
            'fragments': fragments
        }

    def matchMeshFileToImageFile(
        self,
        image_file_path: str,
        mesh_file_path: str,
    ) -> dict:
        if self.detector.method == "roma":
            # RoMa 使用彩色图片
            is_gray = False
        else:
            # LoFTR, sp_lg, xoftr 使用灰度图片
            is_gray = True

        image_data = loadImage(image_file_path, is_gray)

        # TODO: 实现网格与图像的匹配逻辑
        mesh = self.loadMeshFile(mesh_file_path)
        return {}
