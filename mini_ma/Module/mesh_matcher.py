import os
import torch
import trimesh
import numpy as np
from typing import Optional, Union
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
        '''
        self.detector = Detector(
            method=method,
            model_file_path=model_file_path,
        )
        self.device = self.detector.device
        '''

        self.device = 'cuda:7'
        return

    def loadMeshFile(
        self,
        mesh_file_path: str,
    ) -> Optional[Meshes]:
        """
        使用trimesh读取三角网格文件到指定device，支持多种格式（.obj, .ply, .glb, .stl, .off等）

        Args:
            mesh_file_path: 网格文件路径（trimesh支持的所有格式）

        Returns:
            Meshes对象，已移动到指定device
        """
        if not os.path.exists(mesh_file_path):
            print('[ERROR][MeshMatcher::loadMeshFile]')
            print('\t mesh file not exist!')
            print('\t mesh_file_path:', mesh_file_path)
            return None

        # 使用trimesh统一加载所有格式的三角网格文件
        mesh_trimesh = trimesh.load(mesh_file_path)

        # 如果加载的是Scene（包含多个mesh），合并所有mesh
        if isinstance(mesh_trimesh, trimesh.Scene):
            mesh_trimesh = trimesh.util.concatenate(
                [g for g in mesh_trimesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            )

        # 检查是否是Trimesh对象
        if not isinstance(mesh_trimesh, trimesh.Trimesh):
            raise ValueError(f"无法从文件 {mesh_file_path} 中提取三角网格数据")

        # 提取顶点和面数据
        verts = torch.from_numpy(mesh_trimesh.vertices).float()
        faces_idx = torch.from_numpy(mesh_trimesh.faces).long()

        verts = verts.to(self.device)
        faces_idx = faces_idx.to(self.device)

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
        # 转换pos和rot为torch tensor
        if isinstance(pos, (list, np.ndarray)):
            pos = torch.tensor(pos, dtype=torch.float32, device=self.device)
        else:
            pos = pos.to(self.device)

        if isinstance(rot, np.ndarray):
            rot = torch.tensor(rot, dtype=torch.float32, device=self.device)
        elif isinstance(rot, list):
            rot = torch.tensor(rot, dtype=torch.float32, device=self.device)
        else:
            rot = rot.to(self.device)

        # 确保rot是3x3矩阵
        rot = rot.reshape(1, 3, 3)  # [1, 3, 3]
        pos = pos.reshape(1, 3)  # [1, 3]

        R = rot.transpose(-2, -1)  # [1, 3, 3]

        T = torch.bmm(R, pos.unsqueeze(-1)).squeeze(-1)  # [1, 3]

        # 创建相机
        cameras = PerspectiveCameras(
            focal_length=((fx, fy),),
            principal_point=((cx, cy),),
            image_size=((height, width),),
            R=rot,
            T=T,
            device=self.device
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

        mesh = self.loadMeshFile(mesh_file_path)
        return {}
