import torch
import trimesh
import numpy as np
from typing import Tuple, Optional

from camera_control.Module.rgbd_camera import RGBDCamera
from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer

from mini_ma.Module.detector import Detector
from mini_ma.Module.camera_matcher import CameraMatcher


class MeshXYZMatcher(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def extractMatchedTrianglePoint(
        rgbd_camera: RGBDCamera,
        render_dict: dict,
        match_result: dict,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        matched_uv, matched_triangle_idxs = CameraMatcher.extractMatchedUVTriangle(
            render_dict=render_dict,
            match_result=match_result,
        )

        matched_points = rgbd_camera.queryUVPoints(matched_uv)

        return matched_points, matched_triangle_idxs

    @staticmethod
    def queryTrianglePoints(
        mesh: trimesh.Trimesh,
        rgbd_camera: RGBDCamera,
        detector: Detector,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        render_dict = NVDiffRastRenderer.renderNormal(
            mesh,
            rgbd_camera,
            bg_color=[0, 0, 0],
        )

        render_dict['image'] = render_dict['normal_camera']

        match_result = detector.detect(rgbd_camera.image, render_dict['image'])

        if match_result is None:
            print('[ERROR][MeshXYZMatcher::queryTrianglePoints]')
            print('\t matching pairs detect failed!')
            return None, None

        matched_points, matched_triangle_idxs = MeshXYZMatcher.extractMatchedTrianglePoint(
            rgbd_camera, render_dict, match_result)

        return matched_points, matched_triangle_idxs
