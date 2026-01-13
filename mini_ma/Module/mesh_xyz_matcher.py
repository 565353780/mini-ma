from mini_ma.Method.data import toNumpy
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        matched_uv, matched_triangle_idxs = CameraMatcher.extractMatchedUVTriangle(
            render_dict=render_dict,
            match_result=match_result,
        )

        matched_points, valid_mask = rgbd_camera.queryUVPoints(matched_uv)

        return matched_points, matched_triangle_idxs, valid_mask

    @staticmethod
    def queryTrianglePoints(
        mesh: trimesh.Trimesh,
        rgbd_camera: RGBDCamera,
        detector: Detector,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        render_dict = NVDiffRastRenderer.renderNormal(
            mesh,
            rgbd_camera,
            bg_color=[0, 0, 0],
        )

        normal_image_cv = toNumpy(render_dict['normal_camera'] * 255.0, np.uint8)[..., ::-1]

        match_result = detector.detect(rgbd_camera.image_cv, normal_image_cv)

        if match_result is None:
            print('[ERROR][MeshXYZMatcher::queryTrianglePoints]')
            print('\t matching pairs detect failed!')
            return None, None, None

        return MeshXYZMatcher.extractMatchedTrianglePoint(
            rgbd_camera, render_dict, match_result)
