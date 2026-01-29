import torch
import trimesh
import numpy as np
from typing import Tuple, Optional

from camera_control.Module.camera import Camera
from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer

from mini_ma.Method.data import toNumpy
from mini_ma.Module.detector import Detector
from mini_ma.Module.camera_matcher import CameraMatcher


class MeshXYZMatcher(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def extractMatchedTrianglePoint(
        camera: Camera,
        render_dict: dict,
        match_result: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        matched_uv, matched_triangle_idxs = CameraMatcher.extractMatchedUVTriangle(
            render_dict=render_dict,
            match_result=match_result,
        )

        matched_points, confs, valid_mask = camera.queryUVPoints(matched_uv)

        return matched_points, matched_triangle_idxs, confs, valid_mask

    @staticmethod
    def queryTrianglePoints(
        mesh: trimesh.Trimesh,
        camera: Camera,
        detector: Detector,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        render_dict = NVDiffRastRenderer.renderNormal(
            mesh,
            camera,
            bg_color=[0, 0, 0],
        )

        normal_image_cv = toNumpy(render_dict['normal_camera'] * 255.0, np.uint8)[..., ::-1]

        match_result = detector.detect(camera.image_cv, normal_image_cv)

        if match_result is None:
            print('[ERROR][MeshXYZMatcher::queryTrianglePoints]')
            print('\t matching pairs detect failed!')
            return None, None, None, None

        return MeshXYZMatcher.extractMatchedTrianglePoint(
            camera, render_dict, match_result)
