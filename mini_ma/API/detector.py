import os

import numpy as np

from typing import Any, Dict, Optional, Union

from mini_ma.Module.detector import Detector


home = os.environ['HOME']
minima_model_file_path = f'{home}/chLi/Model/MINIMA/minima_lightglue.pth'


def get_default_model_paths() -> dict:
    """Return the default MINIMA pixel-matcher model paths.

    与 ``flux_mv.API.sampler.get_default_model_paths`` 风格保持一致，便于上层
    ``video_pipeline.queryModelPaths()`` 统一聚合所有模块的模型路径。
    """
    return {
        'model_file_path': minima_model_file_path,
    }


def build_detector(
    method: str = 'sp_lg',
    model_file_path: str = minima_model_file_path,
    device: str = 'cuda:0',
    is_offload_cpu: bool = True,
) -> Detector:
    """Build a MINIMA pixel-matcher :class:`Detector`.

    传参与 ``pixel-align-deform/video_pipeline.py`` 中的 ``PixelMatcher``
    构造保持一致。
    """
    return Detector(
        method=method,
        model_file_path=model_file_path,
        device=device,
        is_offload_cpu=is_offload_cpu,
    )


def detect(
    detector: Detector,
    image1: np.ndarray,
    image2: np.ndarray,
) -> Union[Dict[str, Any], None]:
    """Match two BGR images and return the match result dict (or None)."""
    return detector.detect(image1, image2)
