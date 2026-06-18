import os

import numpy as np

from typing import Any, Dict, List, Optional, Tuple, Union

from mini_ma.Method.render import renderMatchResult
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


def detect_batch(
    detector: Detector,
    image_pairs: List[Tuple[np.ndarray, np.ndarray]],
    chunk_size: int = 16,
) -> List[Union[Dict[str, Any], None]]:
    """批量匹配多对 BGR 图像，返回与 ``detect`` 同构的结果列表。

    每个元素的字段/坐标语义与单对 ``detect`` 完全一致，便于上层（如
    ``pixel-align-deform`` 的 ``query_deform_field``）以 ``chunk_size`` 为单位
    做批量推理而无需改动下游消费逻辑。底层不支持真 batch 的 matcher 会自动逐对
    回退。
    """
    return detector.detect_batch(image_pairs, chunk_size=chunk_size)


def render_match_result(
    match_result: Dict[str, Any],
    image1: Union[str, np.ndarray],
    image2: Union[str, np.ndarray],
    show_inliers_only: bool = False,
    dpi: int = 150,
) -> np.ndarray:
    """Render a MINIMA match result as a BGR image."""
    return renderMatchResult(
        match_result=match_result,
        img0=image1,
        img1=image2,
        show_inliers_only=show_inliers_only,
        dpi=dpi,
    )
