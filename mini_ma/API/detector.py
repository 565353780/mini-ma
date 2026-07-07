import os

import numpy as np

from typing import Any, Dict, List, Optional, Tuple, Union

from mini_ma.Method.render import renderMatchResult
from mini_ma.Module.detector import Detector


home = os.environ['HOME']
minima_model_file_path = f'{home}/chLi/Model/MINIMA/minima_lightglue.pth'

# sp_lg 基础特征权重 (SuperPoint / LightGlue) 默认路径。统一采用
# chLi/Model/<模型名>/<权重> 布局, 全部在本 submodule API 里定义, 由上层透传到
# run.py。SuperPoint 权重与 camera-pose-estimate (vggsfm-ba API) 共用同一路径,
# 不重复保存。
superpoint_model_file_path = f'{home}/chLi/Model/SuperPoint/superpoint_v1.pth'
lightglue_model_file_path = (
    f'{home}/chLi/Model/LightGlue/superpoint_lightglue_v0-1_arxiv.pth')

# 服务本地 ckpts 目录 (若权重放这里则优先命中)。可用 FAD_CKPTS_DIR 覆盖。
_CKPTS_DIR_ENV = 'FAD_CKPTS_DIR'


def _first_existing(*paths: str) -> str:
    for path in paths:
        if path and os.path.isfile(path):
            return path
    return paths[0]


def _ckpts_dir() -> Optional[str]:
    ckpts_dir = os.environ.get(_CKPTS_DIR_ENV, '').strip()
    return ckpts_dir or None


def get_default_model_paths() -> dict:
    """Return the default MINIMA + SuperPoint + LightGlue model paths.

    与 ``flux_mv.API.sampler.get_default_model_paths`` 风格保持一致，便于上层
    ``video_pipeline.queryModelPaths()`` 统一聚合所有模块的模型路径。所有权重
    默认路径均在本 submodule API 脚本里定义 (chLi/Model 约定)。若设置了
    ``FAD_CKPTS_DIR`` 且该目录下存在同名权重则优先命中。
    """
    ckpts_dir = _ckpts_dir()

    def _resolve(file_name: str, default_path: str) -> str:
        if ckpts_dir is not None:
            return _first_existing(
                os.path.join(ckpts_dir, file_name), default_path)
        return default_path

    return {
        'model_file_path': _resolve(
            'minima_lightglue.pth', minima_model_file_path),
        'superpoint_model_file_path': _resolve(
            'superpoint_v1.pth', superpoint_model_file_path),
        'lightglue_model_file_path': _resolve(
            'superpoint_lightglue_v0-1_arxiv.pth', lightglue_model_file_path),
    }


def query_missing_model_files() -> List[str]:
    """检查 MINIMA + SuperPoint + LightGlue 默认权重是否就绪, 返回缺失路径。"""
    return [
        path
        for path in get_default_model_paths().values()
        if not os.path.isfile(path)
    ]


def build_detector(
    method: str = 'sp_lg',
    model_file_path: Optional[str] = None,
    device: str = 'cuda:0',
    is_offload_cpu: bool = True,
    superpoint_weights_path: Optional[str] = None,
    lightglue_weights_path: Optional[str] = None,
) -> Detector:
    """Build a MINIMA pixel-matcher :class:`Detector`.

    传参与 ``pixel-align-deform/video_pipeline.py`` 中的 ``PixelMatcher``
    构造保持一致。``model_file_path`` / ``superpoint_weights_path`` /
    ``lightglue_weights_path`` 为 None 时, 回退本 API 脚本
    ``get_default_model_paths()`` 定义的默认路径 (chLi/Model 约定)。
    """
    defaults = get_default_model_paths()
    return Detector(
        method=method,
        model_file_path=model_file_path or defaults['model_file_path'],
        device=device,
        is_offload_cpu=is_offload_cpu,
        superpoint_weights_path=(
            superpoint_weights_path or defaults['superpoint_model_file_path']),
        lightglue_weights_path=(
            lightglue_weights_path or defaults['lightglue_model_file_path']),
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
