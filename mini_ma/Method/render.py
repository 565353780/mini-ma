import cv2
import torch
import numpy as np
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Any, Dict, Union

from mini_ma.Method.plotting import make_matching_figure


@torch.no_grad()
def renderMatchResult(
    match_result: Dict[str, Any],
    img0: Union[str, np.ndarray],
    img1: Union[str, np.ndarray],
    show_inliers_only: bool = False,
    dpi: int = 150,
) -> np.ndarray:
    """
    渲染匹配结果，生成可视化图片
    严格按照 demo.py 中 eval_relapose 的逻辑实现

    Args:
        match_result: detect() 方法返回的匹配结果字典
        image0_path: 第一张图片的文件路径（可选，如果match_result中没有img0）
        image1_path: 第二张图片的文件路径（可选，如果match_result中没有img1）
        save_path: 保存路径（可选），如果提供则保存图片到该路径
        show_inliers_only: 是否只显示内点（默认False，显示所有匹配点）
        dpi: 图片分辨率（默认150）

    Returns:
        numpy数组（BGR格式），可以直接用cv2.imshow()显示或cv2.imwrite()保存
        如果失败返回None
    """
    mkpts0 = match_result.get('mkpts0')
    mkpts1 = match_result.get('mkpts1')
    mconf = match_result.get('mconf')

    if mkpts0 is None or mkpts1 is None:
        print('[ERROR][render::renderMatchResult]')
        print('\t Missing mkpts0 or mkpts1 in match_result!')
        return None

    if isinstance(img0, str):
        img0_color = cv2.imread(img0)
    else:
        img0_color = img0
    if isinstance(img1, str):
        img1_color = cv2.imread(img1)
    else:
        img1_color = img1

    if img0_color is None or img1_color is None:
        print('[ERROR][render::renderMatchResult]')
        print('\t Failed to load images!')
        return None

    img0_color = cv2.cvtColor(img0_color, cv2.COLOR_BGR2RGB)
    img1_color = cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB)

    if len(mconf) > 0:
        conf_min = mconf.min()
        conf_max = mconf.max()
        mconf = (mconf - conf_min) / (conf_max - conf_min + 1e-5)
    color = cm.jet(mconf)

    if len(mkpts0) >= 4:
        ret_H, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC)
    else:
        inliers = None
        ret_H = None

    # print(f"Number of inliers: {inliers.sum() if inliers is not None else 0}")

    if show_inliers_only:
        # 使用 save_matching_figure 的逻辑：只显示内点
        if inliers is None or len(inliers) == 0:
            print('[WARNING][render::renderMatchResult]')
            print('\t No inliers to display!')
            return None

        inlier_mask = inliers.astype(bool).squeeze()

        if inlier_mask is None or len(inlier_mask) == 0:
            print('[WARNING][render::renderMatchResult]')
            print('\t No inliers after filtering!')
            return None

        mkpts0_inliers = mkpts0[inlier_mask]
        mkpts1_inliers = mkpts1[inlier_mask]
        color_inliers = color[inlier_mask]
    else:
        mkpts0_inliers = mkpts0
        mkpts1_inliers = mkpts1
        color_inliers = color

    text = [f'Matches:{len(mkpts0_inliers)}']

    fig = make_matching_figure(
        img0_color, img1_color,
        mkpts0_inliers, mkpts1_inliers,
        color_inliers,
        text=text,
        dpi=dpi,
        path=None
    )

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    # 从canvas获取RGB数据
    try:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape((h, w, 3))
    except (AttributeError, TypeError):
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape((h, w, 4))[:, :, :3]

    img_result = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

    plt.close(fig)

    return img_result
