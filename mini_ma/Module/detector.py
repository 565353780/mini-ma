import os
import cv2
import torch
import numpy as np
import matplotlib

from mini_ma.Method.data import toGrayImage
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional, Union, Dict, Any

from mini_ma.Model.loader import load_model
from mini_ma.Method.io import loadImage
from mini_ma.Method.plotting import make_matching_figure



class Detector(object):
    def __init__(
        self,
        method: str = "roma",
        model_file_path: Optional[str] = None,
        match_threshold: Optional[float] = None,
        fine_threshold: Optional[float] = None,
        thr: Optional[float] = None,
        ckpt2: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        初始化检测器

        Args:
            method: 模型方法名称 ('xoftr', 'loftr', 'roma', 'sp_lg')
            model_file_path: 模型权重文件路径（ckpt）
            match_threshold: XoFTR 粗匹配阈值
            fine_threshold: XoFTR 精细匹配阈值
            thr: LoFTR 匹配阈值
            ckpt2: RoMa 模型类型 ('large' 或其他)
            device: 设备 ('cuda' 或 'cpu')，默认自动选择
        """
        self.method = method
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.matcher = None

        # 创建参数对象
        class Args:
            def __init__(self):
                self.ckpt = model_file_path
                self.match_threshold = match_threshold
                self.fine_threshold = fine_threshold
                self.thr = thr
                self.ckpt2 = ckpt2

        self.args = Args()

        # 设置默认值
        if method == "xoftr":
            if self.args.match_threshold is None:
                self.args.match_threshold = 0.3
            if self.args.fine_threshold is None:
                self.args.fine_threshold = 0.1
            if self.args.ckpt is None:
                self.args.ckpt = "./weights/weights_xoftr_640.ckpt"
            self.is_gray = True
        elif method == "loftr":
            if self.args.thr is None:
                self.args.thr = 0.2
            if self.args.ckpt is None:
                self.args.ckpt = "./weights/minima_loftr.ckpt"
            self.is_gray = True
        elif method == "sp_lg":
            if self.args.ckpt is None:
                self.args.ckpt = "./weights/minima_lightglue.pth"
            self.is_gray = True
        elif method == "roma":
            if self.args.ckpt2 is None:
                self.args.ckpt2 = "large"
            if self.args.ckpt is None:
                self.args.ckpt = './weights/minima_roma.pth'
            self.is_gray = False
        else:
            raise ValueError(f"Unknown method: {method}. Supported methods: 'xoftr', 'loftr', 'roma', 'sp_lg'")

        if model_file_path is not None:
            self.loadModel(model_file_path)
        elif self.args.ckpt is not None:
            if os.path.exists(self.args.ckpt):
                self.loadModel(self.args.ckpt)
        return

    def loadModel(
        self,
        model_file_path: str,
    ) -> bool:
        """
        加载模型

        Args:
            model_file_path: 模型权重文件路径

        Returns:
            是否加载成功
        """
        if not os.path.exists(model_file_path):
            print('[ERROR][Detector::loadModel]')
            print('\t model file not exist!')
            print('\t model_file_path:', model_file_path)
            return False

        self.args.ckpt = model_file_path

        self.matcher = load_model(
            self.method,
            self.args,
            use_path=False,
            test_orginal_megadepth=False,
        )
        print(f'[INFO][Detector::loadModel]')
        print(f'\t Successfully loaded {self.method} model from: {model_file_path}')
        return True

    def detect(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        K0: Optional[np.ndarray] = None,
        K1: Optional[np.ndarray] = None,
        dist0: Optional[np.ndarray] = None,
        dist1: Optional[np.ndarray] = None,
    ) -> Union[Dict[str, Any], None]:
        """
        检测两张图片的匹配点

        Args:
            image1: 第一张图片（numpy 数组，BGR 格式）
            image2: 第二张图片（numpy 数组，BGR 格式）
            K0: 第一张图片的相机内参矩阵（可选）
            K1: 第二张图片的相机内参矩阵（可选）
            dist0: 第一张图片的畸变系数（可选）
            dist1: 第二张图片的畸变系数（可选）

        Returns:
            匹配结果字典，包含：
            - mkpts0: 第一张图片的匹配点坐标
            - mkpts1: 第二张图片的匹配点坐标
            - mconf: 匹配置信度
            - matches: 匹配点对（拼接后的坐标）
            - match_time: 匹配耗时
            如果失败返回 None
        """
        if self.is_gray:
            image1 = toGrayImage(iamge1)
            image2 = toGrayImage(iamge2)
        else:
            image1 = toRGBImage(iamge1)
            image2 = toRGBImage(iamge2)

        result = self.matcher(
            image1, image2,
            K0=K0, K1=K1,
            dist0=dist0, dist1=dist1,
        )
        return result

    def detectImageFilePair(
        self,
        image1_file_path: str,
        image2_file_path: str,
        K0: Optional[np.ndarray] = None,
        K1: Optional[np.ndarray] = None,
        dist0: Optional[np.ndarray] = None,
        dist1: Optional[np.ndarray] = None,
    ) -> Union[Dict[str, Any], None]:
        """
        从文件路径检测图片对
        
        读取图片文件并将其对应数据传入 detect() 来获取结果。
        图片处理方式位于 Method/data_io_*.py
        
        Args:
            image1_file_path: 第一张图片文件路径
            image2_file_path: 第二张图片文件路径
            K0: 第一张图片的相机内参矩阵（可选）
            K1: 第二张图片的相机内参矩阵（可选）
            dist0: 第一张图片的畸变系数（可选）
            dist1: 第二张图片的畸变系数（可选）

        Returns:
            匹配结果字典，如果失败返回 None
        """
        if self.method == "roma":
            # RoMa 使用彩色图片
            is_gray = False
        else:
            # LoFTR, sp_lg, xoftr 使用灰度图片
            is_gray = True

        image1_data = loadImage(image1_file_path, is_gray)
        image2_data = loadImage(image2_file_path, is_gray)

        if image1_data is None:
            print('[ERROR][Detector::detectImageFilePair]')
            print('\t loadImage failed!')
            print('\t image1_file_path:', image1_file_path)
            return None
        if image2_data is None:
            print('[ERROR][Detector::detectImageFilePair]')
            print('\t loadImage failed!')
            print('\t image2_file_path:', image2_file_path)
            return None

        return self.detect(image1_data, image2_data, K0=K0, K1=K1, dist0=dist0, dist1=dist1)

    def renderMatchResult(
        self,
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
            print('[ERROR][Detector::renderMatchResult]')
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
            print('[ERROR][Detector::renderMatchResult]')
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

        print(f"Number of inliers: {inliers.sum() if inliers is not None else 0}")

        if show_inliers_only:
            # 使用 save_matching_figure 的逻辑：只显示内点
            if inliers is None or len(inliers) == 0:
                print('[WARNING][Detector::renderMatchResult]')
                print('\t No inliers to display!')
                return None

            inlier_mask = inliers.astype(bool).squeeze()

            if inlier_mask is None or len(inlier_mask) == 0:
                print('[WARNING][Detector::renderMatchResult]')
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
