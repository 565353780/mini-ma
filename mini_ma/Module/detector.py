import os
import torch
import numpy as np
from typing import Optional, Union, Dict, Any

from mini_ma.Model.loader import load_model
from mini_ma.Method.io import loadImage
from mini_ma.Method.data import toGrayImage, toRGBImage


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
        is_offload_cpu: bool = False,
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
        # offload 模式：matcher 子模型常驻 CPU，``detect()`` 推理窗口内才搬到
        # ``self.device`` 并在 finally 卸载回 CPU；默认模式保持原版本 GPU 常驻。
        self.is_offload_cpu = bool(is_offload_cpu)
        # 当外部显式将 matcher 常驻 GPU（例如 detect() 的批量调用场景）时置位，
        # 此时 detect() 不再在每次推理前后反复搬运 matcher，避免无谓的 CPU<->GPU
        # 拷贝开销；调用方负责在批量结束后再卸载回 CPU。
        self.is_matcher_pinned = False
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

        # offload 模式下，构造完成后立即把 matcher 子模型搬到 CPU；默认模式保持
        # ``load_model`` 已搬到 ``self.device`` 的状态。
        if self.is_offload_cpu:
            self._offloadMatcherToCPU()

        print(f'[INFO][Detector::loadModel]')
        print(f'\t Successfully loaded {self.method} model from: {model_file_path}')
        return True

    def _getInnerMatcher(self) -> Optional[torch.nn.Module]:
        '''Return the underlying ``DataIOWrapper.model`` (nn.Module) if any.

        ``self.matcher`` is the ``from_paths`` / ``from_cv_imgs`` bound method
        produced by ``load_model``; its ``__self__`` is the ``DataIOWrapper``,
        whose ``model`` attribute holds the real matcher network we need to
        ferry between CPU and GPU.
        '''
        if self.matcher is None:
            return None
        wrapper = getattr(self.matcher, '__self__', None)
        if wrapper is None:
            return None
        return getattr(wrapper, 'model', None)

    def _setWrapperDevice(self, device: str) -> None:
        '''Keep ``DataIOWrapper.device`` in sync with the actual model device.

        ``preprocess_image`` in the ``DataIOWrapper`` family of files uses
        ``self.device`` to decide where to allocate input tensors; if we move
        the model without updating it, inputs land on the wrong device.
        '''
        if self.matcher is None:
            return
        wrapper = getattr(self.matcher, '__self__', None)
        if wrapper is None:
            return
        try:
            wrapper.device = torch.device(device)
        except Exception:
            wrapper.device = device

    def _moveMatcherToDevice(self) -> None:
        '''offload 模式下推理前把 matcher 临时搬到 ``self.device``。'''
        if not self.is_offload_cpu:
            return
        inner = self._getInnerMatcher()
        if inner is None:
            return
        try:
            inner.to(self.device)
        except Exception:
            pass
        self._setWrapperDevice(self.device)

    def _offloadMatcherToCPU(self) -> None:
        '''offload 模式下推理结束后把 matcher 卸载回 CPU。'''
        inner = self._getInnerMatcher()
        if inner is not None:
            try:
                inner.to('cpu')
            except Exception:
                pass
        self._setWrapperDevice('cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def toGPU(self) -> None:
        '''供外部调用：把 matcher 常驻到 ``self.device``。

        在需要连续多次 ``detect()`` 的批量场景下，先调用本方法把模型搬到 GPU，
        随后 ``detect()`` 检测到 ``is_matcher_pinned`` 为真便跳过逐次的搬运/卸载。
        '''
        inner = self._getInnerMatcher()
        if inner is not None:
            try:
                inner.to(self.device)
            except Exception:
                pass
        self._setWrapperDevice(self.device)
        self.is_matcher_pinned = True
        return

    def toCPU(self) -> None:
        '''供外部调用：把 matcher 卸载回 CPU 并解除常驻标志。

        与 :meth:`toGPU` 配对使用，批量 ``detect()`` 结束后调用以释放显存。
        '''
        self.is_matcher_pinned = False
        self._offloadMatcherToCPU()
        return

    @torch.no_grad()
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
            image1 = toGrayImage(image1)
            image2 = toGrayImage(image2)
        else:
            image1 = toRGBImage(image1)
            image2 = toRGBImage(image2)

        # matcher 已被外部常驻 GPU 时（is_matcher_pinned），跳过逐次的搬运/卸载，
        # 由调用方在批量结束后统一调用 toCPU()。
        if self.is_matcher_pinned:
            return self.matcher(
                image1, image2,
                K0=K0, K1=K1,
                dist0=dist0, dist1=dist1,
            )

        self._moveMatcherToDevice()
        try:
            result = self.matcher(
                image1, image2,
                K0=K0, K1=K1,
                dist0=dist0, dist1=dist1,
            )
        finally:
            if self.is_offload_cpu:
                self._offloadMatcherToCPU()
        return result

    @torch.no_grad()
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
