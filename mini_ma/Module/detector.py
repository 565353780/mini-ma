import os
import cv2
import torch
import numpy as np
from typing import Optional, Union, Dict, Any

from mini_ma.Model.loader import load_model
from mini_ma.Method.path import createFileFolder


class Detector(object):
    """
    图像匹配检测器
    
    支持多种模型：
    - xoftr: XoFTR 模型
    - loftr: LoFTR 模型
    - roma: RoMa 模型
    - sp_lg: SuperPoint + LightGlue 模型
    """
    
    def __init__(
        self,
        method: str = "sp_lg",
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
        self.matcher_from_paths = None
        self.matcher_from_cv_imgs = None

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
        elif method == "loftr":
            if self.args.thr is None:
                self.args.thr = 0.2
            if self.args.ckpt is None:
                self.args.ckpt = "./weights/minima_loftr.ckpt"
        elif method == "sp_lg":
            if self.args.ckpt is None:
                self.args.ckpt = "./weights/minima_lightglue.pth"
        elif method == "roma":
            if self.args.ckpt2 is None:
                self.args.ckpt2 = "large"
            if self.args.ckpt is None:
                self.args.ckpt = './weights/minima_roma.pth'
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

        try:
            # 加载模型（支持路径和图片数据两种方式）
            self.matcher_from_paths = load_model(
                self.method, 
                self.args, 
                use_path=True, 
                test_orginal_megadepth=False
            )
            self.matcher_from_cv_imgs = load_model(
                self.method, 
                self.args, 
                use_path=False, 
                test_orginal_megadepth=False
            )
            self.matcher = self.matcher_from_paths  # 默认使用路径方式
            print(f'[INFO][Detector::loadModel]')
            print(f'\t Successfully loaded {self.method} model from: {model_file_path}')
            return True
        except Exception as e:
            print('[ERROR][Detector::loadModel]')
            print(f'\t Failed to load model: {e}')
            return False

    def detect(
        self,
        image1: Union[str, np.ndarray],
        image2: Union[str, np.ndarray],
        K0: Optional[np.ndarray] = None,
        K1: Optional[np.ndarray] = None,
        dist0: Optional[np.ndarray] = None,
        dist1: Optional[np.ndarray] = None,
    ) -> Union[Dict[str, Any], None]:
        """
        检测两张图片的匹配点

        Args:
            image1: 第一张图片（文件路径或 numpy 数组）
            image2: 第二张图片（文件路径或 numpy 数组）
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
        # 如果模型未加载，尝试使用默认路径加载
        if self.matcher is None:
            if self.args.ckpt is not None and os.path.exists(self.args.ckpt):
                if not self.loadModel(self.args.ckpt):
                    return None
            else:
                print('[ERROR][Detector::detect]')
                print('\t Model not loaded! Please call loadModel() first or provide model_file_path in __init__.')
                return None

        try:
            # 判断输入类型
            if isinstance(image1, str) and isinstance(image2, str):
                # 文件路径方式
                if not os.path.exists(image1):
                    print('[ERROR][Detector::detect]')
                    print('\t image1 file not exist!')
                    print('\t image1:', image1)
                    return None
                if not os.path.exists(image2):
                    print('[ERROR][Detector::detect]')
                    print('\t image2 file not exist!')
                    print('\t image2:', image2)
                    return None
                
                result = self.matcher_from_paths(
                    image1, image2, 
                    K0=K0, K1=K1, 
                    dist0=dist0, dist1=dist1
                )
            elif isinstance(image1, np.ndarray) and isinstance(image2, np.ndarray):
                # numpy 数组方式
                result = self.matcher_from_cv_imgs(
                    image1, image2,
                    K0=K0, K1=K1,
                    dist0=dist0, dist1=dist1
                )
            else:
                print('[ERROR][Detector::detect]')
                print('\t image1 and image2 must be both file paths or both numpy arrays!')
                return None
            
            return result
        except Exception as e:
            print('[ERROR][Detector::detect]')
            print(f'\t Detection failed: {e}')
            import traceback
            traceback.print_exc()
            return None

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
        return self.detect(image1_file_path, image2_file_path, K0=K0, K1=K1, dist0=dist0, dist1=dist1)
    
    def detectImageDataPair(
        self,
        image1_data: np.ndarray,
        image2_data: np.ndarray,
        K0: Optional[np.ndarray] = None,
        K1: Optional[np.ndarray] = None,
        dist0: Optional[np.ndarray] = None,
        dist1: Optional[np.ndarray] = None,
    ) -> Union[Dict[str, Any], None]:
        """
        从图片数据（numpy 数组）检测图片对
        
        Args:
            image1_data: 第一张图片数据（numpy 数组，BGR 格式）
            image2_data: 第二张图片数据（numpy 数组，BGR 格式）
            K0: 第一张图片的相机内参矩阵（可选）
            K1: 第二张图片的相机内参矩阵（可选）
            dist0: 第一张图片的畸变系数（可选）
            dist1: 第二张图片的畸变系数（可选）
        
        Returns:
            匹配结果字典，如果失败返回 None
        """
        return self.detect(image1_data, image2_data, K0=K0, K1=K1, dist0=dist0, dist1=dist1)
