import cv2
import torch
import numpy as np
from typing import Union


def toNumpy(
    data: Union[torch.Tensor, np.ndarray, list],
    dtype=np.float64,
) -> np.ndarray:
    if isinstance(data, list):
        data = np.asarray(data)
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    data = data.astype(dtype)
    return data

def toTensor(
    data: Union[torch.Tensor, np.ndarray, list],
    dtype=torch.float32,
    device: str = 'cpu',
) -> torch.Tensor:
    if isinstance(data, list):
        data = np.asarray(data)
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data.copy())
    data = data.to(device, dtype=dtype)
    return data

def toGPU(data_dict: dict, device: str = 'cuda:0') -> dict:
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            data_dict[key] = data_dict[key].to(device=device)
    return data_dict

def toGrayImage(image: np.ndarray) -> np.ndarray:
    """
    将输入图片转换为灰度图。
    支持 numpy ndarray 格式。
    如果输入已是灰度图，则直接返回。
    如果输入是BGR/RGB三通道或四通道，使用OpenCV自动检测转换。
    """
    if image.ndim == 2:
        # 已经是单通道灰度
        return image
    elif image.ndim == 3:
        if image.shape[2] == 1:
            # 已经是单通道灰度 HxWx1
            return image[..., 0]
        elif image.shape[2] == 3 or image.shape[2] == 4:
            # OpenCV默认按BGR顺序
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"不支持的图片通道数: {image.shape}")
    else:
        raise ValueError(f"不支持的图片形状: {image.shape}")

def toRGBImage(image: np.ndarray) -> np.ndarray:
    """
    将输入图片转换为RGB格式的三通道图片。
    支持 numpy ndarray 格式。
    如果图片已是三通道，则按BGR默认处理为BGR->RGB。
    如果是单通道灰度，则重复到三通道。
    如果是四通道，则去除Alpha后从BGR转RGB。
    """
    if image.ndim == 2:
        # 单通道灰度图，重复到三通道
        return np.stack([image]*3, axis=-1)
    elif image.ndim == 3:
        if image.shape[2] == 1:
            # HxWx1 -> HxWx3
            return np.concatenate([image]*3, axis=-1)
        elif image.shape[2] == 3:
            # 默认BGR->RGB
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            # BGRA->RGB
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            raise ValueError(f"不支持的图片通道数: {image.shape}")
    else:
        raise ValueError(f"不支持的图片形状: {image.shape}")
