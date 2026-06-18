import time

import cv2
import numpy as np
import torch
# import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
from yacs.config import CfgNode as CN


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


def upper_config(dict_cfg):
    if not isinstance(dict_cfg, dict):
        return dict_cfg
    return {k.upper(): upper_config(v) for k, v in dict_cfg.items()}


class DataIOWrapper(nn.Module):
    """
    Pre-propcess data from different sources
    """

    def __init__(self, model, config, ckpt=None):
        super().__init__()

        self.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
        # NOTE: do NOT call ``torch.set_grad_enabled(False)`` here. That
        # call is thread-local AND persistent, so constructing this
        # wrapper from a worker thread (e.g. the web pipeline's
        # ``mmrecon-worker``) would permanently disable autograd in
        # that thread and break any later trainer (e.g. ``fit_fastgs``'s
        # ``total_loss.backward()``). Gradient computation is disabled
        # locally on the inference entry points via ``@torch.no_grad()``
        # decorators below.
        self.model = model
        self.config = config
        self.img0_size = config['img0_resize']
        self.img1_size = config['img1_resize']
        self.df = config['df']
        self.padding = config['padding']
        self.coarse_scale = config['coarse_scale']

        self.model = self.model.eval().to(self.device)

    def preprocess_image(self, img, device, resize=None, df=None, padding=None, cam_K=None, dist=None, gray_scale=True):
        # xoftr takes grayscale input images
        if gray_scale and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = img.shape[:2]
        new_K = None
        img_undistorted = None
        if cam_K is not None and dist is not None:
            new_K, roi = cv2.getOptimalNewCameraMatrix(cam_K, dist, (w, h), 0, (w, h))
            img = cv2.undistort(img, cam_K, dist, None, new_K)
            img_undistorted = img.copy()

        if resize is not None:
            scale = resize / max(h, w)
            w_new, h_new = int(round(w * scale)), int(round(h * scale))
        else:
            w_new, h_new = w, h

        if df is not None:
            w_new, h_new = map(lambda x: int(x // df * df), [w_new, h_new])

        img = cv2.resize(img, (w_new, h_new))
        scale = np.array([w / w_new, h / h_new], dtype=float)

        if padding:  # padding
            pad_to = max(h_new, w_new)
            img, mask = self.pad_bottom_right(img, pad_to, ret_mask=True)
            mask = torch.from_numpy(mask).to(device)
        else:
            mask = None
        # img = transforms.functional.to_tensor(img).unsqueeze(0).to(device)
        if len(img.shape) == 2:  # grayscale image
            img = torch.from_numpy(img)[None][None].to(self.device).float() / 255.0
        else:  # Color image
            img = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255.0
        return img, scale, mask, new_K, img_undistorted

    @torch.no_grad()
    def from_cv_imgs(self, img0, img1, K0=None, K1=None, dist0=None, dist1=None):
        # print('self.padding', self.padding)
        img0_tensor, scale0, mask0, new_K0, img0_undistorted = self.preprocess_image(
            img0, self.device, resize=self.img0_size, df=self.df, padding=self.padding, cam_K=K0, dist=dist0)
        img1_tensor, scale1, mask1, new_K1, img1_undistorted = self.preprocess_image(
            img1, self.device, resize=self.img1_size, df=self.df, padding=self.padding, cam_K=K1, dist=dist1)
        mkpts0, mkpts1, mconf, match_time = self.match_images(img0_tensor, img1_tensor, mask0, mask1)

        mkpts0 = mkpts0 * scale0
        mkpts1 = mkpts1 * scale1

        matches = np.concatenate([mkpts0, mkpts1], axis=1)
        data = {'matches': matches,
                'mkpts0': mkpts0,
                'mkpts1': mkpts1,
                'mconf': mconf,
                'img0': img0,
                'img1': img1,
                'match_time': match_time
                }
        if K0 is not None and dist0 is not None:
            data.update({'new_K0': new_K0, 'img0_undistorted': img0_undistorted})
        if K1 is not None and dist1 is not None:
            data.update({'new_K1': new_K1, 'img1_undistorted': img1_undistorted})
        return data

    @torch.no_grad()
    def from_paths(self, img0_pth, img1_pth, K0=None, K1=None, dist0=None, dist1=None, read_color=False):

        imread_flag = cv2.IMREAD_COLOR if read_color else cv2.IMREAD_GRAYSCALE

        img0 = cv2.imread(img0_pth, imread_flag)
        img1 = cv2.imread(img1_pth, imread_flag)
        return self.from_cv_imgs(img0, img1, K0=K0, K1=K1, dist0=dist0, dist1=dist1)

    @torch.no_grad()
    def match_images(self, image0, image1, mask0, mask1):
        batch = {'image0': image0, 'image1': image1}
        if mask0 is not None:  # img_padding is True
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                       scale_factor=self.coarse_scale,
                                                       mode='nearest',
                                                       recompute_scale_factor=False)[0].bool()
            batch.update({'mask0': ts_mask_0.unsqueeze(0), 'mask1': ts_mask_1.unsqueeze(0)})

        torch.cuda.synchronize()
        start = time.time()
        pred = self.model(batch)
        torch.cuda.synchronize()
        match_1 = time.time()
        match_time = match_1 - start

        mkpts0 = pred['keypoints0'].cpu().numpy()
        mkpts1 = pred['keypoints1'].cpu().numpy()
        matching_scores0 = pred['matching_scores'].detach().cpu().numpy()

        mconf = matching_scores0

        return mkpts0, mkpts1, mconf, match_time

    def _preprocess_pair(self, img0, img1, K0=None, K1=None, dist0=None, dist1=None):
        """单对图像预处理原子：复用 ``preprocess_image`` 的 resize/df/gray 逻辑。

        返回 (img0_tensor, img1_tensor, scale0, scale1)；不触碰模型，便于批量
        collate 复用同一份预处理实现。sp_lg 路径默认 ``padding=False``，故不返回
        mask（与 ``match_images`` 的单对路径一致）。
        """
        img0_tensor, scale0, _mask0, _new_K0, _img0_undistorted = self.preprocess_image(
            img0, self.device, resize=self.img0_size, df=self.df,
            padding=self.padding, cam_K=K0, dist=dist0)
        img1_tensor, scale1, _mask1, _new_K1, _img1_undistorted = self.preprocess_image(
            img1, self.device, resize=self.img1_size, df=self.df,
            padding=self.padding, cam_K=K1, dist=dist1)
        return img0_tensor, img1_tensor, scale0, scale1

    @torch.no_grad()
    def collate_image_pairs(self, pairs):
        """把 ``[(img0, img1), ...]`` 逐对预处理成模型输入。

        每对图像可有不同分辨率，sp_lg 的 batched forward 逐图 extract，因此这里
        无需 pad 到统一尺寸：返回的是「逐图 tensor 列表」+「逐对 scale」。

        返回 dict:
            image0: List[[1, C, H, W]]
            image1: List[[1, C, H, W]]
            scales0: List[np.ndarray([sx, sy])]
            scales1: List[np.ndarray([sx, sy])]
        """
        image0_list = []
        image1_list = []
        scales0 = []
        scales1 = []
        for pair in pairs:
            img0, img1 = pair[0], pair[1]
            img0_tensor, img1_tensor, scale0, scale1 = self._preprocess_pair(img0, img1)
            image0_list.append(img0_tensor)
            image1_list.append(img1_tensor)
            scales0.append(scale0)
            scales1.append(scale1)
        return {
            'image0': image0_list,
            'image1': image1_list,
            'scales0': scales0,
            'scales1': scales1,
        }

    @torch.no_grad()
    def match_images_batch(self, image0_list, image1_list):
        """对底层模型做单次 batched 调用，返回 per-view 原始结果列表。

        依赖 ``self.model.forward_batch``（见 sp_lg ``Matching``），其返回长度为 N
        的 [{matching_scores, keypoints0, keypoints1}] 列表，坐标为模型输入尺度。
        """
        torch.cuda.synchronize()
        start = time.time()
        preds = self.model.forward_batch({'image0': image0_list, 'image1': image1_list})
        torch.cuda.synchronize()
        match_time = time.time() - start

        per_match_time = match_time / max(1, len(preds))
        outputs = []
        for pred in preds:
            mkpts0 = pred['keypoints0'].cpu().numpy()
            mkpts1 = pred['keypoints1'].cpu().numpy()
            mconf = pred['matching_scores'].detach().cpu().numpy()
            outputs.append((mkpts0, mkpts1, mconf, per_match_time))
        return outputs

    @staticmethod
    def _assemble_match_dict(mkpts0, mkpts1, mconf, scale0, scale1, img0, img1, match_time):
        """把单视角 batched 输出还原成与 ``from_cv_imgs`` 完全一致的结果 dict。"""
        mkpts0 = mkpts0 * scale0
        mkpts1 = mkpts1 * scale1
        matches = np.concatenate([mkpts0, mkpts1], axis=1)
        return {
            'matches': matches,
            'mkpts0': mkpts0,
            'mkpts1': mkpts1,
            'mconf': mconf,
            'img0': img0,
            'img1': img1,
            'match_time': match_time,
        }

    @torch.no_grad()
    def from_cv_imgs_batch(self, pairs):
        """批量版 ``from_cv_imgs``：一次 GPU forward 处理 N 对，返回 List[dict]。

        每个 dict 与单对 ``from_cv_imgs`` 的字段/坐标语义完全一致，因此下游
        ``CameraMatcher`` / ``MeshDeformer`` 无需任何改动即可消费。
        """
        if len(pairs) == 0:
            return []

        collated = self.collate_image_pairs(pairs)
        raw_outputs = self.match_images_batch(
            collated['image0'], collated['image1'])

        results = []
        for i, pair in enumerate(pairs):
            mkpts0, mkpts1, mconf, match_time = raw_outputs[i]
            results.append(self._assemble_match_dict(
                mkpts0=mkpts0,
                mkpts1=mkpts1,
                mconf=mconf,
                scale0=collated['scales0'][i],
                scale1=collated['scales1'][i],
                img0=pair[0],
                img1=pair[1],
                match_time=match_time,
            ))
        return results

    def pad_bottom_right(self, inp, pad_size, ret_mask=False):
        assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
        mask = None
        if inp.ndim == 2:
            padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
            padded[:inp.shape[0], :inp.shape[1]] = inp
            if ret_mask:
                mask = np.zeros((pad_size, pad_size), dtype=bool)
                mask[:inp.shape[0], :inp.shape[1]] = True
        elif inp.ndim == 3:
            padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
            padded[:, :inp.shape[1], :inp.shape[2]] = inp
            if ret_mask:
                mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
                mask[:, :inp.shape[1], :inp.shape[2]] = True
        else:
            raise NotImplementedError()
        return padded, mask
