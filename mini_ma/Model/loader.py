import os
import sys
import torch
from copy import deepcopy

# 获取项目根目录
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# 添加第三方库路径
_third_party_path = os.path.join(_project_root, 'mini_ma/Lib/third_party')
if _third_party_path not in sys.path:
    sys.path.insert(0, _third_party_path)


def load_roma(args, test_orginal_megadepth=False):
    """加载 RoMa 模型"""
    _roma_path = os.path.join(_third_party_path, 'RoMa_minima')
    if _roma_path not in sys.path:
        sys.path.append(_roma_path)
    from romatch import roma_outdoor
    from romatch import tiny_roma_v1_outdoor
    if test_orginal_megadepth:
        from mini_ma.Config.default_for_megadepth_dense import get_cfg_defaults
    else:
        from mini_ma.Config.default import get_cfg_defaults
    from mini_ma.Method.data_io_roma import DataIOWrapper, lower_config
    
    config = get_cfg_defaults(inference=True)
    config = lower_config(config)
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
    
    if args.ckpt2 == 'large':
        if args.ckpt is not None:
            pth_path = args.ckpt
            state_dict = torch.load(pth_path, map_location=device)
            matcher = roma_outdoor(device=device, weights=state_dict)
        else:
            matcher = roma_outdoor(device=device)
    else:
        matcher = tiny_roma_v1_outdoor(device=device)
    
    matcher = DataIOWrapper(matcher, config=config["test"])
    return matcher


def load_loftr(args, test_orginal_megadepth=False):
    """加载 LoFTR 模型"""
    _loftr_path = os.path.join(_third_party_path, 'LoFTR_minima/src')
    if _loftr_path not in sys.path:
        sys.path.insert(0, _loftr_path)
    from loftr import LoFTR, default_cfg
    if test_orginal_megadepth:
        from mini_ma.Config.default_for_megadepth_dense import get_cfg_defaults
    else:
        from mini_ma.Config.default import get_cfg_defaults
    from mini_ma.Method.data_io_loftr import DataIOWrapper, lower_config
    
    config = get_cfg_defaults(inference=True)
    config = lower_config(config)
    _default_cfg = deepcopy(default_cfg)
    filename = os.path.basename(args.ckpt)
    if filename != "outdoor_ds.ckpt":
        _default_cfg['coarse']['temp_bug_fix'] = True

    _default_cfg['match_coarse']['thr'] = args.thr
    matcher = LoFTR(config=_default_cfg)
    matcher.load_state_dict(torch.load(args.ckpt)['state_dict'], strict=True)
    matcher = matcher.eval()

    matcher = DataIOWrapper(matcher, config=config["test"])
    return matcher


def load_sp_lg(args, test_orginal_megadepth=False):
    """加载 SuperPoint + LightGlue 模型"""
    _lightglue_path = os.path.join(_third_party_path, 'LightGlue')
    if _lightglue_path not in sys.path:
        sys.path.insert(0, _lightglue_path)
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
    if test_orginal_megadepth:
        from mini_ma.Config.default_for_megadepth_sparse import get_cfg_defaults
    else:
        from mini_ma.Config.default import get_cfg_defaults
    from mini_ma.Method.data_io_sp_lg import DataIOWrapper, lower_config

    class Matching(torch.nn.Module):
        def __init__(self, sp_conf, lg_conf):
            super().__init__()
            device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
            self.extractor = SuperPoint(**sp_conf).eval().to(device)
            self.matcher = LightGlue(features='superpoint', **lg_conf).eval().to(device)
            n_layers = lg_conf['n_layers']
            ckpt_path = args.ckpt
            state_dict = torch.load(ckpt_path, map_location=device)
            for i in range(n_layers):
                pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            self.matcher.load_state_dict(state_dict, strict=False)

        def _extract(self, image):
            if test_orginal_megadepth:
                return self.extractor.extract(image, resize=None)
            return self.extractor.extract(image)

        def forward(self, batch):
            image0 = batch['image0']
            image1 = batch['image1']
            feats0 = self._extract(image0)
            feats1 = self._extract(image1)

            matches01 = self.matcher({'image0': feats0, 'image1': feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
            matches = matches01['matches']
            points0 = feats0['keypoints'][matches[..., 0]]
            points1 = feats1['keypoints'][matches[..., 1]]
            matching_scores0 = matches01['matching_scores0']
            matching_scores = matching_scores0[matches[..., 0]]

            return {'matching_scores': matching_scores, 'keypoints0': points0, 'keypoints1': points1}

        @staticmethod
        def _stack_features(feat_list):
            """Pad每张图变长的 keypoints/descriptors 到 batch 内同一长度后 stack。

            SuperPoint 在关键点数不一致时无法直接跨 batch ``torch.stack``，因此这里
            逐图 extract（单图 ``extractor.extract`` 已支持），再把 [1, Mi, *] 沿 dim=0
            合并成 [B, max_M, *]，并返回每张图的真实关键点数，便于后续屏蔽 pad 位。

            返回的 feats dict 的 ``keypoints``/``descriptors``/``keypoint_scores`` 均为
            batched tensor；``image_size`` 同样 stack 成 [B, 2]。
            """
            valid_counts = [int(f['keypoints'].shape[1]) for f in feat_list]
            max_count = max(valid_counts) if valid_counts else 0
            max_count = max(max_count, 1)

            keypoints = []
            descriptors = []
            keypoint_scores = []
            image_sizes = []
            for f in feat_list:
                kpts = f['keypoints'][0]            # [Mi, 2]
                desc = f['descriptors'][0]          # [Mi, D]
                if 'keypoint_scores' in f:
                    kscores = f['keypoint_scores'][0]  # [Mi]
                else:
                    kscores = kpts.new_zeros((kpts.shape[0],))

                pad_n = max_count - kpts.shape[0]
                if pad_n > 0:
                    kpts = torch.cat(
                        [kpts, kpts.new_zeros((pad_n, kpts.shape[1]))], dim=0)
                    desc = torch.cat(
                        [desc, desc.new_zeros((pad_n, desc.shape[1]))], dim=0)
                    kscores = torch.cat(
                        [kscores, kscores.new_zeros((pad_n,))], dim=0)

                keypoints.append(kpts)
                descriptors.append(desc)
                keypoint_scores.append(kscores)
                image_sizes.append(f['image_size'][0])

            stacked = {
                'keypoints': torch.stack(keypoints, dim=0),
                'descriptors': torch.stack(descriptors, dim=0),
                'keypoint_scores': torch.stack(keypoint_scores, dim=0),
                'image_size': torch.stack(image_sizes, dim=0),
            }
            return stacked, valid_counts

        def forward_batch(self, batch):
            """一次 GPU forward 处理 N 对图像，返回长度为 N 的 per-view 结果列表。

            ``batch['image0']`` / ``batch['image1']`` 为长度 N 的图像 tensor 列表
            （各自 [1, C, H, W]，允许不同分辨率）。逐图 extract 后 pad+stack 成
            batched feats，再对 LightGlue 做单次 batched 调用；最后按 batch index
            把变长 matches 拆回每张图的 {keypoints0, keypoints1, matching_scores}。
            """
            image0_list = batch['image0']
            image1_list = batch['image1']
            assert len(image0_list) == len(image1_list)
            num = len(image0_list)
            if num == 0:
                return []

            feats0_list = [self._extract(img) for img in image0_list]
            feats1_list = [self._extract(img) for img in image1_list]

            feats0, valid0 = self._stack_features(feats0_list)
            feats1, valid1 = self._stack_features(feats1_list)

            matches_list, matching_scores0 = self._run_matcher_batched(feats0, feats1)
            kpts0 = feats0['keypoints']
            kpts1 = feats1['keypoints']

            results = []
            for i in range(num):
                matches = matches_list[i]
                idx0 = matches[..., 0]
                idx1 = matches[..., 1]
                # 屏蔽落在 pad 区域的关键点对，避免引入伪匹配。
                keep = (idx0 < valid0[i]) & (idx1 < valid1[i])
                idx0 = idx0[keep]
                idx1 = idx1[keep]
                points0 = kpts0[i][idx0]
                points1 = kpts1[i][idx1]
                matching_scores = matching_scores0[i][idx0]
                results.append({
                    'matching_scores': matching_scores,
                    'keypoints0': points0,
                    'keypoints1': points1,
                })
            return results

        @staticmethod
        def _slice_feats(feats, i):
            """从 batched feats dict 取第 i 个样本，重新加上 batch 维（[1, ...]）。"""
            sliced = {}
            for k, v in feats.items():
                if isinstance(v, torch.Tensor) and v.dim() >= 1:
                    sliced[k] = v[i:i + 1]
                else:
                    sliced[k] = v
            return sliced

        def _run_matcher_batched(self, feats0, feats1):
            """对 LightGlue 做 batched 调用，返回 (matches_list, matching_scores0)。

            LightGlue 的「逐点 pruning」在 batch>1 时存在已知缺陷：``ind0`` 仍是
            [1, M]，pruning 路径里按 ``ind0[k, ...]`` 取第 k 个样本会越界
            （IndexError: index k is out of bounds for dimension 0 with size 1）。
            因此 batched 调用时临时关闭 width pruning（width_confidence=-1）绕过该
            路径，匹配结果不变（仅少了 pruning 加速），调用后还原配置。

            为兼容不同 LightGlue 版本，若 batched 调用仍抛出 IndexError（例如其他
            per-batch 索引缺陷），自动回退为逐样本（B=1）调用，保证正确性。
            """
            batch_size = feats0['keypoints'].shape[0]

            saved_width_confidence = getattr(
                self.matcher.conf, 'width_confidence', None)
            disable_pruning = (
                saved_width_confidence is not None
                and saved_width_confidence > 0
                and batch_size > 1
            )
            if disable_pruning:
                self.matcher.conf.width_confidence = -1
            try:
                matches01 = self.matcher({'image0': feats0, 'image1': feats1})
                return matches01['matches'], matches01['matching_scores0']
            except IndexError:
                # 版本兼容兜底：逐样本调用，B=1 与生产 detect 路径完全一致。
                matches_list = []
                scores_list = []
                for i in range(batch_size):
                    single = self.matcher({
                        'image0': self._slice_feats(feats0, i),
                        'image1': self._slice_feats(feats1, i),
                    })
                    matches_list.append(single['matches'][0])
                    scores_list.append(single['matching_scores0'][0])
                return matches_list, scores_list
            finally:
                if disable_pruning:
                    self.matcher.conf.width_confidence = saved_width_confidence

    sp_conf = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "max_num_keypoints": 2048,
        "detection_threshold": 0.0005,
        "remove_borders": 4,
    }
    lg_conf = {
        "name": "lightglue",
        "input_dim": 256,
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "n_layers": 9,
        "num_heads": 4,
        "flash": True,
        "mp": False,
        "depth_confidence": 0.95,
        "width_confidence": 0.99,
        "filter_threshold": 0.1,
        "weights": None,
    }
    matcher = Matching(sp_conf, lg_conf)
    config = get_cfg_defaults(inference=True)
    config = lower_config(config)
    matcher = DataIOWrapper(matcher, config=config["test"])
    return matcher


def load_xoftr(args):
    """加载 XoFTR 模型"""
    _xoftr_path = os.path.join(_third_party_path, 'XoFTR/src')
    if _xoftr_path not in sys.path:
        sys.path.insert(0, _xoftr_path)
    from xoftr import XoFTR
    from mini_ma.Config.default import get_cfg_defaults
    from mini_ma.Method.data_io import DataIOWrapper, lower_config
    
    config = get_cfg_defaults(inference=True)
    config = lower_config(config)
    config["xoftr"]["match_coarse"]["thr"] = args.match_threshold
    config["xoftr"]["fine"]["thr"] = args.fine_threshold
    ckpt = args.ckpt
    matcher = XoFTR(config=config["xoftr"])
    matcher = DataIOWrapper(matcher, config=config["test"], ckpt=ckpt)
    return matcher


def load_model(method: str, args, use_path: bool = True, test_orginal_megadepth: bool = False):
    """
    加载指定的模型
    
    Args:
        method: 模型名称 ('xoftr', 'loftr', 'roma', 'sp_lg')
        args: 包含模型配置的参数对象
        use_path: 是否使用路径方式加载（True: from_paths, False: from_cv_imgs）
        test_orginal_megadepth: 是否使用原始 MegaDepth 配置
    
    Returns:
        模型匹配器对象
    """
    if method == "xoftr":
        matcher = load_xoftr(args)
    elif method == "loftr":
        matcher = load_loftr(args, test_orginal_megadepth=test_orginal_megadepth)
    elif method == "roma":
        matcher = load_roma(args, test_orginal_megadepth=test_orginal_megadepth)
    elif method == "sp_lg":
        matcher = load_sp_lg(args, test_orginal_megadepth=test_orginal_megadepth)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if use_path:
        return matcher.from_paths
    else:
        return matcher.from_cv_imgs

