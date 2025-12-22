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

        def forward(self, batch):
            image0 = batch['image0']
            image1 = batch['image1']
            if test_orginal_megadepth:
                feats0 = self.extractor.extract(image0, resize=None)
                feats1 = self.extractor.extract(image1, resize=None)
            else:
                feats0 = self.extractor.extract(image0)
                feats1 = self.extractor.extract(image1)

            matches01 = self.matcher({'image0': feats0, 'image1': feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
            matches = matches01['matches']
            points0 = feats0['keypoints'][matches[..., 0]]
            points1 = feats1['keypoints'][matches[..., 1]]
            matching_scores0 = matches01['matching_scores0']
            matching_scores = matching_scores0[matches[..., 0]]

            return {'matching_scores': matching_scores, 'keypoints0': points0, 'keypoints1': points1}

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

