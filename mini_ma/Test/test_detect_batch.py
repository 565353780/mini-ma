"""Tests for ``Detector.detect_batch`` 批量推理入口（无需真实权重 / GPU）。

通过给 ``Detector`` 注入 fake matcher，验证：

  * 底层支持 ``forward_batch`` 时走真 batch 路径，按 ``chunk_size`` 切片调用
    ``from_cv_imgs_batch``，且返回数量 / 顺序与输入一致；
  * 底层不支持 ``forward_batch`` 时逐对回退到 ``detect``，结果与逐对调用等价；
  * 空输入返回空列表。

运行：
    python -m unittest mini_ma.Test.test_detect_batch
或：
    python mini-ma/mini_ma/Test/test_detect_batch.py
"""

from __future__ import annotations

import os.path as _osp
import sys
import unittest

_REPO_ROOT = _osp.dirname(_osp.dirname(_osp.dirname(_osp.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import numpy as np
    from mini_ma.Module.detector import Detector
except BaseException as exc:  # pragma: no cover - 环境缺依赖时跳过
    np = None
    Detector = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class _FakeBatchWrapper:
    """暴露 ``model.forward_batch`` 与 ``from_cv_imgs_batch`` 的 fake DataIOWrapper。"""

    def __init__(self):
        self.model = self  # 让 ``getattr(wrapper, 'model')`` 指回自身即可。
        self.batch_calls = []

    # ``_supports_batch`` 通过 ``hasattr(inner, 'forward_batch')`` 探测。
    def forward_batch(self, batch):  # pragma: no cover - 仅作存在性标记
        raise NotImplementedError

    def from_cv_imgs_batch(self, pairs):
        self.batch_calls.append(len(pairs))
        # 用每对第一张图的 [0,0] 像素值作为可区分指纹，便于断言顺序。
        return [{'fingerprint': int(np.asarray(p[0]).flatten()[0])} for p in pairs]


class _FakeSinglePairBound:
    """模拟 ``load_model(use_path=False)`` 返回的 bound ``from_cv_imgs``。

    其 ``__self__`` 指向不含 ``forward_batch`` 的 wrapper，从而触发 fallback。
    """

    def __init__(self):
        class _PlainWrapper:
            model = object()  # 无 forward_batch
        self.__self__ = _PlainWrapper()

    def __call__(self, img0, img1, **kwargs):
        return {'fingerprint': int(np.asarray(img0).flatten()[0])}


def _make_detector():
    # 不加载真实模型：ckpt 路径不存在时 __init__ 不会 loadModel。
    det = Detector(method='sp_lg', model_file_path=None, device='cpu',
                   is_offload_cpu=False)
    return det


def _img(value: int):
    return np.zeros((4, 4, 3), dtype=np.uint8) + value


@unittest.skipIf(Detector is None, f'detector import failed: {_IMPORT_ERROR}')
class TestDetectBatch(unittest.TestCase):
    def test_empty_input(self):
        det = _make_detector()
        self.assertEqual(det.detect_batch([]), [])

    def test_real_batch_path_chunking(self):
        det = _make_detector()
        wrapper = _FakeBatchWrapper()

        # 注入一个 bound-method 风格对象：__self__ 指向支持 batch 的 wrapper。
        class _Bound:
            __self__ = wrapper
        det.matcher = _Bound()
        det.is_gray = True

        pairs = [(_img(i), _img(100 + i)) for i in range(10)]
        results = det.detect_batch(pairs, chunk_size=4)

        # 10 / 4 -> 3 次 batch 调用，对数 [4, 4, 2]。
        self.assertEqual(wrapper.batch_calls, [4, 4, 2])
        # 结果数量与顺序保持。
        self.assertEqual([r['fingerprint'] for r in results], list(range(10)))

    def test_fallback_to_single_detect(self):
        det = _make_detector()
        det.matcher = _FakeSinglePairBound()
        det.is_gray = True

        pairs = [(_img(i), _img(200 + i)) for i in range(5)]
        results = det.detect_batch(pairs, chunk_size=16)

        # fallback 逐对调用 detect -> 数量一致、指纹按输入顺序。
        self.assertEqual(len(results), 5)
        self.assertEqual([r['fingerprint'] for r in results], list(range(5)))


if __name__ == '__main__':
    unittest.main()
