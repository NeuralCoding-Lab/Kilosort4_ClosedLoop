"""Check grouping and transfer batching against frozen drift equations."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from kilosort import datashift


def reference():
    path = Path(__file__).resolve().parents[1] / 'benchmarks/ks4_detection_20261027/baseline/kilosort/datashift.py'
    spec = importlib.util.spec_from_file_location('original_drift', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('order', ['sorted', 'shuffled', 'empty', 'invalid_ids'])
def test_binning_exact(order):
    rng = np.random.default_rng(19)
    st = np.zeros((700, 6))
    st[:, 1] = rng.uniform(0, 150, len(st))
    st[:, 2] = rng.uniform(9, 150, len(st))
    st[:, 4] = rng.integers(0, 13, len(st))
    if order == 'sorted':
        st = st[np.argsort(st[:, 4], kind='stable')]
    elif order == 'empty':
        st = st[:0]
    elif order == 'invalid_ids':
        st[:6, 4] = [-1, 20, np.nan, np.inf, .5, 3.5]
    ops = dict(yc=np.array([0., 150.]), binning_depth=5., Nbatches=13,
               Th_universal=9.)
    expected = reference().bin_spikes(ops, st)
    actual = datashift.bin_spikes(ops, st)
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize('blocks', [1, 3])
def test_alignment_exact_on_cpu(blocks):
    rng = np.random.default_rng(37)
    F = rng.uniform(0, 5, (17, 65, 20))
    ysamp = np.arange(65) * 5.
    ops = dict(Nbatches=17, nblocks=blocks, drift_smoothing=[.5, .5, .5])
    expected = reference().align_block2(F, ysamp, ops, device='cpu')
    actual = datashift.align_block2(F, ysamp, ops, device='cpu')
    for left, right in zip(expected, actual):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        else:
            np.testing.assert_array_equal(left, right)
