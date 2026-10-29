import importlib.util
from pathlib import Path

import numpy as np
import torch

from kilosort import CCG, template_matching


def test_grouping_preserves_order_gaps_and_input():
    labels = np.array([3, 0, 3, 0, 3])
    times = np.array([9., 4., 1., 8., 2.])
    groups = CCG.group_spike_times(labels, times)
    assert [x.tolist() for x in groups] == [[4., 8.], [], [], [9., 1., 2.]]
    groups[0][0] = 99
    assert times.tolist() == [9., 4., 1., 8., 2.]
    assert CCG.group_spike_times([], []) == []


def test_refract_matches_original_full_array_scans():
    rng = np.random.default_rng(711)
    labels = np.repeat([0, 2, 3], [200, 6, 150])
    times = rng.uniform(0, 30, len(labels))
    expected_ref = np.zeros(4)
    expected_rate = np.zeros(4)
    for unit in range(4):
        train = times[labels == unit]
        if len(train) > 10 and train.max() != train.min():
            expected_ref[unit], _, expected_rate[unit] = CCG.check_CCG(train)
    actual = CCG.refract(labels, times)
    np.testing.assert_array_equal(actual[0], expected_ref)
    np.testing.assert_array_equal(actual[1], expected_rate)


def test_merge_cache_tracks_shifted_absorbed_spikes(monkeypatch):
    # The immutable pre-change snapshot provides an independent control flow.
    source = Path(__file__).resolve().parents[1] / 'benchmarks/ks4_speed_20261027/baseline/kilosort/template_matching.py'
    spec = importlib.util.spec_from_file_location('original_template_matching', source)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    torch.manual_seed(51)
    W = torch.eye(7)
    base = torch.tensor([-.001, -.1, -.6, -1., -.6, -.1, -.001])[None, None]
    base = base * torch.tensor([1., 2., 3.])[None, :, None]
    Wall = base.repeat(4, 1, 1)
    Wall[1] = torch.roll(Wall[1] @ W, 1, -1) @ W.T
    labels = np.tile(np.arange(4), 15).astype('int32')
    st = np.column_stack((np.arange(60)*100+50, labels, np.ones(60))).astype('float64')
    features = torch.randn(60, 3, 7)
    ops = dict(nt=7, wPCA=W, fs=30000, settings=dict(acg_threshold=.2, ccg_threshold=.25))
    monkeypatch.setattr(CCG, 'refract', lambda *a, **kw: (np.ones(4), np.zeros(4)))
    calls = []
    def ccg(left, right, **kwargs):
        calls.append((np.sort(left), np.sort(right)))
        return True, True, 0.
    monkeypatch.setattr(CCG, 'check_CCG', ccg)
    expected = original.merging_function(ops, Wall.clone(), labels.copy(), st.copy(), features.clone(),
                                         r_thresh=.2, device=torch.device('cpu'))
    expected_calls = calls.copy()
    calls.clear()
    actual = template_matching.merging_function(ops, Wall.clone(), labels.copy(), st.copy(), features.clone(),
                                                r_thresh=.2, device=torch.device('cpu'))
    assert np.any(actual[3][:, 0] != st[:, 0]), 'Fixture must exercise a nonzero lag shift'
    assert len(calls) == len(expected_calls) > 1
    for pair, control in zip(calls, expected_calls):
        for value, reference in zip(pair, control):
            np.testing.assert_array_equal(value, reference)
    for value, reference in zip(actual, expected):
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        else:
            np.testing.assert_array_equal(value, reference)
