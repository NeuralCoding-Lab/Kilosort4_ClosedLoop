"""Same-input comparisons against the unchanged KS4 detection equations."""
import pytest
import torch
from kilosort import spikedetect
from kilosort.detection import UniversalDetector, kernels


pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not kernels.AVAILABLE,
                                reason='CUDA and optional Triton required')


def example(steps=1603):
    torch.manual_seed(73)
    device = 'cuda'
    X = torch.randn(8, steps, device=device)
    waves = torch.randn(3, 7, device=device)
    waves /= waves.norm(dim=1, keepdim=True)
    ops = dict(nt=7, settings=dict(nt0min=3, n_templates=3),
               wTEMP=waves, Th_universal=1.5)
    iC = torch.randint(8, (4, 9), device=device)
    iC2 = torch.randint(9, (5, 9), device=device)
    weights = torch.randn(2, 4, 9, device=device)
    return X, ops, iC, iC2, weights


@pytest.mark.parametrize('graph', [False, True])
@pytest.mark.parametrize('kind', ['random', 'zero', 'tied', 'nonfinite'])
def test_exact_detection_and_replay(graph, kind):
    X, ops, iC, iC2, weights = example()
    if kind == 'zero':
        X.zero_()
    elif kind == 'tied':
        ops['wTEMP'].zero_()
        ops['wTEMP'][:, 3] = 1
        weights.fill_(1)
        X.zero_()
        X[:, 400:402] = 3
        X[:, 700:702] = -4
        X[:, :2] = 10
        X[:, -2:] = -10
    elif kind == 'nonfinite':
        X[0, 800] = float('nan')
    detector = UniversalDetector(ops, iC, iC2, weights, use_cuda_graph=graph)
    for data in (X, X * .7, torch.zeros_like(X), X):
        expected = spikedetect.template_match(data, ops, iC, iC2, weights)
        actual = detector.match(data)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    assert detector.statistics['batches'] == 4
    if graph:
        assert detector.statistics['graph_hits'] == 3


def test_workspace_rebuilds_on_shape_change():
    X, ops, iC, iC2, weights = example(1600)
    detector = UniversalDetector(ops, iC, iC2, weights)
    detector.match(X)
    X = torch.randn(8, 2400, device='cuda')
    actual = detector.match(X)
    expected = spikedetect.template_match(X, ops, iC, iC2, weights)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert detector.statistics['graph_builds'] == 2


@pytest.mark.parametrize('direction', [-1, 0, 1])
def test_strict_threshold_at_float32_neighbors(direction):
    X, ops, iC, iC2, weights = example()
    X.zero_()
    X[:, 400] = 3.
    ops['wTEMP'].zero_()
    ops['wTEMP'][:, 3] = 1.
    weights.fill_(1.)
    threshold = torch.tensor(12., dtype=torch.float32)
    if direction:
        threshold = torch.nextafter(threshold, torch.tensor(float('inf') * direction))
    ops['Th_universal'] = float(threshold)
    detector = UniversalDetector(ops, iC, iC2, weights)
    expected = spikedetect.template_match(X, ops, iC, iC2, weights)
    actual = detector.match(X)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert len(actual[0]) == (9 if direction < 0 else 0)
