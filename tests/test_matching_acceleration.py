"""Compare accelerated pursuit to the original KS4 equations, including replay."""
import pytest
import torch

from kilosort import template_matching as tm
from kilosort.matching import TemplateMatcher, kernels


def example(device='cpu', *, max_peels=20):
    torch.manual_seed(23)
    ticks = torch.arange(61, device=device, dtype=torch.float32)
    W = -torch.exp(-((ticks - 20) / 3).square())[None]
    W /= W.norm()
    U = torch.zeros((12, 1, 16), device=device)
    for unit in range(12):
        U[unit, 0, unit] = 12 + unit
    ops = dict(nt=61, wPCA=W, max_peels=max_peels, Th_learned=8.)
    trace = torch.zeros((16, 1400), device=device)
    waveforms = torch.einsum('ijk,jl->kil', U, W)
    for center in (200, 600, 1000):
        trace[:, center-30:center+31] += waveforms.sum(1)
    return ops, U, trace


def test_projection_and_exact_sparse_lags():
    ops, U, trace = example()
    # Tiny but nonzero overlap must survive packing.
    U[1, 0, 0] = 1e-10
    ctc = tm.prepare_matching(ops, U)
    matcher = TemplateMatcher(ops, U, ctc)
    neighbors, counts, weights = matcher.packed
    recovered = torch.zeros_like(ctc)
    for source in range(len(U)):
        count = int(counts[source])
        recovered[neighbors[source, :count].long(), source] = weights[source, :count]
    torch.testing.assert_close(recovered, ctc, atol=0, rtol=0)
    expected = torch.nn.functional.conv1d(trace[None], matcher.waveforms.permute(1, 0, 2), padding=30)[0]
    torch.testing.assert_close(matcher.project(trace), expected)
    assert matcher.statistics['sparse_pairs'] < len(U)**2


@pytest.mark.skipif(not torch.cuda.is_available() or not kernels.AVAILABLE,
                    reason='CUDA and optional Triton required')
@pytest.mark.parametrize('graph,max_peels', [(True, 20), (False, 20), (True, 5)])
def test_collision_recovery_capacity_and_graph_replay(graph, max_peels):
    ops, U, trace = example('cuda', max_peels=max_peels)
    ctc = tm.prepare_matching(ops, U)
    expected = tm.run_matching(ops, trace, U, ctc)
    matcher = TemplateMatcher(ops, U, ctc, capacity=2, use_cuda_graph=graph)
    actual = matcher.match(trace)
    assert matcher.statistics['capacity_retries'] == 1
    assert torch.equal(actual.sample_indices, expected[0][:, 0])
    assert torch.equal(actual.template_indices, expected[0][:, 1])
    torch.testing.assert_close(actual.amplitudes, expected[1][:, 0], rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual.scores, expected[2][:, 0], rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual.residual, expected[3], rtol=2e-5, atol=2e-5)
    again = matcher.match(trace)
    assert torch.equal(again.sample_indices, expected[0][:, 0])
    if graph and max_peels % 4 == 0:
        assert matcher.statistics['graph_hits'] == 1
    empty = matcher.match(torch.zeros_like(trace))
    assert empty.sample_indices.numel() == 0
    matcher.close()
    assert not matcher._runners


@pytest.mark.skipif(not torch.cuda.is_available() or not kernels.AVAILABLE,
                    reason='CUDA and optional Triton required')
def test_equal_neighboring_peaks_use_original_fallback():
    ops, U, trace = example('cuda')
    ops['wPCA'].zero_()
    ops['wPCA'][0, 30] = 1
    trace.zero_()
    trace[0, 200:202] = 24
    ctc = tm.prepare_matching(ops, U)
    expected = tm.run_matching(ops, trace, U, ctc)
    matcher = TemplateMatcher(ops, U, ctc)
    actual = matcher.match(trace)
    assert matcher.statistics['tied_peak_fallbacks'] == 1
    assert torch.equal(actual.sample_indices, expected[0][:, 0])
    torch.testing.assert_close(actual.residual, expected[3])
    matcher.close()
