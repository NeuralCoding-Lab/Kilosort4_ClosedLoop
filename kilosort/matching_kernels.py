"""Optional fused GPU operations for KS4 matching pursuit.

The matching equations live in :mod:`kilosort.matching`; these kernels
only fuse memory operations. Triton is optional so the reference path also runs
on CPU. Adapted from ChronoSort's sparse/static/graph matcher.
"""
from __future__ import annotations

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


AVAILABLE = triton is not None


if AVAILABLE:
    @triton.jit
    def _best_score(correlation, norms, scores, identities, dirty,
                    steps: tl.constexpr, templates: tl.constexpr,
                    edge: tl.constexpr, local: tl.constexpr,
                    block_t: tl.constexpr, block_u: tl.constexpr):
        ticks = tl.program_id(0) * block_t + tl.arange(0, block_t)
        units = tl.arange(0, block_u)
        valid_t = ticks < steps
        if local:
            changed = tl.load(dirty + ticks, valid_t, other=0)
            if tl.sum(changed.to(tl.int32), 0) == 0:
                return
        value = tl.load(correlation + units[:, None] * steps + ticks[None, :],
                        (units[:, None] < templates) & valid_t[None, :], other=0.)
        norm = tl.load(norms + units, units < templates, other=1.)
        positive = tl.maximum(value, 0.)
        score = tl.div_rn(positive * positive, norm[:, None])
        score = tl.where((ticks[None, :] >= edge) &
                         (ticks[None, :] < steps - edge), score, 0.)
        score = tl.where(units[:, None] < templates, score, float('-inf'))
        best = tl.max(score, 0)
        winner = tl.min(tl.where(score == best[None, :], units[:, None],
                                 2147483647), 0)
        if local:
            valid_t = valid_t & changed
        tl.store(scores + ticks, best, valid_t)
        tl.store(identities + ticks, winner, valid_t)

    @triton.jit
    def _dirty_times(times, dirty, steps: tl.constexpr,
                     radius: tl.constexpr, width: tl.constexpr):
        center = tl.load(times + tl.program_id(0))
        if center < 0:
            return
        offset = tl.arange(0, width)
        tick = center + offset - radius
        tl.store(dirty + tick, True,
                 (offset < 2 * radius + 1) & (tick >= 0) & (tick < steps))

    @triton.jit
    def _subtract_correlations(correlation, times, ids, amplitudes,
                              neighbors, counts, coefficients,
                              steps: tl.constexpr, slots: tl.constexpr,
                              lags: tl.constexpr, events: tl.constexpr,
                              parity: tl.constexpr, block_s: tl.constexpr,
                              block_l: tl.constexpr):
        event = 2 * tl.program_id(0) + parity
        if event >= events:
            return
        center = tl.load(times + event)
        if center < 0:
            return
        unit = tl.load(ids + event)
        amplitude = tl.load(amplitudes + event)
        slot = tl.program_id(1) * block_s + tl.arange(0, block_s)
        lag = tl.arange(0, block_l)
        count = tl.load(counts + unit)
        valid_slot = (slot < slots) & (slot < count)
        target = tl.load(neighbors + unit * slots + slot, valid_slot, other=0)
        tick = center + lag - lags // 2
        valid = valid_slot[:, None] & (lag[None, :] < lags) & \
                (tick[None, :] >= 0) & (tick[None, :] < steps)
        weight = tl.load(coefficients + (unit * slots + slot[:, None]) * lags
                         + lag[None, :], valid, other=0.)
        address = correlation + target[:, None] * steps + tick[None, :]
        old = tl.load(address, valid, other=0.)
        tl.store(address, old - amplitude * weight, valid)

    @triton.jit
    def _subtract_signal(residual, waveforms, times, ids, amplitudes,
                         steps: tl.constexpr, channels: tl.constexpr,
                         units: tl.constexpr, length: tl.constexpr,
                         events: tl.constexpr, parity: tl.constexpr,
                         block_c: tl.constexpr, block_t: tl.constexpr):
        event = 2 * tl.program_id(0) + parity
        if event >= events:
            return
        center = tl.load(times + event)
        if center < 0:
            return
        unit = tl.load(ids + event)
        amplitude = tl.load(amplitudes + event)
        channel = tl.program_id(1) * block_c + tl.arange(0, block_c)
        offset = tl.arange(0, block_t)
        tick = center + offset - length // 2
        valid = (channel[:, None] < channels) & (offset[None, :] < length) & \
                (tick[None, :] >= 0) & (tick[None, :] < steps)
        wave = tl.load(waveforms + (channel[:, None] * units + unit) * length
                       + offset[None, :], valid, other=0.)
        address = residual + channel[:, None] * steps + tick[None, :]
        old = tl.load(address, valid, other=0.)
        tl.store(address, old - amplitude * wave, valid)


def score_max(correlation, norms, scores, identities, edge, dirty=None):
    units, steps = correlation.shape
    _best_score[(triton.cdiv(steps, 32),)](
        correlation, norms, scores, identities,
        scores if dirty is None else dirty, steps, units, edge,
        dirty is not None, 32, triton.next_power_of_2(units), num_warps=8)


def mark_dirty(times, dirty, radius):
    dirty.zero_()
    _dirty_times[(times.numel(),)](times, dirty, dirty.numel(), radius,
                                  triton.next_power_of_2(2 * radius + 1))


def subtract_matches(correlation, residual, waveforms, times, ids,
                     amplitudes, packed):
    neighbors, counts, coefficients = packed
    slots, lags = coefficients.shape[1:]
    channels, units, length = waveforms.shape
    for parity in range(2):
        events = times.numel()
        if events <= parity:
            continue
        grid = (triton.cdiv(events - parity, 2), triton.cdiv(slots, 8))
        _subtract_correlations[grid](
            correlation, times, ids, amplitudes, neighbors, counts, coefficients,
            correlation.shape[1], slots, lags, events, parity, 8,
            triton.next_power_of_2(lags), num_warps=4, enable_fp_fusion=False)
        _subtract_signal[(triton.cdiv(events - parity, 2), triton.cdiv(channels, 32))](
            residual, waveforms, times, ids, amplitudes, residual.shape[1],
            channels, units, length, events, parity, 32,
            triton.next_power_of_2(length), num_warps=4, enable_fp_fusion=False)
