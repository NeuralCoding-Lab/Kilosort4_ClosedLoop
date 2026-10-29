"""Exact max/argmax memory kernels for universal template detection.

Convolution and spatial dot products remain the original PyTorch operations.
These kernels avoid abs/neighbor-gather buffers and read the original layout.
Triton is optional; spikedetect retains its original implementation.
"""
try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

AVAILABLE = triton is not None


if AVAILABLE:
    @triton.jit
    def _score(A, scores, winners, count: tl.constexpr, steps: tl.constexpr,
               chunk: tl.constexpr, offset, sizes: tl.constexpr,
               waves: tl.constexpr, s0: tl.constexpr, s1: tl.constexpr,
               s2: tl.constexpr, s3: tl.constexpr,
               block_k: tl.constexpr, block_t: tl.constexpr):
        site = tl.program_id(0)
        time = tl.program_id(1) * block_t + tl.arange(0, block_t)
        k = tl.arange(0, block_k)
        value = tl.load(A + (k[:, None] // waves) * s0 + site * s1
                        + (k[:, None] % waves) * s2 + time[None, :] * s3,
                        (k[:, None] < sizes * waves) & (time[None, :] < chunk),
                        other=0.)
        amplitude = tl.abs(value)
        best = tl.max(amplitude, 0)
        best = tl.where(tl.sum((value != value).to(tl.int32), 0) > 0,
                        float('nan'), best)
        # torch.max selects the first flattened size/waveform on ties.
        index = tl.min(tl.where((amplitude == best[None, :]) &
                               (k[:, None] < sizes * waves), k[:, None],
                               2147483647), 0)
        chosen = tl.sum(tl.where(k[:, None] == index[None, :], value, 0.), 0)
        sign = (chosen > 0.).to(tl.int32) - (chosen < 0.).to(tl.int32)
        address = site * steps + offset + time
        tl.store(scores + address, best, time < chunk)
        tl.store(winners + address, (index + 1) * sign, time < chunk)

    @triton.jit
    def _neighbors(scores, neighbors, maximum, sites: tl.constexpr,
                   steps: tl.constexpr, nearest: tl.constexpr,
                   stride0: tl.constexpr, stride1: tl.constexpr,
                   edge: tl.constexpr, block_n: tl.constexpr,
                   block_t: tl.constexpr):
        site = tl.program_id(0)
        time = tl.program_id(1) * block_t + tl.arange(0, block_t)
        n = tl.arange(0, block_n)
        source = tl.load(neighbors + n * stride0 + site * stride1,
                         n < nearest, other=0)
        value = tl.load(scores + source[:, None] * steps + time[None, :],
                        (n[:, None] < nearest) & (time[None, :] < steps),
                        other=float('-inf'))
        best = tl.max(value, 0)
        best = tl.where(tl.sum((value != value).to(tl.int32), 0) > 0,
                        float('nan'), best)
        best = tl.where((time < edge) | (time >= steps - edge), 0., best)
        tl.store(maximum + site * steps + time, best, time < steps)


def score(A, scores, winners, offset, waves):
    sizes, sites, _, chunk = A.shape
    _score[(sites, triton.cdiv(chunk, 128))](
        A, scores, winners, sites, scores.shape[1], chunk, offset, sizes,
        waves, *A.stride(), triton.next_power_of_2(sizes * waves), 128,
        num_warps=4)


def neighbor_max(scores, neighbors, maximum, edge):
    sites, steps = scores.shape
    _neighbors[(sites, triton.cdiv(steps, 128))](
        scores, neighbors, maximum, sites, steps, neighbors.shape[0],
        *neighbors.stride(), edge, triton.next_power_of_2(neighbors.shape[0]),
        128, num_warps=8)
