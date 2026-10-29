"""KS4 matching pursuit with exact sparse lag updates and reusable CUDA graphs.

Adapted from the local ChronoSort low-dimensional matcher. KS4 projection,
normalization, lag orientation and public result shapes are preserved.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import weakref
import torch
from torch.nn import functional as F
from kilosort import matching_kernels as kernels


@dataclass
class Matches:
    sample_indices: torch.Tensor
    template_indices: torch.Tensor
    amplitudes: torch.Tensor
    scores: torch.Tensor
    residual: torch.Tensor
    peels: int


class TemplateMatcher:
    def __init__(self, ops, U, ctc, *, capacity=1024, group_size=4,
                 local_after=8, use_cuda_graph=True):
        self.device = U.device
        self.coefficients = U
        self.basis = ops['wPCA'].contiguous()
        self.length = ops['nt']
        self.norms = (U**2).sum(-1).sum(-1).contiguous()
        self.threshold = float(ops['Th_learned'])
        self.max_peels = int(ops['max_peels'])
        self.capacity = int(capacity)
        self.group_size = int(group_size)
        self.local_after = int(local_after)
        self.waveforms = torch.einsum('ijk,jl->kil', U, self.basis).contiguous()
        # Keep all and only exactly nonzero pairs; no approximation cutoff.
        active = torch.any(ctc != 0, dim=-1).T.cpu()
        sizes = active.sum(1)
        slots = max(1, int(sizes.max()))
        units = len(U)
        neighbors = torch.zeros((units, slots), dtype=torch.int32, device=U.device)
        packed = ctc.new_zeros((units, slots, ctc.shape[-1]))
        for source in range(units):
            target = torch.nonzero(active[source], as_tuple=True)[0].to(U.device)
            neighbors[source, :len(target)] = target.to(torch.int32)
            packed[source, :len(target)] = ctc[target, source]
        self.packed = (neighbors, sizes.to(device=U.device, dtype=torch.int32), packed)
        self.fused = U.is_cuda and kernels.AVAILABLE and hasattr(torch, 'nonzero_static')
        self.use_cuda_graph = bool(use_cuda_graph and self.fused and
            self.max_peels % self.group_size == 0 and self.local_after % self.group_size == 0)
        self._runners = {}
        self.ops = ops
        self.U = U
        self.statistics = dict(graph_builds=0, graph_hits=0, capacity_retries=0,
            tied_peak_fallbacks=0, graph_build_seconds=0., batches=0,
            backend='triton', sparse_pairs=int(sizes.sum()), dense_pairs=units**2)

    def project(self, trace):
        features = F.conv1d(trace.unsqueeze(1), self.basis.unsqueeze(1), padding=self.length // 2)
        return torch.einsum('ijk,kjl->il', self.coefficients, features).contiguous()

    def close(self):
        self._runners.clear()

    def _reference(self, trace):
        from kilosort.template_matching import run_matching
        # Materialize the exact dense tensor only for the rare tied-peak fallback.
        neighbors, sizes, packed = self.packed
        units = len(self.norms)
        ctc = trace.new_zeros((units, units, 2 * self.length + 1))
        for source in range(units):
            count = int(sizes[source])
            ctc[neighbors[source, :count].long(), source] = packed[source, :count]
        st, amplitudes, scores, residual = run_matching(self.ops, trace, self.U, ctc, device=self.device)
        return Matches(st[:, 0], st[:, 1], amplitudes[:, 0], scores[:, 0], residual, 0)

    def match(self, trace):
        trace = trace.contiguous()
        self.statistics['batches'] += 1
        while True:
            key = (tuple(trace.shape), self.capacity)
            runner = self._runners.get(key)
            if runner is None:
                start = time.perf_counter()
                runner = _StaticPursuit(self, trace)
                self._runners[key] = runner
                if self.use_cuda_graph:
                    self.statistics['graph_builds'] += 1
                    self.statistics['graph_build_seconds'] += time.perf_counter() - start
            elif self.use_cuda_graph:
                self.statistics['graph_hits'] += 1
            result, required, tied = runner.run(trace)
            if required > self.capacity:
                self.statistics['capacity_retries'] += 1
                self.capacity = max(2 * self.capacity, required)
                del result, runner
                self._runners.clear()
                continue
            if tied:
                self.statistics['tied_peak_fallbacks'] += 1
                return self._reference(trace)
            return result


class _StaticPursuit:
    """Four fixed-size peels between host checks; capture and replay when enabled."""
    def __init__(self, matcher, example):
        # The owning matcher already holds this workspace in its cache. A weak
        # reverse reference allows immediate GPU release without cyclic GC.
        self.matcher = weakref.proxy(matcher)
        self.input = torch.empty_like(example)
        self.residual = torch.empty_like(example)
        self.correlation = example.new_empty((len(matcher.norms), example.shape[1]))
        self.maximum = example.new_empty(example.shape[1])
        self.winners = torch.empty(example.shape[1], dtype=torch.long, device=example.device)
        self.dirty = torch.empty(example.shape[1], dtype=torch.bool, device=example.device)
        self.previous = torch.empty(matcher.capacity, dtype=torch.long, device=example.device)
        shape = (matcher.group_size, matcher.capacity)
        self.times = torch.empty(shape, dtype=torch.long, device=example.device)
        self.ids = torch.empty_like(self.times)
        self.amps = example.new_empty(shape)
        self.scores = example.new_empty(shape)
        self.counts = torch.empty(matcher.group_size, dtype=torch.int32, device=example.device)
        self.tied = torch.empty(matcher.group_size, dtype=torch.bool, device=example.device)
        self.all_times = torch.empty((matcher.max_peels, matcher.capacity), dtype=torch.long, device=example.device)
        self.all_ids = torch.empty_like(self.all_times)
        self.all_amps = example.new_empty(self.all_times.shape)
        self.all_scores = example.new_empty(self.all_times.shape)
        self.all_counts = torch.empty(matcher.max_peels, dtype=torch.int32, device=example.device)
        self.all_tied = torch.empty(matcher.max_peels, dtype=torch.bool, device=example.device)
        self.graphs = {}
        if matcher.use_cuda_graph:
            self.input.copy_(example)
            # Warm all shape-dependent cuDNN/cuBLAS/Triton operations on a side
            # stream before capture, including both full and dirty score paths.
            stream = torch.cuda.Stream(device=example.device)
            stream.wait_stream(torch.cuda.current_stream(example.device))
            with torch.cuda.stream(stream):
                self._initialize()
                self._group(False)
                self._group(True)
            torch.cuda.current_stream(example.device).wait_stream(stream)
            torch.cuda.synchronize(example.device)
            for name, operation in (('initialize', self._initialize),
                                    ('full', lambda: self._group(False)),
                                    ('local', lambda: self._group(True))):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    operation()
                self.graphs[name] = graph
            torch.cuda.synchronize(example.device)

    def _initialize(self):
        self.correlation.copy_(self.matcher.project(self.input))
        self.residual.copy_(self.input)
        self.previous.fill_(-1)

    def _peel(self, slot, previous, local):
        matcher = self.matcher
        if local:
            kernels.mark_dirty(previous, self.dirty, matcher.length)
        kernels.score_max(self.correlation, matcher.norms, self.maximum,
                           self.winners, matcher.length, self.dirty if local else None)
        pooled = F.max_pool1d(self.maximum[None, None], 2 * matcher.length + 1,
                               stride=1, padding=matcher.length)[0, 0]
        candidate = (pooled > matcher.threshold ** 2) & ((pooled - self.maximum).abs() < 1e-9)
        self.counts[slot] = candidate.sum(dtype=torch.int32)
        times = torch.nonzero_static(candidate, size=matcher.capacity, fill_value=-1)[:, 0]
        self.tied[slot] = ((times[1:] >= 0) &
                           (times[1:] - times[:-1] <= matcher.length)).any()
        safe = times.clamp_min(0)
        ids = self.winners[safe]
        amp = self.correlation[ids, safe] / matcher.norms[ids]
        self.times[slot].copy_(times)
        self.ids[slot].copy_(ids)
        self.amps[slot].copy_(amp)
        self.scores[slot].copy_(pooled[safe].sqrt())
        kernels.subtract_matches(self.correlation, self.residual, matcher.waveforms,
                                  times, ids, amp, matcher.packed)
        return times

    def _group(self, local):
        previous = self.previous
        for slot in range(self.matcher.group_size):
            previous = self._peel(slot, previous, local)
        self.previous.copy_(previous)

    def run(self, trace):
        matcher = self.matcher
        self.input.copy_(trace)
        if self.graphs:
            self.graphs['initialize'].replay()
        else:
            self._initialize()
        used = 0
        for first in range(0, matcher.max_peels, matcher.group_size):
            size = min(matcher.group_size, matcher.max_peels - first)
            local = first >= matcher.local_after
            if self.graphs:
                self.graphs['local' if local else 'full'].replay()
            else:
                previous = self.previous
                for slot in range(size):
                    previous = self._peel(slot, previous, first + slot >= matcher.local_after)
                self.previous.copy_(previous)
            end = first + size
            self.all_times[first:end].copy_(self.times[:size])
            self.all_ids[first:end].copy_(self.ids[:size])
            self.all_amps[first:end].copy_(self.amps[:size])
            self.all_scores[first:end].copy_(self.scores[:size])
            self.all_counts[first:end].copy_(self.counts[:size])
            self.all_tied[first:end].copy_(self.tied[:size])
            used = end
            if int(self.counts[size - 1].item()) == 0:
                break
        required = int(self.all_counts[:used].max().item())
        times = self.all_times[:used].reshape(-1)
        valid = times >= 0
        result = Matches(times[valid], self.all_ids[:used].reshape(-1)[valid],
                               self.all_amps[:used].reshape(-1)[valid],
                               self.all_scores[:used].reshape(-1)[valid],
                               self.residual, used)
        return result, required, bool(self.all_tied[:used].any().item())
