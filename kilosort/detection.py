"""Reusable universal detection workspace, preserving KS4's 40 dot products."""
import torch
import weakref
from torch.nn import functional as F
from kilosort import detection_kernels as kernels


class UniversalDetector:
    def __init__(self, ops, iC, iC2, weigh, *, use_cuda_graph=True):
        self.ops = ops
        self.iC, self.iC2, self.weigh = iC, iC2, weigh
        self.use_cuda_graph = use_cuda_graph
        self.workspace = None
        self.statistics = dict(batches=0, graph_builds=0, graph_hits=0)

    def match(self, X):
        if self.workspace is None or self.workspace.input.shape != X.shape:
            self.workspace = _Workspace(self, X)
            self.statistics['graph_builds'] += int(self.use_cuda_graph)
        elif self.use_cuda_graph:
            self.statistics['graph_hits'] += 1
        self.statistics['batches'] += 1
        return self.workspace.run(X)


class _Workspace:
    def __init__(self, detector, X):
        self.detector = weakref.proxy(detector)
        self.input = torch.empty_like(X)
        shape = (detector.iC.shape[1], X.shape[1])
        self.scores = X.new_empty(shape)
        self.winners = torch.empty(shape, device=X.device, dtype=torch.int64)
        self.maximum = X.new_empty(shape)
        self.graph = None
        if detector.use_cuda_graph:
            self.input.copy_(X)
            stream = torch.cuda.Stream(device=X.device)
            stream.wait_stream(torch.cuda.current_stream(X.device))
            with torch.cuda.stream(stream):
                self.compute()
            torch.cuda.current_stream(X.device).wait_stream(stream)
            torch.cuda.synchronize(X.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                self.compute()
            self.graph = graph

    def compute(self):
        detector = self.detector
        ops = detector.ops
        nt = ops['nt']
        steps = self.input.shape[1]
        width = (steps - 1) // 40 + 1
        # Keep cuDNN convolution, einsum shape and channel reduction unchanged.
        self.B = F.conv1d(self.input.unsqueeze(1), ops['wTEMP'].unsqueeze(1),
                          padding=nt // 2)
        for start in range(0, steps, width):
            A = torch.einsum('ijk,jklm->iklm', detector.weigh,
                             self.B[detector.iC, :, start:start + width])
            kernels.score(A, self.scores, self.winners, start,
                          ops['settings']['n_templates'])
        kernels.neighbor_max(self.scores, detector.iC2, self.maximum, nt)
        self.pooled = F.max_pool1d(self.maximum.unsqueeze(0),
            2 * ops['settings']['nt0min'] + 1, stride=1,
            padding=ops['settings']['nt0min']).squeeze(0)

    def run(self, X):
        self.input.copy_(X)
        if self.graph is None:
            self.compute()
        else:
            self.graph.replay()
        xy = ((self.pooled == self.scores) &
              (self.scores > self.detector.ops['Th_universal'])).nonzero()
        signed = self.winners[xy[:, 0], xy[:, 1]]
        amplitude = self.scores[xy[:, 0], xy[:, 1]]
        sign = signed.sign()
        index = signed.abs() - 1
        nk = self.detector.ops['settings']['n_templates']
        adist = self.B[self.detector.iC[:, xy[:, 0]], index % nk, xy[:, 1]] * sign
        return xy, index, amplitude, adist
