"""CUDA timeline of the unchanged detector and fused/graph implementation."""
import argparse
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kilosort import io, spikedetect
from kilosort.detection import UniversalDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(16)
    saved = torch.load(args.root / 'before/detection_input_1.pt', weights_only=False)
    ops, iC, iC2, weigh = (saved[k] for k in ('ops', 'iC', 'iC2', 'weigh'))
    bfile = io.bfile_from_ops(ops, device=torch.device('cuda'))
    X = bfile.padded_batch_to_torch(450, ops)
    detector = UniversalDetector(ops, iC, iC2, weigh)
    spikedetect.template_match(X, ops, iC, iC2, weigh)
    detector.match(X)
    torch.cuda.synchronize()
    for name in ('before', 'after'):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
            for _ in range(3):
                if name == 'before':
                    spikedetect.template_match(X, ops, iC, iC2, weigh)
                else:
                    detector.match(X)
                torch.cuda.synchronize()
        profile.export_chrome_trace(str(args.root / f'{name}_detection_trace.json'))
        (args.root / f'{name}_gpu_profile.txt').write_text(
            profile.key_averages().table(sort_by='self_cuda_time_total', row_limit=35), encoding='utf-8')
        print(name, profile.key_averages().table(sort_by='self_cuda_time_total', row_limit=12), flush=True)


if __name__ == '__main__':
    main()
