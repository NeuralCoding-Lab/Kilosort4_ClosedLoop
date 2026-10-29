"""Alternate old/new detectors on each identical preprocessed 30-minute batch."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kilosort import io, datashift, spikedetect
from kilosort.detection import UniversalDetector


def timed(operation):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = operation()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--baseline', type=Path, help='Frozen before source directory')
    args = parser.parse_args()
    root = args.root
    torch.set_num_threads(16)
    spec = importlib.util.spec_from_file_location('frozen_datashift', (args.baseline or root / 'baseline') / 'kilosort/datashift.py')
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    reports = []
    for call in range(2):
        checkpoint = torch.load(root / f'before/detection_input_{call}.pt', weights_only=False)
        ops, iC, iC2, weigh = (checkpoint[k] for k in ('ops', 'iC', 'iC2', 'weigh'))
        ops.setdefault('dshift', None)
        bfile = io.bfile_from_ops(ops, device=torch.device('cuda'))
        detector = UniversalDetector(ops, iC, iC2, weigh)
        rows = []
        total = 0
        started = time.perf_counter()
        report = dict(call=call, exact_xy=True, exact_waveform_ids=True,
                      exact_amplitudes=True, exact_adist=True, exact_depths=True,
                      exact_pc_features=True, feature_max_abs=0.)
        collected = []
        wave = torch.arange(-(ops['nt']//2), ops['nt']//2+1, device='cuda')
        for batch in range(min(args.limit or bfile.n_batches, bfile.n_batches)):
            X = bfile.padded_batch_to_torch(batch, ops)
            outputs, durations = {}, {}
            for name in (('before', 'after') if batch % 2 == 0 else ('after', 'before')):
                outputs[name], durations[name] = timed(
                    lambda: spikedetect.template_match(X, ops, iC, iC2, weigh)
                    if name == 'before' else detector.match(X))
            left, right = outputs['before'], outputs['after']
            for index, key in enumerate(('exact_xy', 'exact_waveform_ids', 'exact_amplitudes', 'exact_adist')):
                equal = torch.equal(left[index], right[index])
                report[key] &= equal
                assert equal, (call, batch, key)
            depths = [spikedetect.yweighted(ops['yc'], iC, o[3], o[0]) for o in (left, right)]
            report['exact_depths'] &= torch.equal(*depths)
            assert torch.equal(*depths), (call, batch, 'depths')
            features = [X[iC[:, o[0][:, :1]], o[0][:, 1:2] + wave] @ ops['wPCA'].T for o in (left, right)]
            maximum = float((features[0] - features[1]).abs().max()) if features[0].numel() else 0.
            report['feature_max_abs'] = max(report['feature_max_abs'], maximum)
            report['exact_pc_features'] &= torch.equal(*features)
            assert torch.equal(*features), (call, batch, 'features', maximum)
            xy, imax, amp, _ = left
            block = np.zeros((len(xy), 6))
            block[:, 0] = (xy[:, 1].cpu().numpy() - ops['nt']) / ops['fs'] + batch * ops['batch_size'] / ops['fs']
            block[:, 1] = depths[0].cpu().numpy()
            block[:, 2] = amp.cpu().numpy()
            block[:, 3] = imax.cpu().numpy()
            block[:, 4] = batch
            block[:, 5] = xy[:, 0].cpu().numpy()
            collected.append(block)
            total += len(xy)
            rows.append(dict(batch=batch, spikes=len(xy), **durations))
            if batch % 50 == 0 or batch == bfile.n_batches-1:
                print(f'Pass {call}: {batch+1}/{bfile.n_batches} batches; {total:,} exact spikes', flush=True)
                (root / f'detection_replay_batches_{call}.json').write_text(json.dumps(rows, indent=2))
        st = np.concatenate(collected)
        (old_F, old_y), old_bin = timed(lambda: frozen.bin_spikes(ops, st))
        (new_F, new_y), new_bin = timed(lambda: datashift.bin_spikes(ops, st))
        assert np.array_equal(old_F, new_F) and np.array_equal(old_y, new_y)
        old_align, old_time = timed(lambda: frozen.align_block2(old_F, old_y, ops))
        new_align, new_time = timed(lambda: datashift.align_block2(new_F, new_y, ops))
        align_exact = []
        for old, new in zip(old_align, new_align):
            same = torch.equal(old, new) if isinstance(old, torch.Tensor) else np.array_equal(old, new)
            align_exact.append(same)
            assert same, ('drift alignment differs', call)
        report.update(batches=len(rows), spikes=total,
                      before_seconds=sum(r['before'] for r in rows),
                      after_seconds=sum(r['after'] for r in rows),
                      exact_histogram=True, exact_alignment=align_exact,
                      bin_before_s=old_bin, bin_after_s=new_bin,
                      align_before_s=old_time, align_after_s=new_time,
                      wall_seconds=time.perf_counter()-started,
                      statistics=detector.statistics.copy())
        reports.append(report)
        (root / ('detection_smoke.json' if args.limit else 'detection_validation.json')).write_text(json.dumps(reports, indent=2))
        del detector, checkpoint, bfile, features
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == '__main__':
    main()
