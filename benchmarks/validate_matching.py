"""Validate and repeatedly time both matchers on identical real KS4 batches."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kilosort import io, template_matching as tm
from kilosort.matching import TemplateMatcher


def timed(operation):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = operation()
    torch.cuda.synchronize()
    return value, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(16)
    saved = torch.load(args.checkpoint, weights_only=False, map_location='cuda')
    ops, U = saved['ops'], saved['U']
    bfile = io.bfile_from_ops(ops, device=torch.device('cuda'))
    ctc = tm.prepare_matching(ops, U)
    matcher = TemplateMatcher(ops, U, ctc)
    rows = []
    indices = sorted(set([0, 1, bfile.n_batches//4, bfile.n_batches//2,
                          3*bfile.n_batches//4, bfile.n_batches-1]))
    print(f'VALIDATE templates={len(U)} batches={indices}', flush=True)
    for index in indices:
        X = bfile.padded_batch_to_torch(index, ops)
        expected, original_s = timed(lambda: tm.run_matching(ops, X, U, ctc))
        actual, first_s = timed(lambda: matcher.match(X))
        st, amp, score, residual = expected
        same_times = torch.equal(st[:, 0], actual.sample_indices)
        same_units = torch.equal(st[:, 1], actual.template_indices)
        row = dict(batch=int(index), spikes_original=len(st), spikes_accelerated=len(actual.sample_indices),
                   identical_times=same_times, identical_templates=same_units,
                   original_first_s=original_s, accelerated_first_s=first_s)
        if same_times and same_units:
            row.update(amplitude_max_abs=float((amp[:, 0]-actual.amplitudes).abs().max()),
                       score_max_abs=float((score[:, 0]-actual.scores).abs().max()))
            torch.testing.assert_close(actual.amplitudes, amp[:, 0], rtol=3e-5, atol=3e-5)
            torch.testing.assert_close(actual.scores, score[:, 0], rtol=3e-5, atol=3e-5)
        row['residual_max_abs'] = float((residual-actual.residual).abs().max())
        torch.testing.assert_close(actual.residual, residual, rtol=1e-4, atol=3e-4)
        assert same_times and same_units, row
        repeats = []
        for repeat in range(3):
            # Alternate measurement order to limit temperature/order effects.
            order = ['original', 'accelerated'] if repeat % 2 == 0 else ['accelerated', 'original']
            durations = {}
            for name in order:
                output, durations[name] = timed(lambda: tm.run_matching(ops, X, U, ctc)
                                                if name == 'original' else matcher.match(X))
                del output
            repeats.append(durations)
        row['repeats'] = repeats
        rows.append(row)
        print(json.dumps(row), flush=True)
        args.output.write_text(json.dumps(dict(batches=rows, stats=matcher.statistics), indent=2))
    matcher.close()


if __name__ == '__main__':
    main()
