"""Full-window spike F1 regression check: one-to-one spikes and Hungarian units."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from numba import njit
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@njit(cache=True)
def coincidence(gt_t, gt_l, pred_t, pred_l, ng, npred, tolerance):
    # Same greedy one-to-one pairwise spike kernel as ChronoSort's
    # benchmarks/first_pass/evaluate.py (not an independent best-GT match).
    counts = np.zeros((ng, npred), np.int64)
    last_pred = np.full((ng, npred), -1, np.int64)
    seen = np.full(npred, -1, np.int64)
    first = 0
    for event in range(len(gt_t)):
        lower, upper = gt_t[event]-tolerance, gt_t[event]+tolerance
        while first < len(pred_t) and pred_t[first] < lower:
            first += 1
        candidate = first
        while candidate < len(pred_t) and pred_t[candidate] <= upper:
            unit, cluster = gt_l[event], pred_l[candidate]
            if seen[cluster] != event and candidate > last_pred[unit, cluster]:
                counts[unit, cluster] += 1
                last_pred[unit, cluster] = candidate
                seen[cluster] = event
            candidate += 1
    return counts


def load(folder):
    times = np.load(folder/'spike_times.npy').reshape(-1).astype(np.int64)
    labels = np.load(folder/'spike_clusters.npy').reshape(-1)
    keep = (times >= 0) & (times < 54000000)
    excluded = int((~keep).sum())
    times, labels = times[keep], labels[keep]
    order = np.argsort(times, kind='stable')
    ids, labels, counts = np.unique(labels, return_inverse=True, return_counts=True)
    return times[order], labels[order], ids, counts, excluded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--data', type=Path, default=Path('F:/kilosort_simu/25298815/sim_no_drift_30min'))
    args = parser.parse_args()
    gt_t, gt_l, gt_ids, gt_counts, gt_excluded = load(args.data/'gt')
    summaries = {}
    fig, ax = plt.subplots(figsize=(9,5),layout='constrained')
    for name in ('before','after'):
        pt, pl, ids, counts, excluded = load(args.root/name/'sorting')
        pair = coincidence(gt_t, gt_l, pt, pl, len(gt_ids), len(ids), 12)
        assert np.all(pair <= gt_counts[:,None]) and np.all(pair <= counts[None,:])
        f1 = 2*pair/(gt_counts[:,None]+counts[None,:])
        rows, cols = linear_sum_assignment(f1, maximize=True)
        scores = np.zeros(len(ids)); scores[cols] = f1[rows,cols]
        assigned = {int(j):int(i) for i,j in zip(rows,cols)}
        order = np.lexsort((ids,-scores))
        records=[]
        for rank,j in enumerate(order,1):
            i=assigned.get(int(j))
            records.append(dict(rank=rank, predicted_unit=int(ids[j]),
                gt_unit=int(gt_ids[i]) if i is not None else '', f1=float(scores[j]),
                true_positives=int(pair[i,j]) if i is not None else 0,
                predicted_spikes=int(counts[j]), gt_spikes=int(gt_counts[i]) if i is not None else 0))
        with (args.root/f'{name}_ranked_f1.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,list(records[0]));writer.writeheader();writer.writerows(records)
        np.savez_compressed(args.root/f'{name}_f1_pairs.npz',pair_counts=pair,gt_ids=gt_ids,pred_ids=ids)
        summaries[name]=dict(units=len(ids),gt_units=len(gt_ids),spikes=len(pt),gt_spikes=len(gt_t),
            assigned_units=len(cols),unassigned_units=len(ids)-len(cols),mean_predicted_f1=float(scores.mean()),
            mean_gt_f1=float(scores.sum()/len(gt_ids)),median_f1=float(np.median(scores)),
            f1_ge_090=int((scores>=.9).sum()),f1_ge_080=int((scores>=.8).sum()),
            excluded_gt_spikes=gt_excluded,excluded_predicted_spikes=excluded)
        ax.plot(np.arange(1,len(ids)+1),scores[order],label=f'{name} ({len(ids)} units)',lw=1.8)
        print(name,json.dumps(summaries[name]),flush=True)
    summaries['protocol']=dict(window_samples=[0,54000000],tolerance_samples=12,
        unit_assignment='Hungarian maximum total F1; no quality threshold',
        units='all predicted units; unassigned F1=0; GT at most once',
        initialization='full KS4 30-minute run; default sampling; no GT in sorting')
    (args.root/'quality.json').write_text(json.dumps(summaries,indent=2))
    ax.set(xlabel='Sorted unit rank (descending F1, separately per method)',ylabel='Assigned GT spike-train F1',
           title='Full [0, 1800 s) | one-to-one units | tolerance +/-12 samples',ylim=(0,1.02))
    ax.grid(alpha=.2);ax.legend()
    for ext in ('png','pdf'):fig.savefig(args.root/f'ranked_f1.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':main()
