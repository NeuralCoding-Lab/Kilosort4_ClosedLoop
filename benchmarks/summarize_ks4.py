"""Summarize full-run stage durations and 0.5-second hardware samples."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


LABELS = dict(preprocessing='Preprocessing', drift_correction='Drift correction',
    universal_detection='Universal detection', initial_clustering='Initial clustering',
    template_postprocessing='Template postprocessing', learned_matching='Learned matching',
    final_clustering='Final clustering', final_merge='Final merging', export='Export',
    other='Plots / other')


def load_run(folder):
    events = json.loads((folder / 'stages.json').read_text())
    # spikedetect.run is also called within drift correction. Keep the enclosing
    # drift stage and discard its nested call to avoid double-counting.
    flat = [event for event in events if not any(
        parent is not event and parent['start_s'] < event['start_s'] and
        parent['start_s']+parent['seconds'] >= event['start_s']+event['seconds']
        for parent in events)]
    result = json.loads((folder / 'result.json').read_text())
    if not result.get('success'):
        raise ValueError(f'Unsuccessful run: {folder}')
    hardware = np.genfromtxt(folder / 'hardware.csv', delimiter=',', names=True,
                             dtype=None, encoding='utf-8')
    hardware = np.atleast_1d(hardware)
    if hardware.size == 0:
        raise ValueError(f'No hardware samples: {folder}')
    weights = np.diff(hardware['elapsed_s'], prepend=0.)
    rows = []
    assigned = np.zeros(len(hardware), dtype=bool)
    for event in sorted(flat, key=lambda e: e['start_s']):
        mask = (hardware['elapsed_s'] >= event['start_s']) & (
            hardware['elapsed_s'] < event['start_s']+event['seconds'])
        assigned |= mask
        rows.append(summarize(event['stage'], event['seconds'], hardware[mask], weights[mask]))
    rows.append(summarize('other', max(0, result['wall_seconds']-sum(e['seconds'] for e in flat)), hardware[~assigned], weights[~assigned]))
    rows.append(summarize('total', result['wall_seconds'], hardware, weights))
    return rows, hardware, flat, result


def summarize(stage, seconds, samples, weights):
    row = dict(stage=stage, seconds=seconds, samples=len(samples))
    for key in ('cpu_process_percent', 'cpu_system_percent', 'rss_gib', 'gpu_percent',
                'gpu_memory_controller_percent', 'gpu_used_gib', 'gpu_power_w', 'gpu_temp_c'):
        row[key+'_mean'] = float(np.average(samples[key], weights=weights)) if len(samples) else None
        row[key+'_max'] = float(np.max(samples[key])) if len(samples) else None
    row['gpu_energy_wh_estimate'] = row['gpu_power_w_mean'] * seconds / 3600 if len(samples) else None
    return row


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    runs = {name: load_run(args.root / name) for name in ('before', 'after')}
    metadata = {name: json.loads((args.root / name / 'metadata.json').read_text(encoding='utf-8')) for name in runs}
    for field in ('filename', 'settings', 'run_options', 'threads', 'torch', 'gpu', 'packages', 'instrumentation'):
        if metadata['before'].get(field) != metadata['after'].get(field):
            raise ValueError(f'Incomparable runs: {field} differs')
    table = []
    before = {r['stage']: r for r in runs['before'][0]}
    after = {r['stage']: r for r in runs['after'][0]}
    for stage, row in before.items():
        newer = after[stage]
        table.append(dict(stage=stage, before_s=row['seconds'], after_s=newer['seconds'],
            speedup=row['seconds']/newer['seconds'], saved_percent=(1-newer['seconds']/row['seconds'])*100))
    write_csv(args.root / 'stage_comparison.csv', table)
    write_csv(args.root / 'hardware_summary.csv', [dict(variant=name, **r) for name, run in runs.items() for r in run[0]])
    (args.root / 'comparison.json').write_text(json.dumps(dict(timing=table,
        hardware={name: run[0] for name, run in runs.items()},
        results={name: run[3] for name, run in runs.items()}), indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10})
    stages = [s for s in before if s != 'total']
    fig, ax = plt.subplots(figsize=(11, 6), layout='constrained')
    y = np.arange(len(stages))
    ax.barh(y-.18, [before[s]['seconds'] for s in stages], .36, label='Before', color='#62778c')
    ax.barh(y+.18, [after[s]['seconds'] for s in stages], .36, label='After', color='#128c83')
    for i,s in enumerate(stages):
        ax.text(max(before[s]['seconds'], after[s]['seconds'])+5, i,
                f"{before[s]['seconds']/after[s]['seconds']:.2f}x", va='center')
    ax.set_yticks(y, [LABELS[s] for s in stages]); ax.invert_yaxis()
    ax.set_xlabel('Wall time (seconds)'); ax.legend(); ax.grid(axis='x', alpha=.2)
    ax.set_xlim(0,max(max(before[s]['seconds'],after[s]['seconds']) for s in stages)*1.13)
    ax.set_title(f"KS4 | {metadata['after']['gpu']} | {metadata['after']['threads']} CPU threads")
    for ext in ('png','pdf'):
        fig.savefig(args.root / f'stage_times.{ext}', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True, sharey='row', layout='constrained')
    for col,(name,run) in enumerate(runs.items()):
        hw = run[1]; minutes = hw['elapsed_s']/60
        for row,(field,label) in enumerate([('gpu_percent','GPU utilization (%)'),
                ('cpu_process_percent','Process CPU (%; 100 = one core)'),
                ('gpu_used_gib','Whole-device VRAM (GiB)'), ('rss_gib','Process RSS (GiB)')]):
            ax=axes[row,col]; ax.plot(minutes,hw[field],lw=.8,color='#62778c' if col==0 else '#128c83')
            ax.set_ylabel(label); ax.grid(alpha=.2)
            for event in run[2]:
                ax.axvline(event['start_s']/60,color='gray',alpha=.25,lw=.6)
        axes[0,col].set_title(name.title())
        axes[-1,col].set_xlabel('Elapsed time (minutes)')
    for ext in ('png','pdf'):
        fig.savefig(args.root / f'hardware.{ext}',dpi=160)
    plt.close(fig)
    meta = metadata['after']
    lines = ['# KS4 real-data benchmark', '',
        f"Input: `{meta.get('filename', 'see metadata.json')}`", '',
        f"GPU: {meta['gpu']}; PyTorch: {meta['torch']}; threads: {meta['threads']}", '',
        f"Instrumentation: `{json.dumps(meta.get('instrumentation', {}))}`", '',
        'Settings and export options are recorded in before/after metadata.json.', '',
        '| Stage | Before (s) | After (s) | Speedup | Saved |',
        '|---|---:|---:|---:|---:|']
    for row in table:
        lines.append(f"| {LABELS.get(row['stage'],row['stage'])} | {row['before_s']:.2f} | {row['after_s']:.2f} | {row['speedup']:.2f}× | {row['saved_percent']:.1f}% |")
    lines += ['', '## 硬件占用', '', '| 版本/阶段 | CPU 平均 (单核=100%) | RSS 峰值 GiB | GPU 平均 % | 整卡显存峰值 GiB | GPU 平均 W |', '|---|---:|---:|---:|---:|---:|']
    for name,run in runs.items():
        for r in run[0]:
            if r['samples']:
                lines.append(f"| {name}/{LABELS.get(r['stage'],r['stage'])} | {r['cpu_process_percent_mean']:.1f} | {r['rss_gib_max']:.2f} | {r['gpu_percent_mean']:.1f} | {r['gpu_used_gib_max']:.2f} | {r['gpu_power_w_mean']:.1f} |")
    lines += ['', '## Scope and validation', '',
        '- End-to-end time includes sorting, plotting and export. Checkpoint/profile runs must be reported separately.',
        '- Nested stage times are excluded from stage sums. GPU sampling covers the entire device; CPU 100% means one logical core.',
        '- Compilation and CUDA Graph construction inside sorting are included. Existing disk caches are not cleared; this is not a guaranteed cold-install benchmark.',
        '- Independent runs may differ because of clustering and tied-peak nondeterminism. Output counts do not establish equivalence or ground-truth accuracy.',
        '- This report makes no correctness claim from historical tests. Perform shared-input replay separately.', '']
    for name, run in runs.items():
        result = run[3]
        lines += [f"### {name}", '',
            f"Spikes: {result['spikes']}; units: {result['units']}; good units: {result['good_units']}", '',
            '```json', json.dumps({key: result.get(key) for key in ('matching_stats', 'universal_detection_stats')}, indent=2), '```', '']
    lines += ['![Stage times](stage_times.png)', '', '![Hardware](hardware.png)']
    (args.root/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(table, indent=2))


if __name__ == '__main__':
    main()
