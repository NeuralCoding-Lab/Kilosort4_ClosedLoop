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
    hardware = np.genfromtxt(folder / 'hardware.csv', delimiter=',', names=True,
                             dtype=None, encoding='utf-8')
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
    ax.set_title('KS4 full 30-minute recording | RTX 5090 | 16 CPU threads')
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
    lines=['# KS4 30 分钟完整数据加速测试','',
        '数据：`F:\\kilosort_simu\\25298815\\sim_no_drift_30min`，全部 [0, 1800) 秒；',
        '385 路 int16 / 30 kHz，383 个连接的神经通道。设置沿用数据目录的 ks4_settings.json，其余采用本仓库默认值；保留漂移校正、所有迭代及完整输出。',
        '关键设置：batch_size=60000（900 个 2 秒批次），nblocks=1，max_peels=100，Th_learned=8。白化和通用模板片段按每 25 个批次采样，覆盖全记录（模板片段最多 500000 个）；初始聚类来自全程检测结果，cluster_downsampling=20、max_cluster_subset=25000。', '',
        '硬件：RTX 5090，Intel Core Ultra 7 270K Plus，系统可见 127.27 GiB RAM；PyTorch 2.9.1+cu128；CPU 线程上限 16。每个版本使用独立进程，完整运行一次。','',
        '| 阶段 | 修改前 (秒) | 修改后 (秒) | 加速比 | 耗时减少 |','|---|---:|---:|---:|---:|']
    for row in table:
        lines.append(f"| {LABELS.get(row['stage'],row['stage'])} | {row['before_s']:.2f} | {row['after_s']:.2f} | {row['speedup']:.2f}× | {row['saved_percent']:.1f}% |")
    lines += ['', '## 硬件占用', '', '| 版本/阶段 | CPU 平均 (单核=100%) | RSS 峰值 GiB | GPU 平均 % | 整卡显存峰值 GiB | GPU 平均 W |', '|---|---:|---:|---:|---:|---:|']
    for name,run in runs.items():
        for r in run[0]:
            if r['samples']:
                lines.append(f"| {name}/{LABELS.get(r['stage'],r['stage'])} | {r['cpu_process_percent_mean']:.1f} | {r['rss_gib_max']:.2f} | {r['gpu_percent_mean']:.1f} | {r['gpu_used_gib_max']:.2f} | {r['gpu_power_w_mean']:.1f} |")
    lines += ['', '## 加速内容与正确性', '',
        '- 参照 D:/ChronoSort/src/chronosort/lowdim_matching.py 及 utils/lowdim_matching_kernels.py：缓存重建波形，精确保留非零模板相关项，融合 GPU 评分/更新，8 轮后仅重算受影响时间点，每 4 轮检查一次并复用 CUDA Graph。阈值、最大迭代数和 KS4 时间对齐不变。',
        '- 参照 ChronoSort 的单元时间序列分组/缓存方式：CCG refractory 估计只分组一次，合并时缓存各单元 spike 序列，仅更新被合并的序列及其样本偏移。CCG 内核、阈值与贪心合并顺序不变。',
        '- 新增 8 项测试通过，覆盖 GPU/非 Graph 执行、容量重试、重放、等高峰回退、稀疏相关项、CCG 分组和非零时间偏移的多次合并。',
        '- 仓库原有 test_select_matching_prior_mask_prefers_reference_units 失败，已在修改前快照上复现；其余 12 项相关 CPU 检查通过。详见 acceleration_tests.log、cpu_tests.log 和 baseline_existing_test.log。']
    validation_path = args.root/'matching_validation.json'
    if validation_path.exists():
        validation = json.loads(validation_path.read_text())
        repeats=[r for b in validation['batches'] for r in b['repeats']]
        old=sum(r['original'] for r in repeats); new=sum(r['accelerated'] for r in repeats)
        lines += [f"- 六个同一真实输入批次（首、中、末）各交替重复三次：纯匹配平均 {old/len(repeats)*1000:.2f} → {new/len(repeats)*1000:.2f} ms，合计加速 {old/new:.2f}×。候选时间、模板编号、幅值与评分完全相同；残差最大绝对误差 {max(b['residual_max_abs'] for b in validation['batches']):.3g}。该计时排除批次预处理和特征导出，不能代替完整匹配阶段耗时。",
                  f"- 精确非零模板对 {validation['stats']['sparse_pairs']:,}/{validation['stats']['dense_pairs']:,}，占 {100*validation['stats']['sparse_pairs']/validation['stats']['dense_pairs']:.2f}%；没有近似截断。详见 matching_validation.json。"]
    quality_path=args.root/'quality.json'
    state_path=args.root/'initial_state_comparison.json'
    if state_path.exists():
        state=json.loads(state_path.read_text())
        lines += [f"- 两次独立全流程运行在进入加速匹配前已产生不同初始聚类模板：{state['before_templates']} 与 {state['after_templates']} 个；PCA、白化和漂移数组逐项完全相同。此聚类波动发生在加速匹配之前，因此全流程最终输出的微小差异不能直接归因于加速内核；另用同一原版模板库重放全部 900 批次作为严格对照。"]
    replay_path=args.root/'full_matching_validation.json'
    if replay_path.exists():
        replay=json.loads(replay_path.read_text())
        lines += [f"- **共享原版模板库和相同预处理输入的全部 900 批次对照**：原版/加速版分别产生 {replay['spikes_before']:,}/{replay['spikes_after']:,} 条批次内 spike。{replay['strict_non_tied_batches']} 个未回退批次（{replay['strict_non_tied_spikes']:,} 条 spike）的时间、模板编号完全一致；幅值和评分通过 rtol=atol=3e-5 检查，该范围 tF 最大绝对误差 {replay['feature_max_abs']:.3g}，平均绝对误差 {replay['feature_mean_abs']:.3g}。",
            f"- 其余 {len(replay['legacy_tied_batches'])} 批因相邻等高峰回退到原版 KS4。这些批次不要求逐位重复：原版本身对同一输入重复 10 次会出现 9728/9729 条 spike 以及少数时间变化（batch 660，legacy_tied_repeatability.json）。回退批次的事件差异单独保存，没有将它们删除后声称全记录完全相同。全部批次按精确的（时间，模板）一对一计数，原版/加速版事件一致性 F1 为 {replay['exact_event_f1']:.10f}；这是实现间一致性，不是 GT 质量 F1。",
            f"- 全部批次交替执行顺序，纯匹配累计 {replay['before_seconds']:.2f} → {replay['after_seconds']:.2f} 秒（{replay['speedup']:.2f}×）；计时排除共同预处理和数值比较，不能代替端到端阶段耗时。详见 full_matching_validation.json、full_matching_batches.json。"]
    if quality_path.exists():
        quality=json.loads(quality_path.read_text())
        lines += ['', '## 全时段 GT 排序质量', '',
            '全部 [0, 1800) 秒、所有输出单元、±12 样本（±0.4 ms）容差。每个 unit pair 内 spike 一对一符合，再以 Hungarian 最大总 F1 一对一分配单元，不设质量阈值；未分配输出单元 F1=0。GT 仅用于评价。初始化使用相同 KS4 默认采样范围，排序覆盖全程。', '',
            '| 版本 | 输出单元 | Spike 数 | 平均输出单元 F1 | 平均 GT F1 | F1≥0.9 单元 |',
            '|---|---:|---:|---:|---:|---:|']
        for name in ('before','after'):
            q=quality[name]
            lines.append(f"| {name} | {q['units']} | {q['spikes']:,} | {q['mean_predicted_f1']:.6f} | {q['mean_gt_f1']:.6f} | {q['f1_ge_090']} |")
        lines += ['', '逐单元分配和 F1：before_ranked_f1.csv、after_ranked_f1.csv；计数矩阵：*_f1_pairs.npz；排除统计与协议：quality.json。', '', '![F1 降序曲线](ranked_f1.png)']
    lines += ['', '## 复核说明', '',
        '- 时间边界执行 CUDA 同步；硬件目标每 0.5 秒采样，CPU/Numba 占用 GIL 时实际采样会延迟。平均值按真实采样时间间隔加权，峰值为采样峰值；原始 CSV 保留所有时间戳。嵌套的漂移检测和模板合并归入所属阶段，避免重复计时。Plots / other 包含诊断图和未单独包装的调度步骤。',
        '- GPU 利用率、显存、功耗为整卡 NVML 数据，包含已有的桌面和其他 GPU 进程；RSS 和进程 CPU 为当前 KS4 进程。CPU 100% 代表一个逻辑核，整机有 24 个逻辑核。',
        '- before/after/stages.json 中 PyTorch 峰值来自 KS4 自身重置计数器的行为，不宜直接作为各阶段显存峰值；本报告采用连续采样的 NVML/RSS。',
        '- 单次完整运行的阶段差异可能受操作系统文件缓存、温度和后台负载影响；未修改阶段的变化不能归因于匹配内核。另用相同预处理批次交替重复计时验证匹配加速。',
        '- 原始二进制和 ChronoSort/src 未修改。baseline/kilosort 为修改前快照；各版本 metadata.json 保存实际加载路径、代码哈希及运行配置。', '',
        '![各阶段耗时](stage_times.png)', '', '![硬件占用](hardware.png)', '',
        '原始采样：before/hardware.csv、after/hardware.csv；阶段记录：各版本 stages.json；完整 Phy 与 prior_bundle 输出：各版本 sorting/。']
    (args.root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(table,indent=2))


if __name__ == '__main__':
    main()
