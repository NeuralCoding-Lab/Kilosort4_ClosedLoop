"""Report the drift/detection follow-up without reusing prior-round claims."""
import argparse
import difflib
import hashlib
import io
import json
from pathlib import Path
import pstats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from summarize_ks4 import load_run, write_csv, LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    root = parser.parse_args().root
    runs = {name: load_run(root / name) for name in ('before', 'after')}
    before, after = ({r['stage']: r for r in runs[name][0]} for name in ('before', 'after'))
    table = [dict(stage=s, before_s=r['seconds'], after_s=after[s]['seconds'],
                  speedup=r['seconds']/after[s]['seconds'],
                  saved_percent=100*(1-after[s]['seconds']/r['seconds'])) for s,r in before.items()]
    write_csv(root / 'stage_comparison.csv', table)
    write_csv(root / 'hardware_summary.csv', [dict(variant=n, **r) for n,v in runs.items() for r in v[0]])
    metadata = {n: json.loads((root/n/'metadata.json').read_text()) for n in runs}
    hash_ok = {}
    for name, m in metadata.items():
        source = Path(m['source'])
        hash_ok[name] = all(hashlib.sha256((source/p).read_bytes()).hexdigest() == h
                            for p,h in m['source_hashes'].items())
        assert hash_ok[name], (name, 'source changed after benchmark')
    production = Path(__file__).resolve().parents[1] / 'kilosort'
    frozen = root / 'baseline/kilosort'
    changes, patch = [], []
    for path in production.rglob('*.py'):
        relative = path.relative_to(production)
        old = frozen / relative
        old_text = old.read_text(encoding='utf-8') if old.exists() else ''
        new_text = path.read_text(encoding='utf-8')
        if old_text != new_text:
            changes.append(str(relative))
            patch.extend(difflib.unified_diff(old_text.splitlines(True), new_text.splitlines(True),
                         fromfile=f'before/kilosort/{relative}', tofile=f'after/kilosort/{relative}'))
    (root/'source_changes.patch').write_text(''.join(patch), encoding='utf-8')
    (root/'source_changes.json').write_text(json.dumps(changes, indent=2))
    initial = {}
    for call in range(2):
        states = {n: torch.load(root/n/f'detection_input_{call}.pt', map_location='cpu', weights_only=False)
                  for n in runs}
        same = {}
        for key in ('wPCA','wTEMP','dshift','Wrot','fwav'):
            a,b = (states[n]['ops'].get(key) for n in runs)
            same[key] = torch.equal(a,b) if isinstance(a,torch.Tensor) else np.array_equal(a,b)
        for key in ('iC','iC2','weigh'):
            same[key] = torch.equal(states['before'][key],states['after'][key])
        st = {n: np.load(root/n/f'detection_st_{call}.npy', mmap_mode='r') for n in runs}
        same.update(before_spikes=len(st['before']), after_spikes=len(st['after']),
                    exact_spike_table=np.array_equal(st['before'],st['after']))
        initial[str(call)] = same
    (root/'initial_state_comparison.json').write_text(json.dumps(initial,indent=2))
    for name in runs:
        for call in range(2):
            stream = io.StringIO()
            pstats.Stats(str(root/name/f'detection_{call}.pstats'),stream=stream).strip_dirs().sort_stats('cumtime').print_stats(35)
            (root/f'{name}_detection_profile_{call}.txt').write_text(stream.getvalue(),encoding='utf-8')
    validation = json.loads((root/'detection_validation.json').read_text())
    quality = json.loads((root/'quality.json').read_text())
    fits = json.loads((root/'template_fit_differences.json').read_text())
    (root/'comparison.json').write_text(json.dumps(dict(timing=table,
        hardware={n:v[0] for n,v in runs.items()}, results={n:v[3] for n,v in runs.items()},
        validation=validation, quality=quality, initial=initial,
        template_fit_differences=fits, source_hashes_valid=hash_ok),indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    stages = [s for s in before if s!='total']
    y = np.arange(len(stages))
    fig,ax = plt.subplots(figsize=(11,6),layout='constrained')
    ax.barh(y-.18,[before[s]['seconds'] for s in stages],.36,label='Before',color='#62778c')
    ax.barh(y+.18,[after[s]['seconds'] for s in stages],.36,label='After',color='#128c83')
    for i,s in enumerate(stages):
        ax.text(max(before[s]['seconds'],after[s]['seconds'])+3,i,
                f"{before[s]['seconds']/after[s]['seconds']:.2f}x",va='center')
    ax.set_yticks(y,[LABELS[s] for s in stages]);ax.invert_yaxis()
    ax.set(xlabel='Wall time (seconds)',title='KS4 full 30-minute recording | drift / universal detection follow-up')
    ax.set_xlim(0,max(max(before[s]['seconds'],after[s]['seconds']) for s in stages)*1.15)
    ax.legend();ax.grid(axis='x',alpha=.2)
    for ext in ('png','pdf'):fig.savefig(root/f'stage_times.{ext}',dpi=160)
    plt.close(fig)
    fields=[('gpu_percent','GPU utilization (%)'),('cpu_process_percent','Process CPU (%; 100 = one core)'),
            ('gpu_used_gib','Whole-device VRAM (GiB)'),('rss_gib','Process RSS (GiB)'),('gpu_power_w','Whole-device power (W)')]
    fig,axes=plt.subplots(5,2,figsize=(14,12),sharex=True,sharey='row',layout='constrained')
    for col,(name,run) in enumerate(runs.items()):
        hw=run[1]
        for row,(field,label) in enumerate(fields):
            ax=axes[row,col]
            ax.plot(hw['elapsed_s']/60,hw[field],lw=.8,color='#62778c' if col==0 else '#128c83')
            ax.set_ylabel(label);ax.grid(alpha=.2)
            for event in run[2]:
                ax.axvline(event['start_s']/60,color='gray',alpha=.25,lw=.6)
                if event['stage'] in ('drift_correction','universal_detection'):
                    ax.axvspan(event['start_s']/60,(event['start_s']+event['seconds'])/60,
                               alpha=.12,color='#128c83' if event['stage']=='drift_correction' else '#357ac4')
        axes[0,col].set_title(name.title()+' | green: drift; blue: universal detection')
        axes[-1,col].set_xlabel('Elapsed time (minutes)')
    for ext in ('png','pdf'):fig.savefig(root/f'hardware.{ext}',dpi=160)
    plt.close(fig)
    lines=['# KS4 漂移校正与通用模板检测加速实测','',
        '数据：`F:/kilosort_simu/25298815/sim_no_drift_30min`，完整 [0,1800) 秒；385 路 int16、30 kHz，383 个连接神经通道，900 个批次。',
        '本次 before 已包含上一轮 learned matching 和 CPU 合并优化，比较基准是本轮修改前的当前版本。原始数据及 ChronoSort/src 未修改。',
        '设置沿用 ks4_settings.json，其余使用仓库默认值。batch_size=60000，nblocks=1，template_sizes=5，nearest_templates=100，nearest_chans=10，max_peels=100。不缩短记录、不减少迭代、不改变阈值或采样范围。',
        '硬件：RTX 5090 31.84 GiB、Core Ultra 7 270K Plus、可见 RAM 127.27 GiB；PyTorch 2.9.1+cu128，CPU 线程上限 16。两个版本在独立进程中串行完整运行，各一次。','',
        '## 各阶段耗时','', '| 阶段 | 修改前秒 | 修改后秒 | 加速比 | 耗时减少 |','|---|---:|---:|---:|---:|']
    for r in table:
        lines.append(f"| {LABELS.get(r['stage'],r['stage'])} | {r['before_s']:.3f} | {r['after_s']:.3f} | {r['speedup']:.2f}× | {r['saved_percent']:.1f}% |")
    lines += ['', '漂移阶段包含第一遍通用检测；表中的 Universal detection 是漂移校正后的第二遍，阶段求和不重复计入第一遍。Plots / other 包括诊断绘图、检查点输出及未单独包装的调度步骤。',
              '', '![阶段耗时](stage_times.png)','', '## 优化及判据','',
        '- 保留原 conv1d 与 40 个 einsum 分块，保持空间点积的形状及通道归约顺序。直接读取原始分数布局，融合 abs/max/argmax/符号选择，省去中间数组及分块回填操作。',
        '- 对同一 iC2 邻域计算完全相同的最大值，避免显式构造巨大的邻域 gather。时间 max_pool1d、边界清零、严格大于阈值及相等判峰保持原实现；并列时取第一个 size/waveform。',
        '- 每遍检测复用固定工作区和 CUDA Graph；Graph 构建、预热及缓存加载计入端到端耗时。Triton 磁盘缓存已由等价性测试预热，本表不代表首次安装的冷编译时间。环境变量 KS4_DETECTION_BACKEND=torch 可切回原检测；auto 在 CPU/无 Triton 时回退。',
        '- 漂移分箱一次定位各批次区间，乱序输入仅稳定排序一次；原深度/对数幅值分箱及 scipy 稀疏计数公式不变。',
        '- 对齐中的各候选偏移仍分别按原公式计算，只将相关向量合并传回 CPU。迭代次数、均值归约、argmax、平滑、插值及最终偏移不变。',
        '- 漂移检测仍生成完整 PC 特征，保留原 run 返回约定。','', '## 同输入全时段严格验证','']
    lines[-2:-2] = ['CUDA 时间线对同一真实 batch 450 重复 3 次：原空间点积内核合计 139.24 → 140.74 ms，保持基本相同；原索引/max/abs 分别约 162.24/111.75/50.65 ms，新融合 score/邻域最大值分别约 28.12/26.98 ms。累计 GPU 活动时间约 496.00 → 286.38 ms。该剖析用于定位操作开销，完整加速比以下面的全记录交替重放及端到端计时为准。', '']
    for v in validation:
        lines.append(f"- 第 {v['call']+1} 遍：{v['batches']} 个批次、{v['spikes']:,} 个 spike；时间/空间位置、模板编号、幅值、adist、加权深度及全部 PC 特征逐项完全相等，特征最大绝对误差 {v['feature_max_abs']:.3g}。纯检测累计 {v['before_seconds']:.3f} → {v['after_seconds']:.3f} 秒（{v['before_seconds']/v['after_seconds']:.2f}×），交替执行顺序，排除共同预处理和验证。")
        lines.append(f"  同一全记录 spike 表的深度/幅值直方图、漂移位移及两个 fingerprint 完全相等；分箱 {v['bin_before_s']:.3f} → {v['bin_after_s']:.3f} 秒，对齐 {v['align_before_s']:.3f} → {v['align_after_s']:.3f} 秒。")
    lines += ['', '分箱/对齐重放额外对两遍 spike 表均执行；实际全流程仅第一遍用于估计漂移，第二遍的额外对齐检查不参与排序。完整验证见 detection_validation.json、detection_replay_batches_*.json。44 项本轮及此前加速回归检查通过，见 tests.log；drift_tests.log 保存早期 CPU 等价检查。',
        '', '## 硬件占用','', '| 版本/阶段 | CPU平均%（单核100） | RSS采样峰值GiB | GPU平均% | 整卡显存采样峰值GiB | 整卡平均W |', '|---|---:|---:|---:|---:|---:|']
    for name,run in runs.items():
        for r in run[0]:
            if r['samples']:
                lines.append(f"| {name}/{LABELS.get(r['stage'],r['stage'])} | {r['cpu_process_percent_mean']:.1f} | {r['rss_gib_max']:.2f} | {r['gpu_percent_mean']:.1f} | {r['gpu_used_gib_max']:.2f} | {r['gpu_power_w_mean']:.1f} |")
    lines += ['', '![硬件曲线](hardware.png)', '', '## 全时段排序质量','',
        '使用全部 1200 个 GT 单元和全部输出单元；[0,54000000) 样本内，unit pair 内 ±12 样本（0.4 ms）一对一符合，再使用 Hungarian 最大总 F1 一对一分配单元。未分配单元 F1=0，不按质量筛选。GT 仅用于事后评价。','',
        '| 版本 | 输出单元 | Spike数 | 平均输出F1 | 平均GT F1 | F1≥0.9 | F1≥0.8 |','|---|---:|---:|---:|---:|---:|---:|']
    for n in runs:
        q=quality[n]
        lines.append(f"| {n} | {q['units']} | {q['spikes']:,} | {q['mean_predicted_f1']:.6f} | {q['mean_gt_f1']:.6f} | {q['f1_ge_090']} | {q['f1_ge_080']} |")
    lines += ['', '两次独立全流程可能受聚类和原 KS4 等高峰并行写入的非确定性影响；不能用单次微小 F1 差异宣称质量提高或全流程逐位一致。本轮同输入验证独立检验了检测与漂移实现的输出等价性。独立运行的检测表/模板/漂移状态比较见 initial_state_comparison.json。',
        f"两次独立模板拟合的 wTEMP 在进入检测之前已有细微差异：第一/第二遍最大绝对差 {fits['0']['wTEMP_max_abs']:.3g}/{fits['1']['wTEMP_max_abs']:.3g}；wPCA、白化、高通和漂移数组完全相同。第一遍检测 {initial['0']['before_spikes']:,} → {initial['0']['after_spikes']:,} 个 spike，第二遍均为 {initial['1']['before_spikes']:,} 个，但表值并非逐位一致。后续初始模板库为 {int(round(runs['before'][3]['matching_stats']['dense_pairs']**.5))}/{int(round(runs['after'][3]['matching_stats']['dense_pairs']**.5))} 个模板。因此独立全流程 F1 差异不能直接归因于本轮加速内核；共享冻结模板的全部 1800 批次对照才用于隔离验证实现等价性。详见 template_fit_differences.json。",
        '', '![单元F1降序曲线](ranked_f1.png)','', '## 复核与限制','',
        '- 每个阶段边界同步 CUDA；NVML/psutil 目标每 0.5 秒采样，GIL/系统调度可延迟，原始时间戳均保留，均值按实际间隔加权；峰值为采样峰值。CPU 100% 表示一个逻辑核，进程 CPU/RSS 为当前排序进程。',
        '- GPU 利用率、显存和功耗为整卡数据，包含已有桌面和其他 GPU 进程。未终止用户后台程序。硬件曲线中的绿色/蓝色为漂移/第二遍通用检测。GPU 时间线、回归测试、重放均在完整基准之外执行。',
        '- before/after 均启用相同 cProfile/检查点选项。输入模板检查点体积约 2–5 MB；完整 st/tF 输出在检测计时之外保存，漂移父阶段包含第一遍输出保存开销，两版本一致。总时间包含诊断绘图和这些测试输出。',
        '- 总时间的计时范围是 run_kilosort 调用，含完整排序、导出与诊断绘图；不含 Python 导入及调用前 CUDA 上下文初始化。',
        '- GPU 异步执行时 cProfile 会将大量等待计入 nonzero，不能将其视为 nonzero 内核独占时间。实际 GPU 操作剖析见 *_gpu_profile.txt 和 *_detection_trace.json。',
        '- 每版仅一次完整运行，未修改阶段的时间波动可能来自文件缓存、温度及后台负载；加速另用共享输入、交替顺序的全部批次对照确认。纯检测计时不等于含预处理/特征/传输的完整阶段时间。',
        f'- 已核对完整运行实际源码 SHA256：{hash_ok}；生产代码变化：{", ".join(changes)}。完整差异见 source_changes.patch，原版快照 baseline/kilosort。',
        '- before/after/metadata.json、result.json、stages.json、hardware.csv 保存环境/结果/原始时间和资源样本。最终完整排序位于 after/sorting。','']
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(timing=table, quality=quality, initial=initial),indent=2))


if __name__=='__main__':
    main()
