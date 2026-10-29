"""Full-run, shared-input, hardware and GT report for CPU merge optimization."""
import argparse
import io
import json
from pathlib import Path
import pstats
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from summarize_ks4 import load_run, write_csv, LABELS


def main(root):
    replay={name:json.loads((root/f'replay_{name}/result.json').read_text()) for name in ('before','after')}
    decisions={name:json.loads((root/f'replay_{name}/decisions.json').read_text()) for name in ('before','after')}
    elapsed={name:json.loads((root/f'replay_{name}/stages.json').read_text())[0]['seconds'] for name in ('before','after')}
    control=dict(decisions_identical=decisions['before']==decisions['after'],
        outputs_identical=replay['before']['outputs']==replay['after']['outputs'],
        output_matches={k:replay['before']['outputs'][k]==replay['after']['outputs'][k] for k in replay['before']['outputs']},
        ccg_calls_before=replay['before']['ccg_calls'],ccg_calls_after=replay['after']['ccg_calls'],
        accepted_pairs_before=replay['before']['accepted_pairs'],accepted_pairs_after=replay['after']['accepted_pairs'],
        before_seconds=elapsed['before'],after_seconds=elapsed['after'],speedup=elapsed['before']/elapsed['after'])
    (root/'shared_merge_comparison.json').write_text(json.dumps(control,indent=2))
    runs={name:load_run(root/name) for name in ('before','after')}
    lookup={name:{r['stage']:r for r in run[0]} for name,run in runs.items()}
    initial={}
    for name,run in runs.items():
        console=(root/f'{name}_console.log').read_text(encoding='utf-8',errors='replace')
        initial[name]=dict(universal_spikes=int(re.findall(r'(\d+) spikes extracted',console)[0]),
            initial_clusters=int(re.findall(r'(\d+) clusters found',console)[0]),
            learned_templates=round(np.sqrt(run[3]['matching_stats']['dense_pairs'])))
    (root/'initial_state_comparison.json').write_text(json.dumps(initial,indent=2))
    rows=[]
    for stage,old in lookup['before'].items():
        new=lookup['after'][stage]
        rows.append(dict(stage=stage,before_s=old['seconds'],after_s=new['seconds'],
            speedup=old['seconds']/new['seconds'],saved_percent=(1-new['seconds']/old['seconds'])*100))
    write_csv(root/'stage_comparison.csv',rows)
    write_csv(root/'hardware_summary.csv',[dict(variant=name,**r) for name,run in runs.items() for r in run[0]])
    (root/'comparison.json').write_text(json.dumps(dict(timing=rows,
        hardware={name:run[0] for name,run in runs.items()},results={name:run[3] for name,run in runs.items()}),indent=2))
    profiles={}
    for name in ('before','after'):
        text=io.StringIO()
        stats=pstats.Stats(str(root/name/'merge_profile.pstats'),stream=text)
        stats.strip_dirs().sort_stats('cumtime').print_stats(25)
        (root/f'{name}_profile.txt').write_text(text.getvalue())
        entries=[dict(file=k[0],line=k[1],function=k[2],calls=v[1],self_s=v[2],cumulative_s=v[3])
                 for k,v in stats.stats.items()]
        profiles[name]=sorted(entries,key=lambda x:-x['cumulative_s'])
    (root/'profiles.json').write_text(json.dumps(profiles,indent=2))
    roll_profile={name:next(e for e in entries if e['function']=='roll_features') for name,entries in profiles.items()}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    stages=[r['stage'] for r in rows if r['stage']!='total']
    fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    y=np.arange(len(stages))
    for delta,name,color in ((-.18,'before','#62778c'),(.18,'after','#128c83')):
        ax.barh(y+delta,[lookup[name][s]['seconds'] for s in stages],.36,label=name.title(),color=color)
    for i,s in enumerate(stages):
        ax.text(max(lookup[n][s]['seconds'] for n in runs)+5,i,
                f"{lookup['before'][s]['seconds']/lookup['after'][s]['seconds']:.2f}x",va='center')
    ax.set_yticks(y,[LABELS[s] for s in stages]);ax.invert_yaxis()
    ax.set_xlim(0,max(r['seconds'] for n in runs for r in runs[n][0] if r['stage']!='total')*1.14)
    ax.set_xlabel('Wall time (seconds)');ax.legend();ax.grid(axis='x',alpha=.2)
    ax.set_title('CPU merge optimization | full 30 minutes | RTX 5090 | 16 CPU threads')
    for ext in ('png','pdf'):fig.savefig(root/f'stage_times.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(5,2,figsize=(14,12),sharex=True,sharey='row',layout='constrained')
    for col,(name,run) in enumerate(runs.items()):
        hw=run[1];minutes=hw['elapsed_s']/60
        for row,(field,label) in enumerate([('cpu_process_percent','Process CPU (%; 100 = one core)'),
                ('gpu_percent','Whole-device GPU (%)'),('rss_gib','Process RSS (GiB)'),
                ('gpu_used_gib','Whole-device VRAM (GiB)'),('gpu_power_w','Whole-device GPU power (W)')]):
            ax=axes[row,col];ax.plot(minutes,hw[field],lw=.8,color='#62778c' if col==0 else '#128c83')
            ax.set_ylabel(label);ax.grid(alpha=.2)
            for event in run[2]:
                ax.axvline(event['start_s']/60,color='gray',lw=.6,alpha=.25)
                if event['stage']=='final_merge':
                    ax.axvspan(event['start_s']/60,(event['start_s']+event['seconds'])/60,
                               color='#d6aa52',alpha=.13)
        axes[0,col].set_title(name.title()+' (shaded: final merge)')
        axes[-1,col].set_xlabel('Elapsed time (minutes)')
    for ext in ('png','pdf'):fig.savefig(root/f'hardware.{ext}',dpi=160)
    plt.close(fig)
    control=json.loads((root/'shared_merge_comparison.json').read_text())
    quality=json.loads((root/'quality.json').read_text())
    lines=['# CPU 合并阶段加速：30 分钟完整测试','',
        '基线为上一轮已采用 Triton template matching 和 spike-train 缓存的版本，本轮基线已重新实测。',
        '数据：F:/kilosort_simu/25298815/sim_no_drift_30min；完整 [0,1800) 秒；30 kHz，385 路 int16，383 个连接通道。',
        '硬件：RTX 5090，Core Ultra 7 270K Plus（24 核），系统可见 127.27 GiB RAM；PyTorch 2.9.1+cu128；CPU 线程上限 16。',
        '沿用数据目录 ks4_settings.json 和仓库默认值：batch_size=60000，nblocks=1，max_peels=100，Th_learned=8；保留绘图、合并和完整输出。白化/通用模板每 25 批采样，初始聚类来自全程检测，cluster_downsampling=20、max_cluster_subset=25000。','',
        '## 实现及判据', '',
        '主要优化：roll_features 中先把 NumPy 布尔掩码解析为有序整数行索引，再使用原生 PyTorch 索引复用读取和写回，避免对整条记录反复进行跨 NumPy/PyTorch 的序列索引转换。波形重建、移位、边界填充、PCA 投影和模板更新的算术公式及行顺序不变。',
        '同时复用排序时已复制到 CPU 的完整相似度分数，在 CPU 上执行候选阈值判断，避免重复 GPU 标量比较并同步。阈值显式转换为原分数 dtype，保留 PyTorch 标量转换语义。CCG 内核、ACG/CCG 阈值、相关阈值、排序方式、贪心候选顺序和加权公式均保持原样。', '',
        '26 项定向回归检查通过，包括正/负/零时间偏移、NumPy/Torch 布尔及整数索引、CPU/GPU float32 阈值边界、多次合并和 GPU matching。完整结果见 tests.log。', '',
        '## 全流程速度', '', '| 阶段 | 修改前 s | 修改后 s | 加速比 | 耗时减少 |', '|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {LABELS.get(r['stage'],r['stage'])} | {r['before_s']:.2f} | {r['after_s']:.2f} | {r['speedup']:.2f}× | {r['saved_percent']:.1f}% |")
    lines+=['', '## 同一合并输入的严格对照', '',
        f"使用 before/merge_input.pt 保存的全部合并前 spike、特征、模板和 ops，在独立进程中执行新旧合并。全部 CCG 调用的顺序、输入长度、refractory/cross-refractory 决策及 R12 数值是否完全一致：**{control['decisions_identical']}**。",
        f"Wall、clu、is_ref、st、tF 的形状、dtype 和完整 SHA256 是否均相同：**{control['outputs_identical']}**。共享输入合并耗时 {control['before_seconds']:.2f} → {control['after_seconds']:.2f} 秒，{control['speedup']:.2f}×。",
        '完整记录：replay_before/decisions.json、replay_after/decisions.json、各自 result.json、shared_merge_comparison.json。此对照用于区分实现变化与独立聚类运行的波动。','',
        '## 硬件占用', '', '| 版本/阶段 | CPU 均值 %（单核=100） | RSS 采样峰值 GiB | GPU 均值 % | 整卡显存采样峰值 GiB | GPU 均值 W |','|---|---:|---:|---:|---:|---:|']
    for name,run in runs.items():
        for r in run[0]:
            if r['samples']:lines.append(f"| {name}/{LABELS.get(r['stage'],r['stage'])} | {r['cpu_process_percent_mean']:.1f} | {r['rss_gib_max']:.2f} | {r['gpu_percent_mean']:.1f} | {r['gpu_used_gib_max']:.2f} | {r['gpu_power_w_mean']:.1f} |")
    cpu_work={name:lookup[name]['final_merge']['cpu_process_percent_mean']/100*lookup[name]['final_merge']['seconds'] for name in runs}
    lines+=['',f"合并阶段累计 CPU 工作量（按采样均值×阶段耗时估算的逻辑核秒）{cpu_work['before']:.1f} → {cpu_work['after']:.1f}，减少 {(1-cpu_work['after']/cpu_work['before'])*100:.1f}%。新版平均 CPU 使用率更高，但该阶段持续时间显著缩短。"]
    lines+=['', '## 排序质量', '',
        '全部 [0,1800) 秒，保留全部输出单元；spike 匹配容差 ±12 样本（±0.4 ms），pair 内一对一符合，再以 Hungarian 最大总 F1 做单元一对一分配，不设质量阈值，未分配输出 F1=0。GT 只用于事后评价。', '',
        '| 版本 | 单元 | 良好不应期单元 | Spike | 平均输出单元 F1 | 平均 GT F1 | F1≥0.9 单元 |','|---|---:|---:|---:|---:|---:|---:|']
    for name in ('before','after'):
        q=quality[name];lines.append(f"| {name} | {q['units']} | {runs[name][3]['good_units']} | {q['spikes']:,} | {q['mean_predicted_f1']:.6f} | {q['mean_gt_f1']:.6f} | {q['f1_ge_090']} |")
    lines+=['',
        f"两次独立运行在进入本轮优化的特征移位路径之前，输入已经不同：通用检测 spike {initial['before']['universal_spikes']:,} → {initial['after']['universal_spikes']:,}，初始聚类 {initial['before']['initial_clusters']} → {initial['after']['initial_clusters']}，learned 模板 {initial['before']['learned_templates']} → {initial['after']['learned_templates']}。本轮没有改动检测/聚类算法。该独立全流程的质量差异不能单独归因于本轮合并优化；固定相同合并输入的输出比较见前文。",
        '逐单元分配及排除统计：*_ranked_f1.csv、quality.json；完整计数矩阵：*_f1_pairs.npz。', '',
        '## 计时、热点与复核', '',
        '- before/after 均使用新进程完整运行一次并记录 cProfile；阶段边界同步 CUDA，嵌套阶段按所属大阶段计时，checkpoint IO 与诊断绘图计入 other。另有不含 cProfile 的共享输入合并对照。',
        f"- roll_features 热点：修改前 {roll_profile['before']['calls']} 次累计 {roll_profile['before']['cumulative_s']:.3f} 秒，修改后 {roll_profile['after']['calls']} 次累计 {roll_profile['after']['cumulative_s']:.3f} 秒。两次独立运行的合并次数可能因前序聚类波动而不同，同一输入的严格对照另列于上文。",
        '- cProfile 明细：before_profile.txt、after_profile.txt、profiles.json；原始 profile 在各自目录。',
        '- 硬件目标采样间隔 0.5 秒，受 GIL 等影响会延迟；均值按实际时间间隔加权，峰值是采样峰值。CPU/RSS 为 KS4 进程，GPU/显存/功耗为整卡，包含桌面及其他 GPU 进程。',
        '- 单次全流程差异包含缓存、预热、温度、后台负载和 KS4 初始聚类/等高峰的原有非确定性。未修改阶段的速度或最终质量微小变化不能单独归因于本轮优化；共享输入对照检查判据和输出是否变化。',
        '- 原始二进制和 ChronoSort/src 未修改。baseline/kilosort 保存本轮修改前源码；metadata.json 与 source_changes.patch 保存版本证据。', '',
        '![各阶段耗时](stage_times.png)', '', '![硬件占用](hardware.png)', '', '![排序 F1](ranked_f1.png)', '',
        '修改后完整 Phy 和 prior_bundle 输出：after/sorting/。']
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    main(parser.parse_args().root)
