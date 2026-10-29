"""Plan by default; --execute explicitly starts sequential real-data sorting."""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO / 'benchmarks/configs/260628_260409A05_recording1.json')
    parser.add_argument('--probe', choices=['ProbeA', 'ProbeB'], required=True)
    parser.add_argument('--output', type=Path, required=True, help='New experiment directory')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--tmax', type=float)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--validation', action='store_true', help='Separate checkpoint/profile run, not clean timing')
    args = parser.parse_args()
    if args.repeats < 1 or (args.tmax is not None and args.tmax <= 0):
        parser.error('repeats and tmax must be positive')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    recording = Path(config['recording_directory'])
    structure = json.loads((recording / 'structure.oebin').read_text(encoding='utf-8'))
    stream = next(s for s in structure['continuous'] if s['folder_name'].rstrip('/') == config['streams'][args.probe])
    settings = config['settings']
    if stream['num_channels'] != settings['n_chan_bin'] or stream['sample_rate'] != settings['fs']:
        raise ValueError('Configuration disagrees with Open Ephys channel count/sample rate')
    binary = recording / 'continuous' / config['streams'][args.probe] / 'continuous.dat'
    frame_bytes = 2 * settings['n_chan_bin']
    if binary.stat().st_size % frame_bytes:
        raise ValueError('Incomplete int16 frame in recording')
    duration = binary.stat().st_size / frame_bytes / settings['fs']
    if args.tmax is not None and args.tmax > duration:
        raise ValueError('tmax exceeds recording duration')
    baseline = subprocess.check_output(['git', 'rev-parse', config['baseline_ref'] + '^{commit}'], cwd=REPO, text=True).strip()
    after = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    # A separate snapshot per experiment also supports the existing replay scripts.
    sources = {'before': output / 'baseline', 'after': output / 'accelerated'}
    jobs = []
    for repeat in range(1, args.repeats + 1):
        pair = output / args.probe / f'repeat_{repeat:02d}'
        for variant in (('before', 'after') if repeat % 2 else ('after', 'before')):
            command = [sys.executable, str(REPO / 'benchmarks/benchmark_ks4.py'),
                       '--source', str(sources[variant]), '--output', str(pair / variant),
                       '--config', str(output / 'config.json'), '--probe', args.probe,
                       '--threads', str(config['threads'])]
            if args.tmax is not None:
                command += ['--tmax', str(args.tmax)]
            if args.validation:
                command += ['--checkpoint', '--merge-checkpoint', '--profile-detection']
            jobs.append(dict(variant=variant, pair=str(pair), command=command))
    plan = dict(baseline=baseline, after=after, binary=str(binary), duration_s=duration,
                tmax=args.tmax, validation=args.validation, config=config, jobs=jobs)
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'kilosort', 'setup.py', 'pyproject.toml', 'uv.lock'], cwd=REPO, text=True)
    if dirty.strip():
        raise RuntimeError('Commit sorting/environment changes before benchmarking a frozen revision')
    output.mkdir(parents=True)
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (output / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    for variant, revision in [('before', baseline), ('after', after)]:
        archive = subprocess.check_output(['git', 'archive', '--format=zip', revision], cwd=REPO)
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            bundle.extractall(sources[variant])
    for job in jobs:
        pair = Path(job['pair'])
        pair.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        # Require acceleration: an unavailable backend must fail, not silently fall back.
        backend = 'triton' if job['variant'] == 'after' else 'torch'
        env.update(KS4_MATCHING_BACKEND=backend, KS4_DETECTION_BACKEND=backend)
        with (pair / f"{job['variant']}_console.log").open('w', encoding='utf-8') as log:
            subprocess.run(job['command'], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    for pair in sorted({j['pair'] for j in jobs}):
        subprocess.run([sys.executable, str(REPO / 'benchmarks/summarize_ks4.py'), pair], check=True)


if __name__ == '__main__':
    main()
