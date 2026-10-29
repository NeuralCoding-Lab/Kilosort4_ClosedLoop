"""Reproducible full-recording benchmark; run each variant in a fresh process."""
import argparse
import cProfile
import csv
import ctypes
import functools
import hashlib
import importlib
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time
import traceback

import numpy as np
import psutil
import torch


class Utilization(ctypes.Structure):
    _fields_ = [('gpu', ctypes.c_uint), ('memory', ctypes.c_uint)]


class Memory(ctypes.Structure):
    _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong),
                ('used', ctypes.c_ulonglong)]


class Recorder:
    def __init__(self, output):
        self.output = output
        self.stage = 'startup'
        self.events = []
        self.stop = threading.Event()
        self.process = psutil.Process()
        self.start = time.perf_counter()
        self.nvml = ctypes.CDLL('nvml.dll' if os.name == 'nt' else 'libnvidia-ml.so.1')
        assert self.nvml.nvmlInit_v2() == 0
        self.handle = ctypes.c_void_p()
        assert self.nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)) == 0
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        fields = ['elapsed_s', 'stage', 'cpu_process_percent', 'cpu_system_percent',
                  'rss_gib', 'system_used_gib', 'read_gib', 'write_gib',
                  'gpu_percent', 'gpu_memory_controller_percent', 'gpu_used_gib',
                  'gpu_power_w', 'gpu_temp_c']
        self.process.cpu_percent()
        psutil.cpu_percent()
        with (self.output / 'hardware.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            while not self.stop.is_set():
                util, memory = Utilization(), Memory()
                power, temp = ctypes.c_uint(), ctypes.c_uint()
                self.nvml.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(util))
                self.nvml.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(memory))
                self.nvml.nvmlDeviceGetPowerUsage(self.handle, ctypes.byref(power))
                self.nvml.nvmlDeviceGetTemperature(self.handle, 0, ctypes.byref(temp))
                io = self.process.io_counters()
                writer.writerow(dict(elapsed_s=time.perf_counter() - self.start,
                    stage=self.stage, cpu_process_percent=self.process.cpu_percent(),
                    cpu_system_percent=psutil.cpu_percent(), rss_gib=self.process.memory_info().rss / 2**30,
                    system_used_gib=psutil.virtual_memory().used / 2**30,
                    read_gib=io.read_bytes / 2**30, write_gib=io.write_bytes / 2**30,
                    gpu_percent=util.gpu, gpu_memory_controller_percent=util.memory,
                    gpu_used_gib=memory.used / 2**30, gpu_power_w=power.value / 1000,
                    gpu_temp_c=temp.value))
                stream.flush()
                self.stop.wait(.5)

    def wrap(self, module, name, stage):
        original = getattr(module, name)
        @functools.wraps(original)
        def measured(*args, **kwargs):
            previous = self.stage
            label = stage(*args, **kwargs) if callable(stage) else stage
            torch.cuda.synchronize()
            self.stage = label
            start = time.perf_counter()
            print(f'BENCH START {label}', flush=True)
            try:
                return original(*args, **kwargs)
            finally:
                torch.cuda.synchronize()
                event = dict(stage=label, start_s=start - self.start,
                    seconds=time.perf_counter() - start,
                    cuda_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                    cuda_peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
                self.events.append(event)
                with (self.output / 'stages.json').open('w') as stream:
                    json.dump(self.events, stream, indent=2)
                print(f'BENCH END {label} {event["seconds"]:.3f}s', flush=True)
                self.stage = previous
        setattr(module, name, measured)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('F:/kilosort_simu/25298815/sim_no_drift_30min'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--probe', choices=['ProbeA', 'ProbeB'])
    parser.add_argument('--tmax', type=float)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--checkpoint', action='store_true')
    parser.add_argument('--merge-checkpoint', action='store_true')
    parser.add_argument('--profile-merge', action='store_true')
    parser.add_argument('--profile-detection', action='store_true')
    args = parser.parse_args()
    if args.config and not args.probe:
        parser.error('--config requires --probe')
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source.resolve()))
    ks = importlib.import_module('kilosort.run_kilosort')
    if not Path(ks.__file__).resolve().is_relative_to(args.source.resolve()):
        raise RuntimeError('Loaded kilosort outside requested source directory')
    from kilosort import io, template_matching, spikedetect, clustering_qr
    torch.set_num_threads(args.threads)
    torch.cuda.init()
    config = json.loads(args.config.read_text(encoding='utf-8')) if args.config else None
    if config:
        settings = dict(config['settings'])
        filename = Path(config['recording_directory']) / 'continuous' / config['streams'][args.probe] / 'continuous.dat'
        run_options = dict(config['run_options'])
    else:
        settings = json.loads((args.data / 'ks4_settings.json').read_text())
        filename = args.data / 'sim.imec0.ap.bin'
        run_options = dict(probe=io.load_probe(args.data / 'probe.json'))
    if not filename.is_file():
        raise FileNotFoundError(filename)
    if args.tmax is not None:
        settings['tmax'] = args.tmax
    packages = {}
    for name in ('triton-windows', 'triton', 'numpy', 'scipy'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    metadata = dict(source=str(args.source.resolve()), kilosort_module=ks.__file__,
        packages=packages,
        instrumentation=dict(checkpoint=args.checkpoint, merge_checkpoint=args.merge_checkpoint,
                             profile_merge=args.profile_merge, profile_detection=args.profile_detection),
        filename=str(filename), run_options=run_options if config else {},
        requested_backends={k: os.environ.get(k, 'auto') for k in ('KS4_MATCHING_BACKEND', 'KS4_DETECTION_BACKEND')},
        python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), settings=settings, threads=args.threads,
        platform=platform.platform(), cpu=platform.processor(), logical_cpus=psutil.cpu_count(),
        physical_cpus=psutil.cpu_count(logical=False), ram_gib=psutil.virtual_memory().total / 2**30,
        source_hashes={str(p.relative_to(args.source)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (args.source / 'kilosort').rglob('*.py')})
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    recorder = Recorder(args.output)
    recorder.wrap(ks, 'compute_preprocessing', 'preprocessing')
    recorder.wrap(ks, 'compute_drift_correction', 'drift_correction')
    recorder.wrap(spikedetect, 'run', 'universal_detection')
    recorder.wrap(clustering_qr, 'run', lambda *a, **kw: 'initial_clustering' if kw.get('mode') == 'spikes' else 'final_clustering')
    recorder.wrap(template_matching, 'postprocess_templates', 'template_postprocessing')
    recorder.wrap(template_matching, 'extract', 'learned_matching')
    # The merger nested inside template_postprocessing gets its own label.
    recorder.wrap(template_matching, 'merging_function', lambda *a, **kw: 'initial_template_merge' if recorder.stage == 'template_postprocessing' else 'final_merge')
    recorder.wrap(ks, 'save_sorting', 'export')
    if args.profile_detection:
        measured_detection = spikedetect.run
        detection_call = 0
        original_match = spikedetect.template_match
        def instrument_detection(*values, **kwargs):
            nonlocal detection_call
            call = detection_call
            detection_call += 1
            saved = False
            def capture_match(X, ops, iC, iC2, weigh, **kw):
                nonlocal saved
                if not saved:
                    previous = recorder.stage
                    recorder.stage = 'checkpoint_io'
                    torch.save(dict(ops=ops, iC=iC, iC2=iC2, weigh=weigh),
                               args.output / f'detection_input_{call}.pt')
                    recorder.stage = previous
                    saved = True
                return original_match(X, ops, iC, iC2, weigh, **kw)
            spikedetect.template_match = capture_match
            profile = cProfile.Profile()
            try:
                result = profile.runcall(measured_detection, *values, **kwargs)
            finally:
                spikedetect.template_match = original_match
                profile.dump_stats(str(args.output / f'detection_{call}.pstats'))
            # Output serialization is outside the detection timer.
            previous = recorder.stage
            recorder.stage = 'checkpoint_io'
            if not saved:
                ops = result[2]
                torch.save(dict(ops=ops, iC=ops['iC'], iC2=ops['iC2'], weigh=ops['weigh']),
                           args.output / f'detection_input_{call}.pt')
            np.save(args.output / f'detection_st_{call}.npy', result[0])
            # Drift deliberately discards features; the optimized path may
            # therefore return None, while ordinary detection keeps all PCs.
            if result[1] is not None:
                np.save(args.output / f'detection_features_{call}.npy', result[1])
            recorder.stage = previous
            return result
        spikedetect.run = instrument_detection
    if args.merge_checkpoint or args.profile_merge:
        measured_merge = template_matching.merging_function
        def instrument_merge(*values, **kwargs):
            mode = kwargs.get('mode', values[6] if len(values) > 6 else 'ccg')
            if mode != 'ccg':
                return measured_merge(*values, **kwargs)
            if args.merge_checkpoint:
                previous = recorder.stage
                recorder.stage = 'checkpoint_io'
                torch.save(dict(ops=values[0], Wall=values[1], clu=values[2],
                                st=values[3], tF=values[4], kwargs=kwargs),
                           args.output / 'merge_input.pt')
                recorder.stage = previous
            if not args.profile_merge:
                return measured_merge(*values, **kwargs)
            profile = cProfile.Profile()
            try:
                return profile.runcall(measured_merge, *values, **kwargs)
            finally:
                profile.dump_stats(str(args.output / 'merge_profile.pstats'))
        template_matching.merging_function = instrument_merge
    if args.checkpoint:
        original = template_matching.extract
        def save_checkpoint(ops, bfile, U, **kwargs):
            recorder.stage = 'checkpoint_io'
            torch.save(dict(ops=ops, U=U), args.output / 'matching_input.pt')
            recorder.stage = 'startup'
            return original(ops, bfile, U, **kwargs)
        template_matching.extract = save_checkpoint
    recorder.thread.start()
    started = time.perf_counter()
    status = {}
    try:
        result = ks.run_kilosort(settings=settings, **run_options,
            filename=filename, results_dir=args.output / 'sorting',
            data_dtype='int16', device=torch.device('cuda'), torch_thread_lim=args.threads)
        ops, st, clu, tF, Wall, similar, is_ref, contamination, kept = result
        status = dict(success=True, spikes=int(kept.sum()), units=int(len(np.unique(clu))),
            good_units=int(is_ref.sum()), runtimes={k: v for k, v in ops.items() if k.startswith('runtime')},
            matching_stats=ops.get('matching_stats'),
            universal_detection_stats=ops.get('universal_detection_stats'))
        (args.output / 'effective_settings.json').write_text(
            json.dumps(ops['settings'], indent=2, default=str), encoding='utf-8')
    except BaseException:
        status = dict(success=False, error=traceback.format_exc())
        raise
    finally:
        status['wall_seconds'] = time.perf_counter() - started
        (args.output / 'result.json').write_text(json.dumps(status, indent=2, default=str))
        recorder.stop.set()
        recorder.thread.join()


if __name__ == '__main__':
    main()
