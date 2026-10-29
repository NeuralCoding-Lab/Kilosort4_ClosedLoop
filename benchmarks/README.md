# KS4 matching acceleration

The learned-template extraction uses the optional Triton backend on CUDA when
Triton and `torch.nonzero_static` are available. CPU and installations without
those dependencies retain the original PyTorch path. Set
`KS4_MATCHING_BACKEND=torch` to force the original implementation, or `triton`
to require the accelerated backend. No sorting parameters are changed.

The implementation follows the local ChronoSort matcher: exact nonzero template
pair packing, cached waveform reconstruction, fused score/reduction and residual
updates, dirty-time scoring after eight peels, static candidates, four-peel host
checks, and reusable CUDA graphs. Candidate overflow retries from the untouched
input; adjacent tied peaks use the original KS4 implementation to avoid racing
overlapping writes. All nonzero lag correlations are retained without a cutoff.

CCG refractory estimation groups spike times once. Final template merging reuses
those per-unit trains and updates only the absorbed train, including the original
sample shift. The CCG kernel, thresholds and greedy merge order are unchanged.

`ks4_speed_20261027/baseline/kilosort` is a source snapshot taken before edits.
Run the two variants sequentially in fresh Python processes:

```powershell
$py = 'C:\Users\laohu\anaconda3\envs\kilosort_sort\python.exe'
& $py -u benchmarks/benchmark_ks4.py --source benchmarks/ks4_speed_20261027/baseline --output benchmarks/ks4_speed_20261027/before --checkpoint
& $py -m pytest tests/test_matching_acceleration.py -q
& $py -m pytest tests/test_ccg_grouping.py -q
& $py -u benchmarks/validate_matching.py benchmarks/ks4_speed_20261027/before/matching_input.pt benchmarks/ks4_speed_20261027/matching_validation.json
& $py -u benchmarks/benchmark_ks4.py --source . --output benchmarks/ks4_speed_20261027/after --checkpoint
& $py benchmarks/summarize_ks4.py benchmarks/ks4_speed_20261027
& $py benchmarks/evaluate_ks4.py benchmarks/ks4_speed_20261027
& $py -u benchmarks/replay_matching.py benchmarks/ks4_speed_20261027
```

The output folders must be new; use different names for subsequent experiments.
The full pipeline retains the dataset's settings and all default stages including
drift correction, plots and export. Six identical preprocessed real batches are
also compared against the original equations, then timed three times each with
alternating order. Graph compilation/build time is included in full-run timing
and is separated from steady-state batch timing.

Hardware telemetry uses NVML through ctypes and psutil every 0.5 seconds.
NVML values cover the entire device, including existing desktop applications;
CPU utilization and RSS cover the benchmark process. The process CPU percentage
uses 100% per logical core. Stage boundaries synchronize CUDA. Nested operations
are removed from stage sums to avoid double-counting. Single-run differences in
unchanged stages may reflect caches, thermal conditions and background load.
The sampling thread can be delayed by NumPy/Numba holding the GIL; summaries use
actual interval weights and report sampled peaks, with raw timestamps retained.

Independent full KS4 runs can produce different initial cluster banks even with
identical preprocessing and seeds. The full matching replays reuse the same
baseline checkpoint and identical preprocessed input over every batch, alternating
matcher order, and compare all spike-table and feature entries separately from
the two independent end-to-end sorting runs. Paired kernel timing excludes the
common preprocessing and comparison work; the full-run matching stage includes
preprocessing, feature extraction, host transfer and output array assembly.
Original KS4 can produce different results on identical inputs when three or
more adjacent tied peaks cause overlapping indexed writes. Such batches retain
the original fallback and are recorded separately; strict equality checks apply
to the non-tied path. The batch-660 ten-repeat control demonstrates the original
implementation's own variation (`legacy_tied_repeatability.json`).

## CPU merge follow-up

`ks4_cpu_merge_20261027/baseline` freezes the previous accelerated version.
The principal hotspot was `roll_features`: NumPy boolean masks were repeatedly
used to index PyTorch features for the entire recording. Resolve the mask once
to integer rows and reuse a native Torch index for reading and writing. The
waveform reconstruction, shift, edge filling and projection formulas are kept.
This follow-up also reuses the host scores already transferred for sorting to
avoid CUDA scalar comparisons and synchronization per merge candidate. The host
threshold is cast to the score dtype to preserve PyTorch comparison semantics.
No CCG equations, thresholds, candidate order, averaging or shifts are changed.

Full before/after runs use `benchmark_ks4.py --checkpoint --merge-checkpoint
--profile-merge`; keep their source roots separate. Both use the same settings,
data and 16-thread limit as above. Profile files and the pre-merge checkpoint are
saved alongside the complete sorting. Replay both implementations with one
checkpoint in fresh processes:

```powershell
& $py benchmarks/replay_merge.py --source benchmarks/ks4_cpu_merge_20261027/baseline --checkpoint benchmarks/ks4_cpu_merge_20261027/before/merge_input.pt --output benchmarks/ks4_cpu_merge_20261027/replay_before
& $py benchmarks/replay_merge.py --source . --checkpoint benchmarks/ks4_cpu_merge_20261027/before/merge_input.pt --output benchmarks/ks4_cpu_merge_20261027/replay_after
```

Each replay saves every CCG call's order, decision and R12, plus full-output
SHA256 hashes including tF. This validates the actual merge criterion and output
independently of the earlier sorting stages' nondeterminism. Quality evaluation
uses `evaluate_ks4.py`; the follow-up report uses `summarize_cpu_merge.py`.

## Drift and universal detection follow-up

`ks4_detection_20261027/baseline` freezes the preceding matching/CPU-merge
optimizations. Universal detection keeps the original conv1d and all 40 spatial
einsum chunks, including their channel reduction. Optional Triton kernels read
the untransposed scores directly, select the first size/waveform on ties and
compute the same nearest-template maximum without materializing its large
gather. Temporal pooling, strict amplitude threshold and equality-based peak
selection remain unchanged. A per-run CUDA Graph reuses the fixed workspace;
Graph build/warm-up and cache loading are included in end-to-end timings;
correctness tests warm the Triton disk cache before the after benchmark, so
first-install cold compilation is not represented. Set
`KS4_DETECTION_BACKEND=torch` to use the original detector or `triton` to require
the optional accelerator; default `auto` retains CPU/no-Triton fallback.

Drift binning locates batch spans once (stable sorting only for unordered input).
Depth/amplitude bins and sparse log histogram are unchanged. Alignment transfers
the same individually computed correlation vectors once per iteration rather
than once per candidate shift. Iterations, reductions, smoothing, interpolation
and argmax rules are unchanged. Drift detection still computes PC features, so
the public detection return contract is preserved.

```powershell
& $py benchmarks/benchmark_ks4.py --source benchmarks/ks4_detection_20261027/baseline --output benchmarks/ks4_detection_20261027/before --profile-detection
& $py -m pytest tests/test_universal_detection.py tests/test_drift_memory.py -q
& $py benchmarks/replay_detection.py benchmarks/ks4_detection_20261027
& $py benchmarks/profile_detection_gpu.py benchmarks/ks4_detection_20261027
& $py benchmarks/benchmark_ks4.py --source . --output benchmarks/ks4_detection_20261027/after --profile-detection
& $py benchmarks/evaluate_ks4.py benchmarks/ks4_detection_20261027
& $py benchmarks/summarize_detection.py benchmarks/ks4_detection_20261027
```

The detection profiler saves the two template/geometric states and complete
detected tables/features. Serialization after detection is outside its timer;
the drift parent includes the first dump in both variants. The full replay
alternates execution order on one identical preprocessed batch over both passes
(900 + 900 batches), checks every event, amplitude, depth and PC feature exactly,
then compares complete drift histograms, shifts and fingerprints. CUDA profiling
and correctness replays are separate from full benchmark runs to avoid resource
contention. Full-pipeline GT quality is measured independently over all 30 minutes.
