"""CPU-only harness checks; never execute Kilosort or GPU kernels."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


class ComparisonTests(unittest.TestCase):
    def test_plan_checks_real_metadata_without_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'not_created'
            result = subprocess.run([sys.executable, str(HERE / 'run_comparison.py'),
                '--probe', 'ProbeB', '--output', str(output), '--tmax', '60'],
                capture_output=True, text=True, check=True)
            plan = json.loads(result.stdout)
            self.assertFalse(output.exists())
            self.assertEqual([j['variant'] for j in plan['jobs']], ['before', 'after', 'after', 'before'])
            self.assertAlmostEqual(plan['duration_s'], 4572.3212)
            self.assertFalse(plan['validation'])

    def test_report_excludes_nested_stages_and_old_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for variant, seconds in [('before', 4), ('after', 2)]:
                folder = root / variant
                folder.mkdir()
                (folder / 'stages.json').write_text(json.dumps([
                    dict(stage='drift_correction', start_s=0.1, seconds=seconds),
                    dict(stage='universal_detection', start_s=0.2, seconds=0.1)]))
                (folder / 'result.json').write_text(json.dumps(dict(success=True,
                    wall_seconds=seconds+1, spikes=10, units=2, good_units=1)))
                (folder / 'metadata.json').write_text(json.dumps(dict(gpu='Test GPU',
                    threads=2, torch='test', filename='test.dat', settings={}, run_options={})))
                (folder / 'hardware.csv').write_text(
                    'elapsed_s,cpu_process_percent,cpu_system_percent,rss_gib,gpu_percent,gpu_memory_controller_percent,gpu_used_gib,gpu_power_w,gpu_temp_c\n'
                    '0.5,100,10,1,50,20,2,80,40\n')
            subprocess.run([sys.executable, str(HERE / 'summarize_ks4.py'), str(root)], check=True)
            rows = json.loads((root / 'comparison.json').read_text())['timing']
            self.assertEqual([r['stage'] for r in rows], ['drift_correction', 'other', 'total'])
            self.assertEqual(rows[0]['speedup'], 2)
            report = (root / 'report.md').read_text(encoding='utf-8')
            self.assertIn('Test GPU', report)
            self.assertNotIn('5090', report)
            self.assertNotIn('30 分钟', report)


if __name__ == '__main__':
    unittest.main()
