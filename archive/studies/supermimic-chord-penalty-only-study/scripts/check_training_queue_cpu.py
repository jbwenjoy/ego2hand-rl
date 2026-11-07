"""Small CPU-only failure-path checks; never invokes the research trainer."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('queue_under_test',ROOT/'scripts/run_matched_training_queue.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
OUTPUT=ROOT/'outputs/replication_queue_preparation_v1'
OUTPUT.mkdir(exist_ok=True)

class QueueFailureChecks(unittest.TestCase):
    def setup_fixture(self, path):
        fake=path/'fake_CPU_child.py'
        fake.write_text("import json,sys\nfrom pathlib import Path\nc=json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text())\nPath(c['marker']).write_text('started')\nraise SystemExit(7)\n")
        jobs=[]
        for index in (1,2):
            cfg=path/f'config{index}.json';cfg.write_text(json.dumps({'marker':str(path/f'marker{index}')}))
            jobs.append({'position':index,'method':'A1L','seed':index,'config':str(cfg),'output':str(path/f'run{index}')})
        manifest={'queue_output':str(path/'queue'),'jobs':jobs,'python_executable':sys.executable,'trainer':str(fake),
                  'estimated_total_seconds_from_pilot_costs':1,'per_run_process_watchdog_seconds':10}
        p=path/'manifest.json';p.write_text(json.dumps(manifest));return p,manifest

    def test_real_CPU_child_failure_stops_before_next_job(self):
        with tempfile.TemporaryDirectory(prefix='controller_failure_',dir=OUTPUT) as temporary:
            path=Path(temporary).resolve();self.assertTrue(path.is_relative_to(OUTPUT.resolve()))
            manifest_path,manifest=self.setup_fixture(path)
            with patch.object(queue,'gpu_queue_lock',return_value=contextlib.nullcontext()),patch.object(queue,'validate',return_value={}),patch.object(queue,'reject_external_GPU_work'),patch.object(queue,'audit') as audit:
                with self.assertRaisesRegex(AssertionError,'Training exited 7'):
                    queue.execute(manifest_path,manifest)
            self.assertTrue((path/'marker1').exists());self.assertFalse((path/'marker2').exists());audit.assert_not_called()
            state=queue.read(path/'queue/state.json')
            self.assertEqual(state['status'],'failed_stopped_no_retry');self.assertEqual(len(state['jobs']),1)

    def test_lock_failure_is_recorded_without_launching_child(self):
        with tempfile.TemporaryDirectory(prefix='controller_startup_',dir=OUTPUT) as temporary:
            path=Path(temporary).resolve();self.assertTrue(path.is_relative_to(OUTPUT.resolve()))
            manifest_path,manifest=self.setup_fixture(path)
            with patch.object(queue,'validate',return_value={}),patch.object(queue,'gpu_queue_lock',side_effect=RuntimeError('deliberate lock failure')),patch.object(queue.subprocess,'Popen') as popen:
                with self.assertRaisesRegex(RuntimeError,'deliberate lock failure'):
                    queue.execute(manifest_path,manifest)
            popen.assert_not_called();self.assertEqual(queue.read(path/'queue/state.json')['status'],'failed_no_child_started')

    def test_CIM_identity_and_explicit_missing_status(self):
        payload={'pid':42,'parent_pid':17,'creation_date_utc':'2026-11-02T08:00:00Z','creation_date_local':'2026-11-02T16:00:00+08:00',
                 'command_line':'python D:\\study\\trainer.py --config D:\\study\\config.json','executable_path':'python.exe'}
        with patch.object(queue.subprocess,'run',return_value=SimpleNamespace(stdout=json.dumps(payload))):
            observed=queue.observe_actual_process(42,17,'D:/study/trainer.py','D:/study/config.json')
        self.assertEqual(observed['status'],'observed_exact_PID_and_identity')
        with patch.object(queue.subprocess,'run',side_effect=subprocess.TimeoutExpired('CIM',15)):
            missing=queue.observe_actual_process(42,17,'D:/study/trainer.py','D:/study/config.json')
        self.assertEqual(missing['status'],'missing_or_identity_unverified');self.assertNotIn('creation_date_utc',missing)

stream=io.StringIO()
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(QueueFailureChecks))
report={'status':'passed' if result.wasSuccessful() else 'failed','tests_run':result.testsRun,'log':stream.getvalue(),
        'scope':'Controller only: one real tiny CPU child exits7; no research trainer/simulator/GPU work',
        'queue_runner_sha256':queue.digest(ROOT/'scripts/run_matched_training_queue.py'),'torch_imported': 'torch' in sys.modules}
queue.write(OUTPUT/'controller_checks.json',report);print(json.dumps(report,indent=2))
raise SystemExit(0 if result.wasSuccessful() else 1)
