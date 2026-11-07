"""Tiny real-process lifecycle checks; no research runtime or GPU packages."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_braking_microprobe.py"
spec = importlib.util.spec_from_file_location("c1_watchdog", SCRIPT)
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)


class WatchdogCPU(unittest.TestCase):
    def test_exit7_keeps_partial_and_never_retries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); output = root / "worker"; count = root / "count.txt"
            code = ("from pathlib import Path;import sys;"
                    f"p=Path({str(output)!r});p.mkdir();(p/'partial.txt').write_text('kept');"
                    f"Path({str(count)!r}).open('a').write('one\\n');print('cpu child only',flush=True);sys.exit(7)")
            result = watchdog.watch([sys.executable, "-c", code], output, root / "logs", hard_seconds=3)
            self.assertEqual(result["status"], "failed"); self.assertEqual(result["exit_code"], 7)
            self.assertEqual(count.read_text(), "one\n")
            self.assertTrue((output / "partial.txt").is_file())
            self.assertIn("cpu child only", (root / "logs/stdout.log").read_text())

    def test_timer_kills_real_sleeping_owned_tree_and_preserves_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); output = root / "worker"
            code = ("from pathlib import Path;import time,json,os;"
                    f"p=Path({str(output)!r});p.mkdir();"
                    "(p/'runtime.json').write_text(json.dumps({'pid':os.getpid(),'parent_pid':os.getppid()}));"
                    "(p/'partial.txt').write_text('kept');print('sleeping',flush=True);time.sleep(10)")
            result = watchdog.watch([sys.executable, "-c", code], output, root / "logs", hard_seconds=.6)
            self.assertEqual(result["status"], "timeout")
            self.assertTrue(result["timeout_details"]["cleanup"]["root_reaped"])
            self.assertLess(result["timeout_details"]["kill_requested_elapsed_s"], 1.2)
            self.assertLess(result["elapsed_s"], 4)
            self.assertTrue((output / "partial.txt").is_file())

    @unittest.skipUnless(os.name == "nt", "Windows shared-lock contract")
    def test_other_process_cannot_acquire_held_gpu_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "gpu.lock"
            code = ("import importlib.util,sys;from pathlib import Path;"
                    f"s=importlib.util.spec_from_file_location('w',{str(SCRIPT)!r});"
                    "w=importlib.util.module_from_spec(s);s.loader.exec_module(w);\n"
                    f"try:\n with w.gpu_lock(Path({str(lock)!r})):pass\n"
                    "except OSError:sys.exit(9)\nelse:sys.exit(0)\n")
            with watchdog.gpu_lock(lock):
                result = subprocess.run([sys.executable, "-c", code], timeout=4,
                                        capture_output=True, creationflags=watchdog.HIDDEN)
            self.assertEqual(result.returncode, 9, result.stderr)

    def test_default_preflight_only_passes_validate_flag_and_masks_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); worker = root / "braking_microprobe.py"
            worker.write_text("import os,sys,json\nassert '--validate-only' in sys.argv\n"
                              "assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'\n"
                              "print(json.dumps({'stdlib_only':True}))\n")
            sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            plan = {"process_wall_cap_seconds":240,"total_branches":54,"branch_duration_s":.2,
                    "training_updates":0,"m0":None,"a_brake":None,"parameter_enablement":False,
                    "automatic_retry":False,"source_pins":{
                        "runtime":{"path":str(worker),"sha256":sha(worker)},
                        "watchdog":{"path":str(SCRIPT),"sha256":sha(SCRIPT)}}}
            path = root / "plan.json"; path.write_text(json.dumps(plan))
            result = watchdog.preflight(path, worker, Path(sys.executable))
            self.assertEqual(result["status"], "passed"); self.assertFalse(result["GPU_started"])
            self.assertNotIn("torch", sys.modules)

    @unittest.skipUnless(os.name == "nt", "Windows taskkill failure contract")
    def test_taskkill_exception_falls_back_to_bounded_owned_root_kill(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); output = root / "worker"
            # Direct base interpreter has no venv redirector child. This proves
            # root convergence, while the production record correctly refuses
            # to infer that all possible descendants were terminated.
            python = getattr(sys, "_base_executable", sys.executable)
            code = "import time;print('cpu fault mock',flush=True);time.sleep(10)"
            fault = subprocess.TimeoutExpired(["taskkill"], 10)
            with mock.patch.object(watchdog.subprocess, "run", side_effect=fault):
                result = watchdog.watch([python, "-c", code], output, root / "logs", hard_seconds=.4)
            self.assertEqual(result["status"], "timeout_cleanup_unverified")
            cleanup = result["timeout_details"]["cleanup"]
            self.assertTrue(cleanup["root_reaped"])
            self.assertTrue(cleanup["used_Popen_kill_fallback"])
            self.assertFalse(cleanup["owned_tree_termination_confirmed"])
            self.assertIn("taskkill_error", cleanup)
            self.assertLess(result["elapsed_s"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
