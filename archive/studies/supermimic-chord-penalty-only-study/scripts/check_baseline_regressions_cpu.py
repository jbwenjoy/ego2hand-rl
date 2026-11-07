"""Run original22 geometry/objective +5 hook checks against vendored code."""
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest
from contextlib import redirect_stdout
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[1]
MAIN=Path('D:/workspace/supermimic-chord-study')
sys.path.insert(0,str(ROOT/'src'))
import torch
torch.set_num_threads(2)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    out=ROOT/'outputs/baseline_regressions_cpu_v3';out.mkdir(parents=True,exist_ok=False)
    suite=unittest.TestSuite()
    paths=[MAIN/'tests/test_contact_geometry.py',MAIN/'tests/test_contact_objective.py']
    for index,path in enumerate(paths):
        module=load(f'original_contact_tests_{index}',path)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
        (out/path.name).write_bytes(path.read_bytes())
    import chord_study.contact_objective as tested
    assert Path(tested.__file__).resolve().is_relative_to(ROOT/'src')
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    assert result.wasSuccessful() and result.testsRun==22 and not result.skipped,stream.getvalue()
    entry=load('penalty_training_entry',ROOT/'scripts/train_controlled_pen.py')
    hook=load('original_hook_tests',MAIN/'scripts/check_imagination_contact_hook.py')
    hook.train=entry
    saved=sys.argv
    try:
        sys.argv=[str(MAIN/'scripts/check_imagination_contact_hook.py'),'--output',str(out/'hook5')]
        with redirect_stdout(io.StringIO()):hook.main()
    finally:sys.argv=saved
    original_hook=json.loads((out/'hook5/checks.json').read_text())
    assert original_hook['passed'] and original_hook['tests_run']==5
    (out/'geometry_objective_tests.log').write_text(stream.getvalue())
    report={'status':'passed','original_geometry_objective_tests':22,'original_hook_tests':5,'total':27,'skipped':0,
            'vendored_module_used':tested.__file__,'training_entry_sha256':entry.digest(ROOT/'scripts/train_controlled_pen.py'),
            'original_test_source_sha256':{str(p):entry.digest(p) for p in paths},'CUDA_initialized_flag':torch.cuda.is_initialized(),
            'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'CUDA_device_count':torch.cuda.device_count(),'CUDA_available':torch.cuda.is_available(),
            'scope':'Original regression checks on unchanged vendor/native hook; separate P_ONLY tests exercise new wrapper'}
    assert report['CUDA_device_count']==0 and report['CUDA_available'] is False
    entry.write_json(out/'checks.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
