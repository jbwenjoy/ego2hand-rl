"""Explicit scoped adapter around immutable v1 orchestration; no physics edits."""
from contextlib import contextmanager
import argparse
import json
from pathlib import Path
import runpy
import shutil
import sys
import tomllib

from ..artifacts import file_pin,verify_file_pin
from ..frozen_sources import verify_source_snapshot
from ..strict_json import write_json
from . import pen_solve_exit_observer as observer_module
from ..mechanics import pen_solve_exit_epoch as epoch_module
from .pen_solve_exit_observer_v2 import SolveExitGraphObserverV2
from ..mechanics.pen_solve_exit_epoch_v2 import review as review_v2,physical_view


@contextmanager
def worker_bindings():
    old_class,old_review=observer_module.SolveExitGraphObserver,epoch_module.review
    if old_class.__name__!='SolveExitGraphObserver' or old_review.__module__!='supermimic_research.mechanics.pen_solve_exit_epoch':
        raise ValueError('Unmodified v1 adapter bindings required')
    observer_module.SolveExitGraphObserver=SolveExitGraphObserverV2;epoch_module.review=review_v2
    audit=dict(implementation='PADDED_SOLVER_CONTEXT_V2',observer=SolveExitGraphObserverV2.__module__+'.'+SolveExitGraphObserverV2.__qualname__,epoch=review_v2.__module__+'.'+review_v2.__qualname__,entry_bindings_verified=True)
    try:yield audit
    finally:
        same=observer_module.SolveExitGraphObserver is SolveExitGraphObserverV2 and epoch_module.review is review_v2
        observer_module.SolveExitGraphObserver=old_class;epoch_module.review=old_review
        audit.update(exit_bindings_verified=same,bindings_restored=observer_module.SolveExitGraphObserver is old_class and epoch_module.review is old_review)
        if not same:raise ValueError('V2 bindings changed during explicit scope')


def run_cli(kind,entrypoint):
    root=Path(__file__).resolve().parents[3]
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--replica');parser.add_argument('--register',action='store_true');parser.add_argument('--check-ready',action='store_true');a,_=parser.parse_known_args()
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text())
    if cfg.get('implementation')!='PADDED_SOLVER_CONTEXT_V2':raise ValueError('Explicit V2 config required')
    layout_path=(cp.parent/cfg['context_layout_preflight']).resolve();tests_path=(cp.parent/cfg['layout_tests']).resolve();layout=json.loads(layout_path.read_text())
    if layout['passed']!=layout['total'] or layout['total']!=52 or not all(layout['checks'].values()):raise ValueError('Actual ten-source factory preflight required')
    verify_source_snapshot(json.loads(verify_file_pin(layout['source_receipt']).read_text()),live=True)
    if 'Ran 591 tests' not in tests_path.read_text() or '\nOK\n' not in tests_path.read_text():raise ValueError('V2 full CPU tests required')
    target=(cp.parent/cfg['registration']).resolve().parent if a.register else (cp.parent/cfg['output']).resolve()/str(a.replica)
    existed=target.exists();entrypin=file_pin(entrypoint);audit={};error=None
    def attest_worker(folder,worker_config):
        manifest=json.loads((folder/'context_layout_adapter.json').read_text())
        b=manifest['bindings']
        if (manifest['kind']!='worker' or manifest['config']!=file_pin(worker_config)
                or manifest['body_error'] is not None or b['implementation']!='PADDED_SOLVER_CONTEXT_V2'
                or not all(b[k] for k in ['entry_bindings_verified','exit_bindings_verified','bindings_restored'])):
            raise ValueError('Successful explicit V2 worker attestation required')
        for key in ['entrypoint','entrypoint_snapshot','adapter_source']:verify_file_pin(manifest[key])
        if manifest['layout_preflight']!=file_pin(layout_path) or manifest['layout_tests']!=file_pin(tests_path):raise ValueError('Worker layout evidence changed')
    try:
        if kind=='worker':
            with worker_bindings() as audit:
                runpy.run_path(str(root/'tools/run_pen_solve_exit_replay.py'),run_name='__main__')
        elif kind=='reader':
            wc=(cp.parent/cfg['worker_config']).resolve();wr=(cp.parent/cfg['worker_registration']).resolve()
            attest_worker(wr.parent,wc)
            if not a.register:
                worker=tomllib.loads(wc.read_text());producer=(wc.parent/worker['output']).resolve()/str(a.replica)
                if (producer/'report.json').exists():attest_worker(producer,wc)
            namespace=runpy.run_path(str(root/'tools/review_pen_solve_exit_saved.py'));main=namespace['main'];globals_=main.__globals__;old=globals_['independent_arrays']
            def independent(z,o,ma,c,gd,decoded,cap):return old(z,physical_view(o,z['post_qvel'].shape[-1]),ma,c,gd,decoded,cap)
            globals_['independent_arrays']=independent
            try:
                audit=dict(implementation='PADDED_SOLVER_CONTEXT_V2',physical_gradient_prefix=True,full_raw_copies_preserved=True)
                main(argparse.Namespace(config=cp,replica=a.replica,register=a.register))
            finally:
                globals_['independent_arrays']=old;audit['bindings_restored']=globals_['independent_arrays'] is old
        else:raise ValueError('Declared worker/reader adapter required')
    except BaseException as caught:
        error=dict(type=type(caught).__name__,message=str(caught));raise
    finally:
        if not existed and target.exists():
            verify_file_pin(entrypin);snap=target/'adapter_source_snapshot';snap.mkdir();copy=snap/Path(entrypoint).name;shutil.copyfile(entrypoint,copy)
            if file_pin(copy)['sha256']!=entrypin['sha256']:raise ValueError('Entry point changed while copying')
            write_json(target/'context_layout_adapter.json',dict(schema='solve_exit_layout_adapter:2',kind=kind,config=file_pin(cp),entrypoint=entrypin,entrypoint_snapshot=file_pin(copy),adapter_source=file_pin(__file__),layout_preflight=file_pin(layout_path),layout_tests=file_pin(tests_path),bindings=audit,body_error=error,scope='Version2 raw padded/empty buffers and physical gradient prefix only;old orchestration source preserved,installed solver/physics unchanged.'))
