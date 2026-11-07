"""Pure precision/front contracts; no actual models, datasets, child or GPU."""
from __future__ import annotations
import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
MODULE_SHA = 'b292630b290dd2d05f2987c3f0156bb2d301968680e8282ae086f66fb498964f'


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True, type=Path)
    args = p.parse_args(); out = args.output.resolve()
    if out.exists(): raise ValueError('Fresh CPU proof output required')
    from full151_native.io import pin, sha
    sources = {'entry': ROOT / 'scripts/execute_full151_precision_v1.py',
               'supervisor': ROOT / 'scripts/run_full151_precision_v1.py',
               'module': ROOT / 'src/full151_gradient/precision105_v1.py',
               'checker': Path(__file__).resolve(),
               'analysis': ROOT / 'src/full151_gradient/analysis105_v1.py',
               'native_io': ROOT / 'src/full151_native/io.py',
               'native_metadata': ROOT / 'src/full151_native/metadata.py',
               'atomic_io': ROOT / 'vendor/runtime_io.py',
               'identity_driver': Path('D:/workspace/supermimic-observation-restoration-prefix-study/scripts/run_track_prefix_v3.py'),
               'lifecycle': Path('D:/workspace/supermimic-control-feasibility-draft/scripts/run_braking_microprobe.py')}
    if sha(sources['module']) != MODULE_SHA: raise ValueError('Root MODULE_FREEZE hash changed; do not execute a different source')
    before = {key: pin(path) for key, path in sources.items()}
    out.mkdir(parents=True); folder = out / 'executed_sources'; folder.mkdir()
    for i, (key, path) in enumerate(sources.items()): (folder / ('%02d_%s_%s' % (i, key, path.name))).write_bytes(path.read_bytes())
    (out / 'source_before.json').write_text(json.dumps(before, indent=2), encoding='utf-8')
    checks = {}
    def check(key, value):
        checks[key] = bool(value)
        if not checks[key]: raise AssertionError(key)
    def rejects(key, function):
        try: function()
        except (ValueError, IndexError, KeyError, RuntimeError): check(key, True)
        else: check(key, False)
    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path); value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value
    try:
        for path in sources.values(): ast.parse(path.read_text(encoding='utf-8-sig'))
        check('all_pinned_sources_AST_parse', True)
        from full151_gradient import precision105_v1 as core
        entry = module('precision_entry_CPU_fixture', sources['entry'])
        supervisor = module('precision_supervisor_CPU_fixture', sources['supervisor'])
        check('imports_noTorch_noWarp', not any(k in sys.modules for k in ('torch', 'warp', 'mujoco_warp')))
        cfg = dict(schema=core.SCHEMA, role=entry.ROLE, modes=list(core.MODES), allow_tf32=True, device='cuda:0',
                   evaluation_seed=0, GPU_run_authorized=False, training_authorized=False, Env_or_physics_authorized=False,
                   automatic_retry=False, compile_fallback=False, soft_worker_seconds=60, hard_process_seconds=90,
                   output=str(ROOT / 'outputs/_synthetic_precision_not_created'))
        entry.validate_front_metadata(cfg); core.validate_modes(cfg); check('exact_GPUfalse_namespace_positive', True)
        protocol = dict(schema='full151105_GPU_precision_preregistration:1',execution_authorized=False,
                        **{k:v for k,v in cfg.items() if k in ('modes','allow_tf32','device','evaluation_seed','soft_worker_seconds','hard_process_seconds','automatic_retry','compile_fallback','training_authorized','Env_or_physics_authorized')},
                        expected_counts={**entry.COUNTS,'batch_rows_per_forward':8,'Env_created':False,'compile_used':False})
        entry.validate_protocol_metadata(protocol);check('protocol_count_semantics_positive_synthetic',True)
        rejects('protocol_other_mode_rejected',lambda:entry.validate_protocol_metadata({**protocol,'modes':['other']}))
        rejects('protocol_forward_budget_rejected',lambda:entry.validate_protocol_metadata({**protocol,'expected_counts':{**protocol['expected_counts'],'WM_forward_calls':3}}))
        rejects('protocol_execution_authorized_rejected',lambda:entry.validate_protocol_metadata({**protocol,'execution_authorized':True}))
        for key, bad in [('schema', 'full151105_same_warp_physical_probe_run:2'), ('role', 'other'),
                         ('modes', list(reversed(core.MODES))), ('modes', [core.MODES[0]]),
                         ('allow_tf32', False), ('device', 'cpu'), ('evaluation_seed', True), ('evaluation_seed', 1),
                         ('GPU_run_authorized', 1), ('training_authorized', True), ('Env_or_physics_authorized', True),
                         ('automatic_retry', True), ('compile_fallback', True), ('soft_worker_seconds', 61), ('hard_process_seconds', 91)]:
            changed = {**cfg, key: bad}
            rejects('namespace_negative_%s_%s' % (key, str(bad)), lambda changed=changed: core.validate_modes(changed))
        rejects('outside_owned_output_rejected', lambda: entry.validate_front_metadata({**cfg, 'output': 'C:/outside_precision_output'}))
        rejects('float_budget_front_rejected', lambda: entry.validate_front_metadata({**cfg, 'soft_worker_seconds': 60.0}))
        # Closed form f(u)=b.u+.25||u||^2, analytic gradient b+.5u.
        centers = (np.arange(160).reshape(8,20) * .03125).astype(np.float32)
        b = np.arange(20) * .125
        gradients = (b[None] + .5 * centers.astype(float)).astype(np.float32)
        physical_rows = []; expected = []
        f = lambda u: float(b @ u + .25 * (u @ u))
        for s in range(8):
            for j, joint in enumerate((0,4,8,12)):
                for ei, eps in enumerate((.002,.001)):
                    c = centers[s]; plus = c.copy(); minus = c.copy()
                    plus[joint] += np.float32(eps); minus[joint] -= np.float32(eps)
                    clipped = (s+j+ei) % 3 == 0
                    if clipped: plus[joint] = c[joint] + np.float32(.25*eps)
                    d = (plus.astype(float)-minus.astype(float))/(2*eps)
                    offset = .5*(plus.astype(float)+minus.astype(float))-c.astype(float)
                    slope = (f(plus.astype(float))-f(minus.astype(float)))/(2*eps)
                    expected.append(float(gradients[s].astype(float) @ d))
                    physical_rows.append(dict(snapshot_id=s, joint_index=joint, epsilon_index=ei, epsilon_rad=eps,
                        d_eff=d.tolist(), center_offset=offset.tolist(), clipped=clipped, physical_slope=slope,
                        k=(0,49,99,149)[s%4], stable_nonflat=False, physical_label='SYNTHETIC_NO_QUALIFICATION', sentinel=len(physical_rows)))
                    check('quadratic_asymmetric_bias_%02d' % (len(physical_rows)-1), abs(slope-(expected[-1]+.5*d@offset))<1e-8)
        frozen_rows = copy.deepcopy(physical_rows); projected = core.projected_rows(gradients, physical_rows)
        check('all64_actual_d_eff_projections_exact', [v['WM_AD_effective'] for v in projected] == expected)
        check('physical_rows_not_mutated', physical_rows == frozen_rows)
        check('all64_physical_metadata_retained', all(all(row[k]==old[k] for k in old) for row,old in zip(projected,frozen_rows)))
        check('clip_effective_not_nominal', any(r['clipped'] and abs(r['WM_AD_effective']-r['WM_AD_nominal'])>.01 for r in projected))
        check('terminal149_rows_retained', sum(r['k']==149 for r in projected)==16)
        rejects('63_rows_rejected', lambda: core.projected_rows(gradients, physical_rows[:-1]))
        rejects('FP64_gradient_rejected', lambda: core.projected_rows(gradients.astype(float), physical_rows))
        rejects('nonfinite_gradient_rejected', lambda: core.projected_rows(np.full((8,20),np.nan,np.float32),physical_rows))
        rejects('wrong_gradient_shape_rejected', lambda: core.projected_rows(gradients[:7],physical_rows))
        summary = core.difference_summary([-2.,0.,4.], [0.,1.,0.])
        check('closedform_difference_max_mean_RMS', summary['maximum_absolute']==4 and abs(summary['mean_absolute']-7/3)<1e-15 and abs(summary['RMS']-np.sqrt(7))<1e-15)
        rejects('difference_shape_rejected',lambda:core.difference_summary([1,2],[1]))
        rejects('difference_nonfinite_rejected',lambda:core.difference_summary([np.nan],[1]))
        result = dict(schema='full151105_GPU_precision_diagnostic_report:1', status='COMPLETED_GPU_PRECISION_DIAGNOSTIC',
            modes=list(core.MODES), **entry.COUNTS, GPU_used=True, Env_created=False, compile_used=False,
            physical_gradient_qualified=False, whole_hidden_state_qualified=False,
            models={name:{} for name in core.MODES}, invariants={'synthetic_only':True})
        entry.validate_result(result); check('result_count_namespace_positive_synthetic',True)
        for key in entry.COUNTS:
            rejects('bad_actual_count_'+key,lambda key=key:entry.validate_result({**result,key:result[key]+1}))
        rejects('boolean_count_rejected',lambda:entry.validate_result({**result,'WM_forward_calls':True}))
        rejects('false_invariant_rejected',lambda:entry.validate_result({**result,'invariants':{'x':False}}))
        rejects('physical_qualification_upgrade_rejected',lambda:entry.validate_result({**result,'physical_gradient_qualified':True}))
        # Regex-only coverage does not query CIM or start any worker.
        for name in ('execute_full151_precision_v1','execute_full151_precision_v12','execute_full151_gradient_v2','execute_full151_learning_v3','execute_full151_policy_eval_v2'):
            check('GPU_worker_regex_'+name,re.search(supervisor.WORKER_PATTERN,'D:\\workspace\\'+name+'.py --run') is not None)
        check('worker_regex_rejects_backup',re.search(supervisor.WORKER_PATTERN,'D:/workspace/execute_full151_precision_v1.py.bak --run') is None)
        # Retained-process cleanup fixture: fake Popen object, never an OS child.
        fakecfg = out/'fake_preflight_config.json'
        fakecfg.write_text(json.dumps({'source_pins':{k:before[k] for k in ('entry','supervisor','identity_driver')}}),encoding='utf-8')
        proof = dict(status='CPU_INPUT_SOURCE_PREFLIGHT_ONLY',Torch_imported=False,Warp_imported=False,GPU_started=False,
                     config_sha256=sha(fakecfg),entry_sha256=sha(sources['entry']))
        stopped=[]; launches=[]
        class FakeProcess:
            pid=123456;returncode=0;stdout=None;stderr=None
            def communicate(self,timeout):
                if self.fail and timeout==20:raise subprocess.TimeoutExpired('synthetic_no_child',20)
                return json.dumps(proof),''
        original_popen=supervisor.subprocess.Popen;original_driver=supervisor.driver
        def fake_popen(*args,**kw):
            v=FakeProcess();v.fail=bool(launches);launches.append((args,kw,v));return v
        supervisor.subprocess.Popen=fake_popen
        supervisor.driver=lambda:SimpleNamespace(lifecycle=lambda:SimpleNamespace(stop_owned=lambda process:stopped.append(process) or {'status':'SYNTHETIC_OWNED_STOP'}))
        try:
            got=supervisor.preflight(fakecfg,Path(sys.executable));check('fake_defaultproof_positive',got==proof)
            rejects('fake_timeout_stops_owned_and_aborts',lambda:supervisor.preflight(fakecfg,Path(sys.executable)))
            check('timeout_cleanup_uses_exact_retained_object',len(stopped)==1 and stopped[0] is launches[1][2])
            check('CPUchild_visibility_forced_off_in_fixture',all(v[1]['env']['CUDA_VISIBLE_DEVICES']=='-1' for v in launches))
        finally:
            supervisor.subprocess.Popen=original_popen;supervisor.driver=original_driver
        check('fixture_process_hooks_restored',supervisor.subprocess.Popen is original_popen and supervisor.driver is original_driver)
        for key,path in sources.items():check('source_unchanged_'+key,pin(path)==before[key])
        check('final_noTorch_noWarp_noModel',not any(k in sys.modules for k in ('torch','warp','mujoco_warp')))
        report=dict(schema='full151105_GPU_precision_CPU_contract:1',status='PASS_PURE_CPU_CONTRACT_NO_MODEL',
            passed=sum(checks.values()),total=len(checks),checks=checks,source_pins=before,
            actual_model_loads=0,actual_model_forwards=0,actual_OS_children=0,physics_steps=0,GPU_used=False,
            scope='Pure namespaces/counts/quadraticd_eff/differencealgebra andfakeownedcleanup only. No actual inputmetadata/model/precision qualification.')
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        ready=dict(schema='full151105_GPU_precision_front_READY:1',status='READY_PURE_CPU_CONTRACT_ONLY',source_pins=before,
            cpu_report=pin(out/'report.json'),cpu_checks={'passed':report['passed'],'total':report['total']},
            api={'entry':str(sources['entry']),'supervisor':str(sources['supervisor']),'module':str(sources['module']),
                 'default':'--config CONFIG (no --run)','Root_authorized_only':'--config CONFIG --run'},
            actual_GPU_qualification='PENDING_NOT_RUN',actual_input_preflight='PENDING_ROOT_CONFIG')
        (out/'READY.json').write_text(json.dumps(ready,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps({'passed':report['passed'],'total':report['total'],'report':pin(out/'report.json'),'READY':pin(out/'READY.json')},indent=2))
    except BaseException as exc:
        (out/'failure.json').write_text(json.dumps({'error':repr(exc),'checks_completed':checks,'source_pins':before},indent=2,allow_nan=False),encoding='utf-8')
        raise


if __name__=='__main__':main()
