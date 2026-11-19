"""Registered eighteen fresh frozen-Actor continuous W1 executions."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tomllib

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_frozen_cost_long_design_v1 import settings,schedule,read
from supermimic_research.strict_json import write_json


def main(a):
    started=time.perf_counter(); cp=a.config.resolve(); cfg,paths,inputs,actors,contexts,q=settings(cp); admission=read(paths['admission'])
    if admission['passed']!=admission['total'] or not all(admission['checks'].values()) or admission['actors']!=actors or admission['contexts']!=contexts or admission['actual_native_integrations']!=0:
        raise ValueError('Complete unchanged zero-step actor admission required')
    verify_source_snapshot(read(verify_file_pin(admission['source_receipt'])),live=True); inputs['admission']=file_pin(paths['admission'])
    log=ROOT/'runs/pen_frozen_cost_tests_v1.log'; text=log.read_text(encoding='utf-8-sig')
    if 'Ran 644 tests' not in text or '\nOK (skipped=1)\n' not in text: raise ValueError('Full644 CPU suite required,one explicit CUDA skip')
    inputs['CPU_tests']=file_pin(log)
    sources=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'tools/review_pen_frozen_cost_long_v1.py',cp,paths['protocol'],ROOT/'tests/test_pen_continuous_math_view_v1.py',ROOT/'tests/test_pen_frozen_cost_design_v1.py']
    def freeze(folder):
        receipt=freeze_sources(sources,folder/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()); write_json(folder/'source_receipt.json',receipt); return receipt
    if a.register:
        folder=paths['registration'].parent
        if folder.exists(): raise ValueError('Fresh long evaluation registration required')
        folder.mkdir(); freeze(folder)
        write_json(paths['registration'],dict(schema='pen_frozen_cost_long_registration:1',config=file_pin(cp),inputs=inputs,actors=actors,contexts=contexts,identities=q['identities'],
            cases=schedule(),native_cap=52992,optimizer_budget=0,normalizer_fit_budget=0,source_receipt=file_pin(folder/'source_receipt.json')))
        print(json.dumps(dict(registration=file_pin(paths['registration']),cases=18,native_cap=52992)),flush=True); return
    reg=read(paths['registration']); rr=read(paths['reader_registration'])
    if reg['config']!=file_pin(cp) or reg['inputs']!=inputs or reg['actors']!=actors or reg['cases']!=schedule() or reg['identities']!=q['identities']: raise ValueError('Registered inputs changed')
    if rr['worker_registration']!=file_pin(paths['registration']) or rr['passed']!=rr['total'] or not all(rr['checks'].values()): raise ValueError('Pre-registered saved reader required')
    for r in [reg,rr]: verify_source_snapshot(read(verify_file_pin(r['source_receipt'])),live=True)
    out=paths['output']
    if out.exists() or shutil.disk_usage(ROOT).free<32*1024**3: raise ValueError('Fresh output and32GiB reserve required')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from mjlab.sim.sim import Simulation
    from supermimic_research.runtime.explicit_collision_backend import backend_scope
    from supermimic_research.runtime.pen_stable_friction_profile_v2 import CostProfile
    from supermimic_research.runtime.pen_frozen_cost_rollout_v1 import run_case
    from supermimic_research.runtime.full_rehash_scope import FullRehashVerifierScope,implementation_pins
    from supermimic_research.learning.pen_frozen_cost_actor_v1 import initialize
    from supermimic_research.learning.pen_matched_recovery import WorkCounter
    out.mkdir(); receipt=freeze(out); manifest=read(verify_file_pin(q['factory_manifest'])); counter=WorkCounter(); original=Simulation.step; recorder=None; runtime=None; rows=[]; stage='setup'; profiles=[]
    def activate(value):
        nonlocal recorder
        recorder=value
    def step(sim):
        if recorder is not None and recorder.enabled: recorder.before_step()
        counter.native_call(original,sim)
        if recorder is not None and recorder.enabled: recorder.after_step()
        if counter.native>52992: raise ValueError('Registered native cap exceeded')
    Simulation.step=step
    write_json(out/'intent.json',dict(registration=file_pin(paths['registration']),reader_registration=file_pin(paths['reader_registration']),cases=reg['cases'],native_cap=52992))
    try:
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True); torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        gates=tomllib.loads(paths['gates'].read_text()); objective=tomllib.loads(paths['objective_config'].read_text()); adapter=(paths['objective_config'].parent/objective['adapter_config']).resolve(); reward_cfg=tomllib.loads(adapter.read_text())['reward']
        with backend_scope(paths['backend_manifest']) as collision,torch.no_grad(),torch.jit.optimized_execution(False):
            for decl in reg['cases']:
                stage=decl['name']; folder=out/stage; before=counter.native
                with CostProfile(reg['identities'][decl['algorithm']],manifest) as profile:
                    try:
                        runtime=profile.create_runtime(paths['objective_config'],1)
                        if counter.native!=before: raise ValueError('Constructor advanced native physics')
                        state=torch.load(verify_file_pin(actors[decl['policy']]),map_location='cpu',weights_only=False)
                        actor,audit=initialize(runtime,state,actors[decl['policy']],contexts['SUPPORT' if decl['policy']=='SUPPORT' else 'RESIDUAL'],decl['policy'],profile)
                        with FullRehashVerifierScope('SINGLE_RESOLVE_FULL_REHASH',implementation_pins()) as verifier:
                            case=run_case(runtime,actor,decl,folder,receipt,counter,activate,profile,gates,reward_cfg)
                        write_json(folder/'actor_transfer.json',audit); write_json(folder/'verifier.json',verifier.audit())
                        rows.append(dict(declaration=decl,report=file_pin(folder/'report.json'),actor_transfer=file_pin(folder/'actor_transfer.json'),verifier=file_pin(folder/'verifier.json')))
                    finally:
                        activate(None)
                        if runtime is not None: runtime.close(); runtime=None
                write_json(folder/'profile.json',profile.audit); profiles.append(file_pin(folder/'profile.json'))
                write_json(out/'work.json',dict(stage=stage,completed_cases=len(rows),**counter.metadata()))
                print(json.dumps(dict(case=stage,controls=case['selected_controls'],duration_s=case['selected_duration_s'],valid_prefix_s=case['valid_prefix_s'],stop=case['stop_reason'],success=case['full_task_success'],**counter.metadata())),flush=True)
            checks=dict(all18=len(rows)==18,exact_schedule=[r['declaration'] for r in rows]==reg['cases'],native_exact=counter.native==counter.native_attempted==sum(read(verify_file_pin(r['report']))['native_integrations'] for r in rows)<=52992,
                no_updates=counter.optimizer==counter.optimizer_attempted==0,all_cases=all(read(verify_file_pin(r['report']))['passed']==read(verify_file_pin(r['report']))['total'] for r in rows),
                all_profiles=len(profiles)==18 and all(read(verify_file_pin(p))['bindings_restored'] for p in profiles),collision=collision.verify(cold=True))
        verify_source_snapshot(receipt,live=True)
        write_json(out/'report.json',dict(schema='pen_frozen_cost_long_execution:1',registration=file_pin(paths['registration']),reader_registration=file_pin(paths['reader_registration']),source_receipt=file_pin(out/'source_receipt.json'),
            rows=rows,profiles=profiles,checks=checks,passed=sum(checks.values()),total=len(checks),**counter.metadata(),normalizer_fits=0,wall_s=time.perf_counter()-started,scope='All18 frozen mean-policy executions retained;physical/task outcomes are measured,not execution gates. Separate saved review required;no new training/ego/W32/speed claim.'))
        print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[sum(checks.values()),len(checks)],**counter.metadata())),flush=True)
        if not all(checks.values()): raise ValueError('Final long execution accounting mismatch')
    except BaseException as e:
        write_json(out/'failure.json',dict(stage=stage,type=type(e).__name__,message=str(e),rows=rows,missing=[d for d in reg['cases'] if d['name'] not in {r['declaration']['name'] for r in rows}],**counter.metadata(),resumable=False)); raise
    finally: Simulation.step=original


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--register',action='store_true'); main(p.parse_args())
