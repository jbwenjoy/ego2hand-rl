"""Execute every fixed available model suggestion from its original W1 state."""
import argparse
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.learning.pen_matched_recovery import WorkCounter
from supermimic_research.runtime.explicit_collision_backend import backend_scope,load_contract
from supermimic_research.runtime.backend_pen_runtime import BackendSupportPenRuntime
from supermimic_research.runtime.backend_pen_probe import restore_backend_probe,require_backend_probe
from supermimic_research.runtime.pen_owned_probe_recorder import OwnedProbeRecorder
from supermimic_research.runtime.full_rehash_scope import FullRehashVerifierScope,implementation_pins


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(args):
    started=time.perf_counter();cp=args.config.resolve();cfg=tomllib.loads(cp.read_text())
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['backend_manifest','objective_config','registration','bank_output','output','protocol']}
    reg=read(paths['registration']);bank_path=paths['bank_output']/'report.json';bank=read(bank_path);contract=load_contract(paths['backend_manifest'])
    if (cfg['schema']!=1 or cfg['mode']!='FIXED_ALL_PATH_MODELS_NATIVE_ONE_STEP' or cfg['native_budget_max']!=12000 or cfg['verifier']!='SINGLE_RESOLVE_FULL_REHASH'
        or reg['config']!=file_pin(cp) or bank['config']!=reg['config'] or bank['inputs']!=reg['inputs'] or bank['registration']!=file_pin(paths['registration'])
        or bank['schema']!='pen_frozen_native_suggestion_bank:1' or bank['passed']!=bank['total'] or not all(bank['checks'].values())
        or reg['implementation_pins']!=implementation_pins() or reg['inputs']['backend_manifest']!=file_pin(paths['backend_manifest'])
        or reg['inputs']['objective_config']!=file_pin(paths['objective_config'])):raise ValueError('Exact registered complete fixed bank/backend required')
    for r in [reg,bank]:verify_source_snapshot(read(verify_file_pin(r['source_receipt'])),live=True)
    out=paths['output']
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh native suggestion output required')
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'tools/prepare_pen_native_suggestions.py',cp,paths['protocol']],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt)
    write(out/'intent.json',dict(bank=file_pin(bank_path),registration=file_pin(paths['registration']),declared_controls=1800,available_controls=bank['available_controls'],native_budget_max=12000,optimizer_budget=0,default_changed=False,source_epoch_used=False))
    from mjlab.sim.sim import Simulation
    original=Simulation.step;counter=WorkCounter();runtime=recorder=None;attempted=completed=executed=0;cases=[];scopes=[];stage='construction'
    def step(sim):
        if recorder is not None and recorder.enabled:recorder.before_step()
        value=counter.native_call(original,sim)
        if recorder is not None and recorder.enabled:recorder.after_step()
        return value
    Simulation.step=step
    def restore(source):
        nonlocal attempted,completed
        attempted+=1;audit=restore_backend_probe(runtime,source)
        if not audit['all_array_bytes_equal'] or not audit['all_scalar_values_equal'] or any(not torch.equal(v.cpu(),source['observation'][key]) for key,v in runtime.observation().items()):raise ValueError('Complete source/observation restore mismatch')
        completed+=1;return audit
    def progress():
        elapsed=time.perf_counter()-started;remaining=(bank['available_controls']-executed)*elapsed/executed if executed else None
        value=dict(stage=stage,declared_records=len(cases),executed_controls=executed,available_controls=bank['available_controls'],restore_attempts=attempted,restore_completed=completed,
            wall_s=elapsed,remaining_wall_estimate_s=remaining,**counter.metadata());write(out/'progress.json',value);print(json.dumps(value),flush=True)
    try:
        torch.set_num_threads(4)
        for entry in bank['sources']:
            if not entry['observed']:
                for p in entry['profiles']:cases.append(dict(name=entry['name']+'_'+p['name'],source_name=entry['name'],role=entry['native_role'],cohort=entry['role'],available=False,qualified=False,suggestion=p,reason=p['reason']))
        with backend_scope(paths['backend_manifest']) as backend,torch.no_grad(),torch.jit.optimized_execution(False):
            try:
                for role in ['TRAIN','VALIDATE']:
                    for actor in ['RESIDUAL','SUPPORT']:
                        selected=[s for s in bank['sources'] if s['observed'] and s['role']==role and s['original_source']['actor']==actor]
                        runtime=BackendSupportPenRuntime(paths['objective_config'],1);runtime.set_fixed_frame_features(False);runtime.support_weight=float(runtime.objective_cfg['support_weights'][actor]);recorder=OwnedProbeRecorder(runtime)
                        for entry in selected:
                            source=torch.load(verify_file_pin(entry['original_source']['source']),map_location='cpu',weights_only=False);require_backend_probe(source,contract);template=entry['template']
                            scope=FullRehashVerifierScope(cfg['verifier'],reg['implementation_pins']);source_started=time.perf_counter()
                            with scope:
                                stage=entry['name']+'_WARMUP';restore(source)
                                for p in entry['profiles']:
                                    stage=entry['name']+'_'+p['name']
                                    if not p['available']:
                                        cases.append(dict(name=stage,source_name=entry['name'],role=entry['native_role'],cohort=role,available=False,qualified=False,suggestion=p,reason=p['reason']));continue
                                    folder=out/entry['name']/p['name'];folder.mkdir(parents=True);begin=time.perf_counter();audit=restore(source)
                                    raw=np.asarray(p['raw20'],np.float32);recorder.begin(raw);native_before=counter.native
                                    _,reward,term,trunc,info=runtime.step(torch.tensor(raw,device=runtime.device)[None]);z=recorder.finish();w={k:v.cpu().numpy().copy()[0] for k,v in runtime.contact_views.items()}
                                    actual=z['post_ctrl'][0,template['aids']];error=float(abs(actual-np.asarray(p['expected_PD20'])).max())
                                    tracking=np.float32(w['objective_tracking_reward']);geometry=np.float32(-np.float32(.25)*np.float32(w['objective_geometry_loss'])-np.float32(2.)*np.float32(w['objective_geometry_failure']));residual=np.float32(tracking+geometry)
                                    full_scores=np.asarray([residual,np.float32(residual+np.float32(.2)*np.float32(w['objective_support_bonus']))],np.float64);source_reward=float(reward[0]);expected_reward=full_scores[0 if actor=='RESIDUAL' else 1]
                                    quality=dict(native_count=counter.native-native_before==8,held_PD=bool(np.array_equal(z['post_ctrl'][:,template['aids']],np.tile(actual,(8,1)))),PD=error<=2e-7,
                                        raw=bool(np.array_equal(z['actions20'][0],raw)),eight_samples=int(w['counts'][0])==8,geometry=bool(np.all(w['geometry_max']<=[.002,.0002,.001])),
                                        nonhand=float(w['nonhand_max_sum'][0])<=.1,no_metric_error=not bool(w['errors'].any() or w['global_error'].any()),
                                        finite=all(np.isfinite(v).all() for k,v in z.items() if not k.startswith('legacy_host_')),no_aid=not np.any(z['post_qfrc_applied']) and not np.any(z['post_xfrc_applied']),source_reward=abs(source_reward-expected_reward)<=2e-6)
                                    np.savez_compressed(folder/'trace.npz',**z);np.savez_compressed(folder/'window.npz',**w)
                                    case=dict(template,name=stage,original_name=template['name'],profile=p,source_name=entry['name'],role=entry['native_role'],cohort=role,available=True,
                                        suggestion=p,trace=file_pin(folder/'trace.npz'),window=file_pin(folder/'window.npz'),restore_audit=audit,quality=quality,qualified=all(quality.values()),
                                        actual_PD20=actual.tolist(),PD_error_rad=error,native_integrations=counter.native-native_before,source_reward=source_reward,full_scores=full_scores.tolist(),
                                        original_base_physical_valid=entry['original_base_physical_valid'],terminated=bool(term[0]),truncated=bool(trunc[0]),termination_reasons={k:bool(v[0]) for k,v in info['termination_reasons'].items()},branch_wall_s=time.perf_counter()-begin)
                                    cases.append(case);write(folder/'report.json',case);executed+=1
                                    if executed%10==0:progress()
                            scopes.append(dict(source_name=entry['name'],role=role,wall_s=time.perf_counter()-source_started,verifier=scope.audit()))
                            if not scope.validated:raise ValueError('Verifier scope failed to restore original bindings')
                            progress()
                        runtime.close();runtime=recorder=None
                checks=dict(all1800_declared=len(cases)==1800,all_available_executed=executed==bank['available_controls']<=1500,
                    exact_native_work=counter.native==counter.native_attempted==8*executed<=12000,
                    exact_restores=attempted==completed==executed+30,
                    all30_scopes=len(scopes)==30 and all(s['verifier']['validated'] and s['verifier']['bindings_restored'] and s['verifier']['default_unchanged'] and not s['verifier']['source_epoch_used'] for s in scopes),
                    all_records_preserved=len({c['name'] for c in cases})==1800,
                    no_optimizer=counter.optimizer==counter.optimizer_attempted==0,
                    all_executed_complete=all(c['native_integrations']==8 and c['restore_audit']['all_array_bytes_equal'] and c['restore_audit']['all_scalar_values_equal'] for c in cases if c['available']))
                backend.verify(cold=True);verify_source_snapshot(receipt,live=True)
                result=dict(schema='pen_model_suggestion_native_probes:1',config=file_pin(cp),registration=file_pin(paths['registration']),bank=file_pin(bank_path),source_receipt=file_pin(out/'source_receipt.json'),
                    cases=cases,source_scopes=scopes,checks=checks,passed=sum(checks.values()),total=len(checks),executed_controls=executed,declared_controls=1800,restore_attempts=attempted,restore_completed=completed,
                    **counter.metadata(),backend_audit=backend.audit(),wall_s=time.perf_counter()-started,online_failed_controls=sum(not c['qualified'] for c in cases if c['available']),default_changed=False,source_epoch_used=False,
                    scope='All fixed available single-step suggestions and reverse/base controls recorded;independent original physics/reward/benefit review REQUIRED. Online failures retained. No Actor/trajectory/ego/task/speedup admission.')
                write(out/'report.json',result)
                if not all(checks.values()):raise ValueError('Native suggestion work/coverage discrepancy retained')
                print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],executed=executed,**counter.metadata())),flush=True)
            except Exception as e:
                if recorder is not None and recorder.enabled:
                    try:np.savez_compressed(out/'failure_partial_NOT_RESUMABLE.npz',**recorder.finish())
                    except Exception:pass
                write(out/'failure.json',dict(type=type(e).__name__,message=str(e),stage=stage,cases=cases,restore_attempts=attempted,restore_completed=completed,executed_controls=executed,**counter.metadata(),resumable=False));raise
            finally:
                for rt in list(backend.runtime_set):rt.close()
                runtime=recorder=None
    finally:Simulation.step=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);main(p.parse_args())
