"""OriginalA/candidate/originalB native branches with full per-call source hashes."""
import argparse
import copy
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import warp as wp
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.learning.pen_matched_recovery import WorkCounter
from supermimic_research.learning.world_model import raw_tree_equal
from supermimic_research.runtime.explicit_collision_backend import backend_scope,load_contract
from supermimic_research.runtime.backend_pen_runtime import BackendSupportPenRuntime
from supermimic_research.runtime.backend_pen_probe import restore_backend_probe,require_backend_probe
from supermimic_research.runtime.pen_owned_probe_recorder import OwnedProbeRecorder,perturbed_actor_raw
from supermimic_research.runtime.full_rehash_scope import FullRehashVerifierScope,implementation_pins
from supermimic_research.runtime import owned_pen_state as native
from supermimic_research.runtime.owned_array_writes import Write


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(args):
    started=time.perf_counter();cp=args.config.resolve();cfg=tomllib.loads(cp.read_text())
    if (cfg['schema'],cfg['rounds'],cfg['sources'],cfg['modes'],cfg['native_budget'])!=(1,3,['RESIDUAL_N8841_P024','SUPPORT_N8841_P024'],['ORIGINAL_A','SINGLE_RESOLVE','ORIGINAL_B'],1440):
        raise ValueError('Fixed two-source x3-round x3-mode native benchmark required')
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['probe','objective_config','backend_manifest','protocol','registration','output']}
    inputs={k:file_pin(p) for k,p in paths.items() if k not in ['registration','output']};run=read(paths['probe']);contract=load_contract(paths['backend_manifest'])
    if run['schema']!='backend_controlled_response_probe:1' or run['collision_backend']!=contract or run['passed']!=run['total'] or not all(run['checks'].values()):raise ValueError('Completed same-backend original probe required')
    selected=[next(s for s in run['sources'] if s['name']==name and s['qualified']) for name in cfg['sources']];pins=implementation_pins()
    source_paths=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,paths['protocol']]
    def freeze(out):
        receipt=freeze_sources(source_paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt);return receipt
    if args.register:
        out=paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh native integration registration required')
        out.mkdir();freeze(out);write(paths['registration'],dict(schema='native_full_rehash_integration_registration:1',config=file_pin(cp),inputs=inputs,
            implementation_pins=pins,sources=[s['source'] for s in selected],source_receipt=file_pin(out/'source_receipt.json'),
            expected_positive_restores=198,negative_warmup_restores=4,expected_guard_denials=8,native_budget=1440,optimizer_budget=0))
        print(json.dumps(dict(registration=file_pin(paths['registration']))),flush=True);return
    reg=read(paths['registration'])
    if reg['config']!=file_pin(cp) or reg['inputs']!=inputs or reg['implementation_pins']!=pins or reg['sources']!=[s['source'] for s in selected]:raise ValueError('Registered integration changed')
    verify_source_snapshot(read(verify_file_pin(reg['source_receipt'])),live=True)
    out=paths['output']
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh native integration output required')
    out.mkdir();receipt=freeze(out);basis=np.load(verify_file_pin(run['directions']))['directions']
    from mjlab.sim.sim import Simulation
    original=Simulation.step;counter=WorkCounter();runtime=recorder=None;attempts=completed=0;cases=[];phases=[];negatives=[];stage='construction'
    def step(sim):
        if recorder is not None and recorder.enabled:recorder.before_step()
        value=counter.native_call(original,sim)
        if recorder is not None and recorder.enabled:recorder.after_step()
        return value
    Simulation.step=step
    def restore(source):
        nonlocal attempts,completed
        attempts+=1;audit=restore_backend_probe(runtime,source)
        if not audit['all_array_bytes_equal'] or not audit['all_scalar_values_equal'] or any(not torch.equal(v.cpu(),source['observation'][key]) for key,v in runtime.observation().items()):raise ValueError('Full native/observation readback differs')
        completed+=1;return audit
    try:
        torch.set_num_threads(4)
        with backend_scope(paths['backend_manifest']) as backend,torch.no_grad(),torch.jit.optimized_execution(False):
            try:
                for entry in selected:
                    source=torch.load(verify_file_pin(entry['source']),map_location='cpu',weights_only=False);require_backend_probe(source,contract)
                    runtime=BackendSupportPenRuntime(paths['objective_config'],1);runtime.set_fixed_frame_features(False);runtime.support_weight=float(runtime.objective_cfg['support_weights'][entry['actor']])
                    recorder=OwnedProbeRecorder(runtime)
                    originals=[c for c in run['cases'] if c['source_name']==entry['name'] and c['available'] and c['profile']['direction'] in [-1,0]]
                    if len(originals)!=10:raise ValueError('Original2BASE+8 signed direction0 branches required')
                    for repeat in range(3):
                        for mode in cfg['modes']:
                            stage=f'{entry["name"]}_R{repeat}_{mode}';scope=FullRehashVerifierScope('SINGLE_RESOLVE_FULL_REHASH' if mode=='SINGLE_RESOLVE' else 'ORIGINAL_FULL_REHASH',pins)
                            phase_start=time.perf_counter();before_cases=len(cases)
                            with scope:
                                t=time.perf_counter();restore(source);warmup=time.perf_counter()-t
                                for old in originals:
                                    p=old['profile'];direction=np.zeros(20) if p['direction']<0 else basis[p['direction']]
                                    raw=perturbed_actor_raw(source['base_raw'],direction,p['epsilon'],p['sign']);folder=out/entry['name']/f'round{repeat}'/mode/p['name'];folder.mkdir(parents=True)
                                    torch.cuda.synchronize();wp.synchronize();begin=time.perf_counter();timing={}
                                    t=time.perf_counter();audit=restore(source);timing['restore_and_observation_s']=time.perf_counter()-t
                                    t=time.perf_counter();recorder.begin(raw);timing['recorder_begin_s']=time.perf_counter()-t
                                    before=counter.native;t=time.perf_counter();_,_,term,trunc,info=runtime.step(torch.tensor(raw,device=runtime.device)[None]);timing['recorded8_tick_step_s']=time.perf_counter()-t
                                    t=time.perf_counter();z=recorder.finish();w={k:v.cpu().numpy().copy()[0] for k,v in runtime.contact_views.items()};timing['finish_window_copy_s']=time.perf_counter()-t
                                    actual=z['post_ctrl'][0,old['aids']];error=float(abs(actual-(np.asarray(old['base_PD20'])+p['sign']*p['epsilon']*direction)).max())
                                    quality=dict(native_count=counter.native-before==8,held_PD=bool(np.array_equal(z['post_ctrl'][:,old['aids']],np.tile(actual,(8,1)))),PD=error<=2e-7,eight_samples=int(w['counts'][0])==8,
                                        geometry=bool(np.all(w['geometry_max']<=[.002,.0002,.001])),nonhand=float(w['nonhand_max_sum'][0])<=.1,no_metric_error=not bool(w['errors'].any() or w['global_error'].any()),
                                        finite=all(np.isfinite(v).all() for k,v in z.items() if not k.startswith('legacy_host_')),no_aid=not np.any(z['post_qfrc_applied']) and not np.any(z['post_xfrc_applied']))
                                    t=time.perf_counter();np.savez_compressed(folder/'trace.npz',**z);np.savez_compressed(folder/'window.npz',**w);timing['compressed_IO_s']=time.perf_counter()-t
                                    t=time.perf_counter();torch.cuda.synchronize();wp.synchronize();timing['final_synchronize_s']=time.perf_counter()-t
                                    trace_pin=file_pin(folder/'trace.npz');window_pin=file_pin(folder/'window.npz')
                                    case=dict(old,name=stage+'_'+p['name'],original_name=old['name'],round=repeat,mode=mode,trace=trace_pin,window=window_pin,restore_audit=audit,
                                        quality=quality,qualified=all(quality.values()),actual_PD20=actual.tolist(),PD_error_rad=error,native_integrations=counter.native-before,
                                        terminated=bool(term[0]),truncated=bool(trunc[0]),termination_reasons={k:bool(v[0]) for k,v in info['termination_reasons'].items()},stages=timing,branch_wall_s=time.perf_counter()-begin)
                                    cases.append(case);write(folder/'report.json',case)
                            phases.append(dict(source_name=entry['name'],round=repeat,mode=mode,warmup_s=warmup,phase_wall_s=time.perf_counter()-phase_start,
                                branch_wall_sum_s=sum(c['branch_wall_s'] for c in cases[before_cases:]),scope=scope.audit()))
                            write(out/'progress.json',dict(stage=stage,cases=len(cases),restore_attempts=attempts,restore_completed=completed,**counter.metadata()))
                    # Denial tests are outside ALL performance regions.
                    for mode in ['ORIGINAL_FULL_REHASH','SINGLE_RESOLVE_FULL_REHASH']:
                        scope=FullRehashVerifierScope(mode,pins)
                        with scope:
                            restore(source)
                            for defect in ['SOURCE_HASH','ENGINE_HASH']:
                                stage=entry['name']+'_'+mode+'_'+defect;folder=out/'negative'/stage;folder.mkdir(parents=True)
                                bad=copy.deepcopy(source)
                                if defect=='SOURCE_HASH':
                                    for key in ['observed','snapshot']:bad['native']['sources']['records'][0][key]['sha256']='0'*64
                                else:bad['native']['engine_sources'][0]['sha256']='0'*64
                                before=native.capture(runtime,receipt);torch.save(before,folder/'before.pt');writes=0;old_apply=Write.apply
                                def watched(write):
                                    nonlocal writes
                                    writes+=1;return old_apply(write)
                                denied=False;error=None;native_before=counter.native
                                with patch.object(Write,'apply',watched):
                                    try:restore(bad)
                                    except ValueError as caught:denied=True;error=str(caught)
                                after=native.capture(runtime,receipt);torch.save(after,folder/'after.pt')
                                unchanged=all(raw_tree_equal(before[k],after[k]) for k in ['native','python','rng'])
                                negatives.append(dict(source_name=entry['name'],mode=mode,defect=defect,denied=denied,error=error,
                                    Write_apply_calls=writes,native_integrations=counter.native-native_before,all_state_bytes_unchanged=unchanged,
                                    before=file_pin(folder/'before.pt'),after=file_pin(folder/'after.pt')))
                        if not scope.validated:raise ValueError('Negative verifier scope cleanup failed')
                    runtime.close();runtime=recorder=None
                checks=dict(all180_branches=len(cases)==180,all18_scopes=len(phases)==18 and all(p['scope']['validated'] and p['scope']['bindings_restored'] for p in phases),
                    all202_restores=completed==202 and attempts==210,all8_denials=len(negatives)==8 and all(n['denied'] and not n['Write_apply_calls'] and not n['native_integrations'] and n['all_state_bytes_unchanged'] for n in negatives),
                    all_online_flags=all(c['qualified'] for c in cases),all1440_native=counter.native==counter.native_attempted==1440,no_optimizer=counter.optimizer==counter.optimizer_attempted==0)
                backend.verify(cold=True);verify_source_snapshot(receipt,live=True)
                report=dict(schema='native_full_rehash_integration:1',config=file_pin(cp),registration=file_pin(paths['registration']),inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),
                    cases=cases,phases=phases,negative_tests=negatives,restore_attempts=attempts,restore_completed=completed,checks=checks,passed=sum(checks.values()),total=len(checks),
                    **counter.metadata(),backend_audit=backend.audit(),wall_s=time.perf_counter()-started,default_changed=False,source_epoch_used=False,
                    scope='Bounded same-backend full-rehash candidate:originalA/candidate/originalB x3 rounds,two sources. Per-phase boundary hashes/warmups charged. No default/native physics edit. Independent saved-packet/physical review still required;not task/Actor/ego/training speedup.')
                write(out/'report.json',report)
                if not all(checks.values()):raise ValueError('Keep native verifier integration discrepancy')
                print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[report['passed'],report['total']],**counter.metadata())),flush=True)
            except Exception as error:
                if recorder is not None and recorder.enabled:
                    try:np.savez_compressed(out/'failure_partial_NOT_RESUMABLE.npz',**recorder.finish())
                    except Exception:pass
                write(out/'failure.json',dict(type=type(error).__name__,message=str(error),stage=stage,restore_attempts=attempts,restore_completed=completed,cases=cases,negative_tests=negatives,**counter.metadata(),resumable=False));raise
            finally:
                for rt in list(backend.runtime_set):rt.close()
                runtime=recorder=None
    finally:Simulation.step=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--register',action='store_true');main(p.parse_args())
