"""Current-backend STRICT restore and recorded-branch attribution;no optimization."""
import argparse
import cProfile
import json
import pstats
import subprocess
import sys
import time
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
import torch
import warp as wp
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.learning.pen_matched_recovery import WorkCounter
from supermimic_research.runtime.explicit_collision_backend import backend_scope,load_contract
from supermimic_research.runtime.backend_pen_runtime import BackendSupportPenRuntime
from supermimic_research.runtime.backend_pen_probe import require_backend_probe,restore_backend_probe
from supermimic_research.runtime.pen_owned_probe_recorder import OwnedProbeRecorder,perturbed_actor_raw
from profile_native_collection import OperationCounts


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def stats(profile):
    s=pstats.Stats(profile);rows=[]
    for (filename,line,function),(primitive,calls,total,cumulative,_) in s.stats.items():
        rows.append(dict(file=filename,line=line,function=function,primitive_calls=primitive,calls=calls,self_s=total,cumulative_s=cumulative))
    return dict(total_self_s=float(sum(x['self_s'] for x in rows)),
        top_self=sorted(rows,key=lambda x:x['self_s'],reverse=True)[:50],
        top_cumulative=sorted(rows,key=lambda x:x['cumulative_s'],reverse=True)[:50],
        all_functions=rows,scope='cProfile wall attribution includes waits and instrumentation. Self entries are exclusive;overlapping cumulative entries cannot be summed. Not GPU active time.')


def main(args):
    start=time.perf_counter();cp=args.config.resolve();cfg=tomllib.loads(cp.read_text())
    if (cfg['schema'],cfg['mode'],cfg['restore_repeats'],cfg['source_count'],cfg['branch_native_budget'])!=(1,'STRICT_BACKEND_PIPELINE_PROFILE',3,36,320):
        raise ValueError('Exact declared profiling budget required')
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['TRAIN','VALIDATE','objective_config','backend_manifest','protocol','registration','output']}
    inputs={k:file_pin(p) for k,p in paths.items() if k not in ['registration','output']};contract=load_contract(paths['backend_manifest'])
    reports={role:read(paths[role]) for role in ['TRAIN','VALIDATE']};entries=[]
    for role,r in reports.items():
        if r.get('schema')!='backend_controlled_response_probe:1' or r['collision_backend']!=contract or r['passed']!=r['total'] or not all(r['checks'].values()):raise ValueError('Completed same-backend actual sources required')
        verify_source_snapshot(read(verify_file_pin(r['source_receipt'])),live=True)
        for s in r['sources']:
            if not s['available']:raise ValueError('All36 saved sources,including clipped exclusions,required')
            entries.append(dict(s,cohort=role))
    if len(entries)!=36 or len({s['name'] for s in entries})!=36:raise ValueError('Original36 distinct source identities required')
    source_paths=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'tools/profile_native_collection.py',cp,paths['protocol']]
    def freeze(out):
        receipt=freeze_sources(source_paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        write(out/'source_receipt.json',receipt);return receipt
    if args.register:
        out=paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh profile registration required')
        out.mkdir();freeze(out);write(paths['registration'],dict(schema='backend_STRICT_pipeline_profile_registration:1',config=file_pin(cp),inputs=inputs,
            sources=[s['source'] for s in entries],source_receipt=file_pin(out/'source_receipt.json'),restore_calls=186,
            actual_native_budget=320,optimizer_budget=0,source_epochs_allowed=False))
        print(json.dumps(dict(registration=file_pin(paths['registration']))),flush=True);return
    reg=read(paths['registration'])
    if reg['config']!=file_pin(cp) or reg['inputs']!=inputs or reg['sources']!=[s['source'] for s in entries]:raise ValueError('Registered profile changed')
    verify_source_snapshot(read(verify_file_pin(reg['source_receipt'])),live=True)
    out=paths['output']
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh profiling output required')
    out.mkdir();receipt=freeze(out)
    from mjlab.sim.sim import Simulation
    original=Simulation.step;counter=WorkCounter();runtime=recorder=None;restore_rows=[];cases=[];dispatch=[];checks={};restore_calls=0;phase='construction'
    restore_prof=cProfile.Profile();branch_prof=cProfile.Profile()
    def step(sim):
        if recorder is not None and recorder.enabled:recorder.before_step()
        result=counter.native_call(original,sim)
        if recorder is not None and recorder.enabled:recorder.after_step()
        return result
    Simulation.step=step
    def restore(source):
        nonlocal restore_calls
        audit=restore_backend_probe(runtime,source);restore_calls+=1
        if not audit['all_array_bytes_equal'] or not audit['all_scalar_values_equal'] or any(not torch.equal(v.cpu(),source['observation'][key]) for key,v in runtime.observation().items()):
            raise ValueError('Strict source/observation readback differs')
        return audit
    try:
        torch.set_num_threads(4)
        with backend_scope(paths['backend_manifest']) as backend,torch.no_grad(),torch.jit.optimized_execution(False):
            try:
                for actor in ['RESIDUAL','SUPPORT']:
                    runtime=BackendSupportPenRuntime(paths['objective_config'],1);runtime.set_fixed_frame_features(False)
                    runtime.support_weight=float(runtime.objective_cfg['support_weights'][actor])
                    for entry in [s for s in entries if s['actor']==actor]:
                        phase=entry['name'];source=torch.load(verify_file_pin(entry['source']),map_location='cpu',weights_only=False);require_backend_probe(source,contract)
                        restore(source)
                        for repeat in range(3):
                            torch.cuda.synchronize();wp.synchronize();before=time.perf_counter();restore_prof.enable()
                            audit=restore(source)
                            restore_prof.disable();torch.cuda.synchronize();wp.synchronize()
                            restore_rows.append(dict(source=entry['name'],cohort=entry['cohort'],repeat=repeat,wall_s=time.perf_counter()-before,
                                array_fields=audit['array_fields'],scalar_fields=audit['scalar_fields']))
                        write(out/'progress.json',dict(phase=phase,restore_calls=restore_calls,measured_restores=len(restore_rows),**counter.metadata()))
                    selected=next(s for s in entries if s['name']==actor+'_N8841_P024')
                    source=torch.load(verify_file_pin(selected['source']),map_location='cpu',weights_only=False)
                    phase=selected['name']+'_dispatch';ops=OperationCounts()
                    with ops:restore(source)
                    dispatch.append(dict(source=selected['name'],restore_operation_counts=ops.report()))
                    run=reports['TRAIN'];basis=np.load(verify_file_pin(run['directions']))['directions']
                    originals=[c for c in run['cases'] if c['source_name']==selected['name'] and c['available'] and c['profile']['direction'] in [-1,0]]
                    if len(originals)!=10:raise ValueError('Exactly2 BASE+8 direction0 branches per source required')
                    recorder=OwnedProbeRecorder(runtime)
                    for mode in ['UNINSTRUMENTED','CPROFILE']:
                        for old in originals:
                            phase=mode+'_'+old['name'];p=old['profile'];direction=np.zeros(20) if p['direction']<0 else basis[p['direction']]
                            raw=perturbed_actor_raw(source['base_raw'],direction,p['epsilon'],p['sign']);folder=out/mode/old['name'];folder.mkdir(parents=True)
                            torch.cuda.synchronize();wp.synchronize();begin=time.perf_counter();stages={}
                            if mode=='CPROFILE':branch_prof.enable()
                            t=time.perf_counter();audit=restore(source);stages['strict_restore_and_observation_s']=time.perf_counter()-t
                            t=time.perf_counter();recorder.begin(raw);stages['recorder_begin_s']=time.perf_counter()-t
                            before=counter.native;t=time.perf_counter();_,_,term,trunc,info=runtime.step(torch.tensor(raw,device=runtime.device)[None]);stages['step_with8_recorded_ticks_s']=time.perf_counter()-t
                            t=time.perf_counter();z=recorder.finish();w={k:v.cpu().numpy().copy()[0] for k,v in runtime.contact_views.items()};stages['finish_and_window_copy_s']=time.perf_counter()-t
                            actual=z['post_ctrl'][0,old['aids']];error=float(abs(actual-(np.asarray(old['base_PD20'])+p['sign']*p['epsilon']*direction)).max())
                            quality=dict(native_count=counter.native-before==8,held_PD=bool(np.array_equal(z['post_ctrl'][:,old['aids']],np.tile(actual,(8,1)))),
                                unclipped_PD=error<=2e-7,eight_samples=int(w['counts'][0])==8,
                                geometry=bool(np.all(w['geometry_max']<=[.002,.0002,.001])),nonhand=float(w['nonhand_max_sum'][0])<=.1,
                                no_metric_error=not bool(w['errors'].any() or w['global_error'].any()),
                                finite=all(np.isfinite(v).all() for k,v in z.items() if not k.startswith('legacy_host_')),
                                no_aid=not np.any(z['post_qfrc_applied']) and not np.any(z['post_xfrc_applied']))
                            t=time.perf_counter();np.savez_compressed(folder/'trace.npz',**z);np.savez_compressed(folder/'window.npz',**w);stages['compressed_trace_IO_s']=time.perf_counter()-t
                            t=time.perf_counter();torch.cuda.synchronize();wp.synchronize();stages['final_synchronize_s']=time.perf_counter()-t
                            if mode=='CPROFILE':branch_prof.disable()
                            row=dict(old,name=phase,original_name=old['name'],original_trace=old['trace'],profile_mode=mode,
                                trace=file_pin(folder/'trace.npz'),window=file_pin(folder/'window.npz'),quality=quality,qualified=all(quality.values()),
                                actual_PD20=actual.tolist(),PD_error_rad=error,restore_audit=audit,native_integrations=counter.native-before,
                                terminated=bool(term[0]),truncated=bool(trunc[0]),termination_reasons={k:bool(v[0]) for k,v in info['termination_reasons'].items()},
                                stages=stages,branch_complete_wall_s=time.perf_counter()-begin)
                            cases.append(row);write(folder/'report.json',row)
                    runtime.close();runtime=recorder=None
                checks=dict(all108_restore_measurements=len(restore_rows)==108,all186_strict_restores=restore_calls==186,
                    all40_branches=len(cases)==40,all320_actual_native=counter.native==counter.native_attempted==320,
                    all_online_branch_flags=all(c['qualified'] for c in cases),no_optimizer=counter.optimizer==counter.optimizer_attempted==0)
                restore_prof.dump_stats(str(out/'strict_restore.prof'));branch_prof.dump_stats(str(out/'recorded_branch.prof'))
                backend.verify(cold=True);verify_source_snapshot(receipt,live=True)
                report=dict(schema='backend_STRICT_probe_pipeline_profile:1',config=file_pin(cp),registration=file_pin(paths['registration']),inputs=inputs,
                    source_receipt=file_pin(out/'source_receipt.json'),restore_measurements=restore_rows,restore_calls=restore_calls,cases=cases,dispatch_counts=dispatch,
                    restore_profile=stats(restore_prof),branch_profile=stats(branch_prof),profiles={k:file_pin(out/k) for k in ['strict_restore.prof','recorded_branch.prof']},
                    backend_audit=backend.audit(),checks=checks,passed=sum(checks.values()),total=len(checks),**counter.metadata(),wall_s=time.perf_counter()-start,
                    physical_rows_independently_reviewed=False,source_epoch_used=False,
                    scope='Current-backend STRICT path only;instrumented attribution,not optimized benchmark or speedup. Restore warmups/constructors/dispatch counts outside respective measured regions. All recorded profile rows still need independent EFC/impulse review;no training/gradient/Actor/ego admission.')
                write(out/'report.json',report)
                if not all(checks.values()):raise ValueError('Retain strict profiling discrepancy')
                print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[report['passed'],report['total']],restore_calls=restore_calls,**counter.metadata())),flush=True)
            except Exception as error:
                restore_prof.disable();branch_prof.disable()
                if recorder is not None and recorder.enabled:
                    try:np.savez_compressed(out/'failure_partial_trace_NOT_RESUMABLE.npz',**recorder.finish())
                    except Exception:pass
                write(out/'failure.json',dict(type=type(error).__name__,message=str(error),phase=phase,restore_calls=restore_calls,cases=cases,**counter.metadata(),resumable=False));raise
            finally:
                for rt in list(backend.runtime_set):rt.close()
                runtime=recorder=None
    finally:
        restore_prof.disable();branch_prof.disable();Simulation.step=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--register',action='store_true');main(p.parse_args())
