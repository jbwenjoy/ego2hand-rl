"""Bounded same-solve-exit replay; terminal full54 guard precedes GPU imports."""
import argparse
import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_solve_exit_observer import require_closed_full54
from supermimic_research.runtime.pen_solve_exit_design import schedule
from supermimic_research.strict_json import write_json


def read(p):return json.loads(Path(p).read_text())


def main(a):
    started=time.perf_counter();cp=a.config.resolve();cfg=tomllib.loads(cp.read_text())
    if (cfg['schema'],cfg['mode'],cfg['replicas'],cfg['queries_per_replica'],cfg['native_cap_per_replica'],cfg['optimizer_budget'],cfg['normalizer_fit_budget'],cfg['verifier'])!=(1,'FIXED_TEN_CONDITION_SOLVE_EXIT_REPLAY',['A','B'],60,480,0,0,'SINGLE_RESOLVE_FULL_REHASH'):raise ValueError('Fixed registered replay required')
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['observation_config','preparation','prior_evidence','protocol','registration','output']}
    oc=tomllib.loads(paths['observation_config'].read_text());prep=read(paths['preparation'])
    if prep['passed']!=prep['total'] or not all(prep['checks'].values()) or prep['config']!=file_pin(paths['observation_config']):raise ValueError('Exact completed ten-condition preparation required')
    study_path=(paths['observation_config'].parent/oc['study_config']).resolve();study=tomllib.loads(study_path.read_text());sp={k:(study_path.parent/study[k]).resolve() for k in ['backend_manifest','objective_config']}
    old=read(verify_file_pin(prep['prior']));expected_owner=old['inputs']['registration'];solver_pin=old['inputs']['installed_solver']
    inputs={k:file_pin(p) for k,p in {**sp,**{k:paths[k] for k in ['observation_config','preparation','prior_evidence','protocol']}}.items()};inputs['study_config']=file_pin(study_path);inputs['installed_solver']=solver_pin;inputs['full54_registration']=expected_owner
    for pin in inputs.values():verify_file_pin(pin)
    verify_source_snapshot(read(verify_file_pin(prep['source_receipt'])),live=True)
    sources=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,paths['observation_config'],paths['protocol']]
    def freeze(folder):
        receipt=freeze_sources(sources,folder/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(folder/'source_receipt.json',receipt);return receipt
    if a.register:
        out=paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh replay registration required')
        out.mkdir();freeze(out);write_json(paths['registration'],dict(schema='pen_solve_exit_replay_registration:1',config=file_pin(cp),inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),schedules={rep:schedule(prep['rows'],rep) for rep in cfg['replicas']},native_cap_per_replica=480,total_native_cap=960,optimizer_steps=0));print(json.dumps(dict(registration=file_pin(paths['registration']),native=0)),flush=True);return
    if a.check_ready:
        try:proof=require_closed_full54(paths['observation_config'])
        except ValueError as error:print(json.dumps(dict(ready=False,reason=str(error),native=0)),flush=True);return
        print(json.dumps(dict(ready=True,proof=proof,native=0)),flush=True);return
    # Refuse before simulator import, source loading, output mutation or CUDA.
    proof=require_closed_full54(paths['observation_config'])
    if proof['registration']!=expected_owner:raise ValueError('Closed full54 belongs to a different registration')
    if a.replica not in cfg['replicas']:raise ValueError('Explicit A/B fresh process required')
    reg=read(paths['registration'])
    if reg['config']!=file_pin(cp) or reg['inputs']!=inputs or reg['schedules'][a.replica]!=schedule(prep['rows'],a.replica):raise ValueError('Registered replay inputs/design changed')
    verify_source_snapshot(read(verify_file_pin(reg['source_receipt'])),live=True)
    out=paths['output']/a.replica
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh replica output required')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import numpy as np
    import torch
    from mjlab.sim.sim import Simulation
    from supermimic_research.runtime.explicit_collision_backend import backend_scope
    from supermimic_research.runtime.backend_pen_runtime import BackendSupportPenRuntime
    from supermimic_research.runtime.full_rehash_scope import FullRehashVerifierScope,implementation_pins
    from supermimic_research.runtime.pen_live_packets import LivePenPackets
    from supermimic_research.runtime.pen_owned_probe_recorder import OwnedProbeRecorder
    from supermimic_research.runtime.pen_solve_exit_observer import SolveExitGraphObserver
    from supermimic_research.learning.pen_matched_recovery import WorkCounter
    from supermimic_research.learning.world_model import raw_tree_equal
    from supermimic_research.mechanics.pen_solve_exit_epoch import review as review_epoch
    from supermimic_research.mechanics.warp_packet_decode import decode_saved_case
    class ExitRecorder(OwnedProbeRecorder):
        def begin(self,raw):
            self.exit_rows=[];self.exit_counts=[];super().begin(raw)
        def after_step(self):
            super().after_step();self.exit_counts.append(observer.count())
            if observer.scope=='PACKET_PLUS_SOLVE_EXIT':self.exit_rows.append(observer.read())
    out.mkdir(parents=True);receipt=freeze(out);plan=reg['schedules'][a.replica];write_json(out/'intent.json',dict(registration=file_pin(paths['registration']),proof=proof,schedule=plan,native_cap=480))
    counter=WorkCounter();packets=runtime=observer=None;original=Simulation.step;rows=[];builds=[];scope_pins=[];stage='setup';expected_obs={};known_traces={};setup_extra_native=0;setup_unknown=False
    def step(sim):
        if packets is not None and packets.recorder.enabled:packets.recorder.before_step()
        counter.native_call(original,sim)
        if packets is not None and packets.recorder.enabled:packets.recorder.after_step()
    Simulation.step=step
    try:
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        with backend_scope(sp['backend_manifest']) as backend,torch.no_grad(),torch.jit.optimized_execution(False):
            try:
                for teacher in ['RESIDUAL','SUPPORT']:
                    local=[p for p in plan if p['teacher']==teacher];folder=out/teacher;folder.mkdir();stage=teacher+'_setup'
                    runtime=BackendSupportPenRuntime(sp['objective_config'],1);runtime.set_fixed_frame_features(False);packets=LivePenPackets(runtime,receipt,counter,folder,'SOLVE_EXIT_DIAGNOSTIC_NEVER_FIT');scope=FullRehashVerifierScope(cfg['verifier'],implementation_pins())
                    with scope:
                        packets.restore(prep['rows'][local[0]['row_index']]['case']['source'],'before_graph');before,bp=packets.capture(folder,'before_graph');before_count=counter.metadata();t=time.perf_counter()
                        observer=SolveExitGraphObserver(runtime.env.sim,config_path=paths['observation_config'],installed_solver_pin=solver_pin)
                        previous_unknown=setup_unknown;setup_unknown=True
                        try:observer.build()
                        finally:
                            after,ap=packets.capture(folder,'after_graph');clock0=before['native']['data']['arrays']['time'].astype(np.float64);clock1=after['native']['data']['arrays']['time'].astype(np.float64);delta=(clock1-clock0)/float(before['native']['model']['arrays']['opt.timestep'][0]);estimate=int(np.rint(delta[0]));known=bool(np.all(abs(delta-estimate)<=1e-3) and estimate>=0);setup_extra_native+=estimate if known else 0;setup_unknown=previous_unknown or not known
                            b=dict(teacher=teacher,before=bp,after=ap,wall_s=time.perf_counter()-t,capture_API_calls=1,clock_native_estimate=estimate if known else None,counter_before=before_count,counter_after=counter.metadata(),copy_events=observer.count(),state_equal={k:raw_tree_equal(before[k],after[k]) for k in ['native','python','rng']});builds.append(b);write_json(folder/'graph_build.json',b)
                        if not all(b['state_equal'].values()) or b['copy_events']!=0 or b['counter_before']!=b['counter_after'] or b['clock_native_estimate']!=0:raise ValueError('Graph capture changed physical/owned/RNG state;preserve setup work')
                        packets.recorder=ExitRecorder(runtime)
                        for declaration in local:
                            stage=declaration['name'];qf=folder/stage;qf.mkdir();entry=prep['rows'][declaration['row_index']];oldcase=entry['case'];oldtrace=dict(np.load(verify_file_pin(oldcase['trace'])));observer.select(declaration['scope']);restored=packets.restore(oldcase['source'],stage)
                            obs={k:v.detach().cpu().clone() for k,v in runtime.observation().items()};key=oldcase['name'];same_obs=key not in expected_obs or raw_tree_equal(expected_obs[key],obs);expected_obs.setdefault(key,obs);torch.save(obs,qf/'regenerated_observation.pt')
                            state,record=packets.capture(qf,'source');events=observer.count();native0=counter.native
                            try:case,_=packets.query(oldtrace['actions20'][0].astype(np.float32),qf/'query',stage,state,record)
                            except Exception:
                                torch.save(dict(trace=packets.recorder.trace,contact_rows=packets.recorder.rows,addresses=packets.recorder.addresses,efc=packets.recorder.efc,offsets=packets.recorder.offsets,exit_rows=packets.recorder.exit_rows,exit_counts=packets.recorder.exit_counts),qf/'partial_observer_NOT_RESUMABLE.pt');raise
                            z=dict(np.load(verify_file_pin(case['trace'])));event_count=observer.count()-events;checks=dict(restored_observation=same_obs,source_native_state=raw_tree_equal(restored['native'],state['native']),native8=counter.native-native0==8,actual_PD_exact=np.array_equal(z['post_ctrl'][:,case['aids']],oldtrace['post_ctrl'][:,oldcase['aids']]),copy_events=event_count==(8 if declaration['scope']=='PACKET_PLUS_SOLVE_EXIT' else 0),all8_counts=packets.recorder.exit_counts==[events+(k+1 if declaration['scope']=='PACKET_PLUS_SOLVE_EXIT' else 0) for k in range(8)])
                            row=dict(declaration=declaration,original_case=oldcase,case=file_pin(qf/'query/report.json'),regenerated_observation=file_pin(qf/'regenerated_observation.pt'),source_restore=packets.restores[-1],copy_events=event_count,checks=checks,original_physical_valid=oldcase['physical_audit']['valid'],new_physical_valid=case['physical_audit']['valid'],original_impulse_valid=oldcase['physical_audit']['checks']['EFC_linear_impulse'],new_impulse_valid=case['physical_audit']['checks']['EFC_linear_impulse'])
                            if declaration['scope']=='PACKET_PLUS_SOLVE_EXIT':
                                copies={k:np.stack([r[k] for r in packets.recorder.exit_rows]) for k in observer.buffers};np.savez_compressed(qf/'solve_exit.npz',**copies);metrics,decoded=decode_saved_case(case,z,packets.model,packets.physical);epoch,arrays=review_epoch(z,copies,state['native']['model']['arrays'],case,decoded,packets.model.opt.gravity,int(state['native']['model']['scalars']['opt.iterations']));np.savez_compressed(qf/'balance.npz',**arrays);write_json(qf/'epoch_review.json',epoch);checks['epoch_review']=epoch['passed']==epoch['total'] and all(epoch['checks'].values());row.update(solve_exit=file_pin(qf/'solve_exit.npz'),balance=file_pin(qf/'balance.npz'),epoch_review=file_pin(qf/'epoch_review.json'))
                            def differences(ref):
                                return {k:dict(bytes_equal=z[k].dtype==ref[k].dtype and z[k].shape==ref[k].shape and z[k].tobytes()==ref[k].tobytes(),max_abs=float(np.max(abs(z[k].astype(np.float64)-ref[k].astype(np.float64)))) if z[k].shape==ref[k].shape and z[k].size else None) for k in z}
                            row['differences_from_original']=differences(oldtrace);pairkey=(declaration['repeat'],key)
                            if pairkey in known_traces:row['differences_between_scopes']=differences(known_traces[pairkey])
                            else:known_traces[pairkey]=z
                            row.update(passed=sum(checks.values()),total=len(checks));write_json(qf/'report.json',row);rows.append(dict(name=stage,report=file_pin(qf/'report.json'),declaration=declaration,new_impulse_valid=row['new_impulse_valid']))
                            write_json(out/'work.json',dict(stage=stage,completed=len(rows),**counter.metadata()))
                            if not all(checks.values()):raise ValueError('Replay observation contract discrepancy '+repr(checks))
                            if counter.native>480:raise ValueError('Replica native cap exceeded')
                    write_json(folder/'scope.json',scope.audit());scope_pins.append(file_pin(folder/'scope.json'));observer.close();observer=None;runtime.close();runtime=packets=None
                checks=dict(all60=len(rows)==60,exact_order=[r['name'] for r in rows]==[p['name'] for p in plan],all_build_state=all(all(b['state_equal'].values()) and b['copy_events']==b['clock_native_estimate']==0 for b in builds),native_exact=counter.native==counter.native_attempted==480 and setup_extra_native==0 and not setup_unknown,no_updates=counter.optimizer==counter.optimizer_attempted==0,verifier_scopes=len(scope_pins)==2 and all(read(verify_file_pin(p))['validated'] and read(verify_file_pin(p))['bindings_restored'] for p in scope_pins))
                backend.verify(cold=True);verify_source_snapshot(receipt,live=True);write_json(out/'report.json',dict(schema='pen_solve_exit_replay:1',registration=file_pin(paths['registration']),prerequisites=proof,source_receipt=file_pin(out/'source_receipt.json'),replica=a.replica,rows=rows,builds=builds,verifier_scopes=scope_pins,checks=checks,passed=sum(checks.values()),total=len(checks),**counter.metadata(),normalizer_fits=0,setup_extra_native=setup_extra_native,setup_native_unknown=setup_unknown,wall_s=time.perf_counter()-started,scope='Targeted repeated observations;original/new failures and observer differences retained. Independent saved review required;no engine fix/Actor/ego admission.'))
                if not all(checks.values()):raise ValueError('Replica final accounting discrepancy')
            finally:
                try:
                    if observer is not None:observer.close()
                finally:
                    if runtime is not None:runtime.close()
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error),stage=stage,rows=rows,missing=[p for p in plan if p['name'] not in {r['name'] for r in rows}],builds=builds,**counter.metadata(),recorded_control_native_integrations=counter.native,setup_extra_native=setup_extra_native,setup_native_unknown=setup_unknown,resumable=False)
        failure.update(actual_native_integrations=counter.native+setup_extra_native if not setup_unknown else None,attempted_native_integrations=counter.native_attempted+setup_extra_native if not setup_unknown else None,completed_call_accounting_exact=not setup_unknown and counter.native==counter.native_attempted)
        write_json(out/'failure.json',failure);raise
    finally:Simulation.step=original
    print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[sum(checks.values()),len(checks)],**counter.metadata())),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--replica',choices=['A','B']);g=p.add_mutually_exclusive_group();g.add_argument('--register',action='store_true');g.add_argument('--check-ready',action='store_true');main(p.parse_args())
