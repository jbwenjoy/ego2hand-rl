"""Fresh actor/native4 compressed branches and full BASE-end rollback on GPU."""
import argparse
import copy
import json
import itertools
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools'),str(ROOT)]
import mujoco
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.explicit_collision_backend import backend_scope,load_contract
from supermimic_research.runtime.backend_pen_runtime import BackendSupportPenRuntime
from supermimic_research.runtime.pen_owned_probe_recorder import OwnedProbeRecorder,save_physics,source_frame_gap
from supermimic_research.runtime.full_rehash_scope import FullRehashVerifierScope,implementation_pins
from supermimic_research.runtime import owned_pen_state as native
from supermimic_research.runtime.snapshot_transport import save_owned,load_owned
from supermimic_research.runtime.shared_snapshot_transport import SharedImmutableTransport,load_shared,SCHEMA as SHARED_SCHEMA
from supermimic_research.runtime.owned_array_writes import Write
from supermimic_research.learning.pen_live_decision import from_owned_snapshot
from supermimic_research.learning.pen_matched_recovery import WorkCounter
from supermimic_research.learning.world_model import raw_tree_equal
from supermimic_research.mechanics.probe_row_review import review_recorded_row
from supermimic_research.mechanics.pen_reward_chain import native_components
from supermimic_research.strict_json import write_json
from run_pen_backend_control_study import actor_initializer
from evaluate_pen_long_control import context_pins


def read(p):return json.loads(Path(p).read_text())


def main(args):
    started=time.perf_counter();cp=args.config.resolve();cfg=tomllib.loads(cp.read_text())
    if (cfg['schema'],cfg['mode'],cfg['arms'],cfg['transports'],cfg['branches'],cfg['native_budget'],cfg['alternative_RAW_joint0_delta'],cfg['verifier'])!=(1,'FRESH_W1_OWNED_BRANCH_ROLLBACK_QUALIFICATION',['RESIDUAL','SUPPORT'],['GZIP_FULL','GZIP_SHARED_IMMUTABLE'],['BASE','ALT','CONT_A','CONT_B'],128,.008,'SINGLE_RESOLVE_FULL_REHASH'):raise ValueError('Fixed fresh branch qualification required')
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['study_config','transport_report','shared_transport_report','prior_ledger','protocol','registration','output']};study=tomllib.loads(paths['study_config'].read_text())
    sp={k:(paths['study_config'].parent/study[k]).resolve() for k in ['backend_manifest','objective_config','training_report','plant_context_report']}
    inputs={k:file_pin(p) for k,p in {**sp,**{k:paths[k] for k in ['study_config','transport_report','shared_transport_report','prior_ledger','protocol']}}.items()};contract=load_contract(sp['backend_manifest']);training=read(sp['training_report']);transport=read(paths['transport_report']);shared_transport=read(paths['shared_transport_report'])
    if any(r['passed']!=r['total'] or not all(r['checks'].values()) for r in [training,transport,shared_transport]):raise ValueError('Completed training/transport inputs required')
    contexts=context_pins(read(sp['plant_context_report']));actor_pins={a['arm']:a['checkpoint'] for a in training['arms']}
    source_paths=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,paths['protocol'],ROOT/'tools/run_pen_backend_control_study.py',ROOT/'tools/evaluate_pen_long_control.py']
    def freeze(out):
        receipt=freeze_sources(source_paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt);return receipt
    if args.register:
        out=paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh node registration required')
        out.mkdir();freeze(out);write_json(paths['registration'],dict(schema='pen_live_node_registration:1',config=file_pin(cp),inputs=inputs,actors=actor_pins,contexts=contexts,implementation_pins=implementation_pins(),source_receipt=file_pin(out/'source_receipt.json'),native_budget=128,optimizer_steps=0))
        print(json.dumps(dict(registration=file_pin(paths['registration']))),flush=True);return
    reg=read(paths['registration'])
    if reg['config']!=file_pin(cp) or reg['inputs']!=inputs or reg['actors']!=actor_pins or reg['contexts']!=contexts or reg['implementation_pins']!=implementation_pins():raise ValueError('Registered node inputs changed')
    verify_source_snapshot(read(verify_file_pin(reg['source_receipt'])),live=True)
    out=paths['output']
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh node output required')
    out.mkdir();receipt=freeze(out);counter=WorkCounter();runtime=recorder=shared_store=None;restores=[];captures=[];branches=[];arms=[];denials=[];stage='construction'
    from mjlab.sim.sim import Simulation
    original=Simulation.step
    def step(sim):
        if recorder is not None and recorder.enabled:recorder.before_step()
        value=counter.native_call(original,sim)
        if recorder is not None and recorder.enabled:recorder.after_step()
        return value
    Simulation.step=step
    def load_record(record):return load_shared(record) if record.get('schema')==SHARED_SCHEMA else load_owned(record)
    def capture(folder,name):
        t=time.perf_counter();value=native.capture(runtime,receipt);record=shared_store.save(folder/(name+'.pt.gz'),value) if shared_store is not None else save_owned(folder/(name+'.pt.gz'),value);decoded=load_record(record)
        if not raw_tree_equal(value,decoded):raise ValueError('Owned compressed CPU snapshot differs')
        captures.append(dict(path=str(folder/name),wall_s=time.perf_counter()-t,transport=record));write_json(folder/(name+'.transport.json'),record);return decoded,record
    def restore(record,label):
        t=time.perf_counter();before=counter.metadata();value=load_record(record);audit=native.restore(runtime,value);after=native.capture(runtime,receipt)
        equal={k:raw_tree_equal(value[k],after[k]) for k in ['native','python','rng']}
        restores.append(dict(label=label,audit=audit,complete_state_equal=equal,external_counter_unchanged=before==counter.metadata(),wall_s=time.perf_counter()-t))
        if not audit['all_array_bytes_equal'] or not audit['all_scalar_values_equal'] or not all(equal.values()):raise ValueError('Complete rollback readback differs')
    try:
        torch.set_num_threads(4)
        with backend_scope(sp['backend_manifest']) as backend,torch.no_grad(),torch.jit.optimized_execution(False):
            try:
                for transport_mode,arm in itertools.product(cfg['transports'],cfg['arms']):
                    group=transport_mode+'_'+arm;folder=out/transport_mode/arm;folder.mkdir(parents=True);shared_store=SharedImmutableTransport(folder/'shared_immutable.pt.gz') if transport_mode=='GZIP_SHARED_IMMUTABLE' else None
                    stage=group+'_fresh_actor';state=torch.load(verify_file_pin(actor_pins[arm]),map_location='cpu',weights_only=False)
                    runtime=BackendSupportPenRuntime(sp['objective_config'],1);actor,admission=actor_initializer(runtime,state,actor_pins[arm],contract=contract,plant_pin=contexts[arm],expected_iteration=2000)
                    actor_before={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()};recorder=OwnedProbeRecorder(runtime);physics=save_physics(runtime,folder);model=runtime.env.sim.mj_model;f=runtime.features
                    body=f.body;qa=f.qa;da=int(model.jnt_dofadr[model.body_jntadr[body]])
                    template=dict(dt=.0025,qids=f.qids.cpu().numpy().tolist(),dofs=f.dofs.cpu().numpy().tolist(),aids=runtime.action._position_ctrl_ids.cpu().numpy().tolist(),physics=physics,
                        model=file_pin(folder/'model.mjb'),physical_context=file_pin(folder/'physical_context.npz'),model_layout=dict(object_body=body,object_qposadr=qa,object_dofadr=da,COM_BODY=model.body_ipos[body].tolist()),role='QUALIFY_FRESH_OWNED_NODE_NEVER_FIT')
                    cm=mujoco.MjModel.from_binary_path(str(verify_file_pin(template['model'])));physical=dict(np.load(verify_file_pin(template['physical_context'])));reward_cfg=tomllib.loads(runtime.config_path.read_text())['reward']
                    scope=FullRehashVerifierScope(cfg['verifier'],reg['implementation_pins'])
                    with scope:
                        raw=actor.distribution(runtime.observation()['policy'],validate_args=False).mean[0].cpu().numpy().copy();s0,s0pin=capture(folder,'source0');phase0=int(runtime.motion.time_steps[0]);endb=endbpin=raw_cont=None;source_for_branch=s0;traces={}
                        for label in cfg['branches']:
                            stage=group+'_'+label
                            if label=='ALT':restore(s0pin,group+'_source0');source_for_branch=s0;submitted=raw.copy();submitted[0]+=.008
                            elif label in ['CONT_A','CONT_B']:
                                restore(endbpin,group+'_end_BASE_'+label);source_for_branch=endb
                                if label=='CONT_A':raw_cont=actor.distribution(runtime.observation()['policy'],validate_args=False).mean[0].cpu().numpy().copy()
                                submitted=raw_cont.copy()
                            else:submitted=raw.copy()
                            phase=int(runtime.motion.time_steps[0]);decision=from_owned_snapshot(source_for_branch,phase,template,submitted);bf=folder/label;bf.mkdir();recorder.begin(submitted);before=counter.native
                            _,reward,term,trunc,info=runtime.step(torch.tensor(submitted,device=runtime.device)[None]);z=recorder.finish();w={k:v.cpu().numpy().copy()[0] for k,v in runtime.contact_views.items()}
                            np.savez_compressed(bf/'trace.npz',**z);np.savez_compressed(bf/'window.npz',**w);end,endpin=capture(bf,'end')
                            q=source_for_branch['native']['data']['arrays'];gap=source_frame_gap(q['qpos'][0],qa,q['xmat'][0,body],q['xipos'][0,body])
                            audit_source=dict(native=source_for_branch,phase=phase,source_qpos_R=gap['qpos_R'],source_cached_R=q['xmat'][0,body])
                            actual=z['post_ctrl'][0,template['aids']];quality=dict(eight_ticks=counter.native-before==8,actual_PD=bool(abs(actual-decision['actual_PD20']).max()<=2e-7),no_aid=not bool(np.any(z['post_qfrc_applied']) or np.any(z['post_xfrc_applied'])))
                            case=dict(template,name=stage,available=True,qualified=all(quality.values()),quality=quality,actual_PD20=actual.tolist(),trace=file_pin(bf/'trace.npz'),window=file_pin(bf/'window.npz'),
                                source=s0pin if label in ['BASE','ALT'] else endbpin,end=endpin,phase=phase,source_reward=float(reward[0]),terminated=bool(term[0]),truncated=bool(trunc[0]),native_integrations=counter.native-before)
                            with patch.object(mujoco,'mj_step',side_effect=AssertionError('Saved packet audit')),patch.object(mujoco,'mj_forward',side_effect=AssertionError('Saved packet audit')):
                                packet,physical_audit=review_recorded_row(case,z,w,audit_source,cm,physical,dict(role=case['role'],object_penetration_m=.002,self_penetration_m=.0002,joint_excursion_rad=.001))
                                component,formula,errors=native_components(case,z,w,decision['controller_context'],reward_cfg,runtime.objective_cfg)
                            case.update(physical_audit=physical_audit,reward_checks=formula,reward_errors=errors,source_reward_equal=abs(float(component[-1])-case['source_reward'])<=2e-6,packed_input_bytes=np.array_equal(packet['x'].astype(np.float32),decision['input102']),
                                selected_path_internal_counter=runtime.owned_native_integrations)
                            branches.append(case);traces[label]=z;write_json(bf/'report.json',case);write_json(out/'work.json',dict(stage=stage,**counter.metadata()))
                            if label=='BASE':endb,endbpin=end,endpin
                        before_bad=native.capture(runtime,receipt);bad=copy.deepcopy(s0pin)
                        if shared_store is None:bad['uncompressed_sha256']='0'*64
                        else:bad['payload']['uncompressed_sha256']='0'*64
                        denied=False;writes=0;old_apply=Write.apply
                        def watched(write):
                            nonlocal writes
                            writes+=1;return old_apply(write)
                        with patch('supermimic_research.runtime.snapshot_transport.torch.load',side_effect=AssertionError('Bad transport deserialized')),patch.object(Write,'apply',watched):
                            try:native.restore(runtime,load_record(bad))
                            except ValueError:denied=True
                        after_bad=native.capture(runtime,receipt);denials.append(dict(arm=arm,transport=transport_mode,denied_before_deserialization=denied,Write_apply_calls=writes,state_unchanged=all(raw_tree_equal(before_bad[k],after_bad[k]) for k in ['native','python','rng'])))
                    a,b=traces['CONT_A'],traces['CONT_B'];differences={k:float(abs(a[k]-b[k]).max()) for k in ['post_qpos','post_qvel','post_qacc_warmstart','post_qfrc_constraint']}
                    arms.append(dict(arm=arm,transport=transport_mode,admission=admission,initial_phase=phase0,continuation_phase=next(c['phase'] for c in branches if c['name']==group+'_CONT_A'),
                        scope=scope.audit(),actor_unchanged=raw_tree_equal(actor_before,actor.state_dict()),continuation_inputs_equal=all(np.array_equal(a[k],b[k]) for k in ['actions20','initial_warmstart','initial_native_state','pre_action_phase']),observed_future_max_differences=differences))
                    runtime.close();runtime=recorder=None
                checks=dict(all16_branches=len(branches)==16,all128_native=counter.native==counter.native_attempted==128,no_optimizer=counter.optimizer==counter.optimizer_attempted==0,
                    all12_restores=len(restores)==12 and all(all(v['complete_state_equal'].values()) and v['external_counter_unchanged'] for v in restores),
                    all_input_bytes=all(c['packed_input_bytes'] for c in branches),all_original_physics=all(c['physical_audit']['valid'] for c in branches),all_original_reward_checks=all(all(c['reward_checks'].values()) and c['source_reward_equal'] for c in branches),
                    all_phase_continuation=all(a['continuation_phase']==a['initial_phase']+1 for a in arms),all4_owned_scopes=len(arms)==4 and all(a['scope']['validated'] and a['scope']['bindings_restored'] for a in arms),
                    all_actor_bytes_unchanged=all(a['actor_unchanged'] for a in arms),continuation_inputs_identical=all(a['continuation_inputs_equal'] for a in arms),
                    all4_bad_transport_denials=len(denials)==4 and all(d['denied_before_deserialization'] and not d['Write_apply_calls'] and d['state_unchanged'] for d in denials))
                backend.verify(cold=True);verify_source_snapshot(receipt,live=True)
                result=dict(schema='pen_fresh_owned_node_qualification:1',config=file_pin(cp),registration=file_pin(paths['registration']),inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),checks=checks,passed=sum(checks.values()),total=len(checks),branches=branches,arms=arms,restores=restores,captures=captures,transport_denials=denials,
                    **counter.metadata(),wall_s=time.perf_counter()-started,scope='Fresh2-control decision/rollback qualification only. ALT forcibly rejected;no model proposal,acceptance-controller or full54-case/Actor/ego/task qualification. Future differences reported,not universal determinism.')
                write_json(out/'report.json',result)
                if not all(checks.values()):raise ValueError('Fresh live node discrepancy retained')
                print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],**counter.metadata())),flush=True)
            except Exception as e:
                write_json(out/'failure.json',dict(type=type(e).__name__,message=str(e),stage=stage,branches=branches,restores=restores,**counter.metadata(),resumable=False));raise
            finally:
                for rt in list(backend.runtime_set):rt.close()
                runtime=recorder=None
    finally:Simulation.step=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--register',action='store_true');main(p.parse_args())
