"""New12 fixed controlled families: SOURCE/CPUFalse draft, no execution on import.

The original v2 native Reader/tick/trace acquisition path is reused. New family
outcomes are never selected from WM predictions. Same known poses, not new ego.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,importlib.util,json,sys,time
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent
SCHEMA='phone_new12_controlled_final_acquisition_protocol:1'
RESULT_SCHEMA='phone_new12_controlled_final_acquisition_result:1'
ROLE='NEW_FINAL_VALIDATION_SEALED_BEFORE_ACQUISITION'
PLAN_SHA='bb547651f8b6137d00d15ab63d16ec29c313bc3365a0c9b05f9a8d9ea38e7600'
PARENT_SHA='48f70d650f8f0f197120bc864f25a7c850b37ceae2fd068eea7e91fef40da17d'
READER_SHA='877aa4c5d51bff09e0a6824804eef3f2b335a8dbda26dcc3d0235bb3af57b841'
INPUTS={'design','family_plan','original_family_plan','original_acquisition','original_identities','exposure_policy','reader_contract','model_mjb'}

def pin(path):
 p=Path(path).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(v):
 p=Path(v['path']);assert p.is_absolute() and pin(p)==v,'Changed absolute pin '+str(p);return p
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def array_sha(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def bits(a,b):
 a,b=np.asarray(a),np.asarray(b);return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def input_alias_key(identity,physical_digest):
 r=identity['source_after_action_receipt']
 return (r['sha256'],identity['actual_control20_raw_sha256'],identity['dt_s'],identity['requested_substeps'],physical_digest,r['signature'])

def preflight(path):
 p=read(path)
 assert p['schema']==SCHEMA and set(p['inputs'])==INPUTS
 assert type(p['CPU_run_authorized']) is bool and p['new_training_authorized'] is False and p['GPU_run_authorized'] is False
 assert p['automatic_retry'] is False and p['maximum_steps']==23040 and p['maximum_rows']==240 and p['maximum_episodes']==24
 for v in [*p['inputs'].values(),*p['source_pins'].values()]:checked(v)
 assert p['source_pins']['collector']==pin(__file__)
 assert p['source_pins']['frozen_collector_v2']['sha256']==PARENT_SHA and p['source_pins']['reader']['sha256']==READER_SHA
 assert p['inputs']['family_plan']['sha256']==PLAN_SHA
 plan=read(checked(p['inputs']['family_plan']));design=read(checked(p['inputs']['design']))
 oldplan=read(checked(p['inputs']['original_family_plan']));old=read(checked(p['inputs']['original_acquisition']))
 assert plan['schema']=='phone_new_fresh_final_family_plan:1' and plan['new_families']==12
 assert plan['max_native_steps']==23040 and plan['max_H20ms_rows']==240
 assert [f['family_id'] for f in plan['families']]==list(range(1000,1012)) and len(oldplan['families'])==48
 assert design['family_plan']==p['inputs']['original_family_plan'] and old['family_plan']==p['inputs']['original_family_plan']
 assert old['sample_identities']==p['inputs']['original_identities'] and old['rows']==960
 policy=read(checked(p['inputs']['exposure_policy']))
 assert policy['fresh_final_family_ids']==list(range(1000,1012)) and policy['fresh_final_not_acquired_or_fit'] is True
 qrange=np.asarray(design['actual_limits']['joint_range20x2_F64'],dtype=np.float64)
 urange=np.asarray(design['actual_limits']['actuator_ctrlrange20x2_F64'],dtype=np.float64)
 oldq={f['initial_qpos27_raw_F64_sha256'] for f in oldplan['families']}
 oldu={f['held_ctrl10x20_raw_F64_sha256'] for f in oldplan['families']}
 source_by_branch={f['source_branch']:f['source_record'] for f in oldplan['families']}
 seenq=[];seenu=[];aliases=[]
 for f in plan['families']:
  assert f['split']==ROLE and f['native_dt_s']==[.0003125,.00015625]
  assert f['original48_family_relabelled'] is False and f['acquired'] is False and f['used_for_fit_or_model_selection'] is False
  assert f['source_record']==source_by_branch[f['source_branch']]
  src=read(checked(f['source_record']))['native_probe']['raw']
  q=np.asarray(f['initial_qpos27_F64'],dtype=np.float64);qv=np.asarray(f['initial_qvel26_F64'],dtype=np.float64)
  u=np.asarray(f['held_ctrl10x20_F64'],dtype=np.float64);base=np.asarray(f['base_ctrl20_F64'],dtype=np.float64)
  delta=np.asarray(f['requested_delta10x20_F64'],dtype=np.float64);alpha=np.asarray(f['whole_curve_alpha20_F64'],dtype=np.float64)
  assert q.shape==(27,) and qv.shape==(26,) and base.shape==(20,) and u.shape==delta.shape==(10,20) and alpha.shape==(20,)
  assert all(np.isfinite(x).all() for x in [q,qv,u,base,delta,alpha])
  assert bits(qv,np.zeros(26,dtype=np.float64)) and np.all(q[:20]>=qrange[:,0]) and np.all(q[:20]<=qrange[:,1])
  assert np.all(u>=urange[:,0]) and np.all(u<=urange[:,1]) and np.all(base>=urange[:,0]) and np.all(base<=urange[:,1])
  assert np.max(abs(q[:8]-np.asarray(src['qpos27'])[:8]))<=.005+1e-12
  assert bits(q[8:],np.asarray(src['qpos27'],dtype=np.float64)[8:]) and bits(base,np.asarray(src['ctrl20'],dtype=np.float64))
  assert abs(q[23:]@q[23:]-1)<=1e-12 and np.all((alpha>=0)&(alpha<=1))
  assert bits(u,base+delta*alpha) and np.max(abs(u[:,:8]-base[:8]))<=.05+1e-12 and np.max(abs(u[:,8:]-base[8:]))<=.025+1e-12
  qh,uh=array_sha(q),array_sha(u)
  assert qh==f['initial_raw_sha256'] and uh==f['control_raw_sha256']
  aliases.append(dict(family_id=f['family_id'],initial_alias_original=qh in oldq,control_curve_alias_original=uh in oldu))
  seenq.append(qh);seenu.append(uh)
 assert len(set(seenq))==len(set(seenu))==12 and not any(a['initial_alias_original'] or a['control_curve_alias_original'] for a in aliases)
 proof=read(checked(p['inputs']['reader_contract']))
 assert proof['status']=='PASS_192_NATIVE_STEP_GOLDEN_READER_CONTRACT' and proof['passed']==proof['total']==4 and all(v is True for v in proof['checks'].values())
 assert proof['counts']['integration_steps']==192 and proof['reader']==p['source_pins']['reader']
 out=Path(p['output']);assert out.is_absolute() and out.parent.resolve()==H and not out.exists()
 expected=dict(original_family_count=48,new_family_count=12,new_family_ids=list(range(1000,1012)),unique_initial_hashes=12,unique_whole_control_hashes=12,initial_aliases_original48=0,control_curve_aliases_original48=0,raw_alias_checks=aliases)
 assert p['metadata_alias_receipt']==expected
 return p,design,plan,out

def load_frozen_collector(p):
 path=checked(p['source_pins']['frozen_collector_v2']);sys.path.insert(0,str(path.parent))
 spec=importlib.util.spec_from_file_location('fresh12_frozen_acquisition_v2',path)
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 assert pin(sys.modules[mod.NativePhoneReader.__module__].__file__)==p['source_pins']['reader']
 return mod

def run_analysis(protocol_path):
    protocol,design,plan,out=preflight(protocol_path)
    assert protocol['CPU_run_authorized'] is True
    frozen=load_frozen_collector(protocol)
    NativePhoneReader=frozen.NativePhoneReader
    write,save_npz,receipt_json=frozen.write,frozen.save_npz,frozen.receipt_json
    reset_with_physical_diagnostics=frozen.reset_with_physical_diagnostics
    assert protocol['CPU_run_authorized'] is True
    out.mkdir();write(out/'protocol.json',protocol);started=time.perf_counter()
    readers={};rows=[];identities=[];episodes=[];source_aliases=defaultdict(list);phase='load_models'
    try:
        for h in [.0003125,.00015625]:
            readers[h]=NativePhoneReader(checked(protocol['inputs']['model_mjb']),dt_s=h)
        assert len({r.shared_physics_digest for r in readers.values()})==1
        old_report=read(checked(protocol['inputs']['original_acquisition']))
        old_identities=read(checked(protocol['inputs']['original_identities']))
        original_input_aliases=defaultdict(list);fresh_input_aliases=defaultdict(list)
        for row in old_identities:
            digest=old_report['model_audits'][str(row['dt_s'])]['shared_physics_digest_without_dt']
            assert digest==readers[row['dt_s']].shared_physics_digest
            original_input_aliases[input_alias_key(row,digest)].append(row['sample_id'])
        for family in plan['families']:
            fid=family['family_id'];q0=np.array(family['initial_qpos27_F64']);base=np.array(family['base_ctrl20_F64'])
            commands=np.array(family['held_ctrl10x20_F64'])
            for hi,h in enumerate(family['native_dt_s']):
                ep=fid*2+hi;phase=f'episode{ep}';folder=out/f'episode_{ep:03d}';folder.mkdir()
                r=readers[h];initial=reset_with_physical_diagnostics(r,q0,base)
                initial_path=folder/'initial.npz'
                save_npz(initial_path,qpos27=initial['qpos27'],qvel26=initial['qvel26'],ctrl20=initial['ctrl20'],
                    state58=initial['state58'],integration_state=initial['receipt']['integration_state'],
                    contact_rows24=initial['initial_contact_rows24'],pad_normal_N2=initial['initial_pad_normal_N2'],
                    phone_nonpad_normal_N=np.array(initial['initial_phone_nonpad_normal_N']),
                    table_normal_N=np.array(initial['initial_table_normal_N']))
                e=dict(episode_id=ep,family_id=fid,split=family['split'],source_branch=family['source_branch'],
                    source_record=family['source_record'],profile=family['profile'],initialization_seed=family['initialization_seed'],
                    control_seed=family['control_seed'],dt_s=h,substeps=r.n,initial=pin(initial_path),
                    initial_receipt=receipt_json(initial['receipt']),initial_pad_normal_N2=initial['initial_pad_normal_N2'],
                    reset_contact_phase=initial['contact_phase'],interval_row_ids=[],termination='MAX10_INTERVALS',
                    reset_termination=initial.get('reset_termination','INITIAL_READY'),
                    reset_error=initial.get('reset_error'),initial_warning_numbers=np.asarray(r.d.warning.number))
                active_commands=commands if not r.terminated else commands[:0]
                if r.terminated:e['termination']=initial['reset_termination']
                for interval,u in enumerate(active_commands):
                    v=r.step_control(u);sm=v['summary'];sample=len(rows)
                    tracepath=folder/f'interval_{interval:02d}.npz'
                    target=v['target_state58'] if v['complete_H20ms'] else np.full(58,np.nan)
                    arrays=dict(v['trace'])|dict(source_state58=v['source_state58'],observed_prefix_target58=v['target_state58'],
                        target_at_H_state58=target,actual_control20=v['actual_control20'],physical_world_delta52=v['physical_world_delta52'],
                        source_integration_state=v['source_receipt']['integration_state'],
                        source_after_action_integration_state=v['source_after_action_receipt']['integration_state'],
                        target_integration_state=v['target_receipt']['integration_state'])
                    save_npz(tracepath,**arrays)
                    contact7=np.r_[sm['pad_coverage2'],sm['pad_min_normal_N2'],sm['pad_endpoint_normal_N2'],sm['phone_nonpad_max_normal_N']]
                    row=dict(x58_F64=v['source_state58'],u20_F64=u,y58_F64=target,
                        target_delta52_F64=v['physical_world_delta52'],native_dt_s_F64=h,n_substeps_i64=r.n,
                        executed_substeps_i64=v['returned_substeps'],complete_H_bool=v['complete_H20ms'],
                        training_eligible_bool=v['training_eligible'],row_id_i64=sample,family_id_i64=fid,
                        episode_id_i64=ep,interval_index_i64=interval,contact_summary_F64=contact7,split=family['split'])
                    rows.append(row);e['interval_row_ids'].append(sample)
                    identity=dict(sample_id=sample,family_id=fid,episode_id=ep,interval_index=interval,split=family['split'],
                        source_branch=family['source_branch'],source_record=family['source_record'],profile=family['profile'],
                        dt_s=h,requested_substeps=r.n,executed_substeps=v['returned_substeps'],
                        complete_H20ms=v['complete_H20ms'],training_eligible=v['training_eligible'],
                        exclusion_reasons=[] if v['training_eligible'] else [
                            k for k,cond in [('partial_H',not v['complete_H20ms']),('nonfinite',not v['finite']),
                                ('native_warning',bool(np.any(v['warning_numbers'])))] if cond],
                        termination=v['termination'],force_phase=v['force_phase'],trace=pin(tracepath),
                        source_receipt=receipt_json(v['source_receipt']),source_after_action_receipt=receipt_json(v['source_after_action_receipt']),
                        target_receipt=receipt_json(v['target_receipt']),actual_control20_raw_sha256=array_sha(u),
                        summary=sm,warning_numbers=v['warning_numbers'])
                    identities.append(identity)
                    fresh_input_aliases[input_alias_key(identity,r.shared_physics_digest)].append(sample)
                    key=(v['source_after_action_receipt']['sha256'],v['target_receipt']['sha256'],array_sha(u),
                        h,r.n,r.shared_physics_digest,'phone_COM_world_state58:1',v['source_receipt']['signature'])
                    source_aliases[key].append(sample)
                    if v['termination']!='COMPLETE_INTERVAL':e['termination']=v['termination'];break
                episodes.append(e)
                print(json.dumps(dict(episode=ep,family=fid,split=family['split'],rows=len(e['interval_row_ids']),
                                      steps=r.tick,termination=e['termination'])),flush=True)
        phase='write_grouped_data';outputs={}
        f64={'x58_F64','u20_F64','y58_F64','target_delta52_F64','native_dt_s_F64','contact_summary_F64'}
        bools={'complete_H_bool','training_eligible_bool'}
        widths=dict(x58_F64=58,u20_F64=20,y58_F64=58,target_delta52_F64=52,contact_summary_F64=7)
        keys=['x58_F64','u20_F64','y58_F64','target_delta52_F64','native_dt_s_F64','n_substeps_i64',
              'executed_substeps_i64','complete_H_bool','training_eligible_bool','row_id_i64','family_id_i64',
              'episode_id_i64','interval_index_i64','contact_summary_F64']
        for split in [ROLE]:
            chosen=[v for v in rows if v['split']==split]
            a={k:np.asarray([r[k] for r in chosen],dtype=np.float64 if k in f64 else np.bool_ if k in bools else np.int64) for k in keys}
            for key,width in widths.items():a[key]=a[key].reshape(-1,width)
            dest=out/'fresh_final.npz';save_npz(dest,**a);outputs['fresh_final']=pin(dest)
        aliases=[dict(sample_ids=rr,family_ids=sorted({identities[i]['family_id'] for i in rr}),
                      splits=sorted({identities[i]['split'] for i in rr})) for rr in source_aliases.values() if len(rr)>1]
        cross=[v for v in aliases if len(v['splits'])>1]
        write(out/'sample_identities.json',identities);write(out/'episodes.json',episodes)
        raw_alias_report=dict(
            schema='phone_fresh12_raw_identity_alias_receipt:1',
            initial_and_control_plan_preflight=protocol['metadata_alias_receipt'],
            fresh_same_input_action_groups=[dict(sample_ids=ids) for ids in fresh_input_aliases.values() if len(ids)>1],
            cross_original960_same_input_action=[dict(fresh_sample_ids=ids,original_sample_ids=original_input_aliases[key]) for key,ids in fresh_input_aliases.items() if key in original_input_aliases],
            no_deduplication_or_reassignment=True,
            key_scope='source_after_action completeINTEGRATION SHA+actualu bytes+actualdt/n+sharedphysics+signature; visible state equality alone is insufficient. Same receipt/action does not remove shared lineages or prove state58 Markov sufficiency.')
        write(out/'raw_identity_aliases.json',raw_alias_report)
        audits={str(h):r.audit()|dict(body_ipos_body_m=r.ipos.copy(),joint_range20x2=r.qrange.copy(),actuator_ctrlrange20x2=r.urange.copy()) for h,r in readers.items()}
        counts={k:sum(r.counts[k] for r in readers.values()) for k in next(iter(readers.values())).counts}
        assert counts['integration_steps']<=23040 and len(episodes)==24 and len(rows)<=240
        summary=dict(schema=RESULT_SCHEMA,status='COMPLETE_NEW12_CONTROLLED_FINAL_NATIVE_ACQUISITION',
            protocol=pin(args.protocol),design=protocol['inputs']['design'],family_plan=protocol['inputs']['family_plan'],
            state_schema='phone_COM_world_state58:1',delta_schema='phone_physical_world_delta52:2',
            array_schema='phone_episode_H20ms_arrays:1',outputs=outputs,
            sample_identities=pin(out/'sample_identities.json'),episodes=pin(out/'episodes.json'),
            counts=counts,model_audits=audits,rows=len(rows),families=12,episode_count=24,
            split_counts={s:dict(families=len({e['family_id'] for e in episodes if e['split']==s}),
                rows=sum(r['split']==s for r in rows),eligible_rows=sum(r['split']==s and r['training_eligible_bool'] for r in rows),
                partial_rows=sum(r['split']==s and not r['complete_H_bool'] for r in rows)) for s in [ROLE]},
            exact_alias_groups=aliases,cross_partition_alias_groups=cross,partition_qualified=not cross,
            raw_identity_alias_report=pin(out/'raw_identity_aliases.json'),
            cross_original_input_action_alias_groups=len(raw_alias_report['cross_original960_same_input_action']),
            raw_identity_aliases_absent=(not aliases and not raw_alias_report['fresh_same_input_action_groups'] and not raw_alias_report['cross_original960_same_input_action']),
            planned_maximum_rows=240,planned_maximum_integration_steps=23040,
            unexecuted_planned_intervals=240-len(rows),fresh_controlled_family_validation_only=True,
            no_deduplication_or_reassignment=True,original_holding_gate_qualified=False,
            independent_ego_trajectory_or_real_dynamics_GT_qualified=False,Markov_state58_qualified=False,
            first_new_learning_training_started=False,elapsed_s=time.perf_counter()-started,
            scope='New controlled initial q/actions at the same two known source poses and original model. Bothdt remain grouped. Partial/abort/reset failures retained; no learned training, independent ego/scene, Markov, policy or holding qualification.')
        write(out/'report.json',summary)
        print(json.dumps(dict(status=summary['status'],rows=len(rows),counts=counts,partition_qualified=not cross)),flush=True)
        return summary
    except BaseException as err:
        write(out/'failure.json',dict(phase=phase,error=repr(err),episodes=episodes,returned_rows=len(rows),
            counts={str(h):r.counts for h,r in readers.items()},automatic_retry=False));raise

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',action='store_true');args=ap.parse_args()
 if args.run:run_analysis(args.protocol)
 else:
  protocol,design,plan,out=preflight(args.protocol)
  print(json.dumps(dict(status='METADATA_ONLY_NO_NATIVE_TORCH_NN_ARRAY_PAYLOAD',families=12,episodes=24,max_H_rows=240,max_steps=23040,CPU_run_authorized=protocol['CPU_run_authorized'],raw_plan_aliases=protocol['metadata_alias_receipt'])))
