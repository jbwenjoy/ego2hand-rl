"""Execute the frozen48-family plan and export grouped H20ms phone transitions."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,json,time
import numpy as np
from native_phone_reader import NativePhoneReader,pin

H=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def clean(v):
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,np.bool_):return bool(v)
    return v
def write(p,j):
    with Path(p).open('x',encoding='utf8') as f:json.dump(clean(j),f,indent=2,allow_nan=False)
def save_npz(p,**arrays):
    with Path(p).open('xb') as f:np.savez_compressed(f,**arrays)
def checked(r):
    assert pin(r['path'])==r,'Changed pinned input '+r['path'];return Path(r['path'])
def array_sha(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def receipt_json(r):
    return {k:v for k,v in r.items() if k!='integration_state'}|{
        'dtype':r['integration_state'].dtype.str,'shape':list(r['integration_state'].shape)}

def reset_with_physical_diagnostics(reader,qpos,ctrl):
    """Preserve a physically invalid initial episode without hiding structural errors."""
    try:return reader.reset(qpos,ctrl)
    except ValueError as err:
        known=str(err) in ['Known no-aid initial native state','Known finite initial contacts']
        d=reader.d
        if not known or d is None or np.any(d.xfrc_applied) or np.any(d.qfrc_applied):raise
        if str(err)=='Known no-aid initial native state' and not np.any(d.warning.number):raise
        # These are additional reads of the failed RESET_FORWARD context only.
        # No forward, pose overwrite, native step or retry of this episode.
        contact=[]
        for ci in range(d.ncon):
            c=d.contact[ci];g1,g2=map(int,c.geom);f=np.empty(6)
            reader.mj.mj_contactForce(reader.m,d,ci,f)
            reader.counts['contact_force_reads']+=1;reader.counts['initial_contact_force_reads']+=1
            contact.append(np.r_[0,g1,g2,int(c.dim),int(c.efc_address),c.dist,c.pos,c.frame,f])
        try:x=reader.state58()
        except ValueError:x=np.full(58,np.nan)
        reader.terminated=True
        return dict(state58=x,receipt=reader.receipt(),qpos27=d.qpos.copy(),qvel26=d.qvel.copy(),ctrl20=d.ctrl.copy(),
            initial_contact_rows24=np.asarray(contact,dtype=np.float64).reshape(-1,24),
            initial_pad_normal_N2=np.full(2,np.nan),initial_phone_nonpad_normal_N=float('nan'),
            initial_table_normal_N=float('nan'),contact_phase='FAILED_RESET_FORWARD; force summaries unknown; raw retained',
            reset_termination='ABORT_INITIAL_NATIVE_WARNING' if np.any(d.warning.number) else 'ABORT_INITIAL_NONFINITE_CONTACT',
            reset_error=str(err),warning_numbers=np.array(d.warning.number))

def preflight(p):
    protocol=read(p)
    assert protocol['schema']=='phone_episode_acquisition_protocol:1'
    assert type(protocol['CPU_run_authorized']) is bool and protocol['new_training_authorized'] is False
    for r in [*protocol['source_pins'].values(),*protocol['inputs'].values()]:checked(r)
    assert protocol['source_pins']['collector']==pin(__file__)
    design=read(checked(protocol['inputs']['design']));plan=read(checked(protocol['inputs']['family_plan']))
    assert design['family_plan']==protocol['inputs']['family_plan']
    assert plan['schema']=='phone_episode_family_plan:1' and len(plan['families'])==48
    assert plan['maximum_native_integration_steps']==92160 and plan['intervals_per_episode']==10
    assert Counter(f['split'] for f in plan['families'])=={'train':36,'validation':6,'test':6}
    qrange=np.array(design['actual_limits']['joint_range20x2_F64'])
    urange=np.array(design['actual_limits']['actuator_ctrlrange20x2_F64'])
    qhash=[];uhash=[]
    for i,f in enumerate(plan['families']):
        assert f['family_id']==i and f['native_dt_s']==[.0003125,.00015625] and f['substeps_per_H']==[64,128]
        old=read(checked(f['source_record']))['native_probe']['raw']
        q=np.array(f['initial_qpos27_F64']);u=np.array(f['held_ctrl10x20_F64']);base=np.array(f['base_ctrl20_F64'])
        assert q.shape==(27,) and u.shape==(10,20) and base.shape==(20,)
        assert np.isfinite(q).all() and np.isfinite(u).all()
        assert np.all(q[:20]>=qrange[:,0]) and np.all(q[:20]<=qrange[:,1])
        assert np.all(u>=urange[:,0]) and np.all(u<=urange[:,1])
        assert np.max(abs(q[:8]-np.array(old['qpos27'])[:8]))<=.005+1e-12
        assert q[8:].tobytes()==np.array(old['qpos27'])[8:].tobytes()
        assert base.tobytes()==np.array(old['ctrl20']).tobytes()
        assert np.max(abs(u[:,:8]-base[:8]))<=.05+1e-12 and np.max(abs(u[:,8:]-base[8:]))<=.025+1e-12
        assert array_sha(q)==f['initial_qpos27_raw_F64_sha256']
        assert array_sha(u)==f['held_ctrl10x20_raw_F64_sha256']
        assert not any(f['initial_qvel26_F64'])
        qhash.append(array_sha(q));uhash.append(array_sha(u))
    assert len(set(qhash))==48 and len(set(uhash))==48
    proof=read(checked(protocol['inputs']['reader_contract']))
    assert proof['status']=='PASS_192_NATIVE_STEP_GOLDEN_READER_CONTRACT'
    assert proof['passed']==proof['total']==4 and all(v is True for v in proof['checks'].values())
    assert proof['counts']['integration_steps']==192
    assert proof['reader']==protocol['source_pins']['reader']
    out=Path(protocol['output']);assert out.is_absolute() and not out.exists()
    return protocol,design,plan,out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',action='store_true');args=ap.parse_args()
    protocol,design,plan,out=preflight(args.protocol)
    if not args.run:
        print(json.dumps(dict(status='METADATA_ONLY_NO_MODEL',families=48,episodes=96,max_steps=92160)));return
    assert protocol['CPU_run_authorized'] is True
    out.mkdir();write(out/'protocol.json',protocol);started=time.perf_counter()
    readers={};rows=[];identities=[];episodes=[];source_aliases=defaultdict(list);phase='load_models'
    try:
        for h in plan['native_dt_s']:
            readers[h]=NativePhoneReader(checked(protocol['inputs']['model_mjb']),dt_s=h)
        assert len({r.shared_physics_digest for r in readers.values()})==1
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
        for split in ['train','validation','test']:
            chosen=[v for v in rows if v['split']==split]
            a={k:np.asarray([r[k] for r in chosen],dtype=np.float64 if k in f64 else np.bool_ if k in bools else np.int64) for k in keys}
            for key,width in widths.items():a[key]=a[key].reshape(-1,width)
            dest=out/f'{split}.npz';save_npz(dest,**a);outputs[split]=pin(dest)
        aliases=[dict(sample_ids=rr,family_ids=sorted({identities[i]['family_id'] for i in rr}),
                      splits=sorted({identities[i]['split'] for i in rr})) for rr in source_aliases.values() if len(rr)>1]
        cross=[v for v in aliases if len(v['splits'])>1]
        write(out/'sample_identities.json',identities);write(out/'episodes.json',episodes)
        audits={str(h):r.audit()|dict(body_ipos_body_m=r.ipos.copy(),joint_range20x2=r.qrange.copy(),actuator_ctrlrange20x2=r.urange.copy()) for h,r in readers.items()}
        counts={k:sum(r.counts[k] for r in readers.values()) for k in next(iter(readers.values())).counts}
        assert counts['integration_steps']<=92160 and len(episodes)==96
        summary=dict(schema='phone_episode_acquisition_result:1',status='COMPLETE_GROUPED_NATIVE_ACQUISITION',
            protocol=pin(args.protocol),design=protocol['inputs']['design'],family_plan=protocol['inputs']['family_plan'],
            state_schema='phone_COM_world_state58:1',delta_schema='phone_physical_world_delta52:2',
            array_schema='phone_episode_H20ms_arrays:1',outputs=outputs,
            sample_identities=pin(out/'sample_identities.json'),episodes=pin(out/'episodes.json'),
            counts=counts,model_audits=audits,rows=len(rows),families=48,episode_count=96,
            split_counts={s:dict(families=len({e['family_id'] for e in episodes if e['split']==s}),
                rows=sum(r['split']==s for r in rows),eligible_rows=sum(r['split']==s and r['training_eligible_bool'] for r in rows),
                partial_rows=sum(r['split']==s and not r['complete_H_bool'] for r in rows)) for s in ['train','validation','test']},
            exact_alias_groups=aliases,cross_partition_alias_groups=cross,partition_qualified=not cross,
            no_deduplication_or_reassignment=True,original_holding_gate_qualified=False,
            independent_ego_trajectory_or_real_dynamics_GT_qualified=False,Markov_state58_qualified=False,
            first_new_learning_training_started=False,elapsed_s=time.perf_counter()-started,
            scope='Local native simulated episodes around two mapped ego-frame poses. Bothdt in each family together. Partial records retained with NaN H targets; eligibility is full finite H/no-warning, not safety or holding. No learned model or policy benefit.')
        write(out/'report.json',summary)
        print(json.dumps(dict(status=summary['status'],rows=len(rows),counts=counts,partition_qualified=not cross)),flush=True)
    except BaseException as err:
        write(out/'failure.json',dict(phase=phase,error=repr(err),episodes=episodes,returned_rows=len(rows),
            counts={str(h):r.counts for h,r in readers.items()},automatic_retry=False));raise

if __name__=='__main__':main()
