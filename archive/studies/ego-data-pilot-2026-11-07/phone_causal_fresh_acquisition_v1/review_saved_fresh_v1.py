"""Light saved-file integrity review; no simulator, Torch, model or replay."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,importlib.util,json,sys
import numpy as np
H=Path(__file__).resolve().parent
P=H.parent
ROLE='NEW_FINAL_VALIDATION_SEALED_BEFORE_ACQUISITION'
READS=Counter()

def pin(path):
    path=Path(path).resolve(strict=True);b=path.read_bytes()
    return dict(path=str(path),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(v):
    path=Path(v['path']);assert path.is_absolute() and pin(path)==v
    return path
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def load(v,kind,keys=None):
    READS[kind]+=1
    with np.load(checked(v),allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}
def bits(a,b):
    a,b=np.asarray(a),np.asarray(b)
    assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def receipt(meta,array,tick):
    assert array.dtype==np.float64 and array.shape==(300,)
    assert meta['signature']==16383 and meta['size']==300 and meta['shape']==[300] and meta['dtype']=='<f8'
    assert meta['sha256']==sha(array) and meta['global_tick']==tick
def alias(row,digest):
    return (row['source_after_action_receipt']['sha256'],row['actual_control20_raw_sha256'],row['dt_s'],row['requested_substeps'],digest,row['source_after_action_receipt']['signature'])
def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)

def main():
    out=H/'saved_review_v1';out.mkdir(exist_ok=False)
    try:
        reportpath=H/'acquired_v1/report.json';r=read(reportpath);p=read(checked(r['protocol']))
        assert r['schema']=='phone_new12_controlled_final_acquisition_result:1'
        assert p['CPU_run_authorized'] is True and p['new_training_authorized'] is False
        for v in [*p['inputs'].values(),*p['source_pins'].values()]:checked(v)
        terminal=read(H/'process_run1/terminal.json')
        assert terminal['returncode']==0 and terminal['protocol']==r['protocol'] and terminal['source']==p['source_pins']['collector']
        assert checked(terminal['stderr']).stat().st_size==0;checked(terminal['stdout']);checked(terminal['launcher'])
        plan=read(checked(r['family_plan']));families={f['family_id']:f for f in plan['families']}
        assert set(families)==set(range(1000,1012)) and r['family_plan']==p['inputs']['family_plan']
        ids=read(checked(r['sample_identities']));episodes=read(checked(r['episodes']))
        assert len(episodes)==24 and len(ids)==r['rows']==240
        z=load(r['outputs']['fresh_final'],'aggregate_NPZ_reads')
        widths=dict(x58_F64=58,u20_F64=20,y58_F64=58,target_delta52_F64=52,contact_summary_F64=7)
        for k,a in z.items():
            dtype=np.float64 if k.endswith('_F64') else np.bool_ if k.endswith('_bool') else np.int64
            assert a.dtype==dtype and a.shape==((240,widths[k]) if k in widths else (240,))
        bits(z['row_id_i64'],np.arange(240,dtype=np.int64))
        old=read(checked(p['inputs']['original_acquisition']));oldids=read(checked(p['inputs']['original_identities']))
        for v in old['outputs'].values():checked(v)
        oldkeys=defaultdict(list)
        for row in oldids:
            digest=old['model_audits'][str(row['dt_s'])]['shared_physics_digest_without_dt']
            oldkeys[alias(row,digest)].append(row['sample_id'])
        allkeys=defaultdict(list);transitionkeys=defaultdict(list);seen=[];counts=Counter();terms=Counter();rows=[]
        keys=['source_state58','actual_control20','target_at_H_state58','physical_world_delta52','observed_prefix_target58',
              'source_integration_state','source_after_action_integration_state','target_integration_state',
              'pre_tick','post_tick','post_qpos27','post_qvel26','post_state58','pre_energy2','contact_rows24']
        for ei,e in enumerate(episodes):
            fid=e['family_id'];f=families[fid];dt=e['dt_s'];n=int(round(.02/dt))
            assert e['episode_id']==fid*2+([.0003125,.00015625].index(dt))==2000+ei
            assert e['split']==f['split']==ROLE and e['source_record']==f['source_record']
            assert len(e['interval_row_ids'])==10 and e['reset_termination']=='INITIAL_READY' and not any(e['initial_warning_numbers'])
            initial=load(e['initial'],'initial_NPZ_reads');bits(initial['qpos27'],np.asarray(f['initial_qpos27_F64'],dtype=np.float64))
            bits(initial['qvel26'],np.zeros(26));bits(initial['ctrl20'],np.asarray(f['base_ctrl20_F64'],dtype=np.float64))
            receipt(e['initial_receipt'],initial['integration_state'],0)
            prev=initial['integration_state'];x=initial['state58'];tick=0
            counts['initial_contact_force_reads']+=len(initial['contact_rows24'])
            digest=r['model_audits'][str(dt)]['shared_physics_digest_without_dt']
            assert r['model_audits'][str(dt)]['passed'] is True and digest==old['model_audits'][str(dt)]['shared_physics_digest_without_dt']
            for interval,sid in enumerate(e['interval_row_ids']):
                row=ids[sid];seen.append(sid)
                assert row['sample_id']==sid and row['family_id']==fid and row['episode_id']==e['episode_id'] and row['interval_index']==interval and row['split']==ROLE
                t=load(row['trace'],'trace_NPZ_reads',keys);N=len(t['post_tick']);assert 0<N<=n
                assert row['executed_substeps']==z['executed_substeps_i64'][sid]==N
                assert row['requested_substeps']==z['n_substeps_i64'][sid]==n and z['native_dt_s_F64'][sid]==dt
                assert z['family_id_i64'][sid]==fid and z['episode_id_i64'][sid]==e['episode_id'] and z['interval_index_i64'][sid]==interval
                u=np.asarray(f['held_ctrl10x20_F64'][interval],dtype=np.float64)
                bits(z['u20_F64'][sid],u);bits(t['actual_control20'],u);assert sha(u)==row['actual_control20_raw_sha256']
                bits(z['x58_F64'][sid],x);bits(t['source_state58'],x);bits(t['source_integration_state'],prev)
                bits(t['pre_tick'],np.arange(tick,tick+N,dtype=np.int64));bits(t['post_tick'],np.arange(tick+1,tick+N+1,dtype=np.int64))
                for name,key,tt in [('source_receipt','source_integration_state',tick),('source_after_action_receipt','source_after_action_integration_state',tick),('target_receipt','target_integration_state',tick+N)]:receipt(row[name],t[key],tt)
                finite=all(np.isfinite(t[k]).all() for k in ['post_qpos27','post_qvel26','post_state58','pre_energy2','contact_rows24'])
                complete=N==n;eligible=bool(complete and finite and not any(row['warning_numbers']))
                assert complete==bool(z['complete_H_bool'][sid])==row['complete_H20ms']
                assert eligible==bool(z['training_eligible_bool'][sid])==row['training_eligible']
                if complete:bits(z['y58_F64'][sid],t['post_state58'][-1]);bits(t['target_at_H_state58'],t['post_state58'][-1])
                else:assert np.isnan(z['y58_F64'][sid]).all() and np.isnan(t['target_at_H_state58']).all()
                if eligible:assert np.isfinite(z['target_delta52_F64'][sid]).all()
                else:assert np.isnan(z['target_delta52_F64'][sid]).all()
                bits(z['target_delta52_F64'][sid],t['physical_world_delta52'])
                bits(t['observed_prefix_target58'],t['post_state58'][-1])
                assert 'solver_PRE' in row['force_phase'] and finite
                if row['termination']!='COMPLETE_INTERVAL':assert interval==9 and row['termination']==e['termination']
                allkeys[alias(row,digest)].append(sid)
                transitionkeys[(row['source_after_action_receipt']['sha256'],row['target_receipt']['sha256'],sha(u),dt,n,digest,'phone_COM_world_state58:1',row['source_receipt']['signature'])].append(sid)
                counts['integration_steps']+=N;counts['step_contact_force_reads']+=len(t['contact_rows24'])
                rows.append(dict(sample_id=sid,family_id=fid,dt_s=dt,interval=interval,steps=N,complete=complete,eligible=eligible,termination=row['termination']))
                prev=t['target_integration_state'];x=t['post_state58'][-1];tick+=N
            counts['MjData']+=1;counts['initial_forward']+=1;counts['receipt_reads']+=31;terms[e['termination']]+=1
        assert sorted(seen)==list(range(240))
        counts.update(MJB_reads=2,observer_forward=0,NN_training_GPU=0)
        counts['contact_force_reads']=counts['initial_contact_force_reads']+counts['step_contact_force_reads']
        assert dict(counts)==r['counts'] and counts['integration_steps']==22543<=23040
        assert int(z['training_eligible_bool'].sum())==228 and int((~z['complete_H_bool']).sum())==12
        assert all(len(v)==1 for v in transitionkeys.values()) and all(len(v)==1 for v in allkeys.values())
        assert not any(k in oldkeys for k in allkeys)
        ar=read(checked(r['raw_identity_alias_report']));assert ar['fresh_same_input_action_groups']==[] and ar['cross_original960_same_input_action']==[]
        assert r['exact_alias_groups']==[] and r['cross_partition_alias_groups']==[] and r['raw_identity_aliases_absent'] is True
        assert not any(k in sys.modules for k in ['torch','mujoco','warp'])
        checks={k:True for k in ['source_protocol_terminal_and_original_data_pins','all12fixedfamilies_24episodes_240planned_rows_retained','typed_aggregate_and_true_commands_initials_match_plan','all240source_afteraction_target300receipts_rawhash_continuity','all_integer_PRE_POST_ticks_steps_and_contact_read_counts','228eligible12partial_NaN_Htargets_exact_masks','same_original_physics_bothdt_model_audit_bindings','raw_initial_control_and_fullinput_action_alias_checks_no_dedup','source_no_olddata_writes_and_no_model_result_selection','review_zero_native_Torch_NN_replay']}
        write(out/'row_integrity_summary.json',rows)
        result=dict(schema='phone_fresh12_acquisition_saved_review:1',status='PASS_LIGHT_SAVED_FRESH12_INTEGRITY_NOT_INDEPENDENT_EGO',engineering_qualified=True,checks=checks,passed=len(checks),total=len(checks),producer_report=pin(reportpath),protocol=r['protocol'],source_auditor=pin(__file__),terminal=pin(H/'process_run1/terminal.json'),arrays=r['outputs'],sample_identities=r['sample_identities'],family_plan=r['family_plan'],rows=240,eligible_rows=228,partial_rows=12,episodes=24,families=12,counts=dict(counts),saved_reads=dict(READS),episode_terminations=dict(terms),original_input_action_aliases=0,new_input_action_aliases=0,row_summary=pin(out/'row_integrity_summary.json'),
            scope=['Reviewer authored the thin collector; this is author saved-integrity review independent of Root actual execution, not an independent native investigator.','All aggregate rows,24initial snapshots and240receipt/control/POST/tick traces opened; no simulator or NN. Contact rows used only for counts/finite checks, not new per-pair force/wrench arithmetic.','Full integration receipt/continuity bytes are verified; opaque MuJoCo internal scratch completeness is not newly qualified.','Old48 report/plan/identities and all3 aggregate file hashes verified unchanged; source writes only fresh output, source-backed not a filesystem write audit.','Known2source poses/model; controlled new q/actions only. Partial/abort outcomes retained; no holding, real-force, independent-ego, policy or state58-Markov qualification.'],actual_new_native_Torch_NN_steps=0)
        write(out/'report.json',result);print(json.dumps(dict(report=pin(out/'report.json'),passed=len(checks),total=len(checks),counts=dict(counts),saved_reads=dict(READS))))
    except BaseException as e:
        write(out/'failure.json',dict(error=repr(e),source=pin(__file__),saved_reads=dict(READS),automatic_retry=False,new_native_NN_calls=0));raise
if __name__=='__main__':main()
