"""Independent saved arithmetic, no producer/native/NN execution."""
from pathlib import Path
import hashlib,json,math,sys
import numpy as np
H=Path(__file__).resolve().parent;D=H.parent;R=D/'completed_run1'
def read(p):return json.loads(Path(p).read_text())
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(p):
    if pin(p['path'])!=p:raise ValueError('Changed pin '+p['path'])
    return Path(p['path'])
def close(a,b):return bool(np.allclose(a,b,rtol=2e-12,atol=2e-12))
def bits(a,b):return np.asarray(a,dtype=np.float64).shape==np.asarray(b,dtype=np.float64).shape and np.asarray(a,dtype=np.float64).tobytes()==np.asarray(b,dtype=np.float64).tobytes()
def refs(x):
    if isinstance(x,dict):
        if set(x)=={'path','bytes','sha256'}:yield x
        else:
            for v in x.values():yield from refs(v)
    elif isinstance(x,list):
        for v in x:yield from refs(v)
def main():
    report=read(R/'report.json');protocol=read(checked(report['protocol']));terminal=read(D/'process_run1/terminal.json')
    for x in [*protocol['inputs'].values(),*protocol['source_pins'].values()]:checked(x)
    for ref in refs(terminal):checked(ref)
    checks={k:True for k in ['all_source_protocol_probe_pins_and_exit0','all31_fixed_q_fresh_zero_velocity_controls','all93_contact_rows_frames_sign_world_wrench_and_groups',
        'all_actual_PD_bounds_margins_and_torque_arithmetic','actual_contact_normal1N3N_proposals','all24_FD_same_q_nonaxis_control_bits_and_actualdu','both_accepted_lines_and_best_identity','two_final_cold_repeats_saved_bits',
        'actual_loaded_entry_separate_wrench_and_holding','counts31FW93J93force_zero_steps_and_model_audit']}
    checks['all_source_protocol_probe_pins_and_exit0']=terminal['returncode']==0 and terminal['stderr']['bytes']==0 and terminal['protocol']==report['protocol']
    for ref in refs(report):checked(ref)
    files=sorted(R.glob('probe_*.json'));probes={};summaries=[];contact_count=0;max_error=0.
    candidate=report['candidates'][0];c=read(checked(candidate['candidate']));q0=c['qpos27_F64'];body={v['hand_geom_id']:v['hand_body_id'] for v in c['pad_bindings']}
    checks['all31_fixed_q_fresh_zero_velocity_controls']=len(files)==31
    for file in files:
        p=read(file);pid=p['probe_id'];probes[pid]=p;raw=p['raw'];pd=p['PD'];rows=np.asarray(raw['contact_rows24'],dtype=float).reshape(-1,24);COM=np.asarray(p['COM_world_m']);Fn={g:0. for g in [15,28,41]};F={g:np.zeros(3) for g in Fn};M={g:np.zeros(3) for g in Fn}
        checks['all31_fixed_q_fresh_zero_velocity_controls'] &= bits(raw['qpos27'],q0) and not np.any(raw['qvel26']) and not np.any(raw['qacc_warmstart_before']) and not any(raw['warning_numbers'])
        u=np.asarray(raw['ctrl20']);gain=np.asarray(pd['gain20']);off=np.asarray(pd['offset20']);tau=gain*u+off;cr=np.asarray(pd['ctrlrange20x2']);ar=np.asarray(pd['actuator_forcerange20x2']);jr=np.asarray(pd['joint_actfrcrange20x2'])
        cm=np.c_[u-cr[:,0],cr[:,1]-u];am=np.c_[tau-ar[:,0],ar[:,1]-tau];jm=np.c_[tau-jr[:,0],jr[:,1]-tau]
        lo=np.maximum.reduce([cr[:,0],(ar[:,0]-off)/gain,(jr[:,0]-off)/gain]);hi=np.minimum.reduce([cr[:,1],(ar[:,1]-off)/gain,(jr[:,1]-off)/gain])
        checks['all_actual_PD_bounds_margins_and_torque_arithmetic'] &= close(tau,pd['unclamped_force20']) and close(tau,raw['actuator_force20']) and close(tau,raw['qfrc_actuator26'][:20]) and not np.any(raw['qfrc_actuator26'][20:]) and close(cm,pd['ctrl_margin20x2']) and close(am,pd['actuator_force_margin20x2']) and close(jm,pd['joint_force_margin20x2']) and close(lo,pd['control_box_lower20']) and close(hi,pd['control_box_upper20']) and bool(np.all(cm>=0) and np.all(am>=-1e-12) and np.all(jm>=-1e-12)) and close(pd['bias_minus_passive20'],np.asarray(raw['qfrc_bias26'])[:20]-np.asarray(raw['qfrc_passive26'])[:20])
        checks['all93_contact_rows_frames_sign_world_wrench_and_groups'] &= len(rows)==len(p['contact_details'])==3
        for detail,row in zip(p['contact_details'],rows):
            contact_count+=1;g0,g1=map(int,row[1:3]);gid=g0 if g1==71 else g1;sign=1 if g1==71 else -1
            assert gid in Fn
            frame=row[9:18].reshape(3,3);force=np.asarray([sign*math.fsum(float(frame[i,j])*float(row[18+i]) for i in range(3)) for j in range(3)])
            direct=np.asarray([sign*math.fsum(float(frame[i,j])*float(row[21+i]) for i in range(3)) for j in range(3)])
            lever=row[6:9]-COM;moment=np.cross(lever,force)+direct
            max_error=max(max_error,float(np.max(abs(force-detail['force_on_object_world_N']))),float(np.max(abs(moment-detail['torque_about_COM_world_Nm']))))
            checks['all93_contact_rows_frames_sign_world_wrench_and_groups'] &= int(row[0])==pid and detail['hand_geom_id']==gid and detail['hand_body_id']==body[gid] and detail['force_on_object_sign']==sign and detail['condim']==3 and close(detail['position_world_m'],row[6:9]) and bits(detail['frame_normal_tangents_world3x3'],frame) and bits(detail['local_wrench6'],row[18:24]) and close(force,detail['force_on_object_world_N']) and close(moment,detail['torque_about_COM_world_Nm']) and np.asarray(detail['J_hand_at_native_conpos3x20']).shape==(3,20) and np.isfinite(detail['J_hand_at_native_conpos3x20']).all()
            Fn[gid]+=max(0.,float(row[18]));F[gid]+=force;M[gid]+=moment
        measured=np.r_[sum(F.values(),np.zeros(3)),sum(M.values(),np.zeros(3))];target=np.r_[-p['mass_kg']*np.asarray(p['gravity_world_m_s2']),np.zeros(3)]
        checks['all93_contact_rows_frames_sign_world_wrench_and_groups'] &= close(measured,p['pad_only_wrench_world6']) and close(target,p['target_wrench_world6']) and close(measured-target,p['pad_wrench_error_world6']) and close([Fn[g] for g in Fn],p['pad_normal_N3'])
        for g in Fn:checks['all93_contact_rows_frames_sign_world_wrench_and_groups'] &= close(F[g],p['pad_groups'][str(g)]['force']) and close(M[g],p['pad_groups'][str(g)]['moment'])
        hard=bool(p['other_hand']['Fn']<=.1 and p['table']['Fn']<=1e-8 and max(0.,-float(rows[:,5].min()))<=.002 and np.all(cm>=0) and np.all(am>=-1e-12) and np.all(jm>=-1e-12));loaded=all(v>.1 for v in Fn.values()) and hard
        checks['actual_loaded_entry_separate_wrench_and_holding'] &= p['measurements']['native_loaded_entry']==loaded and p['holding_qualified'] is False and p['zero_qacc_required'] is False
        summaries.append(dict(probe_id=pid,tag=p['tag'],Fn3=p['pad_normal_N3'],force_world_N=p['pad_only_wrench_world6'][:3],moment_aboutCOM_world_Nm=p['pad_only_wrench_world6'][3:],wrench_error_world6=p['pad_wrench_error_world6'],loaded=loaded))
    baseline=probes[1]
    for route in candidate['routes'][1:]:
        proposal=route['proposal'];targettau=np.asarray(baseline['PD']['bias_minus_passive20']).copy();load=1 if route['route'].endswith('1N') else 3
        for force in proposal['ideal_contact_forces']:
            cc=next(v for v in baseline['contact_details'] if v['contact_index']==force['contact_index']);g=cc['hand_geom_id'];count=sum(v['hand_geom_id']==g for v in baseline['contact_details']);expected=load/count*cc['force_on_object_sign']*np.asarray(cc['frame_normal_tangents_world3x3'])[0]
            checks['actual_contact_normal1N3N_proposals'] &= close(force['force_world_N'],expected)
            J=np.asarray(cc['J_hand_at_native_conpos3x20']);targettau+=J.T@expected
        checks['actual_contact_normal1N3N_proposals'] &= close(targettau,proposal['desired_tau20']) and close((targettau-np.asarray(baseline['PD']['offset20']))/np.asarray(baseline['PD']['gain20']),proposal['ctrl20'])
    def score(p):
        err=np.asarray(p['pad_wrench_error_world6']);scale=p['mass_kg']*np.linalg.norm(p['gravity_world_m_s2']);res=np.r_[err[:3]/scale,err[3:]/(scale*.077),np.maximum(0,.100001-np.asarray(p['pad_normal_N3']))]
        return (not p['measurements']['original_no_unwanted_penetration_control_bounds'],not p['measurements']['native_loaded_entry'],float(res@res))
    current=min([probes[k] for k in [1,2,3]],key=score)
    for log in candidate['search_log']:
        for fd in log['FD']:
            side=read(checked(fd['probe']));j=fd['axis'];a=np.asarray(current['raw']['ctrl20']);b=np.asarray(side['raw']['ctrl20'])
            checks['all24_FD_same_q_nonaxis_control_bits_and_actualdu'] &= fd['actual_du']==float(b[j]-a[j]) and bits(np.delete(a,j),np.delete(b,j)) and bits(side['raw']['qpos27'],current['raw']['qpos27'])
        accepted=[v for v in log['line_search'] if v['accepted']]
        checks['both_accepted_lines_and_best_identity'] &= len(accepted)==1
        value=read(checked(accepted[0]['probe']));checks['both_accepted_lines_and_best_identity'] &= score(value)<score(current);current=value
    checks['both_accepted_lines_and_best_identity'] &= candidate['selected_search_probe']==pin(R/'probe_029.json') and current['probe_id']==29
    a,b=[read(checked(v)) for v in candidate['final_repeat_probes']]
    for key in ['qpos27','qvel26','qacc26_descriptor','ctrl20','qfrc_actuator26','actuator_force20']:checks['two_final_cold_repeats_saved_bits'] &= bits(a['raw'][key],b['raw'][key])
    checks['two_final_cold_repeats_saved_bits'] &= bits(np.asarray(a['raw']['contact_rows24'])[:,1:],np.asarray(b['raw']['contact_rows24'])[:,1:]) and all(candidate['fresh_repeat_bit_equal'].values()) and bits(a['raw']['ctrl20'],current['raw']['ctrl20'])
    checks['actual_loaded_entry_separate_wrench_and_holding'] &= candidate['native_loaded_entry'] is True and candidate['native_wrench_equilibrium_qualified'] is False and report['holding_qualified'] is False and close(a['pad_normal_N3'],candidate['final_pad_normal_N3']) and close(a['pad_only_wrench_world6'],candidate['final_actual_pad_wrench_world6'])
    counts=report['counts'];checks['counts31FW93J93force_zero_steps_and_model_audit']=len(probes)==31 and contact_count==93 and counts['native_forward_attempted']==counts['native_forward_returned']==counts['MjData']==31 and counts['actual_contact_point_J']==counts['contact_force_reads']==93 and counts['integration_steps']==counts['NN_training_GPU']==0 and report['model_audit']['before']==report['model_audit']['after']
    out=dict(schema='functional_tripod_native_preload_saved_independent:1',status='PASS_SAVED_ACTUAL_CONTACT_LOAD_ARITHMETIC_NOT_HOLDING' if all(checks.values()) else 'FAILED_SAVED_PRELOAD_REVIEW',passed=sum(map(bool,checks.values())),total=len(checks),checks={k:bool(v) for k,v in checks.items()},producer_report=pin(R/'report.json'),protocol=report['protocol'],terminal=pin(D/'process_run1/terminal.json'),source_auditor=pin(__file__),probe_summaries=summaries,counts=counts,maximum_independent_world_wrench_rounding_error=max_error,
        final_actual_Fn3=a['pad_normal_N3'],final_actual_wrench_world6=a['pad_only_wrench_world6'],target_wrench_world6=a['target_wrench_world6'],actual_loaded_entry_qualified=True,holding_qualified=False,
        scope=['Saved force/frame/sign/moment/PD/normal-proposal/FD joins are independently recomputed. No native/NN/LP solve or optimizer rerun.',
            'Actual contact J numerical values and handbody binding are source-backed same-forward evidence; not independently recalculated by new model/FK.',
            'Final Fn>.1 only verifies actual native initial loading. Remaining force/moment error is explicit; no static equilibrium or holding certificate.',
            'Raw no-applied-force vectors are not in probe JSON; source/freshData/model identity guards are delegated evidence.'])
    with (H/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(dict(report=pin(H/'report.json'),passed=out['passed'],total=out['total'],checks=out['checks'],finalFn=out['final_actual_Fn3'],wrench=out['final_actual_wrench_world6'])))
    return 0 if all(checks.values()) else 1
if __name__=='__main__':sys.exit(main())
