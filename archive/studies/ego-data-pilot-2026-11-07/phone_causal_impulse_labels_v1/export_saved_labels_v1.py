"""22-pair physical supervision from saved causal traces; no native/NN calls.

Impulse axes are the FROZEN source BODY axes; torque is about the MOVING COM.
Absent observed contacts have measured zero labels. Unexecuted sides remain
NaN/unknown. These future labels never become current model forward inputs.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def checked(r):assert pin(r['path'])==r,'Changed input '+r['path'];return Path(r['path'])
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def rms(a):return float(np.sqrt(np.mean(np.sum(a*a,axis=-1))))

def decode_window(contact_rows,pre_ticks,COM_pre,R_source,h,pair_ids,phone_gid=71):
    cr=np.asarray(contact_rows,dtype=np.float64);ticks=np.asarray(pre_ticks,dtype=np.int64);com=np.asarray(COM_pre);R=np.asarray(R_source)
    assert cr.ndim==2 and cr.shape[1]==24 and com.shape==(len(ticks),3) and R.shape==(3,3) and len(ticks)>0
    assert np.all(np.diff(ticks)==1) and np.isfinite(cr).all() and np.max(abs(R.T@R-np.eye(3)))<1e-10
    index={int(g):i for i,g in enumerate(pair_ids)};n=len(ticks);J=np.zeros((22,6));Fn=np.zeros((n,22));generated=np.zeros((n,22),dtype=np.int64);world_tick=np.zeros((n,6))
    mask=(cr[:,1]==phone_gid)|(cr[:,2]==phone_gid);c=cr[mask]
    if len(c):
        assert np.all((c[:,1]==phone_gid)^(c[:,2]==phone_gid))
        ti=np.searchsorted(ticks,c[:,0].astype(np.int64));assert np.all(ti<len(ticks)) and np.array_equal(ticks[ti],c[:,0])
        other=np.where(c[:,1]==phone_gid,c[:,2],c[:,1]).astype(np.int64)
        assert all(int(g) in index for g in other),'Every phone boundary contact needs a canonical pair'
        slot=np.array([index[int(g)] for g in other]);sign=np.where(c[:,2]==phone_gid,1.,-1.)
        frame=c[:,9:18].reshape(-1,3,3);force=c[:,18:24]
        F=sign[:,None]*np.einsum('nji,nj->ni',frame,force[:,:3])
        direct=sign[:,None]*np.einsum('nji,nj->ni',frame,force[:,3:])
        M=np.cross(c[:,6:9]-com[ti],F)+direct;W=np.c_[F,M]
        BODY=np.c_[F@R,M@R]
        np.add.at(J,slot,h*BODY);np.add.at(Fn,(ti,slot),np.maximum(force[:,0],0.))
        np.add.at(generated,(ti,slot),1);np.add.at(world_tick,ti,W)
    return dict(impulse_source_BODY_moving_COM22x6=J,mean_positive_Fn22_N=Fn.mean(0),min_positive_Fn22_N=Fn.min(0),
        max_positive_Fn22_N=Fn.max(0),last_PRE_positive_Fn22_N=Fn[-1],Fn_gt_0p1_coverage22=(Fn>.1).mean(0),
        generated_contact_coverage22=(generated>0).mean(0),world_tick_wrench6=world_tick,phone_contact_rows=len(c))

def analytic_checks():
    pair=[15,28,41]+list(range(100,119));ticks=np.array([0]);com=np.zeros((1,3));R=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    def row(g1,g2,frame,tick=0,point=(0,.1,0)):
        return np.r_[tick,g1,g2,3,0,-.000075,point,np.asarray(frame).ravel(),1,0,0,0,0,0]
    a=decode_window(np.array([row(15,71,np.eye(3))]),ticks,com,R,.02,pair)
    b=decode_window(np.array([row(71,15,np.diag([-1,-1,1]))]),ticks,com,R,.02,pair)
    expected=np.array([0.,-.02,0,0,0,-.002]);checks={}
    checks['geom_order_and_frame_sign_fixture']=bool(np.array_equal(a['impulse_source_BODY_moving_COM22x6'],b['impulse_source_BODY_moving_COM22x6']))
    checks['frozen_BODY_axes_and_lever_torque_fixture']=bool(np.max(abs(a['impulse_source_BODY_moving_COM22x6'][0]-expected))<1e-16)
    c=np.array([row(15,71,np.eye(3),tick=0),row(15,71,np.eye(3),tick=1)])
    cc=np.array([[0.,0.,0.],[0.,.01,0.]])
    v=decode_window(c,np.array([0,1]),cc,np.eye(3),.01,pair)
    checks['moving_COM_origin_fixture']=abs(v['impulse_source_BODY_moving_COM22x6'][0,5]+.0019)<1e-16
    checks['observed_absent_pairs_are_zero_only_after_execution']=bool(np.array_equal(v['impulse_source_BODY_moving_COM22x6'][1:],np.zeros((21,6))))
    return checks

def main():
    out=H/'completed_v1';protocol=H/'protocol_v1.json';assert not out.exists() and not protocol.exists()
    pr=dict(schema='causal_phone22_saved_impulse_export_protocol:1',source=pin(__file__),
        inputs={'acquisition_report':pin(P/'phone_causal_train_acquisition_v1/acquired_v1/report.json'),
            'geometry_metadata':pin(P/'phone_geometry_features_v1/completed_v1/compiled_constants.json'),
            'physical_parameters':pin(P/'phone_object_wrench_labels_v1/completed_v1/physical_parameters.json')},
        outputs_scope='All2952 planned rows; 2942 actual sourceBODY/movingCOM observations,10unavailable remainNaN. Current geometry inputs exclude these future targets.',
        normal_load_threshold_N=.1,native_or_NN_training_authorized=False)
    write(protocol,pr);out.mkdir();start=time.perf_counter();checks=analytic_checks();assert all(checks.values())
    meta=read(checked(pr['inputs']['geometry_metadata']));acq=read(checked(pr['inputs']['acquisition_report']));pars=read(checked(pr['inputs']['physical_parameters']))
    ids=[int(v['other_geom_id']) for v in meta['pairs']];names=[v['name'] for v in meta['pairs']]
    assert len(ids)==len(set(ids))==22 and acq['all72_baselines_passed_before_any_side'] is True and acq['partial_executed_branches']==0
    records=read(checked(acq['records']));assert len(records)==2952
    mass=pars['mass_kg'];I=np.array(pars['I_body_3x3']);gravity=np.array(pars['gravity_world_m_s2'])
    fields=['impulse_source_BODY_moving_COM22x6','mean_positive_Fn22_N','min_positive_Fn22_N','max_positive_Fn22_N','last_PRE_positive_Fn22_N','Fn_gt_0p1_coverage22','generated_contact_coverage22']
    values={k:np.full((2952,22,6) if k==fields[0] else (2952,22),np.nan,dtype=np.float64) for k in fields}
    duration=np.zeros(2952);known=np.zeros(2952,dtype=np.bool_);linear=np.full((2952,3),np.nan);angular=linear.copy();wrench_match=0.;read_count=0;ticks_count=0;contacts_count=0;rows=[]
    for i,r in enumerate(records):
        assert r['branch_id']==i
        rows.append({k:r[k] for k in ['branch_id','context_id','partition','original_family_id','kind','executed']})
        if not r['executed']:continue
        assert r['complete_H20ms'] is True and r['training_eligible'] is True
        with np.load(checked(r['output']),allow_pickle=False) as z:
            a={k:z[k].copy() for k in ['source_state58','observed_prefix_target58','post_state58','pre_tick','contact_rows24','pre_contact_wrench_about_COM_world6']}
        read_count+=1;n=len(a['pre_tick']);assert n==r['executed_substeps']==r['n_substeps']
        s=a['source_state58'];y=a['observed_prefix_target58'];R=s[43:52].reshape(3,3);R1=y[43:52].reshape(3,3)
        comp=np.vstack([s[None,40:43],a['post_state58'][:-1,40:43]]);h=r['native_h_s'];T=h*n
        v=decode_window(a['contact_rows24'],a['pre_tick'],comp,R,h,ids,phone_gid=meta['phone_geom_id'])
        for key in fields:values[key][i]=v[key]
        known[i]=True;duration[i]=T;ticks_count+=n;contacts_count+=v['phone_contact_rows']
        wrench_match=max(wrench_match,float(np.max(abs(v['world_tick_wrench6']-a['pre_contact_wrench_about_COM_world6']))))
        impulse=v[fields[0]].sum(0);linear[i]=mass*(y[52:55]-s[52:55])-mass*gravity*T-R@impulse[:3]
        L0=R@I@R.T@s[55:58];L1=R1@I@R1.T@y[55:58];angular[i]=L1-L0-R@impulse[3:]
    checks['all2952_rows_2942_observed_10_unknown']=bool(known.sum()==2942 and (~known).sum()==10 and read_count==2942)
    checks['unavailable_impulse_labels_NaN_not_fake_zero']=bool(np.isnan(values[fields[0]][~known]).all())
    checks['decoded_world_wrench_matches_frozen_reader']=wrench_match<1e-12
    checks['all_actual_H20ms_and_tick_budget']=bool(np.all(duration[known]==.02) and ticks_count==282432)
    arrays=values|dict(branch_id=np.arange(2952,dtype=np.int64),physical_labels_known=known,actual_duration_s=duration,
        linear_momentum_residual_world_Ns=linear,angular_momentum_residual_world_Nms=angular,pair_other_geom_ids=np.array(ids,dtype=np.int64))
    dest=out/'all2952_pair22_physical_labels.npz';np.savez_compressed(dest,**arrays);write(out/'row_identities.json',rows)
    report=dict(schema='causal_phone22_saved_impulse_export_result:1',status='PASS' if all(checks.values()) else 'FAILED_LABEL_CONTRACT',protocol=pin(protocol),
        checks=checks,passed=sum(bool(v) for v in checks.values()),total=len(checks),rows=2952,executed_rows=int(known.sum()),unexecuted_rows=int((~known).sum()),
        pair_names=names,pair_other_geom_ids=ids,mass_kg=mass,gravity_world=gravity.tolist(),physical_parameters_measured=False,
        decoded_wrench_max_abs_difference=wrench_match,linear_momentum_vectorRMS_Ns=rms(linear[known]),angular_momentum_vectorRMS_Nms=rms(angular[known]),
        counts=dict(saved_native_branch_NPZ_reads=read_count,decoded_PRE_ticks=ticks_count,phone_contact_rows=contacts_count,native_FK_forward_steps_NN_AD_training_calls=0),
        frame='FROZEN source BODY axes, torque integrated about each tick MOVING COM; rotate both impulse blocks by sourceR to world',
        coverage_scope='fraction of actualPRE ticks: generated contact rows vs summed positive Fn>.1N are separate fields; no hypothetical zero-contact certificate or holding claim',
        exposure_scope='OriginalTRAIN state families with fit/knownstate-actioncheck roles unchanged. No originalVAL/TEST/freshfinal acquisition/fit.',
        outputs={'all_labels':pin(dest),'row_identities':pin(out/'row_identities.json')},elapsed_s=time.perf_counter()-start)
    write(out/'report.json',report);print(json.dumps(dict(report=pin(out/'report.json'),passed=report['passed'],total=report['total'],reads=read_count)))
    assert all(checks.values()),'Failed export retained without retry'
if __name__=='__main__':main()
