"""UNFROZEN: measured native three-pad preload, fixed pose, zero integration.

Construction/evaluate explicitly load MuJoCo and run native FW. Import is inert.
No material point A is used for forces/Jacobians. Ideal normal proposals are
separate from fresh native measurements; no all-qacc-zero entry criterion.
"""
from pathlib import Path
import hashlib,importlib.util,json,math,sys
H=Path(__file__).resolve().parent;P=H.parent
HOLDING=P/'functional_tripod_dynamics_v1/run_functional_tripod_dynamics_v3.py'
HOLDING_SHA='d433a51df307bf45b40e70bc2c74776654eb2ece65bd2de67ae8163c201fd960'
PADS=[15,28,41]
class ProbeBudget(RuntimeError):pass
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def clean(x):
    if hasattr(x,'tolist'):return clean(x.tolist())
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return None
    return x
def write(p,value):
    with Path(p).open('x',encoding='utf8') as f:json.dump(clean(value),f,indent=2,allow_nan=False)
def checked(r):
    if pin(r['path'])!=r:raise ValueError('Changed pin '+r['path'])
    return Path(r['path'])

class NativeTripodPreloadEvaluator:
    def __init__(self,model_pin,binding_pin,*,dt_s=.00015625,max_forward_per_candidate=40,output_dir=None):
        import mujoco
        import numpy as np
        if pin(HOLDING)['sha256']!=HOLDING_SHA:raise ValueError('Frozen original physics helper')
        spec=importlib.util.spec_from_file_location('tripod_preload_frozen_physics',HOLDING);holding=importlib.util.module_from_spec(spec);spec.loader.exec_module(holding)
        self.mj,self.np,self.h=mujoco,np,holding;binding=json.loads(checked(binding_pin).read_text())
        if binding['model']!=model_pin:raise ValueError('Same compiled model authority')
        if dt_s!=.00015625 or type(max_forward_per_candidate) is not int or max_forward_per_candidate!=40:raise ValueError('Fixed fine-dt/40FW bounded experiment')
        self.m=mujoco.MjModel.from_binary_path(str(checked(model_pin)));m=self.m
        self.phone,self.phone_geom,self.table,self.pads=holding.assert_model(m,mujoco,np,binding)
        m.opt.timestep=dt_s;self.model_digest=holding.model_digest_except_dt(m,np)
        self.max_forward=max_forward_per_candidate;self.output_dir=None if output_dir is None else Path(output_dir)
        self.per_candidate={};self.counts=dict(MJB_reads=1,MjData=0,native_forward_attempted=0,native_forward_returned=0,actual_contact_point_J=0,contact_force_reads=0,integration_steps=0,NN_training_GPU=0)
    def control_box(self,qpos27):
        np,m=self.np,self.m;q=np.asarray(qpos27,dtype=np.float64)
        if q.shape!=(27,) or not np.isfinite(q).all() or np.any(q[:20]<m.jnt_range[:20,0]) or np.any(q[:20]>m.jnt_range[:20,1]) or abs(q[23:27]@q[23:27]-1)>1e-12:raise ValueError('Finite original hardware pose/native quaternion')
        gain=m.actuator_gainprm[:,0].copy();offset=m.actuator_biasprm[:,0]+m.actuator_biasprm[:,1]*q[:20]
        if np.any(gain<=0):raise ValueError('Positive native gains')
        lo=np.maximum.reduce([m.actuator_ctrlrange[:,0],(m.actuator_forcerange[:,0]-offset)/gain,(m.jnt_actfrcrange[:20,0]-offset)/gain])
        hi=np.minimum.reduce([m.actuator_ctrlrange[:,1],(m.actuator_forcerange[:,1]-offset)/gain,(m.jnt_actfrcrange[:20,1]-offset)/gain])
        if np.any(lo>hi):raise ValueError('Empty real PD/control box')
        return lo,hi
    def evaluate(self,qpos27,ctrl20,*,candidate_id,tag,target_wrench_world6=None):
        np,m,mj=self.np,self.m,self.mj;q=np.asarray(qpos27,dtype=np.float64).copy();u=np.asarray(ctrl20,dtype=np.float64).copy();lo,hi=self.control_box(q)
        if u.shape!=(20,) or not np.isfinite(u).all() or np.any(u<lo) or np.any(u>hi):raise ValueError('Actual command outside real PD/actuator/generalized bounds; no clamp repair')
        used=self.per_candidate.get(candidate_id,0)
        if used>=self.max_forward:raise ProbeBudget('Total40 nativeFW per candidate includes routes/search/final repeats')
        self.per_candidate[candidate_id]=used+1;self.counts['native_forward_attempted']+=1;index=self.counts['native_forward_attempted']
        d=mj.MjData(m);self.counts['MjData']+=1;d.qpos[:]=q;d.qvel[:]=0;d.ctrl[:]=u;d.qacc_warmstart[:]=0
        mj.mj_forward(m,d);self.counts['native_forward_returned']+=1
        if not all(np.isfinite(v).all() for v in [d.qpos,d.qvel,d.qacc,d.qfrc_bias,d.qfrc_passive,d.qfrc_actuator,d.actuator_force]):raise ValueError('Nonfinite native probe')
        if np.any(d.qvel) or np.any(d.qfrc_applied) or np.any(d.xfrc_applied) or np.any(d.warning.number):raise ValueError('Probe velocity/aid/warning')
        if not np.allclose(d.actuator_length,q[:20],rtol=0,atol=1e-14):raise ValueError('Direct-joint native actuator coordinate mismatch')
        gain=m.actuator_gainprm[:,0].copy();offset=m.actuator_biasprm[:,0]+m.actuator_biasprm[:,1]*d.actuator_length+m.actuator_biasprm[:,2]*d.actuator_velocity
        rawPD=gain*u+offset;cm=np.c_[u-m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]-u]
        am=np.c_[rawPD-m.actuator_forcerange[:,0],m.actuator_forcerange[:,1]-rawPD];jm=np.c_[rawPD-m.jnt_actfrcrange[:20,0],m.jnt_actfrcrange[:20,1]-rawPD]
        COM=d.xipos[self.phone].copy();mg=float(m.body_mass[self.phone])*m.opt.gravity.copy()
        target=np.r_[-mg,np.zeros(3)] if target_wrench_world6 is None else np.asarray(target_wrench_world6,dtype=np.float64)
        if target.shape!=(6,) or not np.isfinite(target).all():raise ValueError('Explicit finite world target wrench')
        group={str(g):dict(Fn=0.,force=np.zeros(3),moment=np.zeros(3),contact_count=0) for g in PADS}
        other=dict(Fn=0.,force=np.zeros(3),moment=np.zeros(3));table=dict(Fn=0.,force=np.zeros(3),moment=np.zeros(3));rows=[];details=[];penetration=selfpen=0.
        for ci in range(d.ncon):
            c=d.contact[ci];g0,g1=map(int,c.geom);force=np.empty(6);mj.mj_contactForce(m,d,ci,force);self.counts['contact_force_reads']+=1
            if not np.isfinite(np.r_[c.dist,c.pos,c.frame,force]).all():raise ValueError('Nonfinite contact evidence')
            rows.append(np.r_[index,g0,g1,int(c.dim),int(c.efc_address),c.dist,c.pos,c.frame,force]);penetration=max(penetration,-float(c.dist))
            if self.phone_geom not in (g0,g1):
                if self.table not in (g0,g1):selfpen=max(selfpen,-float(c.dist))
                continue
            other_geom=g0 if g1==self.phone_geom else g1;sign=1 if g1==self.phone_geom else -1
            frame=c.frame.reshape(3,3);F=sign*(frame.T@force[:3]);direct=sign*(frame.T@force[3:]);M=np.cross(c.pos-COM,F)+direct
            bucket=group[str(other_geom)] if other_geom in PADS else table if other_geom==self.table else other
            bucket['Fn']+=max(0.,float(force[0]));bucket['force']+=F;bucket['moment']+=M
            if other_geom in PADS:bucket['contact_count']+=1
            J=None
            if other_geom!=self.table:
                jp=np.zeros((3,m.nv));mj.mj_jac(m,d,jp,None,c.pos,int(m.geom_bodyid[other_geom]));self.counts['actual_contact_point_J']+=1;J=jp[:,:20].copy()
            details.append(dict(contact_index=ci,hand_geom_id=other_geom,hand_body_id=int(m.geom_bodyid[other_geom]),designated=other_geom in PADS,
                position_world_m=c.pos.copy(),frame_normal_tangents_world3x3=frame.copy(),force_on_object_sign=sign,
                friction5=c.friction.copy(),condim=int(c.dim),efc_address=int(c.efc_address),distance_m=float(c.dist),local_wrench6=force.copy(),
                force_on_object_world_N=F,direct_torque_on_object_world_Nm=direct,torque_about_COM_world_Nm=M,J_hand_at_native_conpos3x20=J,
                scope='Generated native con.pos, not a proven unique closest surface pair or finite pressure patch. No positiveFn filter removes raw contacts.'))
        padwrench=np.r_[sum((v['force'] for v in group.values()),np.zeros(3)),sum((v['moment'] for v in group.values()),np.zeros(3))]
        Fn=np.array([group[str(g)]['Fn'] for g in PADS]);loaded=bool(np.all(Fn>.1));all_generated=all(group[str(g)]['contact_count']>0 for g in PADS)
        hard=bool(other['Fn']<=.1 and table['Fn']<=1e-8 and penetration<=.002 and selfpen<=.0002 and np.all(cm>=0) and np.all(am>=-1e-12) and np.all(jm>=-1e-12))
        out=dict(schema='functional_tripod_native_preload_probe:1',probe_id=index,candidate_id=candidate_id,tag=tag,phase='FRESH_ZERO_VELOCITY_NATIVE_FORWARD',
            raw=dict(qpos27=q,qvel26=d.qvel.copy(),qacc26_descriptor=d.qacc.copy(),ctrl20=u,qacc_warmstart_before=np.zeros(26),qacc_warmstart_after=d.qacc_warmstart.copy(),qfrc_bias26=d.qfrc_bias.copy(),qfrc_passive26=d.qfrc_passive.copy(),qfrc_actuator26=d.qfrc_actuator.copy(),actuator_force20=d.actuator_force.copy(),contact_rows24=np.asarray(rows).reshape(-1,24),warning_numbers=d.warning.number.copy()),
            contact_details=details,COM_world_m=COM,phone_R_body_world=d.xmat[self.phone].reshape(3,3).copy(),mass_kg=float(m.body_mass[self.phone]),gravity_world_m_s2=m.opt.gravity.copy(),
            pad_groups=group,pad_normal_N3=Fn,pad_only_wrench_world6=padwrench,target_wrench_world6=target,
            pad_wrench_error_world6=padwrench-target,other_hand=other,table=table,
            PD=dict(gain20=gain,offset20=offset,bias_minus_passive20=d.qfrc_bias[:20]-d.qfrc_passive[:20],control_box_lower20=lo,control_box_upper20=hi,
                ctrlrange20x2=m.actuator_ctrlrange.copy(),actuator_forcerange20x2=m.actuator_forcerange.copy(),joint_actfrcrange20x2=m.jnt_actfrcrange[:20].copy(),
                unclamped_force20=rawPD,ctrl_margin20x2=cm,actuator_force_margin20x2=am,joint_force_margin20x2=jm),
            measurements=dict(all3_generated=bool(all_generated),all3_Fn_gt_0p1N=loaded,original_no_unwanted_penetration_control_bounds=hard,
                native_loaded_entry=loaded and hard,max_native_penetration_m=penetration,max_self_penetration_m=selfpen),
            holding_qualified=False,zero_qacc_required=False,counts=dict(self.counts))
        if self.output_dir is not None:
            f=self.output_dir/('probe_'+str(index).zfill(3)+'.json');write(f,out);out['saved_probe']=pin(f)
        return out
    def audit_model(self):
        now=self.h.model_digest_except_dt(self.m,self.np)
        if now!=self.model_digest:raise ValueError('Original model changed')
        return dict(before=self.model_digest,after=now,unchanged_except_explicit_dt=True,counts=dict(self.counts),per_candidate_forward=dict(self.per_candidate))

def normal_contact_proposal(probe,normal_N_each_pad,*,target_wrench_world6=None):
    """Per-pad total normal split over its ACTUAL generated contacts; ideal only."""
    import numpy as np
    if normal_N_each_pad not in (1.,3.):raise ValueError('Only preregistered1N/3N')
    contacts=[c for c in probe['contact_details'] if c['designated']]
    if any(not any(c['hand_geom_id']==g for c in contacts) for g in PADS):return dict(available=False,reason='MISSING_NATIVE_DESIGNATED_CONTACT',ctrl20=None)
    tau=np.asarray(probe['PD']['bias_minus_passive20'],dtype=np.float64).copy();forces=[]
    for c in contacts:
        count=sum(v['hand_geom_id']==c['hand_geom_id'] for v in contacts)
        F=normal_N_each_pad/count*c['force_on_object_sign']*np.asarray(c['frame_normal_tangents_world3x3'])[0]
        tau+=np.asarray(c['J_hand_at_native_conpos3x20']).T@F;forces.append(dict(contact_index=c['contact_index'],geom_id=c['hand_geom_id'],force_world_N=F))
    pd=probe['PD'];u=(tau-np.asarray(pd['offset20']))/np.asarray(pd['gain20']);lo=np.asarray(pd['control_box_lower20']);hi=np.asarray(pd['control_box_upper20'])
    available=bool(np.isfinite(u).all() and np.all(u>=lo) and np.all(u<=hi))
    return dict(available=available,reason=None if available else 'INVERSE_PD_OUT_OF_TRUE_BOUNDS_NO_CLAMP',ctrl20=u if available else None,raw_inverse_ctrl20=u,
        desired_tau20=tau,ideal_contact_forces=forces,target_wrench_world6=probe['target_wrench_world6'] if target_wrench_world6 is None else target_wrench_world6,
        actual_source_probe=probe.get('saved_probe'),native_force_or_holding_qualified=False,
        scope='Ideal normal load at same-forward actual native con.pos; fresh native evaluate is required. Per-pad total1N/3N, never per-contact multiplied budget.')
