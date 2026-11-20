"""Recompute multiclip reference geometry and nominal support on the actual guarded GPU backend."""
from pathlib import Path
import json,sys
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import mujoco
import collect_guarded_finger_response as b
from build_pen_capacity_targets import compute,balanced_directions
from supermimic_research.mechanics.multiclip_wrench_screen import contact_rays,min_normal
from supermimic_research.mechanics.pen_contact_groups import normal_group
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json

read=lambda p:json.loads(verify_file_pin(p).read_bytes())

def main(out):
    out.mkdir(exist_ok=False);root=b.ROOT/'runs'
    src=file_pin(root/'pen_catalog_screen_v2/report.json');cpu=read(src);source=read(read(cpu['intent'])['case'])
    cp_ref=file_pin(root/'pen_capacity_targets_v1/report.json');cpu_refs=read(cp_ref)
    write_json(out/'intent.json',dict(cpu_screen=src,cpu_capacity=cp_ref,scope='Same fixed reference frames and fixed mount as the CPU catalog screen. Guarded W1 forward graph only, zero velocity and zero controls. No policy, no time integration, no changes to the train/development/validation/reserve split. Bounded normal force is only used to scale directional support; prior wrench LP uses unbounded nonnegative normal sum.'))
    dirs=balanced_directions();results={};count=0
    with torch.no_grad(),torch.jit.optimized_execution(False),b.backend_scope(b.COLLISION) as col:
        manifest=b.factory();ident=b.execution_identity('STABLE_SAME_REGIME_FRICTION_OR',col.contract['backend_id'],manifest)
        with b.CostProfile(ident,manifest) as profile:
            rt=profile.create_runtime(b.OBJECTIVE,1)
            try:
                sim=rt.env.sim;m=sim.mj_model;body=rt.features.body;qa=rt.features.qa;qids=rt.features.qids.cpu().numpy();jids=m.actuator_trnid[:,0]
                np.testing.assert_array_equal(m.jnt_qposadr[jids],qids);b.require_structure(m)
                b.save_physics(rt,out)
                with np.load(verify_file_pin(source['physical_context'])) as old,np.load(out/'physical_context.npz') as cur:
                    for k in cur.files:np.testing.assert_array_equal(old[k],cur[k])
                with np.load(verify_file_pin(source['trace'])) as z:mount=(z['initial_mocap_pos'][0],z['initial_mocap_quat'][0]);q0=z['initial_qpos']
                def put(key,v):
                    buf=getattr(sim.data,key);buf.copy_(torch.as_tensor(v,dtype=buf.dtype,device=buf.device).reshape(buf.shape))
                for name,entry in cpu['results'].items():
                    with np.load(verify_file_pin(entry['source'])) as f:hand=f['reference_qpos']
                    with np.load(verify_file_pin(entry['series'])) as f:
                        pos=f['COM'];quat=f['quat_wxyz'];oldmetrics=f['metrics'];prior=f['nominal_wrench']
                    with np.load(verify_file_pin(cpu_refs['refs'][name]['series'])) as f:hcpu=f['capacity'];maskcpu=f['mask'];old_eligible=f['eligible'];omega=f['omega_rad_s']
                    sim.reset();torch.cuda.synchronize();put('mocap_pos',mount[0]);put('mocap_quat',mount[1])
                    assert not sim.wp_data.qvel.numpy().any() and not sim.wp_data.qfrc_applied.numpy().any() and not sim.wp_data.xfrc_applied.numpy().any()
                    bound=m.jnt_range[jids];rows=[];hs=[];masks=[];normal=[];unresolved=[]
                    for i,v in enumerate(hand):
                        q=q0.copy();q[qids]=v;q[qa:qa+7]=np.r_[pos[i],quat[i]];put('qpos',q)
                        torch.cuda.synchronize();sim.forward();torch.cuda.synchronize();count+=1
                        d=sim.wp_data;n=int(d.nacon.numpy()[0]);c=d.contact
                        assert n<=d.naconmax and not d.overflow.numpy().any() and float(d.time.numpy()[0])==0.
                        com=d.xipos.numpy()[0,body];np.testing.assert_allclose(com,pos[i],atol=2e-7,rtol=0)
                        rot=Rotation.from_quat(quat[i,[1,2,3,0]]).as_matrix();gg=c.geom.numpy()[:n];ds=c.dist.numpy()[:n]
                        cs=[];signs=[];mask=0;rays=[];pen=selfpen=otherpen=0.
                        xyz=c.pos.numpy()[:n];fr=c.frame.numpy()[:n];mu=c.friction.numpy()[:n];dim=c.dim.numpy()[:n]
                        for k,(g,h) in enumerate(gg):
                            on=m.geom_bodyid[[g,h]]==body;gn=m.geom(g).name or '';hn=m.geom(h).name or ''
                            if on.sum()==1:
                                pen=max(pen,-ds[k]);digit=normal_group(hn if on[0] else gn)
                                if digit<0:otherpen=max(otherpen,-ds[k])
                                elif ds[k]<=0:
                                    contact=SimpleNamespace(dim=int(dim[k]),frame=fr[k],pos=xyz[k],friction=mu[k]);sgn=-1 if on[0] else 1
                                    cs.append(contact);signs.append(sgn);mask|=1<<digit;rays.extend(contact_rays(contact,sgn,com))
                            elif gn.startswith('robot/') and hn.startswith('robot/'):selfpen=max(selfpen,-ds[k])
                        joint=float(max(np.max(bound[:,0]-v),np.max(v-bound[:,1]),0))
                        legal=pen<=.002 and selfpen<=.0002 and joint<=.001 and otherpen==0
                        valid=bool(oldmetrics[i,6]);load=None
                        if valid:
                            try:load=min_normal(rays,prior[i])
                            except (ValueError,RuntimeError) as error:unresolved.append(dict(frame=i,error=str(error)))
                        rows.append([pen,selfpen,joint,len(cs),legal,valid,load is not None])
                        normal.append(load if load is not None else np.nan);masks.append(mask);hs.append(compute(cs,signs,com,rot,dirs))
                    rows=np.asarray(rows);masks=np.asarray(masks);hs=np.asarray(hs);switch=masks[1:]!=masks[:-1]
                    eligible=(rows[:-1,4]>0)&(rows[1:,4]>0)&~switch&(masks[:-1]!=0)
                    stage=np.where(switch,'switch',np.where(masks[:-1]==0,'no_contact',np.where(omega>=.5,'turn','slow')))
                    f=out/(name+'.npz');np.savez_compressed(f,metrics=rows,normal_N=normal,mask=masks,capacity=hs,directions=dirs,eligible=eligible,stage=stage)
                    results[name]=dict(source=entry['source'],series=file_pin(f),frames=len(hand),geometric=int(rows[:,4].sum()),initial_geometry=bool(rows[0,4]),
                        feasible=int(rows[:,6].sum()),both=int((rows[:,4]*rows[:,6]).sum()),eligible=int(eligible.sum()),eligible_changed=int(np.sum(eligible!=old_eligible)),
                        mask_changes=int(np.sum(masks!=maskcpu)),capacity_L1=float(np.mean(np.abs(hs-hcpu))),unresolved=unresolved)
            finally:rt.close()
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),forward_calls=count,native_integrations=0,metrics=['object_pen_m','self_pen_m','joint_rad','hand_contacts','geometry_legal','valid_prior','cone_feasible'],results=results))
    print([(k,v['both'],v['eligible'],v['mask_changes']) for k,v in results.items()])
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
