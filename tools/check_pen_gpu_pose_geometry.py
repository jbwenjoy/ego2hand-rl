"""Replay observed or predicted substep configurations through the guarded W1 forward graph."""
from pathlib import Path
from types import SimpleNamespace
import json,sys
import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation
import collect_guarded_finger_response as base
from build_pen_capacity_targets import compute,balanced_directions
from supermimic_research.mechanics.pen_substep_secant import load_local_secant
from supermimic_research.mechanics.controlled_pen_records import native_state55
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json


def compare(root,out):
    out.mkdir(exist_ok=False);master=file_pin(root/'report.json');report=json.loads(verify_file_pin(master).read_bytes())
    ck=json.loads((root.parent/'pen_substep_secant_v1/report.json').read_bytes())['checkpoint']
    net=load_local_secant(torch.load(verify_file_pin(ck),map_location='cpu',weights_only=True))
    write_json(out/'intent.json',dict(data=master,secant=ck,population='Both previously held out policy cohorts, all unique selected sources; paired valid nonterminal nonzero PD branches.',
        mode='Guarded W1 forward graph only; reset Data before each branch/forecast. Sole time-varying input is supplied measured or forecast pre-step qpos. Both replays hold the single source qvel; no native integration, object forces, actor or optimizer updates.'))
    rows=[];calls=0;dirs=balanced_directions()
    with torch.no_grad(),torch.jit.optimized_execution(False),base.backend_scope(base.COLLISION) as collision:
        manifest=base.factory()
        ident=base.execution_identity('STABLE_SAME_REGIME_FRICTION_OR',collision.contract['backend_id'],manifest)
        with base.CostProfile(ident,manifest) as profile:
            rt=profile.create_runtime(base.OBJECTIVE,1)
            try:
                sim=rt.env.sim;m=sim.mj_model;body=rt.features.body;qa=rt.features.qa;da=int(m.jnt_dofadr[m.body_jntadr[body]])
                qids=rt.features.qids.cpu().tolist();dofs=rt.features.dofs.cpu().tolist()
                model=mujoco.MjModel.from_binary_path(str(verify_file_pin(report['actors']['TEST_RESIDUAL']['model'])))
                assert m.ngeom==model.ngeom and all(m.geom(i).name==model.geom(i).name for i in range(m.ngeom))
                physdir=out/'live_plant';physdir.mkdir();base.save_physics(rt,physdir)
                with np.load(physdir/'physical_context.npz') as live:
                    for role in ['TEST_RESIDUAL','TEST_SUPPORT']:
                        with np.load(verify_file_pin(report['actors'][role]['physical_context'])) as old:
                            for key in old.files:np.testing.assert_array_equal(live[key],old[key])
                def put(k,v):
                    buf=getattr(sim.data,k);buf.copy_(torch.as_tensor(v,dtype=buf.dtype,device=buf.device).reshape(buf.shape))
                def shape(q):
                    nonlocal calls
                    put('qpos',q);torch.cuda.synchronize();sim.forward();torch.cuda.synchronize();calls+=1
                    d=sim.wp_data;n=int(d.nacon.numpy()[0]);c=d.contact
                    if n>d.naconmax or d.overflow.numpy().any():raise ValueError('Native buffer overflow')
                    gg=c.geom.numpy()[:n];ds=c.dist.numpy()[:n];mu=c.friction.numpy()[:n];dim=c.dim.numpy()[:n]
                    point=c.pos.numpy()[:n];frame=c.frame.numpy()[:n];cs=[];signs=[];pairs=[]
                    for i,(g,h) in enumerate(gg):
                        on=m.geom_bodyid[[g,h]]==body
                        if on.sum()!=1 or ds[i]>0:continue
                        other=int(h if on[0] else g)
                        if not (m.geom(other).name or '').startswith('robot/'):continue
                        cs.append(SimpleNamespace(pos=point[i],frame=frame[i],friction=mu[i],dim=int(dim[i])))
                        signs.append(-1 if on[0] else 1);pairs.append((int(g if on[0] else h),other))
                    rot=Rotation.from_quat(q[qa+np.array([4,5,6,3])]).as_matrix()
                    return compute(cs,signs,d.xipos.numpy()[0,body],rot,dirs),sorted(pairs)
                for sample in report['rows']:
                    if not sample['actor'].startswith('TEST'):continue
                    branches={b['name']:b for b in sample['branches']}
                    if not all(b['valid'] and not b['terminal'] for b in branches.values()):continue
                    if np.linalg.norm(np.subtract(branches['plus']['pd'],branches['minus']['pd']))<1e-7:continue
                    outcomes={}
                    for side in ['minus','plus']:
                        b=branches[side]
                        with np.load(verify_file_pin(b['trace'])) as z:
                            poses=z['pre_qpos'];v=z['pre_qvel'][0];warm=z['initial_warmstart'];mp=z['post_mocap_pos'][0];mq=z['post_mocap_quat'][0]
                            geom_raw=z['raw_contact_tick_plus23'];pairs=[[] for _ in range(8)]
                            for row in geom_raw:
                                tick,g,h=map(int,row[:3]);on=m.geom_bodyid[[g,h]]==body
                                if on.sum()==1 and row[5]<=0 and (m.geom(h if on[0] else g).name or '').startswith('robot/'):
                                    pairs[tick].append((g,h) if on[0] else (h,g))
                        x=native_state55(poses[0],v,qids,dofs,qa,da)
                        pred=net.predict_qpos(*[torch.as_tensor(vec,dtype=torch.float64) for vec in (x,warm,b['pd'])]).numpy()
                        outcomes[side]={'native':b['capacity']}
                        for mode,qs in [('true_pose',poses),('secant',pred)]:
                            sim.reset();torch.cuda.synchronize()
                            for key,val in [('mocap_pos',mp),('mocap_quat',mq),('qvel',v),('ctrl',b['pd']),('qacc_warmstart',warm)]:put(key,val)
                            caps=[];match=[]
                            for i,q in enumerate(qs):
                                cap,ids=shape(q);caps.append(cap);match.append(ids==sorted(pairs[i]))
                            outcomes[side][mode]=np.mean(caps,axis=0).tolist()
                            outcomes[side][mode+'_matching_ticks']=int(sum(match))
                    rows.append(dict(role=sample['actor'],phase=sample['phase'],risk=sample['risk_selected'],uniform=sample['uniform_selected'],endpoints=outcomes))
            finally:rt.close()
    summary={}
    for role in ['TEST_RESIDUAL','TEST_SUPPORT']:
        sel=[r for r in rows if r['role']==role]
        effects=lambda key:np.stack([np.subtract(r['endpoints']['plus'][key],r['endpoints']['minus'][key]) for r in sel])
        d=effects('native');s={}
        for mode in ['true_pose','secant']:
            y=effects(mode)
            s[mode]=dict(n=len(sel),relative_error=float(np.linalg.norm(y-d)/np.linalg.norm(d)),positive_dot=int(np.sum((d*y).sum(1)>0)),matched_ticks=sum(r['endpoints'][side][mode+'_matching_ticks'] for r in sel for side in ('minus','plus')),total_ticks=16*len(sel))
        summary[role]=s
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),forward_calls=calls,summary=summary,rows=rows));print(summary)
if __name__=='__main__':compare(*map(Path,sys.argv[1:]))
