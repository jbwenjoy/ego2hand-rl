"""Transfer pinned clip-specific hand poses and servo targets to the guarded W1 plant."""
import json,sys
from pathlib import Path
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import collect_guarded_finger_response as b
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json


def read(pin):return json.loads(verify_file_pin(pin).read_bytes())


def trial(rt,folder,target,start,case,physical,model,qids,aids,body,qa,mount):
    folder.mkdir()
    sim=rt.env.sim;sim.reset();torch.cuda.synchronize()
    def put(key,value):
        buf=getattr(sim.data,key);buf.copy_(torch.as_tensor(value,device=buf.device,dtype=buf.dtype).reshape(buf.shape))
    q0=start.copy();ctrl=np.zeros(model.nu);ctrl[aids]=target
    put('qpos',q0);put('ctrl',ctrl);put('mocap_pos',mount[0]);put('mocap_quat',mount[1])
    torch.cuda.synchronize();sim.forward();torch.cuda.synchronize()
    d=sim.wp_data
    assert np.array_equal(d.qpos.numpy()[0],q0.astype(np.float32)) and not d.qvel.numpy().any() and not d.time.numpy().any()
    rec=b.OwnedProbeRecorder(rt);qs=[];audits=[];contacts=[]
    for k in range(10):
        rec.begin(np.zeros(20))
        for _ in range(8):
            rec.before_step();sim.step();rec.after_step()
        z=rec.finish();metrics,_=b.decode_saved_case(case,z,model,physical)
        assert np.all(z['post_ctrl'][:,aids]==ctrl[aids].astype(np.float32)) and not np.any(z['post_xfrc_applied']) and not np.any(z['post_qfrc_applied'])
        assert np.allclose(z['post_mocap_pos'],mount[0],atol=1e-7) and np.allclose(z['post_mocap_quat'],mount[1],atol=1e-7)
        assert np.allclose(z['post_GPU_time'],(k*8+np.arange(1,9))*.0025,atol=5e-7)
        valid=all(metrics['checks'].values()) and not z['overflow'].any() and max(z['pre_penetration_m'])<=.002 and max(z['pre_self_penetration_m'])<=.0002 and max(z['post_joint_limit_excursion_rad'])<=.001
        sq=z['post_qpos'];qs.extend(sq)
        for i in range(8):
            normal=0.
            for row in z['raw_contact_tick_plus23']:
                if int(row[0])!=i:continue
                g,h=map(int,row[1:3]);on=model.geom_bodyid[[g,h]]==body
                if on.sum()==1 and (model.geom(h if on[0] else g).name or '').startswith('robot/'):normal+=max(0.,row[18])
            contacts.append(normal)
        path=folder/f'window_{k+1:02}.npz';np.savez_compressed(path,**z)
        audits.append(dict(trace=file_pin(path),valid=bool(valid),physics=metrics,penetration_m=float(max(z['pre_penetration_m'])),self_m=float(max(z['pre_self_penetration_m'])),joint_rad=float(max(z['post_joint_limit_excursion_rad']))))
    angle=np.rad2deg((Rotation.from_quat(np.asarray(qs)[:,qa+np.array([4,5,6,3])])*Rotation.from_quat(q0[qa+np.array([4,5,6,3])]).inv()).magnitude())
    drift=1000*np.linalg.norm(np.asarray(qs)[:,qa:qa+3]-q0[qa:qa+3],axis=1)
    result=dict(max_mm=float(max(drift)),max_deg=float(max(angle)),valid_windows=sum(a['valid'] for a in audits),contact_ticks=int(np.count_nonzero(np.array(contacts)>.1)),audits=audits)
    write_json(folder/'report.json',result);return result


def main(out,preload_path=None,names=('clip_003','clip_004')):
    out.mkdir(exist_ok=False);run=b.ROOT/'runs'
    preload=file_pin(preload_path or run/'pen_dev_static_preload_v1/report.json');pr=read(preload)
    proj=read(read(pr['intent'])['source']);screen=read(read(proj['intent'])['screen'])
    screen_intent=read(screen['intent']);case=read(screen_intent['case']); model=mujoco.MjModel.from_binary_path(str(verify_file_pin(screen_intent.get('cpu_model',case['model']))))
    with np.load(verify_file_pin(case['trace'])) as f:
        qbase=f['initial_qpos'];mount=(f['initial_mocap_pos'][0],f['initial_mocap_quat'][0])
    with np.load(verify_file_pin(case['physical_context'])) as f:physical=dict(f)
    qids=np.array(case['qids']);aids=np.array(case['aids']);layout=case['model_layout'];body=layout['object_body'];qa=layout['object_qposadr']
    inputs={};expected={}
    for n in names:
        source=next(r for r in proj['rows'] if r['clip']==n)
        with np.load(verify_file_pin(source['variants']['projection']['trace'])) as f:
            q=qbase.copy();q[qids]=f['initial_hand_q'];q[qa:qa+7]=np.r_[f['object_pos'],f['object_quat']]
        for arm,h in next(r for r in pr['rows'] if r['clip']==n)['holds'].items():
            with np.load(verify_file_pin(h['trace'])) as f:inputs[n,arm]=(q,f['ctrl'])
            expected[n,arm]=[h['max_mm'],h['max_deg']]
    write_json(out/'intent.json',dict(preload=preload,reps=3,ticks_per_trial=80,scope='Guarded fixed-control GPU diagnostic to 0.2s for all cases including physically invalid windows. CPU identical targets. Recorder action field is an unused zero placeholder. No RL or reference advancement.'))
    rows=[];manifest=b.factory()
    with torch.no_grad(),torch.jit.optimized_execution(False),b.backend_scope(b.COLLISION) as col:
        identity=b.execution_identity('STABLE_SAME_REGIME_FRICTION_OR',col.contract['backend_id'],manifest)
        with b.CostProfile(identity,manifest) as profile:
            rt=profile.create_runtime(b.OBJECTIVE,1)
            try:
                b.require_structure(rt.env.sim.mj_model)
                b.save_physics(rt,out)
                with np.load(out/'physical_context.npz') as f:
                    for key,v in physical.items():np.testing.assert_array_equal(v,f[key])
                for i in range(3):
                    for key in (list(inputs) if i%2==0 else list(reversed(inputs))):
                        q,target=inputs[key];folder=out/f'{i}_{key[0]}_{key[1]}'
                        result=trial(rt,folder,target,q,case,physical,model,qids,aids,body,qa,mount)
                        rows.append(dict(rep=i,clip=key[0],arm=key[1],cpu=expected[key],report=file_pin(folder/'report.json'),**{k:v for k,v in result.items() if k!='audits'}))
            finally:rt.close()
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),native_ticks=len(rows)*80,rows=rows))
    for r in rows:print(r['clip'],r['arm'],r['rep'],round(r['max_mm'],2),round(r['max_deg'],2),r['valid_windows'],r['contact_ticks'])
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path);parser.add_argument('--preload',type=Path);parser.add_argument('--clips',nargs='+',default=['clip_003','clip_004'])
    a=parser.parse_args();main(a.out.resolve(),a.preload,a.clips)
