"""Open-loop 20..320 ms pose diagnostics with saved measured PD inputs."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from fit_guarded_finger_response import groups,fit
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.controlled_pen_records import native_state55
from supermimic_research.mechanics.transition_records import state_rotation
from supermimic_research.strict_json import write_json


def advance(x,y):
    R=Rotation.from_rotvec(y[43:46]).as_matrix()@state_rotation(x)
    return np.r_[x[:43]+y[:43],R[:,0],R[:,1],x[49:]+y[46:52]]


def main(root,out):
    out.mkdir(exist_ok=False)
    report=json.loads((root/'report.json').read_bytes());anchors=[]
    for key in ('TRAIN_A','TRAIN_B'):
        arm=json.loads(verify_file_pin(report['arms'][key]).read_bytes());ids=arm['selected'];anchors+=fit(groups(arm),ids)
    states=np.stack([a[1] for a in anchors]);scale=np.maximum(states.std(0),1e-3)
    ev=root.parent/'pen_batch_updates_v1/W4096_U32/eval_final_actor/report.json'
    rows=[]
    for rep,entry in enumerate(json.loads(ev.read_bytes())['cases']):
        c=json.loads(verify_file_pin(entry['report']).read_bytes())
        with np.load(verify_file_pin(c['trace'])) as file: z=dict(file)
        qa=c['model_layout']['object_qposadr'];da=c['model_layout']['object_dofadr']
        def state(q,v):return native_state55(q,v,c['qids'],c['dofs'],qa,da)
        for phase in [160,180,192,200,206,210,214,218,222,226]:
            start=(phase-1)*8
            x=state(z['pre_qpos'][start],z['pre_qvel'][start]);initial=x.copy()
            for k in range(1,17):
                idx=start+k*8-1
                a=anchors[np.argmin(np.mean(((states-x)/scale)**2,axis=1))]
                pd=z['post_ctrl'][idx,c['aids']]
                x=advance(x,a[3]+(pd[ids]-a[2])@a[4])
                if k not in (1,2,4,8,16):continue
                truth=state(z['post_qpos'][idx],z['post_qvel'][idx])
                if k==1:
                    exact=np.r_[truth[:43]-initial[:43],Rotation.from_matrix(state_rotation(truth)@state_rotation(initial).T).as_rotvec(),truth[49:]-initial[49:]]
                    np.testing.assert_allclose(advance(initial,exact),truth,atol=1e-7)
                R=Rotation.from_matrix(state_rotation(truth));Ri=Rotation.from_matrix(state_rotation(initial))
                for name,pos,orientation in [('model',x[40:43],Rotation.from_matrix(state_rotation(x))),
                    ('constant_velocity',initial[40:43]+initial[49:52]*k*.02,Rotation.from_rotvec(initial[52:55]*k*.02)*Ri),
                    ('constant_pose',initial[40:43],Ri)]:
                    rows.append(dict(repeat=rep,phase=phase,horizon_ms=k*20,method=name,com_mm=1000*np.linalg.norm(pos-truth[40:43]),
                                     angle_deg=np.rad2deg((orientation*R.inv()).magnitude())))
    summary={}
    for ms in (20,40,80,160,320):
        summary[ms]={}
        for name in ('model','constant_velocity','constant_pose'):
            cases=[r for r in rows if r['method']==name and r['horizon_ms']==ms]
            summary[ms][name]={k:float(np.sqrt(np.mean([r[k]**2 for r in cases]))) for k in ['com_mm','angle_deg']}
    write_json(out/'report.json',dict(training=file_pin(root/'report.json'),evaluation=file_pin(ev),rows=rows,summary=summary,
        scope='Measured PD open-loop input. One initial native state, no intermediate state injection. State-only nearest-anchor 8-axis PD response. Thirty overlapping windows per horizon from three numerical repeats of one reference.'))
    print(json.dumps(summary))
if __name__=='__main__':main(*map(Path,sys.argv[1:]))

