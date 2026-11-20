"""Distinguish a poor learned wrench from 20ms integration error using a measured-wrench oracle."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from check_pen_momentum_closure import train,infer,DT,PHASES
from check_guarded_multistep import advance,state_rotation,native_state55
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json


def rigid(x,F,T,m,I,g):
    R=state_rotation(x);v=x[49:52];w=x[52:55];L=R@I@R.T@w
    vn=v+DT*(F/m+g)
    mid=Rotation.from_rotvec(.5*DT*w).as_matrix()@R
    wm=np.linalg.solve(mid@I@mid.T,L+.5*DT*T)
    Q=Rotation.from_rotvec(DT*wm).as_matrix()@R
    wn=np.linalg.solve(Q@I@Q.T,L+DT*T)
    return np.r_[x[40:43]+.5*DT*(v+vn),Q[:,0],Q[:,1],vn,wn]


def main(root,out):
    out.mkdir(exist_ok=False);anchors,ids=train(root)
    states=np.stack([a[1] for a in anchors]);scale=np.maximum(states.std(0),.001)
    rows=[];checks=[];pins=[]
    for cohort in ('W4096_U32','W2048_U32'):
        ep=root.parent/'pen_batch_updates_v1'/cohort/'eval_final_actor/report.json';pins.append(file_pin(ep))
        for rep,entry in enumerate(json.loads(ep.read_bytes())['cases']):
            case=json.loads(verify_file_pin(entry['report']).read_bytes())
            with np.load(verify_file_pin(case['trace'])) as f:z=dict(f)
            layout=case['model_layout'];p=case['physics'];mass=p['mass'];I=np.array(p['inertia_BODY']);g=np.array(p['gravity'])
            def state(q,v):return native_state55(q,v,case['qids'],case['dofs'],layout['object_qposadr'],layout['object_dofadr'])
            def actual(i):return state(z['post_qpos'][i],z['post_qvel'][i])
            loads={}
            for phase in sorted({s+k for s in PHASES for k in range(16)}):
                c=json.loads(verify_file_pin(case['controls'][phase-1]).read_bytes())
                with np.load(verify_file_pin(c['window'])) as f:
                    assert f['counts'][0]==8
                    loads[phase]=f['wrench_sum'][0]/8
            for phase in PHASES:
                start=(phase-1)*8
                for mode in ('predicted','measured_world_oracle'):
                    x=state(z['pre_qpos'][start],z['pre_qvel'][start])
                    for k in range(1,17):
                        j=start+k*8-1;pd=z['post_ctrl'][j,case['aids']]
                        a=anchors[np.argmin(np.mean(((states-x)/scale)**2,axis=1))]
                        delta=a[3]+(pd[ids]-a[2])@a[4];R=state_rotation(x)
                        wrench=np.r_[R@delta[78:81],R@delta[81:84]]
                        observed=loads[phase+k-1]
                        if k==1 and mode=='predicted':
                            checks.append(dict(cohort=cohort,phase=phase,rep=rep,
                                force_error_N=float(np.linalg.norm(observed[:3]-wrench[:3])),
                                torque_error_Nm=float(np.linalg.norm(observed[3:]-wrench[3:]))))
                        if mode=='measured_world_oracle':wrench=observed
                        nxt=advance(x,delta)
                        nxt[40:]=rigid(x,wrench[:3],wrench[3:],mass,I,g)
                        x=nxt
                        if k in (1,2,4,8,16):
                            truth=actual(j);error=Rotation.from_matrix(state_rotation(x)@state_rotation(truth).T).magnitude()
                            rows.append(dict(cohort=cohort,rep=rep,phase=phase,k=k,mode=mode,
                                mm=1000*float(np.linalg.norm(x[40:43]-truth[40:43])),deg=float(np.rad2deg(error))))
    summary={}
    for cohort in ('W4096_U32','W2048_U32'):
        summary[cohort]={}
        for mode in ('predicted','measured_world_oracle'):
            summary[cohort][mode]=[np.sqrt(np.mean(np.square([[r['mm'],r['deg']] for r in rows if
                r['cohort']==cohort and r['mode']==mode and r['k']==k]),axis=0)).tolist() for k in (1,2,4,8,16)]
    write_json(out/'report.json',dict(source=file_pin(root/'report.json'),evaluations=pins,summary=summary,
        horizons_ms=[20,40,80,160,320],units=['mm_rms','deg_rms'],rows=rows,first_step_wrench_errors=checks,
        oracle_scope='Future measured 20ms mean world wrench supplied as privileged input. No state corrections. Not an available control model. Same integrator in both conditions.'))
    print(json.dumps(summary))
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
