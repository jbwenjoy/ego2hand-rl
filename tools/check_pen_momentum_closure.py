"""Compare velocity-residual and wrench/momentum closures using fixed training anchors."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.spatial.transform import Rotation
from check_guarded_multistep import advance, state_rotation,native_state55
from fit_guarded_finger_response import groups,fit,HEADS
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json
DT=.02
PHASES=[160,180,192,200,206,210,214,218,222,226]
MODES=['direct','velocity_residual','momentum_residual','constant_velocity','constant_pose']


def infer(x,pd,a,mode,ids,mass,inertia,gravity):
    _,s,u,y,j,defect,dj=a
    du=pd[ids]-u
    delta=y+du@j
    if mode=='direct':return advance(x,delta)
    # A residual in the same world-left rotation-vector chart as the saved labels.
    delta[:20]+=DT*(x[20:40]-s[20:40])
    delta[40:43]+=DT*(x[49:52]-s[49:52])
    delta[43:46]+=DT*(x[52:55]-s[52:55])
    nxt=advance(x,delta)
    if mode=='momentum_residual':
        R=state_rotation(x);Q=state_rotation(nxt)
        defect_hat=defect+du@dj
        nxt[49:52]=x[49:52]+DT*(R@delta[78:81]/mass+gravity)+defect_hat[:3]/mass
        L=(R@inertia@R.T)@x[52:55]+DT*(R@delta[81:84])+defect_hat[3:]
        nxt[52:55]=np.linalg.solve(Q@inertia@Q.T,L)
    return nxt


def train(root):
    report=json.loads((root/'report.json').read_bytes());anchors=[]
    for arm in ('TRAIN_A','TRAIN_B'):
        spec=json.loads(verify_file_pin(report['arms'][arm]).read_bytes());ids=spec['selected']
        data=groups(spec);hooks={}
        for r in spec['rows']:
            with np.load(verify_file_pin(r['packet'])) as p:
                hooks[r['phase'],r['name']]=p['heads'][0,46:52]
        learned=fit(data,ids)
        for a in learned:
            phase=a[0];dx=[];dy=[]
            for axis in ids:
                plus=f'j{axis}_plus';minus=f'j{axis}_minus'
                dx.append((data[phase][plus]['pd']-data[phase][minus]['pd'])[ids])
                dy.append(hooks[phase,plus]-hooks[phase,minus])
            anchors.append((*a,hooks[phase,'sham'],np.linalg.lstsq(dx,dy,rcond=1e-6)[0]))
    return anchors,ids


def run(root,out):
    out.mkdir(exist_ok=False)
    cohorts=['W4096_U32','W2048_U32']
    write_json(out/'intent.json',dict(training=file_pin(root/'report.json'),cohorts=cohorts,phases=PHASES,modes=MODES,
        scope='First cohort previously inspected. Second held out of fitting and first use for this comparison. No test-time adaptation, no intermediate state correction.'))
    anchors,ids=train(root)
    states=np.stack([a[1] for a in anchors]);scale=np.maximum(states.std(0),.001)
    rows=[];sources=[]
    for cohort in cohorts:
        ep=root.parent/'pen_batch_updates_v1'/cohort/'eval_final_actor/report.json';sources.append(file_pin(ep))
        for rep,item in enumerate(json.loads(ep.read_bytes())['cases']):
            c=json.loads(verify_file_pin(item['report']).read_bytes())
            with np.load(verify_file_pin(c['trace'])) as f:z=dict(f)
            mass=c['physics']['mass'];I=np.asarray(c['physics']['inertia_BODY']);g=np.asarray(c['physics']['gravity'])
            def state(q,v):return native_state55(q,v,c['qids'],c['dofs'],c['model_layout']['object_qposadr'],c['model_layout']['object_dofadr'])
            if rep==0:
                for a in anchors:
                    _,s,u,y,*_=a; pd=np.zeros(20);pd[ids]=u
                    exact=advance(s,y)
                    np.testing.assert_allclose(infer(s,pd,a,'momentum_residual',ids,mass,I,g),exact,atol=1e-7)
            for phase in PHASES:
                start=(phase-1)*8;x0=state(z['pre_qpos'][start],z['pre_qvel'][start])
                for mode in MODES:
                    x=x0.copy()
                    for k in range(1,17):
                        idx=start+k*8-1
                        if mode.startswith('constant'):
                            x=x0.copy()
                            if mode=='constant_velocity':
                                x[40:43]+=k*DT*x0[49:52]
                                R=Rotation.from_rotvec(k*DT*x0[52:55]).as_matrix()@state_rotation(x0)
                                x[43:49]=np.r_[R[:,0],R[:,1]]
                        else:
                            a=anchors[np.argmin(np.mean(((states-x)/scale)**2,axis=1))]
                            x=infer(x,z['post_ctrl'][idx,c['aids']],a,mode,ids,mass,I,g)
                        if not np.isfinite(x).all():raise ValueError('Nonfinite extrapolation')
                        if k in (1,2,4,8,16):
                            truth=state(z['post_qpos'][idx],z['post_qvel'][idx])
                            err=Rotation.from_matrix(state_rotation(x)@state_rotation(truth).T).magnitude()
                            rows.append(dict(cohort=cohort,rep=rep,phase=phase,k=k,mode=mode,mm=float(1000*np.linalg.norm(x[40:43]-truth[40:43])),deg=float(np.rad2deg(err))))
    summary={}
    for cohort in cohorts:
        summary[cohort]={}
        for mode in MODES:
            vals=[]
            for k in (1,2,4,8,16):
                scores=[[r['mm'],r['deg']] for r in rows if r['cohort']==cohort and r['mode']==mode and r['k']==k]
                assert len(scores)==30
                vals.append(np.sqrt(np.mean(np.square(scores),axis=0)).tolist())
            summary[cohort][mode]=vals
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),sources=sources,anchors=len(anchors),horizons_ms=[20,40,80,160,320],
        rms_columns=['COM_mm','angle_deg'],summary=summary,rows=rows))
    for cohort in cohorts:
        print(cohort,{m:np.round(np.array(v)[[0,2,4]],2).tolist() for m,v in summary[cohort].items()})
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
