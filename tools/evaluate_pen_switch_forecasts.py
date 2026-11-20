"""Frozen substep forecasts on risk-selected and uniform native sources."""
import json,sys
from pathlib import Path
import numpy as np
import torch
import mujoco
from check_pen_capacity_geometry_oracle import geometry,balanced_directions
from check_pen_native_load_suggestions import native_state55
from supermimic_research.mechanics.pen_substep_pose import load_substep_pose
from supermimic_research.mechanics.pen_substep_secant import load_local_secant
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json

read=lambda pin:json.loads(verify_file_pin(pin).read_bytes())


def main(root,out):
    out.mkdir(exist_ok=False);runs=root.parent;bank=read(file_pin(root/'report.json'));dirs=balanced_directions()
    fit=read(file_pin(runs/'pen_substep_pose_fit_v1/report.json'))
    models={k:load_substep_pose(torch.load(verify_file_pin(fit['models'][k]['checkpoint']),map_location='cpu',weights_only=True),dtype=torch.float64).eval() for k in ('absolute','paired')}
    models['secant']=load_local_secant(torch.load(verify_file_pin(read(file_pin(runs/'pen_substep_secant_v1/report.json'))['checkpoint']),map_location='cpu',weights_only=True)).eval()
    torch.set_num_threads(2)
    write_json(out/'intent.json',dict(data=file_pin(root/'report.json'),model_refs=[file_pin(runs/'pen_substep_pose_fit_v1/report.json'),file_pin(runs/'pen_substep_secant_v1/report.json')],
        scope='Models frozen before this bank. Three learned forecasts, stored integration-state CPU open-loop predictor and measured substep pose oracle, all using the same geometry backend. Mask/multiplicity variation differs from force/drift; stratify per selection arm. No test-time fitting.'))
    rows=[]
    for role,meta in bank['actors'].items():
        m=mujoco.MjModel.from_binary_path(str(verify_file_pin(meta['model'])));layout=meta['layout'];body=layout['object_body'];qa=layout['object_qposadr'];da=layout['object_dofadr']
        qids=m.jnt_qposadr[m.actuator_trnid[:,0]];dofs=m.jnt_dofadr[m.actuator_trnid[:,0]]
        for src in [row for row in bank['rows'] if row['actor']==role]:
            actual={};pred={k:{} for k in (*models,'cpu','observed')}
            branches={r['name']:r for r in src['branches']}
            if not all(r['valid'] and not r['terminal'] for r in branches.values()):continue
            if np.linalg.norm(np.subtract(branches['plus']['pd'],branches['minus']['pd']))<1e-7:continue
            # Read one pre-action state only for all prospective forecasts; future poses are used by the oracle alone.
            with np.load(verify_file_pin(branches['sham']['trace'])) as z:
                x=native_state55(z['pre_qpos'][0],z['pre_qvel'][0],qids,dofs,qa,da)
                warm=z['initial_warmstart'];state=z['initial_native_state']
                obs=mujoco.MjData(m);mujoco.mj_setState(m,obs,state,mujoco.mjtState.mjSTATE_INTEGRATION)
                geo=mujoco.MjData(m);geo.mocap_pos[:]=obs.mocap_pos;geo.mocap_quat[:]=obs.mocap_quat
            for side in ('plus','minus'):
                b=branches[side];pd=np.asarray(b['pd']);actual[side]=np.asarray(b['capacity'])
                poses={}
                for k,net in models.items():
                    inp=[torch.as_tensor(v,dtype=torch.float64) for v in (x,warm,pd)]
                    with torch.no_grad():poses[k]=net.predict_qpos(*inp).numpy()
                d=mujoco.MjData(m);mujoco.mj_setState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION);d.ctrl[:]=pd
                assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
                cpu=[]
                for _ in range(8):
                    cpu.append(d.qpos.copy());mujoco.mj_step(m,d)
                poses['cpu']=cpu
                with np.load(verify_file_pin(b['trace'])) as z:poses['observed']=z['pre_qpos']
                for k,seq in poses.items():
                    pred[k][side]=np.mean([geometry(m,geo,q,body,dirs) for q in seq],axis=0)
            signal=actual['plus']-actual['minus'];effects={}
            for k,d in pred.items():
                dy=d['plus']-d['minus']
                effects[k]=dict(error_l2=float(np.linalg.norm(dy-signal)),magnitude=float(np.linalg.norm(dy)),positive_dot=bool(dy@signal>0))
            noise=np.linalg.norm(np.subtract(branches['sham']['capacity'],branches['sham_repeat']['capacity']))
            rows.append(dict(role=role,phase=src['phase'],risk=src['risk_selected'],uniform=src['uniform_selected'],signal_l2=float(np.linalg.norm(signal)),noise_l2=float(noise),effects=effects))
    summary={}
    for role in bank['actors']:
        for arm in ('risk','uniform'):
            rs=[r for r in rows if r['role']==role and r[arm]];signal=sum(r['signal_l2']**2 for r in rs)
            summary[role+'_'+arm]={k:dict(n=len(rs),error_over_zero=float(np.sqrt(sum(r['effects'][k]['error_l2']**2 for r in rs)/signal)) if signal else None,
                positive_dot=sum(r['effects'][k]['positive_dot'] for r in rs)) for k in (*models,'cpu','observed')}
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),summary=summary,rows=rows))
    print({k:[round(a['error_over_zero'],3) if a['error_over_zero'] is not None else None for a in v.values()] for k,v in summary.items()})
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
