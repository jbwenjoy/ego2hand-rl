"""Factor object-pose midpoint bias vs action-induced motion at changing contacts."""
from pathlib import Path
import json,sys
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from check_pen_capacity_geometry_oracle import geometry,balanced_directions
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json


def parts(pair,qa):
    xyz=pair[:,:,qa:qa+3]
    r=[Rotation.from_quat(z[:,qa+np.array([4,5,6,3])]) for z in pair]
    dv=(r[1]*r[0].inv()).as_rotvec();mid=Rotation.from_rotvec(dv*.5)*r[0]
    return xyz.mean(0),mid,xyz[1]-xyz[0],dv


def main(root,secant,out):
    out.mkdir(exist_ok=False);report=json.loads((root/'report.json').read_bytes());role=report['roles']['TEST']
    ref=json.loads((secant/'report.json').read_bytes())
    write_json(out/'intent.json',dict(data=file_pin(root/'report.json'),secant_fit=file_pin(secant/'report.json'),scope='Post-hoc geometry diagnostic on previously inspected pairs. Hold true hand path fixed. Combine position and world-left geodesic rotation midpoint from either true or predicted actions with signed difference from either. Uses future measured objects/hands solely to attribute prediction error.'))
    with np.load(verify_file_pin(ref['predictions'])) as f:
        pred={(int(p),str(n)):q for p,n,q in zip(f['phase'],f['name'],f['local_secant'])}
    model=mujoco.MjModel.from_binary_path(str(root/'native/TEST/model.mjb'));body=model.body('object/cube').id;qa=model.jnt_qposadr[model.body_jntadr[body]]
    rows=[]
    groups={}
    for row in role['rows']:groups.setdefault(row['phase'],{})[row['name']]=row
    dirs=balanced_directions()
    for phase,group in groups.items():
        for name in ['mixed']+[f'j{i}' for i in role['selected']]:
            true=[];forecast=[];cap=[];poses={}
            for side in ('minus','plus'):
                row=group[f'{name}_{side}']
                if not row['valid'] or row['terminal']:break
                with np.load(verify_file_pin(row['trace'])) as z:
                    true.append(z['pre_qpos']);mp=z['post_mocap_pos'][0];mq=z['post_mocap_quat'][0]
                forecast.append(pred[phase,row['name']])
                with np.load(verify_file_pin(row['packet'])) as f:cap.append(f['capacity'])
            if len(true)!=2:continue
            true=np.stack(true).astype(np.float64);forecast=np.stack(forecast).astype(np.float64)
            x,R,dx,dw=parts(true,qa);xh,Rh,dxh,dwh=parts(forecast,qa)
            for mid in ('true','forecast'):
                for effect in ('true','forecast'):
                    po,ro=(x,R) if mid=='true' else (xh,Rh)
                    dp,dr=(dx,dw) if effect=='true' else (dxh,dwh)
                    scores=[]
                    for hand,sign in zip(true,(-.5,.5)):
                        q=hand.copy();q[:,qa:qa+3]=po+sign*dp
                        q[:,qa+3:qa+7]=(Rotation.from_rotvec(sign*dr)*ro).as_quat()[:,[3,0,1,2]]
                        if mid==effect=='true':
                            np.testing.assert_allclose(q[:,qa:qa+3],hand[:,qa:qa+3],atol=1e-9)
                            assert np.min(abs((q[:,qa+3:qa+7]*hand[:,qa+3:qa+7]).sum(1)))>1-1e-6
                            q[:,qa:qa+7]=hand[:,qa:qa+7]
                        if mid==effect=='forecast':q[:,qa:qa+7]=forecast[0 if sign<0 else 1,:,qa:qa+7]
                        d=mujoco.MjData(model);d.mocap_pos[:]=mp;d.mocap_quat[:]=mq
                        scores.append(np.mean([geometry(model,d,v,body,dirs) for v in q],axis=0))
                    v=scores[1]-scores[0];native=cap[1]-cap[0]
                    poses[mid+'_mid_'+effect+'_effect']=dict(magnitude=float(np.linalg.norm(v)),error=float(np.linalg.norm(v-native)))
            rows.append(dict(phase=phase,name=name,native=float(np.linalg.norm(cap[1]-cap[0])),cases=poses))
    summary={}
    for subset,rs in [('mixed',[r for r in rows if r['name']=='mixed']),('basis',[r for r in rows if r['name']!='mixed']),('phase192_mixed',[r for r in rows if r['name']=='mixed' and r['phase']==192])]:
        summary[subset]={c:float(np.sqrt(sum(r['cases'][c]['error']**2 for r in rs)/sum(r['native']**2 for r in rs))) for c in rs[0]['cases']}
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),summary=summary,rows=rows));print(summary)
if __name__=='__main__':main(*map(Path,sys.argv[1:]))

