"""Independent strict errors and raw-contact/EFC readback of learned policies."""
import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main(a):
    rp=a.report.resolve();run=json.loads(rp.read_text());out=a.output.resolve()
    cp=verify_file_pin(run['config']);cfg=tomllib.loads(cp.read_text());gp=(cp.parent/cfg['gates']).resolve();gates=tomllib.loads(gp.read_text())
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh independent evaluation review')
    out.mkdir();receipt=freeze_sources([Path(__file__).resolve(),cp,gp],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    checks={};cases=[]
    for c in run['cases']:
        z=dict(np.load(verify_file_pin(c['trace'])));raw=dict(np.load(verify_file_pin(c['raw_contacts'])));m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['model'])))
        n=len(z['post_qpos']);t=np.arange(1,n+1)*c['dt'];body=m.body('object/cube').id;qa=m.jnt_qposadr[m.body_jntadr[body]]
        jids=[m.joint(f'robot/right_finger{i}_joint{j}').id for i in range(1,6) for j in range(1,5)];qids=m.jnt_qposadr[jids]
        position=z['post_qpos'][:,qa:qa+3]
        actual=Rotation.from_quat(z['post_qpos'][:,qa+np.array([4,5,6,3])]);ref=Rotation.from_quat(z['quaternion_wxyz'][:,[1,2,3,0]])
        target=Slerp(z['time_s'],ref)(t);targetp=np.stack([np.interp(t,z['time_s'],z['COM_world_m'][:,j]) for j in range(3)],-1)
        distance=np.linalg.norm(position-targetp,axis=-1);angle=(target.inv()*actual).magnitude()
        force=np.zeros((len(raw['dim']),6));valid=True
        for i,dim in enumerate(raw['dim']):
            tick=int(raw['tick'][i]);dim=int(dim);addr=raw['efc_address'][i];offset=int(raw['efc_offsets'][tick]);count=int(raw['efc_offsets'][tick+1])-offset
            if addr[0]<0:continue
            width=1 if dim==1 else (2*(dim-1) if c['cone']==0 else dim)
            ids=addr[:width].astype(int)
            if len(ids)!=width or np.any(ids<0) or np.any(ids>=count):valid=False;continue
            values=raw['efc_force'][offset+ids].astype(np.float64)
            if c['cone']==0 and dim>1:
                force[i,0]=sum(values)
                for j in range(dim-1):force[i,j+1]=(values[2*j]-values[2*j+1])*float(raw['friction'][i,j])
            else:force[i,:dim]=values
        pen=np.zeros(n);selfpen=np.zeros(n);hand=np.zeros(n);nonhand=np.zeros(n)
        for i,(g,h) in enumerate(raw['geom']):
            g,h=int(g),int(h);tick=int(raw['tick'][i]);ob=m.geom_bodyid[[g,h]]==body;dist=float(raw['dist'][i]);normal=max(0.,force[i,0])
            gn=m.geom(g).name or '';hn=m.geom(h).name or ''
            if ob.sum()==1:
                pen[tick]=max(pen[tick],-dist);other=hn if ob[0] else gn
                if other.startswith('robot/'):hand[tick]+=normal
                else:nonhand[tick]+=normal
            elif gn.startswith('robot/') and hn.startswith('robot/'):selfpen[tick]=max(selfpen[tick],-dist)
        joint=np.max(np.maximum(np.maximum(m.jnt_range[jids,0]-z['post_qpos'][:,qids],z['post_qpos'][:,qids]-m.jnt_range[jids,1]),0),axis=-1)
        metrics=dict(actual_duration_s=float(t[-1]),COM_RMS_m=float(np.sqrt(np.mean(distance**2))),COM_max_m=float(distance.max()),
            angle_RMS_rad=float(np.sqrt(np.mean(angle**2))),angle_max_rad=float(angle.max()),max_penetration_m=float(pen.max()),
            max_self_penetration_m=float(selfpen.max()),max_joint_limit_excursion_rad=float(joint.max()),
            max_object_nonhand_normal_force_N=float(nonhand.max()),hand_contact_coverage_over_0_1N=float(np.mean(hand>.1)))
        local=dict(record_coverage=n==c['native_integrations']==c['controls']*8,
            COM_origin_matches=np.array_equal(position,z['post_COM']),raw_active_addresses=valid,
            raw_local_force_matches=bool(np.max(abs(force-raw['local_force6']))<=1e-10),
            hand_load_matches=bool(np.max(abs(hand-z['pre_hand_normal_N']))<=1e-10),
            nonhand_load_matches=bool(np.max(abs(nonhand-z['pre_object_nonhand_normal_N']))<=1e-10),
            object_penetration_matches=np.array_equal(pen,z['pre_penetration_m']),self_penetration_matches=np.array_equal(selfpen,z['pre_self_penetration_m']),
            no_post_start_reset=c['post_start_resets']==0,fixed_mount=c['fixed_mount'],physical_matches=all(c['physical_matches'].values()),
            no_force=not np.any(z['post_qfrc_applied']) and not np.any(z['post_xfrc_applied']),finite=all(np.isfinite(v).all() for v in z.values()),
            no_overflow=not np.any(z['overflow']),source_phase=np.array_equal(z['pre_action_phase'],np.arange(1,c['controls']+1)))
        local.update({key+'_metric':bool(np.isclose(value,c['summary']['metrics'][key],atol=1e-12,rtol=1e-12)) for key,value in metrics.items()})
        recomputed={key:metrics[key]<=gates[key] for key in ['COM_RMS_m','COM_max_m','angle_RMS_rad','angle_max_rad','max_penetration_m',
            'max_self_penetration_m','max_joint_limit_excursion_rad','max_object_nonhand_normal_force_N']}
        local.update({key+'_gate':value==c['summary']['gates'][key] for key,value in recomputed.items()})
        first={}
        for key,values in [('COM_max_m',distance),('angle_max_rad',angle),('max_penetration_m',pen),('max_self_penetration_m',selfpen)]:
            ids=np.flatnonzero(values>gates[key]);first[key]=None if not len(ids) else dict(native_index=int(ids[0]),time_s=float(t[ids[0]]),value=float(values[ids[0]]),limit=gates[key])
        prefix=t<=.14+1e-12
        cases.append(dict(name=c['name'],metrics=metrics,first_breaches=first,
            common_0_14s=dict(COM_RMS_m=float(np.sqrt(np.mean(distance[prefix]**2))),angle_RMS_rad=float(np.sqrt(np.mean(angle[prefix]**2))),
                max_penetration_m=float(pen[prefix].max()),max_self_penetration_m=float(selfpen[prefix].max())),checks={key:bool(v) for key,v in local.items()}))
        checks.update({c['name']+'_'+key:bool(v) for key,v in local.items()})
    verify_source_snapshot(receipt,live=True)
    result=dict(schema='native_pen_PPO_independent_strict_review:1',evaluation=file_pin(rp),source_receipt=file_pin(out/'source_receipt.json'),
        cases=cases,checks=checks,passed=sum(checks.values()),total=len(checks),native_integrations=0,optimizer_steps=0,
        scope='Independent raw EFC/contact/pose/gate readback on three same-training-reference single episodes. Matching failed gates is a review pass, not task success or CHORD/ego benefit.')
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],
        durations=[c['metrics']['actual_duration_s'] for c in cases])))
    if not all(checks.values()):raise ValueError('Keep independent readback discrepancy')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
