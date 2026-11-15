"""Compare fresh-process continuation against original physical tolerances."""
import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def maximum(v):return float(np.max(v)) if np.asarray(v).size else 0.


def contact_difference(a,b):
    def groups(c):
        result={}
        for i,(w,g,d) in enumerate(zip(c['worldid'],c['geom'],c['dim'])):
            result.setdefault((int(w),int(g[0]),int(g[1]),int(d)),[]).append(i)
        return result
    A,B=groups(a),groups(b);same=A.keys()==B.keys() and all(len(A[k])==len(B[k]) for k in A)
    force=torque=position=0.
    # Rows/EFC allocation order can differ. Match only within the exact world,
    # ordered geom pair and dimension, then minimize contact-position distance.
    for key in A.keys()&B.keys():
        ia=np.array(A[key]);ib=np.array(B[key]);cost=np.linalg.norm(a['pos'][ia,None,:]-b['pos'][ib][None,:,:],axis=-1)
        x,y=linear_sum_assignment(cost);delta=abs(a['local_force6'][ia[x]]-b['local_force6'][ib[y]])
        force=max(force,maximum(delta[:,:3]));torque=max(torque,maximum(delta[:,3:]));position=max(position,maximum(cost[x,y]))
    return dict(same_contact_groups=same,matched_force_max_N=force,matched_torque_max_Nm=torque,matched_position_max_m=position,
        WORLD_force_max_N=maximum(abs(a['wrench_WORLD_COM'][:,:3]-b['wrench_WORLD_COM'][:,:3])),
        WORLD_torque_max_Nm=maximum(abs(a['wrench_WORLD_COM'][:,3:]-b['wrench_WORLD_COM'][:,3:])),
        Fn6_max_N=maximum(abs(a['Fn6']-b['Fn6'])))


def compare_packets(A,B,gates):
    rows=[];checks={}
    checks['equal_substep_coverage']=len(A['substeps'])==len(B['substeps'])==64
    checks['equal_control_coverage']=len(A['transitions'])==len(B['transitions'])==8
    for i,(a,b) in enumerate(zip(A['substeps'],B['substeps'])):
        p,q=a['qpos'],b['qpos'];v,w=a['qvel'],b['qvel']
        if p.shape[-1]!=27 or v.shape[-1]!=26:raise ValueError('Original controlled free7+hand20 layout')
        ar=Rotation.from_quat(p[:,[4,5,6,3]]);br=Rotation.from_quat(q[:,[4,5,6,3]])
        errors=dict(hand_position_rad=maximum(abs(p[:,7:]-q[:,7:])),object_position_m=maximum(np.linalg.norm(p[:,:3]-q[:,:3],axis=-1)),
            object_rotation_rad=maximum((ar*br.inv()).magnitude()),hand_velocity_rad_s=maximum(abs(v[:,6:]-w[:,6:])),
            object_velocity_m_s=maximum(np.linalg.norm(v[:,:3]-w[:,:3],axis=-1)),
            object_omega_rad_s=maximum(np.linalg.norm(ar.apply(v[:,3:6])-br.apply(w[:,3:6]),axis=-1)),control_rad=maximum(abs(a['ctrl']-b['ctrl'])))
        contact=contact_difference(a['contact'],b['contact'])
        flags={key:value<=gates[key] for key,value in errors.items()}
        flags.update(clock_exact=np.array_equal(a['time'],b['time']),contact_topology=contact['same_contact_groups'],
            contact_force=contact['same_contact_groups'] and contact['matched_force_max_N']<=gates['EFC_force_N'],
            contact_torque=contact['same_contact_groups'] and contact['matched_torque_max_Nm']<=gates['EFC_torque_Nm'],
            contact_position=contact['same_contact_groups'] and contact['matched_position_max_m']<=gates['object_position_m'],
            WORLD_force=contact['WORLD_force_max_N']<=gates['EFC_force_N'],WORLD_torque=contact['WORLD_torque_max_Nm']<=gates['EFC_torque_Nm'],
            grouped_Fn6=contact['Fn6_max_N']<=gates['EFC_force_N'],
            producer_contact_indices=all(a['contact']['flags'].values()),restorer_contact_indices=all(b['contact']['flags'].values()))
        rows.append(dict(substep=i,checks={k:bool(v) for k,v in flags.items()},state_errors=errors,contact=contact,
            warmstart_max=maximum(abs(a['warmstart']-b['warmstart']))))
        checks.update({f'substep{i}_'+key:bool(v) for key,v in flags.items()})
    control=[]
    for i,(a,b) in enumerate(zip(A['transitions'],B['transitions'])):
        flags={key:np.array_equal(a[key],b[key]) for key in ['sample','applied','EMA','target','phase','terminated','truncated']}
        control.append(dict(k=a['k'],checks={key:bool(v) for key,v in flags.items()},
            reward_max=maximum(abs(a['reward']-b['reward'])),policy_max=maximum(abs(a['policy']-b['policy'])),critic_max=maximum(abs(a['critic']-b['critic']))))
        checks.update({f'control{i}_'+key:bool(v) for key,v in flags.items()})
    return dict(rows=rows,control=control,checks=checks,passed=int(sum(checks.values())),total=len(checks),
        first_failed_substep=next((r['substep'] for r in rows if not all(r['checks'].values())),None),
        maximum_state_errors={key:max(r['state_errors'][key] for r in rows) for key in rows[0]['state_errors']},
        maximum_contact_errors={key:max(r['contact'][key] for r in rows) for key in rows[0]['contact'] if key!='same_contact_groups'})


def main(a):
    out=a.output.resolve();cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());gp=(cp.parent/cfg['gates']).resolve();gates=tomllib.loads(gp.read_text())
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh review output')
    producer=json.loads(a.producer.read_text());restorer=json.loads(a.restorer.read_text())
    if producer.get('config')!=file_pin(cp) or restorer.get('config')!=file_pin(cp):raise ValueError('Explicit matching continuation protocol required')
    if producer['checkpoint']!=restorer['checkpoint'] or producer['worlds']!=restorer['worlds']:raise ValueError('Matching checkpoint/worlds')
    A=torch.load(verify_file_pin(producer['continuation']),weights_only=False);B=torch.load(verify_file_pin(restorer['continuation']),weights_only=False)
    out.mkdir();receipt=freeze_sources([Path(__file__).resolve(),cp,gp],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    result=compare_packets(A,B,gates)
    rng_equal=json.loads(verify_file_pin(producer['rng_witness']).read_text())==json.loads(verify_file_pin(restorer['rng_witness']).read_text())
    result['checks'].update(immediate_array_bytes=restorer['immediate']['all_array_bytes_equal'],RNG_next_values_equal=rng_equal,
        actual_integration_budget=producer['observed_integrations']==16*producer['worlds']*8 and restorer['observed_integrations']==8*restorer['worlds']*8)
    result.update(schema='pen_native_portable_continuation_review:1',producer=file_pin(a.producer),restorer=file_pin(a.restorer),
        config=file_pin(cp),gates=file_pin(gp),source_receipt=file_pin(out/'source_receipt.json'),
        passed=int(sum(result['checks'].values())),total=len(result['checks']),
        scope='Original physical/force tolerances retained, group-matched contact order. Array copy/RNG equality is separate from continuation. Full Goal incomplete; no policy/ego/speedup claim.')
    verify_source_snapshot(receipt,live=True);write(out/'report.json',result)
    print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],
        first_failed_substep=result['first_failed_substep'],maximum_state_errors=result['maximum_state_errors'],maximum_contact_errors=result['maximum_contact_errors'])))
    if not all(result['checks'].values()):raise ValueError('Keep failed physical continuation; do not promote training readiness')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--producer',type=Path,required=True)
    p.add_argument('--restorer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
