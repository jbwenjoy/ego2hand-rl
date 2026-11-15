"""Post-hoc descriptive common-horizon comparison of all registered executions."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    evaluation=json.loads(a.evaluation.read_text());review=json.loads(a.review.read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh common-window analysis')
    if review['preview']!=file_pin(a.evaluation) or review['passed']!=review['total']:raise ValueError('Independent original-clock/raw review required')
    out.mkdir();receipt=freeze_sources([Path(__file__).resolve()],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    cases=evaluation['cases'];final=[c for c in cases if c['policy']!='INITIAL'];groups=dict(ALL_COMMON=cases,FINAL_COMMON=final);rows=[];checks={}
    for group,cohort in groups.items():
        horizon=min(c['summary']['metrics']['actual_duration_s'] for c in cohort)
        for c in cohort:
            z=dict(np.load(verify_file_pin(c['trace'])));time=np.arange(1,len(z['post_qpos'])+1)*.0025;use=time<=horizon+1e-12
            actual=Rotation.from_quat(z['post_quaternion_wxyz'][:,[1,2,3,0]]);ref=Rotation.from_quat(z['quaternion_wxyz'][:,[1,2,3,0]])
            target=Slerp(z['time_s'],ref)(time);p=np.stack([np.interp(time,z['time_s'],z['COM_world_m'][:,j]) for j in range(3)],-1)
            distance=np.linalg.norm(z['post_COM']-p,axis=-1);angle=(target.inv()*actual).magnitude()
            metrics=dict(COM_RMS_m=float(np.sqrt(np.mean(distance[use]**2))),COM_max_m=float(distance[use].max()),
                angle_RMS_rad=float(np.sqrt(np.mean(angle[use]**2))),angle_max_rad=float(angle[use].max()),
                max_penetration_m=float(z['pre_penetration_m'][use].max()),max_self_penetration_m=float(z['pre_self_penetration_m'][use].max()))
            checks[group+'_'+c['name']+'_coverage']=int(use.sum())==round(horizon/.0025)
            rows.append(dict(group=group,name=c['name'],policy=c['policy'],repetition=c['repetition'],horizon_s=horizon,metrics=metrics))
    aggregate=[]
    for group in groups:
        for policy in ['INITIAL','RESIDUAL','SUPPORT']:
            selected=[r for r in rows if r['group']==group and r['policy']==policy]
            if selected:
                aggregate.append(dict(group=group,policy=policy,horizon_s=selected[0]['horizon_s'],metrics={key:dict(min=float(np.min([r['metrics'][key] for r in selected])),median=float(np.median([r['metrics'][key] for r in selected])),max=float(np.max([r['metrics'][key] for r in selected]))) for key in selected[0]['metrics']}))
    verify_source_snapshot(receipt,live=True)
    result=dict(schema='pen_support_descriptive_common_horizon:1',evaluation=file_pin(a.evaluation),independent_review=file_pin(a.review),source_receipt=file_pin(out/'source_receipt.json'),
        rules='Post-hoc descriptive rule: ALL_COMMON=min duration across all9; FINAL_COMMON=min duration across all6 final actors. Keep every repeat and original full failed outcome.',
        rows=rows,aggregate=aggregate,checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,
        scope='Matched-time descriptive metrics for numerical repeats, not preregistered statistical benefit, independent training seeds, task success, physical WM gradients or ego generalization.')
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],aggregate=aggregate)),flush=True)
    if not all(checks.values()):raise ValueError('Retain comparison coverage discrepancy')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
