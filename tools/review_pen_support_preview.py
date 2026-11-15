"""Independent saved-pose/EFC/mean-wrench readback and failure ordering."""
import argparse,json,subprocess,sys,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import mujoco
import numpy as np
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.mechanics.saved_pen_diagnostics import diagnose


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    run=json.loads(a.preview.read_text());out=a.output.resolve();gp=ROOT/'configs/controlled_evaluation_v4.toml';gates=tomllib.loads(gp.read_text())
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh CPU review output')
    out.mkdir();paths=[Path(__file__).resolve(),ROOT/'src/supermimic_research/mechanics/saved_pen_diagnostics.py',gp]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    checks={};cases=[]
    for c in run['cases']:
        z=dict(np.load(verify_file_pin(c['trace'])));raw=dict(np.load(verify_file_pin(c['raw_contacts'])));windows=dict(np.load(verify_file_pin(c['control_windows'])))
        m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['model'])))
        result=diagnose(m,z,raw,windows,gates);local=result['checks']
        target=windows['objective_target_source'];measured=windows['objective_measured_source']
        loss=(((measured[:,:3]-target[:,:3])/.3)**2).mean(-1)+(((measured[:,3:]-target[:,3:])/.003)**2).mean(-1)
        score=np.exp(-.5*loss)
        if c.get('checkpoint_schema')=='pen_geometry_support_PPO_training:4':
            baseline=np.exp(-.5*(((target[:,:3]/.3)**2).mean(-1)+((target[:,3:]/.003)**2).mean(-1)))
            score-=baseline
        expected=np.where(windows['objective_support_eligible'],score,0)
        local['declared_support_score']=bool(np.allclose(expected,windows['objective_support_bonus'],atol=1e-6,rtol=1e-6))
        local.update({key+'_metric':bool(np.isclose(value,c['summary']['metrics'][key],atol=1e-12,rtol=1e-12)) for key,value in result['metrics'].items()})
        for key in ['COM_RMS_m','COM_max_m','angle_RMS_rad','angle_max_rad','max_penetration_m','max_self_penetration_m','max_joint_limit_excursion_rad','max_object_nonhand_normal_force_N']:
            local[key+'_gate']=(result['metrics'][key]<=gates[key])==c['summary']['gates'][key]
        local.update(no_force=not np.any(z['post_qfrc_applied']) and not np.any(z['post_xfrc_applied']),finite=all(np.isfinite(v).all() for v in z.values()),
            no_overflow=not np.any(z['overflow']),no_post_start_reset=c['post_start_resets']==0,fixed_mount=c['fixed_mount'],physical_matches=all(c['physical_matches'].values()),
            native_count=c['native_integrations']==c['controls']*8==len(z['post_qpos']))
        cases.append(dict(name=c['name'],**result));checks.update({c['name']+'_'+k:v for k,v in local.items()})
    verify_source_snapshot(receipt,live=True)
    report=dict(schema='pen_support_preview_independent_review:1',preview=file_pin(a.preview),source_receipt=file_pin(out/'source_receipt.json'),cases=cases,
        checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,
        scope='Independent original-clock full-SO3, active-EFC, geometry, signed hand WORLD/moving-COM wrench and mean support loss readback. Failure ordering is descriptive, not causal or a successful task/ego/WM-gradient claim.')
    write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[report['passed'],report['total']],first_breaches=[c['first_breaches'] for c in cases])),flush=True)
    if not all(checks.values()):raise ValueError('Retain independent readback discrepancy')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preview',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
