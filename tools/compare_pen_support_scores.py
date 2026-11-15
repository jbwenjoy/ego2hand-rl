"""Preregistered same-row comparison; score invariants are not policy outcomes."""
import argparse,json,subprocess,sys,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.mechanics.support_score import support_score


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def stats(x):
    a=np.asarray(x,dtype=np.float64)
    if not a.size:return dict(count=0,mean=None,min=None,max=None,positive_fraction=None)
    return dict(count=int(a.size),mean=float(a.mean()),min=float(a.min()),max=float(a.max()),positive_fraction=float(np.mean(a>0)))


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh offline study output')
    out.mkdir();paths=[Path(__file__).resolve(),ROOT/'src/supermimic_research/mechanics/support_score.py',cp,ROOT/'docs/support-score-study-protocol-v1.md']
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    training_path=(cp.parent/cfg['training_report']).resolve();preview_path=(cp.parent/cfg['preview_report']).resolve();ledger_path=(cp.parent/cfg['prior_ledger']).resolve()
    training=json.loads(training_path.read_text());preview=json.loads(preview_path.read_text());ledger=json.loads(ledger_path.read_text())
    if ledger['roots']['training']!=file_pin(training_path) or ledger['roots']['preview_raw']!=file_pin(preview_path):raise ValueError('Prior ledger input pin mismatch')
    checks={};analytic=[];elig=torch.ones(3,dtype=torch.bool)
    targets=torch.tensor([[0.,0,.3,0,0,0],[0,0,0,0,.003,0],[.12,-.2,.3,.001,.003,-.002]],dtype=torch.float64)
    for variant in cfg['variants']:
        def score(w,normal=.5):return support_score(w,targets,elig,cfg['force_scale_N'],cfg['torque_scale_Nm'],variant,
            normal_mean_N=torch.full((3,),normal),contact_threshold_N=cfg['contact_threshold_N'])['score']
        rows=torch.stack([score(alpha*targets) for alpha in cfg['alpha_grid']]);zero=score(torch.zeros_like(targets));weak=score(torch.zeros_like(targets),0.)
        before=score(targets,cfg['contact_threshold_N']-1e-6);after=score(targets,cfg['contact_threshold_N']+1e-6)
        invariant=dict(zero_net_at_strong_contact=bool((zero.abs()<1e-14).all()),zero_net_at_weak_contact=bool((weak.abs()<1e-14).all()),
            threshold_continuity=bool(((before-after).abs()<1e-14).all()),exact_target_unique_best=bool((rows.argmax(0)==cfg['alpha_grid'].index(1.)).all()) and all(bool((rows[cfg['alpha_grid'].index(1.)]>rows[j]).all()) for j in range(len(rows)) if cfg['alpha_grid'][j]!=1.),
            opposite_target_negative=bool((rows[0]<0).all()))
        analytic.append(dict(variant=variant,alpha_grid=cfg['alpha_grid'],scores=rows.tolist(),invariants=invariant,passes_selection=all(invariant.values())))
    selected=[r['variant'] for r in analytic if r['passes_selection']];checks['preregistered_unique_selection']=selected==['CENTERED']
    # Rotation, component sensitivity, eligibility and exact zero-target behavior.
    from scipy.spatial.transform import Rotation
    R=torch.from_numpy(Rotation.from_euler('xyz',[.3,-.7,.8]).as_matrix())
    rotate=lambda w:torch.cat([w[:,:3]@R.T,w[:,3:]@R.T],-1)
    base=support_score(.4*targets,targets,elig,.3,.003,'CENTERED')['score']
    checks['rotation_invariance']=torch.allclose(base,support_score(rotate(.4*targets),rotate(targets),elig,.3,.003,'CENTERED')['score'],atol=1e-14,rtol=0)
    ideal=support_score(targets,targets,elig,.3,.003,'CENTERED')['score'];perturb=targets.clone();perturb[:,:3]+=.3
    checks['force_sensitivity']=bool((support_score(perturb,targets,elig,.3,.003,'CENTERED')['score']<ideal).all());perturb=targets.clone();perturb[:,3:]+=.003
    checks['torque_sensitivity']=bool((support_score(perturb,targets,elig,.3,.003,'CENTERED')['score']<ideal).all())
    checks['invalid_disabled']=bool((support_score(targets,targets,~elig,.3,.003,'CENTERED')['score']==0).all())
    checks['zero_target_zero_measurement']=bool((support_score(torch.zeros_like(targets),torch.zeros_like(targets),elig,.3,.003,'CENTERED')['score']==0).all())
    cohorts=[];source_pins=[]
    for arm in training['arms']:
        for log in arm['logs']:
            pin=log['memory'];source_pins.append(pin);m=torch.load(verify_file_pin(pin),map_location='cpu',weights_only=False)
            measured=m['contact_objective_measured_source'];target=m['contact_objective_target_source'];eligible=m['contact_objective_support_eligible']
            normal=m['contact_Fn6_sum'].sum(-1)/m['contact_counts'][...,0];weak=eligible&(normal<=cfg['contact_threshold_N']);strong=eligible&~weak;values={}
            for variant in cfg['variants']:
                result=support_score(measured,target,eligible,.3,.003,variant,normal_mean_N=normal,contact_threshold_N=cfg['contact_threshold_N'])
                value=result['score'];values[variant]=dict(eligible=stats(value[eligible]),weak=stats(value[weak]),strong=stats(value[strong]))
                if variant=='ABSOLUTE':checks[arm['arm']+str(log['iteration'])+'_v1_reconstruction']=torch.allclose(value.float(),m['contact_objective_support_bonus'],atol=1e-6,rtol=1e-6)
                if variant=='CENTERED':
                    v=value.numpy();t=target.numpy();w=measured.numpy();loss=(((w[...,:3]-t[...,:3])/.3)**2).mean(-1)+(((w[...,3:]-t[...,3:])/.003)**2).mean(-1)
                    zero=((t[...,:3]/.3)**2).mean(-1)+((t[...,3:]/.003)**2).mean(-1);expected=np.where(eligible.numpy(),np.exp(-.5*loss)-np.exp(-.5*zero),0)
                    checks[arm['arm']+str(log['iteration'])+'_independent_numpy']=bool(np.allclose(v,expected,atol=1e-14,rtol=1e-14))
            cohorts.append(dict(arm=arm['arm'],iteration=log['iteration'],memory=pin,variants=values))
    saved=[]
    for case in preview['cases']:
        pin=case['control_windows'];source_pins.append(pin);w={k:torch.from_numpy(v) for k,v in dict(np.load(verify_file_pin(pin))).items()}
        normal=w['Fn6_sum'].sum(-1)/w['counts'][:,0];rows=[]
        for variant in cfg['variants']:
            value=support_score(w['objective_measured_source'],w['objective_target_source'],w['objective_support_eligible'],.3,.003,variant,normal_mean_N=normal)['score']
            rows.append(dict(variant=variant,control_end_s=(np.arange(1,len(value)+1)*.02).tolist(),scores=value.tolist(),mean_hand_normal_N=normal.tolist()))
        saved.append(dict(name=case['name'],variants=rows))
    checks={k:bool(v) for k,v in checks.items()};verify_source_snapshot(receipt,live=True)
    result=dict(schema='pen_support_score_offline_comparison:1',config=file_pin(cp),training=file_pin(training_path),preview=file_pin(preview_path),prior_ledger=file_pin(ledger_path),
        source_receipt=file_pin(out/'source_receipt.json'),inputs=source_pins,analytic=analytic,selected_variant=selected,cohorts=cohorts,executions=saved,
        checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,scope=cfg['scope'])
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),selected=selected,checks=[result['passed'],result['total']],analytic=[dict(variant=r['variant'],invariants=r['invariants']) for r in analytic])),flush=True)
    if not all(checks.values()):raise ValueError('Retain offline comparison discrepancy; do not train')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
