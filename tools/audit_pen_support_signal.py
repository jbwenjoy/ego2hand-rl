"""Offline zero-wrench support-score audit; never modify pilot rewards/weights."""
import argparse,json,subprocess,sys,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def stats(x):
    a=np.asarray(x,dtype=np.float64)
    if not a.size:return dict(count=0,min=None,median=None,mean=None,max=None)
    return dict(count=a.size,min=float(a.min()),median=float(np.median(a)),mean=float(a.mean()),max=float(a.max()))


def scores(target,cfg):
    loss=(target[...,:3]/cfg['force_scale_N'])**2
    torque=(target[...,3:]/cfg['torque_scale_Nm'])**2
    return np.exp(-.5*(loss.mean(-1)+torque.mean(-1)))


def main(a):
    run=json.loads(a.training.read_text());preview=json.loads(a.preview.read_text());qp=verify_file_pin(run['qualification']);qual=json.loads(qp.read_text())
    config=verify_file_pin(run['config']);cfg=tomllib.loads(config.read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh offline signal audit')
    out.mkdir();receipt=freeze_sources([Path(__file__).resolve(),config],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    table=dict(np.load(verify_file_pin(qual['support_reference'])));valid=table['valid'];zero=scores(table['targets'][0,valid],cfg)
    checks=dict(reference_mask=int(valid.sum())==365,zero_score_bounded=bool(np.all((zero>=0)&(zero<=1))))
    reference=dict(valid_intervals=int(valid.sum()),zero_wrench_score=stats(zero),fraction_zero_score_over_0_8=float(np.mean(zero>.8)),
                   fraction_zero_score_over_0_5=float(np.mean(zero>.5)))
    cohorts=[]
    for arm in run['arms']:
        for log in arm['logs']:
            m=torch.load(verify_file_pin(log['memory']),map_location='cpu',weights_only=False)
            target=m['contact_objective_target_source'].numpy();measured=m['contact_objective_measured_source'].numpy()
            loss=((measured[...,:3]-target[...,:3])/cfg['force_scale_N'])**2
            torque=((measured[...,3:]-target[...,3:])/cfg['torque_scale_Nm'])**2
            expected=np.exp(-.5*(loss.mean(-1)+torque.mean(-1)));eligible=m['contact_objective_support_eligible'].numpy()
            actual=m['contact_objective_support_bonus'].numpy();zero_score=scores(target,cfg)
            mean_load=m['contact_Fn6_sum'].numpy().sum(-1)/m['contact_counts'].numpy()[...,0]
            weak=(mean_load<=.1)&eligible;strong=(mean_load>.1)&eligible
            check_name=arm['arm']+'_iter'+str(log['iteration'])
            checks[check_name+'_saved_score']=bool(np.allclose(np.where(eligible,expected,0),actual,atol=1e-6,rtol=1e-6))
            checks[check_name+'_saved_loss']=bool(np.allclose(loss.mean(-1)+torque.mean(-1),m['contact_objective_support_loss'].numpy(),atol=1e-6,rtol=1e-6))
            cohorts.append(dict(arm=arm['arm'],iteration=log['iteration'],memory=log['memory'],eligible_rows=int(eligible.sum()),
                weak_load_eligible_rows=int(weak.sum()),weak_load_score=stats(actual[weak]),weak_load_zero_wrench_score=stats(zero_score[weak]),
                strong_load_score=stats(actual[strong]),eligible_score_minus_zero=stats((actual-zero_score)[eligible]),
                weak_load_fraction_score_over_0_8=None if not weak.any() else float(np.mean(actual[weak]>.8))))
    executions=[]
    for c in preview['cases']:
        w=dict(np.load(verify_file_pin(c['control_windows'])));mean_load=w['Fn6_sum'].sum(-1)/w['counts'][:,0]
        zero_score=scores(w['objective_target_source'],cfg);eligible=w['objective_support_eligible'];weak=(mean_load<=.1)&eligible
        rows=[]
        for k in np.flatnonzero(weak):
            rows.append(dict(end_s=float((k+1)*.02),mean_hand_normal_N=float(mean_load[k]),actual_score=float(w['objective_support_bonus'][k]),
                counterfactual_zero_wrench_score=float(zero_score[k])))
        executions.append(dict(name=c['name'],weak_load_rows=rows))
    verify_source_snapshot(receipt,live=True)
    report=dict(schema='pen_support_zero_wrench_offline_audit:1',training=file_pin(a.training),preview=file_pin(a.preview),qualification=file_pin(qp),
        reference=reference,cohorts=cohorts,executions=executions,checks=checks,passed=sum(checks.values()),total=len(checks),source_receipt=file_pin(out/'source_receipt.json'),
        actual_native_integrations=0,optimizer_steps=0,
        interpretation='A bounded absolute exp score can be positive for zero wrench. These are saved-data diagnostics, not an intervention or causal explanation of the failed policy. Do not relabel weak mean load as all native samples having zero contact.',
        next_protocol='Before another support-training sweep, preregister contact-load eligibility and/or a zero-wrench-centered score, test physical feasibility/normalization and report equal budgets. Keep original tracking/geometry gates and v1 pilot unchanged.')
    write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[report['passed'],report['total']],reference=reference,executions=executions)),flush=True)
    if not all(checks.values()):raise ValueError('Retain offline audit discrepancy')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--training',type=Path,required=True);p.add_argument('--preview',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
