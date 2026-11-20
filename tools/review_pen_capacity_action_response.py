"""Independently audit first-control geometry-capacity responses on existing shaft-selected actions."""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.mechanics.controlled_pen_records import native_state55
from supermimic_research.mechanics.pen_capacity_trace import packet_capacity
from supermimic_research.mechanics.pen_wrench_capacity import balanced_directions
from supermimic_research.mechanics.pen_capacity_forecast import load_capacity_head,capacity_mismatch
from supermimic_research.mechanics.pen_reward_chain import A,ACTION
from supermimic_research.strict_json import write_json


def read(pin):
    return json.loads(verify_file_pin(pin).read_bytes())


def prepare(root,targets,shaft_review,out):
    out.mkdir(exist_ok=False,parents=True)
    source_pin=file_pin(root/'report.json')
    safety_pin=file_pin(shaft_review/'report.json')
    producer=read(source_pin); verified=read(safety_pin)
    if verified['source']!=source_pin:raise ValueError('Shaft audit belongs to another run')
    intent=read(producer['intent'])
    target_pin=file_pin(targets/'report.json')
    target_series=read(target_pin)['refs']['clip_001']['series']
    with np.load(verify_file_pin(target_series)) as f:
        reference=f['capacity'];eligible=f['eligible'];stage=f['stage']
        np.testing.assert_array_equal(f['directions'],balanced_directions())
    rows=[]
    for arm,armpin in producer['arms'].items():
        armdata=read(armpin);audit=verified['arms'][arm]
        if audit['source']!=armpin:raise ValueError('Arm audit mismatch')
        model=mujoco.MjModel.from_binary_path(str(verify_file_pin(audit['model_observed'])))
        with np.load(verify_file_pin(audit['physics_observed'])) as f:friction=f['geom_friction']
        props=read(armdata['proposals'])
        for phase in intent['phases']:
            k=phase-1
            row=dict(arm=arm,phase=phase,k=k,stage=str(stage[k]),eligible=bool(eligible[k]),target=reference[k].tolist(),cases=[])
            rows.append(row)
            audit_phase=next(r for r in audit['phases'] if r['phase']==phase)
            source=next((r for r in props['sources'] if r['phase']==phase),None)
            if not source or 'skip' in source:
                row['status']=audit_phase['status'];continue
            snap=torch.load(verify_file_pin(source['source']),weights_only=False,map_location='cpu')
            feat=(A('env'),A('_native_full_pose_features'))
            py=snap['python'];qa=py['scalars'][feat+(A('qa'),)];b=py['scalars'][feat+(A('body'),)]
            qids=py['tensors'][feat+(A('qids'),)].numpy();dofs=py['tensors'][feat+(A('dofs'),)].numpy()
            aids=py['tensors'][ACTION+(A('_position_ctrl_ids'),)].numpy();da=model.jnt_dofadr[model.body_jntadr[b]]
            layout=dict(object_body=b,object_qposadr=qa)
            nq=snap['native']['data']['arrays']
            state=native_state55(nq['qpos'][0],nq['qvel'][0],qids,dofs,qa,da)
            row.update(source=source['source'],state=state.tolist())
            for branch in armdata['cases']:
                if branch['phase']!=phase:continue
                first=branch['steps'][0]
                aud=next(q for q in audit_phase['branches'] if q['slot']==branch['rep'])['steps'][0]
                assert aud['trace']==first['trace'] and aud['valid']==first['valid']
                with np.load(verify_file_pin(first['trace'])) as f:
                    np.testing.assert_array_equal(f['pre_qpos'][0],nq['qpos'][0]);np.testing.assert_array_equal(f['pre_qvel'][0],nq['qvel'][0])
                    sub=packet_capacity(f,model,layout,friction,length_scale_m=.10)
                    pd=f['post_ctrl'][0,aids].copy()
                    np.testing.assert_array_equal(f['post_ctrl'][:,aids],np.broadcast_to(pd,(8,20)))
                cap=sub['capacity'].mean(0)
                permitted=first['valid'] and not first['terminal'] and sub['maximum_containment_excess']<=2e-5 and sub['positive_gap_skipped_max_load_N']<=1e-5
                row['cases'].append(dict(slot=branch['rep'],kind=branch['kind'],trace=first['trace'],valid=bool(permitted),
                    capacity=cap.tolist(),pd=pd.tolist(),native_l1=float(np.abs(cap-reference[k]).mean()),tick_mask=sub['mask'].astype(int).tolist(),
                    max_containment_excess=sub['maximum_containment_excess'],max_positive_gap_load=sub['positive_gap_skipped_max_load_N']))
            complete=sorted(q['slot'] for q in row['cases'])==list(range(6)) and all(q['valid'] for q in row['cases'])
            row['status']='invalid_or_missing_first_control' if not complete else ('included' if eligible[k] else 'ineligible_reference')
    write_json(out/'report.json',dict(source=source_pin,safety_review=safety_pin,target_source=target_pin,target_series=target_series,
        source_packets=rows,protocol='Closed-loop frozen source states from a shaft-gradient-selected prospective study. First 20ms only. Eligible[k] and reference[k], k=phase-1. Capacity is mean BODY support over eight ticks; loss is mean absolute directional discrepancy.'))
    print('prepared',len(rows),{a:sum(r['status']=='included' and r['arm']==a for r in rows) for a in producer['arms']})


def contains_pin(node,pin):
    if isinstance(node,dict):return node==pin or any(contains_pin(v,pin) for v in node.values())
    if isinstance(node,list):return any(contains_pin(v,pin) for v in node)
    return False


def summarize(rows):
    if not rows:return {'n':0}
    actual=np.array([r['actual_change'] for r in rows])
    predicted=np.array([r['linear_prediction'] for r in rows])
    nonlinear=np.array([r['model_change'] for r in rows])
    signal=float(np.linalg.norm(actual));noise=np.array([r['sham_spread'] for r in rows]);above=np.abs(actual)>noise+1e-10
    return dict(n=len(rows),exceeds_base_repeat_spread=int(above.sum()),sign_agrees=int((actual*predicted>0).sum()),
                sign_agrees_above_spread=int(((actual*predicted>0)&above).sum()),model_finite_sign_agrees=int((actual*nonlinear>0).sum()),
                predicted_norm_over_observed=float(np.linalg.norm(predicted)/signal) if signal else None,
                jvp_error_over_zero_response=float(np.linalg.norm(predicted-actual)/signal) if signal else None,
                finite_error_over_zero_response=float(np.linalg.norm(nonlinear-actual)/signal) if signal else None,
                native_change_rms=float(np.sqrt(np.mean(actual**2))),predicted_change_rms=float(np.sqrt(np.mean(predicted**2))),
                base_repeat_spread_rms=float(np.sqrt(np.mean(noise**2))))


def evaluate(prepared,fit,out):
    if out.exists():raise ValueError('Fresh capacity response review required')
    native_pin=file_pin(prepared/'report.json');native=read(native_pin)
    fit_pin=file_pin(fit/'report.json');fit_report=read(fit_pin)
    ckpin=file_pin(fit/'state.pt')
    if not contains_pin(fit_report,ckpin):raise ValueError('Unverified state-only checkpoint')
    fit_intent=read(fit_report['intent'])
    if fit_intent['reference']!=native['target_series']:
        raise ValueError('Target capacity reference differs from the forecast training protocol')
    collection=read(read(fit_intent['data'])['collection'])
    behavior_pins=read(collection['intent'])['actors']
    probe_actors=read(read(native['source'])['intent'])['actors']
    if probe_actors!={'residual':behavior_pins['TEST2'],'support':behavior_pins['TEST3']}:
        raise ValueError('Source actors differ from the declared holdout roles')
    train_pins=[behavior_pins[k]['sha256'] for k in ('A','B','C','D','VAL')]
    if any(p['sha256'] in train_pins for p in probe_actors.values()):
        raise ValueError('Evaluation policy also owns a TRAIN or VAL trajectory')
    head=load_capacity_head(torch.load(verify_file_pin(ckpin),weights_only=True,map_location='cpu'),dtype=torch.float64,device='cpu').requires_grad_(False)
    if head.use_history:raise ValueError('State-only capacity model required')
    t=lambda v:torch.as_tensor(v,dtype=torch.float64)
    all_rows=[];summary={}
    for arm in ('residual','support'):
        rows=[];exclusions=[]
        for g in native['source_packets']:
            if g['arm']!=arm:continue
            if g['status']!='included':
                exclusions.append(dict(phase=g['phase'],stage=g['stage'],status=g['status']));continue
            state=t(g['state']);target=t(g['target'])
            grouped={k:sorted([q for q in g['cases'] if q['kind']==k],key=lambda r:r['slot']) for k in ('DESCEND','ASCEND','BASE')}
            pd={k:t(v[0]['pd']) for k,v in grouped.items()}
            for k,v in grouped.items():np.testing.assert_array_equal(v[0]['pd'],v[1]['pd'])
            if not torch.allclose((pd['DESCEND']+pd['ASCEND'])/2,pd['BASE'],atol=2e-7,rtol=0):raise ValueError('Asymmetric transmitted PD probe')
            def loss(u):return capacity_mismatch(head(state,u),target)
            u=pd['BASE'].clone().requires_grad_(True)
            grad=torch.autograd.grad(loss(u),u)[0].detach()
            pred={k:float(loss(v).detach()) for k,v in pd.items()}
            real={k:float(np.mean([q['native_l1'] for q in v])) for k,v in grouped.items()}
            observed_capacity={k:np.mean([q['capacity'] for q in v],axis=0) for k,v in grouped.items()}
            spread=abs(grouped['BASE'][0]['native_l1']-grouped['BASE'][1]['native_l1'])
            for a,b in [('DESCEND','ASCEND'),('DESCEND','BASE'),('ASCEND','BASE')]:
                direction=pd[a]-pd[b];jvp=float(grad@direction)
                _,jvp72=torch.autograd.functional.jvp(lambda p:head(state,p),u,direction)
                delta=observed_capacity[a]-observed_capacity[b]
                rows.append(dict(arm=arm,phase=g['phase'],stage=g['stage'],contrast=a+'-'+b,actual_change=real[a]-real[b],
                    repeat_changes=[grouped[a][i]['native_l1']-grouped[b][i]['native_l1'] for i in range(2)],
                    native_l1=real,model_l1=pred,linear_prediction=jvp,model_change=pred[a]-pred[b],sham_spread=spread,
                    pd_displacement=float(direction.norm()),gradient_norm=float(grad.norm()),native_capacity_delta_norm=float(np.linalg.norm(delta)),
                    predicted_capacity_jvp_norm=float(jvp72.norm()),capacity_jvp_error_norm=float(np.linalg.norm(jvp72.numpy()-delta))))
        groups={key:summarize([r for r in rows if r['contrast']==key]) for key in ('DESCEND-ASCEND','DESCEND-BASE','ASCEND-BASE')}
        summary[arm]=dict(declared=sum(r['arm']==arm for r in native['source_packets']),excluded=exclusions,groups=groups)
        all_rows.extend(rows)
    out.mkdir();write_json(out/'report.json',dict(inputs=dict(native=native_pin,fit=fit_pin,model=ckpin),reviewer=file_pin(__file__),summary=summary,rows=all_rows,
        scope='All HOST computational review, no native steps, policy updates or post-hoc source selection. ASCEND/DESCEND are shaft-loss directions. Error ratios compare the scalar capacity mismatch differential against null zero-change prediction. Same-source no-perturb BASE and its repeat establish an empirical numerical spread, not a confidence bound.'))
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);cmd=p.add_subparsers(dest='mode',required=True)
    q=cmd.add_parser('prepare');q.add_argument('producer',type=Path);q.add_argument('targets',type=Path);q.add_argument('shaft_review',type=Path);q.add_argument('out',type=Path)
    q=cmd.add_parser('evaluate');q.add_argument('prepared',type=Path);q.add_argument('fit',type=Path);q.add_argument('out',type=Path)
    args=p.parse_args()
    if args.mode=='prepare':prepare(args.producer,args.targets,args.shaft_review,args.out)
    else:evaluate(args.prepared,args.fit,args.out)

