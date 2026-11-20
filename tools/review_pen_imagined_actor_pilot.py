"""Saved CPU review of the paired imagined-Actor pilot, including native, common-prefix scores."""
import argparse
import json
from pathlib import Path
import sys
import tomllib
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from build_pen_capacity_targets import L,compute
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import verify_source_snapshot
from supermimic_research.learning.native_pen_ppo import model_pair,cpu_tree
from supermimic_research.learning.pen_matched_recovery import saved_bytes_equal
from supermimic_research.mechanics.pen_capacity_forecast import load_capacity_head
from supermimic_research.mechanics.pen_contact_groups import normal_group
from supermimic_research.mechanics.pen_load_forecast import LoadForecast
from supermimic_research.mechanics.pen_paired_imagination import imagined_objectives
from supermimic_research.mechanics.pen_task_contracts import tracking_series
from supermimic_research.mechanics.pen_wrench_capacity import balanced_directions
from supermimic_research.runtime.controlled_evaluation import summarize_trace
from supermimic_research.strict_json import write_json

SEEDS=(7411,7412,7413)
TERMS=('COM_RMS_mm','angle_RMS_deg','shaft_RMS_deg','capacity_L1')
KEYS=['parent','A1_7411','A2_7411','A2_7412','A1_7412','A1_7413','A2_7413']


def read(pin):return json.loads(verify_file_pin(pin).read_bytes())


def require(value,why):
    if not value:raise ValueError(why)


def cpu(pin):return torch.load(verify_file_pin(pin),map_location='cpu',weights_only=False)


def verify_training(run,intent,trained):
    require([k for k,_ in trained['actors']]==KEYS,'Unexpected native arm ordering')
    require(intent['seeds']==list(SEEDS) and [intent[k] for k in ('minibatch','updates','bptt_controls','starts','raw_trust_norm','lr')]==[10,24,3,30,.02,.00003],'Unexpected training design')
    torch.set_num_threads(2)
    pins=intent['pins'];actors=dict(trained['actors'])
    start_report=read(file_pin(ROOT/'runs/pen_imagined_actor_starts_v1/report.json'))
    require(start_report['pack']==pins['starts'],'Actor start pack differs')
    expected_load=read(file_pin(ROOT/'runs/pen_load_forecast_v1/report.json'))['outcomes']['state']['model']
    expected_cap=read(file_pin(ROOT/'runs/pen_capacity_fit_v1/report.json'))['variants']['state']['model']
    require(pins['load']==expected_load and pins['capacity']==expected_cap,'Frozen model identity mismatch')
    require(actors['parent']==pins['actor'],'Parent pin differs')
    base=cpu(pins['actor'])
    obj_cfg=tomllib.loads((ROOT/'configs/pen_full_pose_native_v1.toml').read_text())['reward']
    weights=cpu(pins['load'])
    load=LoadForecast(torch.zeros(2,len(weights['center'])),torch.zeros(2,len(weights['out_center'])))
    load.load_state_dict(weights);load=load.double().eval().requires_grad_(False)
    cap=load_capacity_head(cpu(pins['capacity']),dtype=torch.float64,device='cpu').requires_grad_(False)
    with np.load(verify_file_pin(pins['starts'])) as f:
        t={k:torch.as_tensor(f[k]).double() for k in ('state55','previous_raw','previous_ema','ref_joints','ref_pose_wxyz','ref_links','mount_wxyz','encoder_bias','default_joint_pos','default_joint_vel','lower','upper','capacity_start','capacity_eligible','observation161')}
        mass=torch.tensor(f['mass_kg'][0],dtype=torch.float64);inertia=torch.from_numpy(f['inertia_BODY'][0]).double();grav=torch.from_numpy(f['gravity_WORLD'][0]).double()
    from supermimic_research.mechanics.pen_imagined_observation import pen_imagined_observation
    obs=pen_imagined_observation(t['state55'],t['previous_raw'],t['ref_joints'][:,0],t['ref_pose_wxyz'][:,0],t['ref_links'][:,0],t['mount_wxyz'],t['encoder_bias'],t['default_joint_pos'],t['default_joint_vel'])
    torch.testing.assert_close(obs,t['observation161'],rtol=0,atol=1e-6)
    result=[]
    for key,actor_pin in trained['actors'][1:]:
        arm,seed=key.split('_');seed=int(seed)
        artifact=file_pin(run/(key+'.json'));log=read(artifact)
        require(log['actor']==actor_pin and log['seed']==seed and log['label']==arm,'Training summary differs')
        exported=cpu(actor_pin);first_pin=file_pin(run/(key+'_first.pt'));first=cpu(first_pin)
        rng=torch.Generator().manual_seed(seed)
        order=torch.cat([torch.randperm(30,generator=rng).reshape(3,10) for _ in range(8)])
        require(torch.equal(order,first['schedule']) and log['order']==order.tolist(),'Minibatch order mismatch')
        actor,_,_=model_pair('cpu',SimpleNamespace(**base['config']));actor.load_state_dict(base['actor']);actor=actor.double().eval().requires_grad_(False)
        last=actor.net[-1];last.requires_grad_(True)
        opt=torch.optim.Adam([last.weight,last.bias],lr=intent['lr'])
        W=last.weight.detach().clone();B=last.bias.detach().clone()
        mask=torch.zeros_like(B);mask[4:12]=1
        with torch.no_grad():start=actor.distribution(obs).mean.clone()
        def compare_terms(expected):
            with torch.no_grad(): terms=imagined_objectives(actor,load,cap,t,obj_cfg,mass,inertia,grav)
            require(terms.keys()==expected.keys(),'Imagined objective channels changed')
            for k,v in terms.items():np.testing.assert_allclose(float(v),expected[k],rtol=1e-10,atol=1e-11)
        compare_terms(log['before'])
        max_drift=0.
        require(len(log['history'])==24,'Missing optimization history')
        for step,(idx,reported) in enumerate(zip(order,log['history']),1):
            terms=imagined_objectives(actor,load,cap,{k:v[idx] for k,v in t.items()},obj_cfg,mass,inertia,grav)
            loss=-terms['base']+(.2 if arm=='A2' else 0.)*terms['capacity_error']
            opt.zero_grad();loss.backward();last.weight.grad.mul_(mask[:,None]);last.bias.grad.mul_(mask)
            grad=torch.nn.utils.clip_grad_norm_([last.weight,last.bias],1.,error_if_nonfinite=True)
            opt.step()
            with torch.no_grad():
                drift=float((actor.distribution(obs).mean-start).norm(dim=-1).max())
                alpha=min(1.,.02/max(drift,1e-15))
                if alpha<1:
                    last.weight.copy_(W+(last.weight-W)*alpha);last.bias.copy_(B+(last.bias-B)*alpha)
                drift=float((actor.distribution(obs).mean-start).norm(dim=-1).max())
                max_drift=max(max_drift,drift)
            require(reported['step']==step and drift<=.02000001,'Invalid step or trust bound')
            np.testing.assert_allclose([float(loss.detach()),float(grad),drift],[reported[k] for k in ('loss','grad','drift')],rtol=1e-10,atol=1e-11)
            if step==1:
                require(first['step']==1 and saved_bytes_equal(first['actor'],cpu_tree(actor.state_dict())) and saved_bytes_equal(first['optimizer'],cpu_tree(opt.state_dict())),'First-step optimizer/Actor differs from CPU replay')
                actor.load_state_dict(first['actor']);opt.load_state_dict(first['optimizer'])
        compare_terms(log['after'])
        for k,v in actor.state_dict().items():
            require(torch.equal(v.to(base['actor'][k].dtype),exported['actor'][k]),'Final CPU replay differs '+k)
            if k in ('net.6.weight','net.6.bias'):
                keep=torch.ones(len(v),dtype=torch.bool);keep[4:12]=False
                require(torch.equal(base['actor'][k][keep],exported['actor'][k][keep]),'Nonintervention rows changed')
            else:require(torch.equal(base['actor'][k],exported['actor'][k]),'Frozen parameter changed')
        require(saved_bytes_equal(exported['config'],base['config']) and saved_bytes_equal(exported['batch_backend'],base['batch_backend']) and exported['iteration']==base['iteration']==2781,'Export metadata mismatch')
        require(all(int(s['step'])==24 for s in opt.state.values()),'Adam budget')
        result.append(dict(key=key,actor=actor_pin,log=artifact,first=first_pin,max_training_support_drift=max_drift,
                        replayed_steps=24,first_checkpoint_equal=True,final_export_equal=True,frozen_rows_equal=True,
                        before=log['before'],after=log['after']))
    parent_export=file_pin(run/'parent.pt')
    require(saved_bytes_equal(cpu(parent_export),base),'Re-exported parent differs')
    return dict(rows=result,parent_equal=True,parent_export=parent_export,optimizer_replay_steps=144)


def first_capacity(z,model,layout,friction,dirs):
    body=layout['object_body'];qa=layout['object_qposadr']
    R=Rotation.from_quat(z['pre_qpos'][0,qa+np.array([4,5,6,3])]).as_matrix()
    com=z['pre_object_COM'][0];raw=z['raw_contact_tick_plus23']
    cons=[];signs=[];w=np.zeros(6);normal=0.;margin=0.
    for row in raw[raw[:,0]==0]:
        g,h=row[1:3].astype(int);on=model.geom_bodyid[[g,h]]==body
        if on.sum()!=1 or normal_group(model.geom(int(h if on[0] else g)).name or '')<0:continue
        if row[5]>0:
            margin=max(margin,abs(row[18]));continue
        pg,ph=model.geom_priority[[g,h]]
        mu=friction[g] if pg>ph else friction[h] if ph>pg else np.maximum(friction[g],friction[h])
        frame=row[9:18].reshape(3,3);sign=1 if on[1] else -1
        cons.append(SimpleNamespace(dim=int(row[3]),pos=row[6:9],frame=frame,friction=mu[[0,0,1,2,2]]));signs.append(sign)
        force=sign*frame.T@row[18:21]
        torque=sign*frame.T@row[21:24]+np.cross(row[6:9]-com,force)
        w+=np.r_[R.T@force,R.T@torque/L];normal+=row[18]
    H=compute(cons,signs,com,R,dirs)
    violation=max(0.,float((dirs@(w/normal)-H).max())) if normal>1e-8 else 0.
    require(violation<=2e-5 and margin<=1e-5,'Unqualified native contact containment')
    return H,violation,margin,normal


def prefix(series,capacities,n):
    if not n:return dict(valid_controls=0,seconds=0,eligible_capacity_controls=0,**{k:None for k in TERMS},shaft_at_4p44_deg=None)
    s=slice(0,n*8)
    rms=lambda x:float(np.sqrt(np.mean(x*x)))
    caps=[x for x in capacities[:n] if x is not None]
    return dict(valid_controls=n,seconds=.02*n,eligible_capacity_controls=len(caps),
                COM_RMS_mm=1000*rms(series['COM_error_m'][s]),angle_RMS_deg=np.rad2deg(rms(series['full_angle_rad'][s])),
                shaft_RMS_deg=np.rad2deg(rms(series['shaft_angle_rad'][s])),capacity_L1=float(np.mean(caps)) if caps else None,
                shaft_at_4p44_deg=float(np.rad2deg(series['shaft_angle_rad'][1775])) if n>=222 else None)


def review_case(pin,entry,weights,target,eligible,dirs,gates):
    c=read(pin)
    require(c['weights']==weights and c['passed']==c['total'] and all(c['checks'].values()),'Invalid endpoint/weights')
    require(saved_bytes_equal(cpu(c['actor_after']),cpu(weights)['actor']),'Actor/norm differs after evaluation')
    m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['model'])))
    with np.load(verify_file_pin(c['physical_context'])) as f:friction=f['geom_friction'][0]
    require(not m.npair and int(m.opt.cone)==int(mujoco.mjtCone.mjCONE_PYRAMIDAL),'Unexpected friction model')
    with np.load(verify_file_pin(c['trace'])) as f:z=dict(f)
    series=tracking_series(z,.0025)
    caps=[];eligible_indices=[];invalid=[];violation=margin=0.;loaded=0
    count=len(c['controls']);require(0<count<=368 and c['native_integrations']==8*count,'Native budget mismatch')
    for i,q in enumerate(c['controls']):
        row=read(q);require(row['phase']==i+1 and row['native_integrations']==8,'Phase/decimation mismatch')
        tracepath=verify_file_pin(row['trace'])
        if not row['physical_audit']['valid']:
            invalid.append(i);caps.append(None);continue
        with np.load(tracepath) as f:
            np.testing.assert_array_equal(f['post_qpos'],z['post_qpos'][8*i:8*i+8])
            H,v,g,fn=first_capacity(f,m,c['model_layout'],friction,dirs)
        violation=max(violation,v);margin=max(margin,g);loaded+=int(fn>1e-8)
        caps.append(float(np.mean(np.abs(H-target[i]))) if eligible[i] else None)
        if eligible[i]:eligible_indices.append(i+1)
    stop=c['stop_reason'];complete=stop=='REFERENCE_COMPLETION' and count==368
    expected_stop='PHYSICAL_AUDIT_STOP' if invalid else 'TASK_FAILURE' if row['terminated'] else 'REFERENCE_COMPLETION' if row['truncated'] and count==368 else None
    require(stop==expected_stop and (invalid==[count-1] if invalid else True),'Stop classification or prefix mismatch')
    n=count-len(invalid);require(c['valid_prefix_controls']==n and c['valid_prefix_s']==entry['valid_s']==.02*n,'Wrong valid prefix')
    independent=summarize_trace(z,z,.0025,complete,gates)
    require(independent['gates']==c['summary']['gates'] and independent['qualified']==entry['success']==c['full_task_success'] and entry['stop']==stop,'Strict qualification disagreement')
    np.testing.assert_allclose([independent['metrics'][k] for k in ('COM_RMS_m','angle_RMS_rad')],[c['summary']['metrics'][k] for k in ('COM_RMS_m','angle_RMS_rad')],rtol=1e-12,atol=1e-14)
    own=prefix(series,caps,n)
    return (dict(source=pin,rep=c['repetition'],stop=stop,full_success=entry['success'],missing_controls=368-count,
                 own_prefix=own,eligible_control_phases=eligible_indices,failed_gates=[k for k,v in independent['gates'].items() if not v],
                 full_horizon=(dict(COM_RMS_mm=own['COM_RMS_mm'],angle_RMS_deg=own['angle_RMS_deg']) if complete else None),
                 legible_terminal_reasons=row['termination_reasons'],containment=dict(loaded=loaded,max_excess=violation,max_positive_gap_load_N=margin)),
            (series,caps))


def main(run,targets,out,check_training_only=False):
    run=run.resolve();out=out.resolve()
    if not check_training_only and not (run/'report.json').is_file():
        complete=[k for k in KEYS if (run/('eval_'+k)/'report.json').is_file()]
        print(json.dumps(dict(state='incomplete',completed_evaluations=complete,missing=[k for k in KEYS if k not in complete],failure=(run/'failure.json').exists(),wrote_output=False)));return
    if out.exists():raise FileExistsError(out)
    if check_training_only:
        training=verify_training(run,read(file_pin(run/'intent.json')),read(file_pin(run/'trained.json')))
        print(json.dumps(dict(state='cpu_training_replay_verified',steps=training['optimizer_replay_steps'],endpoints=len(training['rows']),wrote_output=False)));return
    result_pin=file_pin(run/'report.json');result=read(result_pin)
    rp=file_pin(run/'source_receipt.json');verify_source_snapshot(read(rp))
    training=verify_training(run,read(result['intent']),read(result['training']))
    require([e['key'] for e in result['evaluated']]==KEYS,'Missing/unordered evaluations')
    targetpin=file_pin(targets/'report.json');tseries=read(targetpin)['refs']['clip_001']['series']
    dirs=balanced_directions()
    with np.load(verify_file_pin(tseries)) as f:
        target=f['capacity'];eligible=f['eligible'];np.testing.assert_array_equal(f['directions'],dirs)
    gatepin=file_pin(ROOT/'configs/controlled_evaluation_v4.toml');gates=tomllib.loads(verify_file_pin(gatepin).read_text())
    store={};evaluations={}
    for e in result['evaluated']:
        key=e['key'];ev=read(e['evaluation'])
        require(len(ev['cases'])==ev['total']==2 and ev['cases']==e['cases'] and ev['successes']==sum(v['success'] for v in ev['cases']),'Wrong evaluation accounting')
        weights=training['parent_export'] if key=='parent' else e['actor']
        require(weights==file_pin(run/(key+'.pt')),'Evaluated Actor mismatch')
        cases=[];store[key]=[]
        for j,entry in enumerate(ev['cases']):
            case,data=review_case(entry['report'],entry,weights,target,eligible,dirs,gates)
            require(case['rep']==j,'Unexpected repetition order')
            cases.append(case);store[key].append(data)
        evaluations[key]=dict(source=e['evaluation'],weights=weights,cases=cases)
    pairs={}
    for seed in SEEDS:
        keys=['parent',f'A1_{seed}',f'A2_{seed}']
        # Fixed common horizon across both numerical repetitions and all three policies.
        n=min(c['own_prefix']['valid_controls'] for k in keys for c in evaluations[k]['cases'])
        fields=(*TERMS,'shaft_at_4p44_deg')
        values={k:[prefix(*a,n) for a in store[k]] for k in keys}
        paired={k:{f:float(np.mean([r[f] for r in vals])) if all(r[f] is not None for r in vals) else None for f in fields} for k,vals in values.items()}
        pairs[str(seed)]=dict(common_controls=n,native_seconds=.02*n,eligible_controls=values['parent'][0]['eligible_capacity_controls'],
                per_repeat=values,repeat_mean=paired,A2_minus_A1={f:paired[keys[2]][f]-paired[keys[1]][f] if paired[keys[1]][f] is not None else None for f in fields})
    common_n=min(c['own_prefix']['valid_controls'] for e in evaluations.values() for c in e['cases'])
    global_prefix=dict(controls=common_n,seconds=.02*common_n,per_repeat={
        k:[prefix(*sample,common_n) for sample in store[k]] for k in KEYS})
    out.mkdir()
    write_json(out/'report.json',dict(producer=result_pin,source_receipt=rp,capacity_targets=tseries,gate_config=gatepin,training=training,evaluations=evaluations,pairs=pairs,global_prefix=global_prefix,
        scope='CPU replay checks exact trained weights, Adam history and first saved Adam state. Native eval records replayed from saved data, zero new simulator calls. Capacity L1 first tick per valid reference-eligible control, distinct from imagined mean over eight ticks. Joint prefix per sampling seed includes paired arms, parent and both repeats. Same reference and warm start; three start minibatch permutations, not independent pretraining seeds.'))
    print(json.dumps({s:dict(seconds=p['native_seconds'],A2_minus_A1=p['A2_minus_A1']) for s,p in pairs.items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('targets',type=Path);p.add_argument('out',type=Path);p.add_argument('--check-training-only',action='store_true')
    a=p.parse_args();main(a.run,a.targets,a.out,a.check_training_only)
