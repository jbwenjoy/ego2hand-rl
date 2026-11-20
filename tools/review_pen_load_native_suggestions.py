"""Saved-only, independently recomputed shaft-angle review of predeclared RAW20 suggestions."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import tomllib

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import verify_source_snapshot
from supermimic_research.mechanics.pen_reward_chain import A,ACTION,MOTION
from supermimic_research.mechanics.warp_packet_decode import decode_saved_case
from supermimic_research.strict_json import write_json

FIELDS=('loss','shaft_deg','tracking','support','reward')
PHASES=(160,180,192,200,206,210,214,218,222,226)


def read(pin):
    return json.loads(verify_file_pin(pin).read_bytes())


def require(condition,message):
    if not condition: raise ValueError(message)


def aggregate(group):
    by_kind={k:[r for r in group if r['kind']==k] for k in ('BASE','DESCEND','ASCEND')}
    means={k:{f:float(np.mean([r['mean'][f] for r in rows])) for f in FIELDS} for k,rows in by_kind.items()}
    spread=abs(by_kind['BASE'][1]['mean']['loss']-by_kind['BASE'][0]['mean']['loss'])
    margin={k:means[k]['loss']-means['DESCEND']['loss'] for k in ('BASE','ASCEND')}
    worst_margin=min(r['mean']['loss'] for k in ('BASE','ASCEND') for r in by_kind[k])-max(r['mean']['loss'] for r in by_kind['DESCEND'])
    reps=[]
    for v in (0,1):
        vals={k:rows[v]['mean'] for k,rows in by_kind.items()}
        reps.append(dict(repeat=v,order_slots={k:rows[v]['slot'] for k,rows in by_kind.items()},
                         wins=all(vals[k]['loss']-vals['DESCEND']['loss']>spread for k in ('BASE','ASCEND'))))
    return dict(mean=means,sham_loss_spread=spread,margin_over_descend=margin,
                beats_both=all(x>0 for x in margin.values()),beats_both_outside_spread=all(x>spread for x in margin.values()),
                separates_all_repeats_outside_spread=worst_margin>spread,repeats=reps,
                descend_minus_base={f:means['DESCEND'][f]-means['BASE'][f] for f in FIELDS})


def review_arm(arm,folder,arm_pin,intent,receipt,objective_pin,config_pins):
    report=read(arm_pin);bank=read(report['proposals'])
    require(report['arm']==arm and tuple(bank['declared'])==PHASES,'Arm/phase declaration differs')
    H=intent['H'];eps=intent['raw_L2']
    sources={s['phase']:s for s in bank['sources']}
    require(len(sources)==len(bank['sources']) and set(sources)<=set(PHASES),'Invalid declared source bank')
    cases=defaultdict(list)
    for c in report['cases']:cases[c['phase']].append(c)
    require(set(cases)<=set(sources),'Case without source')
    m=mujoco.MjModel.from_binary_path(str(folder/'model.mjb'))
    with np.load(folder/'physical_context.npz') as z: physical=dict(z)
    results=[];errors=defaultdict(float);traces=0;source_i=-1
    for phase in PHASES:
        item=dict(phase=phase)
        if phase not in sources:
            results.append(dict(**item,status='missing_source'));continue
        source_i+=1
        s=sources[phase];snap=torch.load(verify_file_pin(s['source']),weights_only=False,map_location='cpu')
        require(snap['sources']==receipt and snap['metadata']['objective_config']==objective_pin,'Snapshot source/objective mismatch')
        control_pin=snap['config'];control_cfg=tomllib.loads(verify_file_pin(control_pin).read_text());config_pins[control_pin['path']]=control_pin
        py=snap['python'];t=py['tensors'];feat=(A('env'),A('_native_full_pose_features'))
        pose=py['arrays'][(A('support_table'),A('pose'))];group_q=t[MOTION+(A('group'),A('joint_pos'))].numpy()
        qa=py['scalars'][feat+(A('qa'),)];body=py['scalars'][feat+(A('body'),)];dof=int(m.jnt_dofadr[m.body_jntadr[body]])
        native=snap['native']['data']['arrays'];aids=t[ACTION+(A('_position_ctrl_ids'),)].numpy();qids=t[feat+(A('qids'),)].numpy()
        require(int(t[MOTION+(A('time_steps'),)][0])==phase,'Invalid snapshot phase')
        if 'skip' in s:
            require(not cases[phase],'Skipped phase contains executed cases')
            results.append(dict(**item,status='skipped_direction',why=s['skip']));continue
        props=s['proposals'];base=np.array(props['BASE']['raw']);delta=np.array(props['DESCEND']['raw'])-base
        np.testing.assert_allclose(np.array(props['ASCEND']['raw'])+delta,base,atol=1e-12,rtol=0)
        np.testing.assert_allclose(np.linalg.norm(delta),eps,atol=1e-12)
        np.testing.assert_allclose(delta[np.setdiff1d(np.arange(20),s['active_axes'])],0,atol=1e-12)
        slots=['BASE','DESCEND','ASCEND','ASCEND','DESCEND','BASE'] if source_i%2==0 else ['BASE','ASCEND','DESCEND','DESCEND','ASCEND','BASE']
        item.update(proposed_cost={k:p['cost'] for k,p in props.items()},order=slots,branches=[],missing_slots=[])
        found={c['rep']:c for c in cases[phase]}
        require(len(found)==len(cases[phase]) and set(found)<=set(range(6)),'Duplicate or extra slot')
        for slot,kind in enumerate(slots):
            if slot not in found:item['missing_slots'].append(slot);continue
            c=found[slot];require(c['kind']==kind and 0<len(c['steps'])<=H,'Wrong order or horizon')
            steps=[];q0=native['qpos'][0];v0=native['qvel'][0]
            for j,z0 in enumerate(c['steps']):
                with np.load(verify_file_pin(z0['trace'])) as saved:z=dict(saved)
                traces+=1
                require(len(z['post_qpos'])==8 and int(z['pre_action_phase'][0])==phase+j,'Wrong decimation/phase')
                np.testing.assert_array_equal(z['actions20'][0],np.asarray(props[kind if j==0 else 'BASE']['raw'],np.float32))
                np.testing.assert_array_equal(z['pre_qpos'][0],q0);np.testing.assert_array_equal(z['pre_qvel'][0],v0)
                np.testing.assert_array_equal(z['pre_qpos'][1:],z['post_qpos'][:-1])
                np.testing.assert_allclose(z['post_ctrl'][:,aids],np.broadcast_to(props[kind]['pd'][j],(8,20)),atol=2e-7,rtol=0)
                q0=z['post_qpos'][-1];v0=z['post_qvel'][-1]
                ref=pose[phase+j];R=Rotation.from_quat(q0[qa+np.array([4,5,6,3])]);T=Rotation.from_quat(ref[[4,5,6,3]])
                actual,target=R.as_matrix()[:,0],T.as_matrix()[:,0]
                dot=float(actual@target)
                loss=1-dot;deg=float(np.degrees(np.arctan2(np.linalg.norm(np.cross(actual,target)),dot)))
                rc=control_cfg['reward'];dist=np.linalg.norm(ref[:3]-q0[qa:qa+3]);angle=(T.inv()*R).magnitude()
                hand=np.mean((group_q[phase+j]-q0[qids])**2);raw=z['actions20'].clip(-1,1)
                tracking=rc['COM_weight']*np.exp(-(dist/rc['COM_scale_m'])**2)+rc['angle_weight']*np.exp(-(angle/rc['angle_scale_rad'])**2)+rc['hand_weight']*np.exp(-hand/rc['hand_scale_rad']**2)-rc['action_weight']*np.mean(raw**2)
                meta=dict(dt=.0025,model_layout=dict(object_body=body,object_qposadr=qa,object_dofadr=dof,COM_BODY=m.body_ipos[body]))
                checks,decoded=decode_saved_case(meta,z,m,physical); geom=np.array([z[k].max() for k in ('pre_penetration_m','pre_self_penetration_m','post_joint_limit_excursion_rad')])
                rows=z['raw_contact_tick_plus23'];nonhand=0.
                for tick,g,h,fn in rows[:,[0,1,2,18]]:
                    on=m.geom_bodyid[[int(g),int(h)]]==body
                    if on.sum()==1 and not (m.geom(int(h if on[0] else g)).name or '').startswith('robot/'):nonhand+=max(fn,0)
                ok=bool(all(checks['checks'].values()) and (geom<=[.002,.0002,.001]).all() and not np.any(z['overflow']) and not np.any(z['post_qfrc_applied']) and not np.any(z['post_xfrc_applied']) and nonhand<=1e-9)
                require(checks['checks']==z0['checks']['checks'] and ok==z0['valid'],'Inconsistent raw-physics validity')
                if j+1<len(c['steps']):require(ok and not z0['terminal'],'Control after a stop')
                for f,calc,tol in [('loss',loss,2e-12),('shaft_deg',deg,1e-7),('tracking',tracking,2e-5)]:
                    err=abs(float(z0[f])-float(calc));errors[f]=max(errors[f],err);require(err<=tol,f'{f} disagreement {err}')
                steps.append(dict(loss=loss,shaft_deg=deg,tracking=float(tracking),support=z0['support'],reward=z0['reward'],
                                  valid=ok,terminal=z0['terminal'],trace=z0['trace']))
            complete=len(steps)==H and all(q['valid'] and not q['terminal'] for q in steps)
            item['branches'].append(dict(kind=kind,slot=slot,complete=complete,steps=steps,mean={f:float(np.mean([q[f] for q in steps])) for f in FIELDS}))
        if item['missing_slots'] or not all(b['complete'] for b in item['branches']):
            item['status']='incomplete_or_physically_invalid'
        else:
            item['status']='complete';item.update(aggregate(item['branches']))
        results.append(item)
    complete=[r for r in results if r['status']=='complete']
    summary=dict(declared=len(PHASES),complete=len(complete),excluded=[{'phase':r['phase'],'reason':r['status']} for r in results if r['status']!='complete'],
        wins=sum(r['beats_both'] for r in complete),wins_outside_sham=sum(r['beats_both_outside_spread'] for r in complete),
        all_repeats_wins_outside_sham=sum(r['separates_all_repeats_outside_spread'] for r in complete),
        predicted_descent=sum(r['proposed_cost']['DESCEND']<r['proposed_cost']['BASE'] for r in complete),traces=traces,
        by_order={name:dict(n=len(subset),wins=sum(r['beats_both_outside_spread'] for r in subset)) for name in ['DESCEND','ASCEND'] for subset in [[r for r in complete if r['order'][1]==name]]},
        by_repeat=[dict(n=len(complete),wins=sum(r['repeats'][i]['wins'] for r in complete)) for i in range(2)],
        median_change_descend_minus_base={k:float(np.median([r['descend_minus_base'][k] for r in complete])) if complete else None for k in FIELDS},max_independent_errors=dict(errors))
    return dict(source=arm_pin,model_observed=file_pin(folder/'model.mjb'),physics_observed=file_pin(folder/'physical_context.npz'),phases=results,summary=summary)


def main(run,out):
    run=run.resolve();out=out.resolve()
    require((run/'report.json').is_file() and not (run/'failure.json').exists(),'Completed run required')
    rp=file_pin(run/'report.json');r=read(rp);intent=read(r['intent'])
    require(tuple(intent['phases'])==PHASES and (intent['H'],intent['raw_L2'],intent['repetitions'])==(3,.02,2),'Unexpected protocol')
    model_path=verify_file_pin(intent['model']);reference_fit=read(file_pin(ROOT/'runs/pen_load_forecast_v1/report.json'))
    require(reference_fit['outcomes']['state']['model']==intent['model'],'Incorrect load forecaster')
    for pin in intent['actors'].values():verify_file_pin(pin)
    receipt_path=file_pin(run/'source_receipt.json');receipt=read(receipt_path)
    verify_source_snapshot(receipt)
    config_pins={};obj=file_pin(ROOT/'configs/pen_long_control_training_v1.toml')
    require(any(rec['observed']==obj for rec in receipt['records']),'Objective not in source receipt')
    config_pins[obj['path']]=obj
    arms={arm:review_arm(arm,run/arm,r['arms'][arm],intent,receipt,obj,config_pins) for arm in ('residual','support')}
    ticks=sum(a['summary']['traces']*8 for a in arms.values())
    require(ticks<=r['native'],'Trace count exceeds native steps')
    out.mkdir(exist_ok=False)
    write_json(out/'report.json',dict(source=rp,intent=r['intent'],source_receipt=receipt_path,configs=list(config_pins.values()),actors=intent['actors'],
        model=file_pin(model_path),reviewer=file_pin(__file__),arms=arms,native_in_producer=r['native'],branch_ticks=ticks,prefix_ticks=r['native']-ticks,
        criterion='Per-phase mean BASE and mean ASCEND loss each exceed mean DESCEND loss by strictly more than absolute difference of the two BASE losses. Each branch loss is the mean across the three control endpoints. Strong separation requires all four controls to beat worst DESCEND by that spread (lower DESCEND is better).',
        scope='Saved-only. Shaft, tracking, PD delivery, source state chain and EFC physical checks recomputed. Total native reward, support bonus and terminal signals are reported from the pinned producer records. Previously inspected one-reference cohorts; two numerical repetitions are not independent training seeds.'))
    print(json.dumps({k:a['summary'] for k,a in arms.items()},ensure_ascii=True))


if __name__=='__main__':main(*map(Path,sys.argv[1:]))
