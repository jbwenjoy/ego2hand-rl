"""CPU-only one-control coverage at two recorded filter failures and their preceding states."""
import argparse
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics import pen_contact_action_filter as af
from supermimic_research.mechanics import pen_corrective_lookahead as shooting
from supermimic_research.strict_json import write_json


PHASES={'clip_002':(8,9),'clip_003':(50,51)}
METRICS=('object_mm','self_mm','joint_excursion_rad','nonhand_N','COM_mm','shaft_deg')
GATES=np.array([2.,.2,.001,.1,50.,60.])
SEED=112023


def read(pin):
    return json.loads(verify_file_pin(pin).read_bytes())


def original_raw(actor,previous,ring):
    pool=np.tile(actor.clip(-1,1),(6,1))
    last=previous.clip(-1,1)
    pool[1,ring]=.5*(pool[0,ring]+last[ring])
    pool[2,ring]=last[ring]
    pool[3]=.5*(pool[0]+last)
    pool[4]=last
    pool[5]=0
    return pool


def transmitted(x,raw):
    return np.clip(x['refq']+(.5*x['previous_ema']+.25*raw)-x['encoder_bias'],x['lower'],x['upper'])


def simulate(model,x,phase,targets,filter_model):
    data=mujoco.MjData(model)
    data.qpos[:]=x['qpos']
    data.qvel[:]=x['qvel']
    data.qacc_warmstart[:]=x['warmstart']
    data.mocap_pos[:]=x['mocap_pos']
    data.mocap_quat[:]=x['mocap_quat']
    data.time=(phase-1)*.02
    state=np.empty(mujoco.mj_stateSize(model,shooting.STATE))
    mujoco.mj_getState(model,data,state,shooting.STATE)
    pose=x['target_pose']
    rows=shooting.score_candidates(model,state,targets[:,None,:],pose[None,:3],pose[None,3:],
        body=filter_model.body,qa=filter_model.qa,da=filter_model.da)
    metrics=[]
    for row in rows:
        tick=row['intervals'][0]
        metrics.append([1000*tick['max_object_penetration_m'],1000*tick['max_self_penetration_m'],
            tick['max_joint_excursion_rad'],tick['max_nonhand_normal_N'],tick['mm'],tick['shaft_deg']])
    return np.asarray(metrics,dtype=float),np.array([r['finite'] for r in rows]),sum(r['native_steps'] for r in rows)


def assess(out,clip,phase,unit,model,filter_model,case):
    inputs,decisions=case['inputs'],case['decisions']
    with np.load(verify_file_pin(inputs)) as z:
        x={k:z[k][phase-1] for k in z.files}
    logged=read(decisions)[phase-1]
    if logged['phase']!=phase:
        raise ValueError('Recorded control phase mismatch')
    six=original_raw(x['actor_raw'],x['previous_raw'],filter_model.ring)
    pd=transmitted(x,six)
    chosen=af.LABELS.index(logged['choice'])
    np.testing.assert_array_equal(six[chosen],x['selected_raw'])
    np.testing.assert_array_equal(pd[chosen],logged['pd'])
    np.testing.assert_array_equal(np.repeat(pd[chosen:chosen+1],8,axis=0),x['actual_pd_by_tick'])
    old,ok_old,old_ticks=simulate(model,x,phase,pd,filter_model)
    if not ok_old.all():
        raise ValueError('Nonfinite conventional candidate in parity replay')
    original=filter_model._peaks(pd,x['qpos'],x['qvel'],x['warmstart'],
        (x['mocap_pos'],x['mocap_quat']),None)
    np.testing.assert_allclose(old[:,:2],original,rtol=0,atol=1e-8)
    log_errors=[]
    for label,pair in logged['evaluations'].items():
        diff=old[af.LABELS.index(label),:2]-[pair['object_mm'],pair['self_mm']]
        np.testing.assert_allclose(diff,0,rtol=0,atol=1e-8)
        log_errors.append(float(abs(diff).max()))
    if phase==PHASES[clip][-1] and (len(log_errors)!=6 or np.any((old[:,:2]<=GATES[:2]).all(axis=1))):
        raise ValueError('Requested failure did not exhaust all six geometry forecasts')
    actor=six[0]
    local=np.clip(actor+.35*(2*unit[:128]-1),-1,1).astype(np.float32)
    global_raw=(2*unit[128:]-1).astype(np.float32)
    extra=np.vstack([local,global_raw])
    extra_pd=transmitted(x,extra)
    more,ok_more,extra_ticks=simulate(model,x,phase,extra_pd,filter_model)
    raw=np.vstack([six,extra])
    targets=np.vstack([pd,extra_pd])
    metrics=np.vstack([old,more])
    finite=np.r_[ok_old,ok_more]
    passed=finite & (metrics<=GATES).all(axis=1)
    clearance=(metrics[:,:2]/GATES[:2]).max(axis=1)
    raw_dist=np.linalg.norm(raw-actor,axis=1)
    pd_dist=np.linalg.norm(targets-pd[0],axis=1)
    names=[*af.LABELS,*[f'local_{i}' for i in range(128)],*[f'global_{i}' for i in range(128)]]
    def candidate(i):
        return dict(label=names[i],raw=raw[i].tolist(),pd=targets[i].tolist(),metrics=dict(zip(METRICS,metrics[i].tolist())),
            raw_l2=float(raw_dist[i]),pd_l2_rad=float(pd_dist[i]))
    valid=np.flatnonzero(passed)
    finite_ids=np.flatnonzero(finite)
    other=finite & (metrics[:,1:]<=GATES[1:]).all(axis=1)
    best_object=np.flatnonzero(other)
    data=out/f'{clip}_phase{phase}.npz'
    np.savez_compressed(data,raw=raw,pd=targets,metrics=metrics,finite=finite,all_gates=passed,
        raw_l2=raw_dist,pd_l2=pd_dist,labels=np.array(names))
    return dict(clip=clip,phase=phase,source_case=case['native_case'],source_inputs=inputs,source_decisions=decisions,
        parity=dict(logged_count=len(log_errors),max_logged_forecast_gap_mm=max(log_errors),
            six_max_independent_forecast_gap_mm=float(abs(old[:,:2]-original).max()),applied_pd_error_rad=0.),
        ticks=dict(independent_original_six=old_ticks,forecast_parity_six=8*6,expanded=extra_ticks),
        tried=len(raw),distinct_raw=len(np.unique(raw,axis=0)),distinct_pd=len(np.unique(targets,axis=0)),
        passing=int(passed.sum()),distinct_passing_pd=len(np.unique(targets[passed],axis=0)),
        passing_by_group=dict(original=int(passed[:6].sum()),local=int(passed[6:134].sum()),global_raw=int(passed[134:].sum())),
        closest_raw=candidate(valid[np.argmin(raw_dist[valid])]) if valid.size else None,
        closest_pd=candidate(valid[np.argmin(pd_dist[valid])]) if valid.size else None,
        min_object_mm=float(metrics[finite_ids,0].min()),min_symmetric_geometry_ratio=float(clearance[finite_ids].min()),
        min_object_mm_with_other_gates=float(metrics[best_object,0].min()) if best_object.size else None,
        candidates=file_pin(data))


def run(out):
    begin=time.perf_counter()
    source=file_pin(ROOT/'runs/pen_online_contact_filter_v1/report.json')
    old=read(source)
    intent=read(old['intent'])
    model_pin=intent['anchors']['model']
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(model_pin)))
    filter_model=af.PenContactActionFilter(model,'cpu_dynamics')
    out.mkdir(parents=True,exist_ok=False)
    unit=qmc.Sobol(20,scramble=True,seed=SEED).random_base2(8)
    design=out/'sobol.npz'
    np.savez_compressed(design,unit=unit)
    write_json(out/'intent.json',dict(source=source,code=[file_pin(__file__),file_pin(shooting.__file__),file_pin(af.__file__)],
        model=model_pin,phases=PHASES,source_arm='cpu_dynamics',rep=0,design=file_pin(design),seed=SEED,
        metrics=METRICS,limits=GATES.tolist(),local_half_width=.35,groups=[6,128,128],ticks_per_candidate=8,
        protocol='Replay the six conventional RAW commands and validate the transmitted PD, original logged forecasts and a separate six-candidate replay first. Apply frozen Sobol samples 0:128 in an actor-centered RAW box clipped to [-1,1] and 128:256 over the full RAW cube. Every candidate holds the genuine bounded PD target over one 20ms control from the same recorded free-object state and warm start. Four per-tick physical gates and two native endpoint task gates. Rank feasible actions by RAW L2 and by applied PD L2 separately, with candidate order breaking ties. No original W1 episode is continued.'))
    rows=[]
    for clip,phases in PHASES.items():
        case=old['clips'][clip]['cases']['cpu_dynamics'][0]
        if case['rep']!=0:
            raise ValueError('Expected the first numerical replicate')
        for p in phases:
            state_start=time.perf_counter()
            row=assess(out,clip,p,unit,model,filter_model,case)
            row['wall_s']=time.perf_counter()-state_start
            rows.append(row)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=rows,
        unique_candidate_ticks=sum(r['ticks']['independent_original_six']+r['ticks']['expanded'] for r in rows),
        parity_ticks=sum(r['ticks']['forecast_parity_six'] for r in rows),
        total_cpu_ticks=sum(sum(r['ticks'].values()) for r in rows),wall_s=time.perf_counter()-begin,
        scope='Finite one-control CPU coverage on four previously observed states; no closed-loop validation, physical impossibility certificate or training.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    run(p.parse_args().out.resolve())
