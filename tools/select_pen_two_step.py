"""Freeze two candidate controls from one measured state and static reference frames."""
import argparse
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.stats import qmc

from audit_pen_filter_action_coverage import GATES, METRICS, ROOT, read, shooting, transmitted
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.strict_json import write_json


PHASES = {'clip_002':8,'clip_003':50}
SEED = 112024
LIVE = ('qpos','qvel','warmstart','mocap_pos','mocap_quat','actor_raw','previous_raw','previous_ema',
        'refq','target_pose','encoder_bias','lower','upper')


def second_raw(actor, previous, first, unit):
    local_actor = np.clip(actor + .35*unit[4:16], -1, 1)
    local_first = np.clip(first + .35*unit[16:24], -1, 1)
    return np.vstack([first,actor,previous,np.zeros_like(first),local_actor,local_first,unit[24:32]]).astype(np.float32)


def run_clip(cfg, model, unit, out, clip, phase):
    with np.load(verify_file_pin(cfg['live']), allow_pickle=False) as z:
        x = {key:z[key][phase-1] for key in LIVE}
    with np.load(verify_file_pin(cfg['reference_joints']), allow_pickle=False) as z:
        q = z['reference_qpos'][phase:phase+2].astype(np.float32)
    with np.load(verify_file_pin(cfg['reference_object']), allow_pickle=False) as z:
        pose = np.c_[z['COM'][phase:phase+2],z['quat_wxyz'][phase:phase+2]]
    np.testing.assert_array_equal(x['refq'],q[0])
    np.testing.assert_allclose(x['target_pose'][:3],pose[0,:3],atol=1e-6,rtol=0)
    assert abs(np.dot(x['target_pose'][3:],pose[0,3:])) > 1 - 1e-6
    with np.load(verify_file_pin(cfg['first_pool']), allow_pickle=False) as z:
        raw,pd,first,labels = (z[s] for s in ('raw','pd','metrics','labels'))
        valid = z['finite'] & np.isfinite(first).all(1) & (first<=GATES).all(1)
        np.testing.assert_array_equal(valid,z['all_gates'])
    np.testing.assert_array_equal(transmitted(x,raw), pd)
    body = model.body('object/cube').id
    joint = model.body_jntadr[body]
    args = dict(body=body,qa=int(model.jnt_qposadr[joint]),da=int(model.jnt_dofadr[joint]))
    actor,previous = x['actor_raw'].clip(-1,1), x['previous_raw'].clip(-1,1)
    state = np.empty(mujoco.mj_stateSize(model,shooting.STATE))
    pegs = np.flatnonzero(valid)
    scores = np.full((len(pegs),32,6),np.nan)
    raw_ledger = np.empty((len(pegs),32,20),np.float32)
    native_ticks = finite_children = passing_children = parents_with_continuation = 0
    best = None
    for k,parent in enumerate(pegs):
        d = mujoco.MjData(model)
        d.qpos[:],d.qvel[:],d.qacc_warmstart[:] = x['qpos'],x['qvel'],x['warmstart']
        d.mocap_pos[:],d.mocap_quat[:] = x['mocap_pos'],x['mocap_quat']
        d.time = (phase-1)*.02
        d.ctrl[:] = pd[parent]
        mujoco.mj_step(model,d,nstep=8)
        native_ticks += 8
        if np.any(d.warning.number):
            raise ValueError('Unexpected warning on previously feasible first action')
        mujoco.mj_getState(model,d,state,shooting.STATE)
        second = second_raw(actor,previous,raw[parent],unit)
        raw_ledger[k] = second
        ema1 = .5*x['previous_ema'] + .25*raw[parent]
        targets = np.clip(q[1] + (.5*ema1 + .25*second) - x['encoder_bias'], x['lower'],x['upper'])
        outcomes = shooting.score_candidates(model,state,targets[:,None],pose[1:,:3],pose[1:,3:],**args)
        continuation = False
        for j,row in enumerate(outcomes):
            native_ticks += row['native_steps']
            if not row['finite']:
                continue
            interval = row['intervals'][0]
            metrics = np.array([1000*interval['max_object_penetration_m'],1000*interval['max_self_penetration_m'],
                                interval['max_joint_excursion_rad'],interval['max_nonhand_normal_N'],
                                interval['mm'],interval['shaft_deg']])
            scores[k,j] = metrics
            finite_children += 1
            if not (metrics<=GATES).all():
                continue
            passing_children += 1
            continuation = True
            scaled = np.stack([first[parent],metrics])/GATES
            peak = scaled.max()
            ref_cost = np.mean(scaled[:,4:]**2)
            action_cost = np.mean((np.stack([raw[parent],second[j]]) - actor)**2)
            key = (float(peak + .25*ref_cost + .05*action_cost),float(peak),int(parent),j)
            if best is None or key < best[0]:
                best = (key,dict(first_id=int(parent),first_label=str(labels[parent]),second_id=j,
                        raw_sequence=[raw[parent].tolist(),second[j].tolist()],
                        pd_sequence=[pd[parent].tolist(),targets[j].tolist()],
                        predicted=[dict(zip(METRICS,row.tolist())) for row in (first[parent],metrics)],
                        worst_gate_ratio=float(peak),reference_cost=float(ref_cost),raw_cost=float(action_cost),
                        objective=key[0]))
        parents_with_continuation += int(continuation)
    path = out/f'{clip}_branches.npz'
    np.savez_compressed(path,first_ids=pegs,second_raw=raw_ledger,second_metrics=scores)
    selected = best[1] if best is not None else dict(raw_sequence=None,pd_sequence=None,predicted=None,
                                                       first_id=None,second_id=None)
    return dict(phase=phase,**selected,first_candidates=len(raw),first_passing=len(pegs),
                second_attempts=32*len(pegs),second_finite=finite_children,
                both_passing=passing_children,first_with_continuation=parents_with_continuation,
                native_cpu_ticks=native_ticks,branches=file_pin(path))


def run(out):
    r = ROOT/'runs'
    fixed = read(file_pin(r/'pen_prospective_prefix_v1/registration.json'))
    coverage = read(file_pin(r/'pen_filter_action_coverage_v1/intent.json'))
    cfg = {}
    for clip,phase in PHASES.items():
        item = fixed['clips'][clip]['item']
        cfg[clip] = dict(phase=phase,first_pool=file_pin(r/'pen_filter_action_coverage_v1'/f'{clip}_phase{phase}.npz'),
                         live=file_pin(r/'pen_online_contact_filter_v1'/clip/'cpu_dynamics_0/filter_inputs.npz'),
                         reference_joints=item['source'],reference_object=item['series'])
    out.mkdir(parents=True,exist_ok=False)
    reg = out/'registration.json'
    write_json(reg,dict(clips=cfg,code=[file_pin(__file__),file_pin(shooting.__file__),
                   file_pin(ROOT/'tools/audit_pen_filter_action_coverage.py')],model=coverage['model'],
        first='All previously feasible first-step rows in ascending saved ID order. Reuse first metrics, replay each parent once.',
        second='32 per parent. IDs 0..3: repeat parent, held source Actor, held source previous RAW, zero. '
               '4..15 source Actor + .35 Sobol offsets; 16..23 parent RAW + .35 Sobol offsets; 24..31 full RAW cube. '
               'Shared scrambled 32x20 Sobol net, mapped to [-1,1], same points on each parent and clip, clamp RAW.',
        seed=SEED,metrics=METRICS,limits=GATES.tolist(),score='Require both intervals finite, within all six gates. '
            'Minimize worst_gate_ratio across both steps + .25 mean squared COM/shaft gate ratios '
            '+ .05 mean squared deviation of both RAW vectors from held source Actor. Break ties by worst ratio, '
            'first ID, second ID. No winner returns null.',
        state='Only phase-1 index from measured qpos, qvel, warmstart, fixed mount, RAW, Actor mean, EMA and limits. '
              'PD2 uses static next reference and branch EMA; dynamics continue from each parent integration state. '
              'Observed subsequent states or Actor actions are never used.',
        budget='8 CPU ticks per eligible parent plus 8 per second action, up to 32 per parent. '
               'Previously recorded first-screen ticks are reused and not charged as new work.'))
    model = mujoco.MjModel.from_binary_path(str(verify_file_pin(coverage['model'])))
    unit = 2*qmc.Sobol(20,scramble=True,seed=SEED).random_base2(5)-1
    clips = {}
    start=time.monotonic()
    for clip,phase in PHASES.items():
        result = run_clip(cfg[clip],model,unit,out,clip,phase)
        clips[clip] = result
        print(clip,result['first_passing'],result['both_passing'],result['first_id'],result['second_id'],flush=True)
    write_json(out/'selection.json',dict(registration=file_pin(reg),clips=clips,wall_s=time.monotonic()-start,
                                         total_cpu_ticks=sum(row['native_cpu_ticks'] for row in clips.values())))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'runs/pen_two_step_recovery_v1/search')
    run(p.parse_args().out)
