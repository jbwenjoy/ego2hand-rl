"""Fresh W1 replay of two preselected RAW actions inside the original CPU filter policy."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import evaluate_pen_online_contact_filter as online
from evaluate_pen_dev_clip import b,evaluate_cases,file_pin,read,verify_file_pin,write_json


STARTS={'clip_002':8,'clip_003':50}
METRICS=('object_mm','self_mm','joint_excursion_rad','nonhand_N','COM_mm','shaft_deg')


@contextmanager
def scheduled_policy(folder,model,anchors,reference,phase,raw_sequence):
    original=online.af.PenContactActionFilter
    class ScheduledCPUFilter(original):
        """The policy owns only a decision counter; the live environment owns every state."""
        count=0
        def select_action(self,**x):
            self.count+=1
            i=self.count-phase
            if i not in (0,1):
                return super().select_action(**x)
            raw=raw_sequence[i].copy()
            pd=np.clip(x['refq']+(.5*x['previous_ema']+.25*raw)-x['encoder_bias'],x['lower'],x['upper'])
            # Shadow forecast of the fixed action for auditing local model error.
            # It does not participate in selection or change the live state.
            peak=self._peaks(pd[None],x['qpos'],x['qvel'],x['warmstart'],
                (x['mocap_pos'],x['mocap_quat']),None)[0]
            name=f'frozen_step_{i+1}'
            return raw,dict(choice=name,triggered=False,pd=pd.tolist(),planning_ticks=8,predicted_poses=0,
                evaluations={name:dict(object_mm=float(peak[0]),self_mm=float(peak[1]))})
    with patch.object(online,'ARMS',('cpu_dynamics',)), patch.object(online.af,'PenContactActionFilter',ScheduledCPUFilter):
        with online.filter_policy(folder,model,anchors,reference,None) as cases:
            yield cases


def case_details(case,selected,baseline):
    p=selected['phase']
    native=read(case['native_case'])
    details=read(case['decisions'])
    forced=[]
    with np.load(verify_file_pin(case['inputs'])) as z, np.load(verify_file_pin(native['trace'])) as tr:
        for i,step in enumerate((p,p+1)):
            if step>case['controls']:
                forced.append(dict(phase=step,executed=False,passed=False))
                continue
            d=details[step-1]
            if d['choice']!=f'frozen_step_{i+1}':
                raise ValueError('The frozen control was not delivered at its registered phase')
            np.testing.assert_array_equal(z['selected_raw'][step-1],np.asarray(selected['raw_sequence'][i],dtype=np.float32))
            tick=8*step-1
            tick_slice=slice(tick-7,tick+1)
            target=z['target_pose'][step-1]
            q=tr['post_quaternion_wxyz'][tick]
            shaft=(Rotation.from_quat(np.roll(target[3:],-1)).inv()*Rotation.from_quat(np.roll(q,-1))).apply([1.,0.,0.])
            metrics=dict(object_mm=d['actual_object_mm'],self_mm=d['actual_self_mm'],
                joint_excursion_rad=float(tr['post_joint_limit_excursion_rad'][tick_slice].max()),
                nonhand_N=float(tr['pre_object_nonhand_normal_N'][tick_slice].max()),
                COM_mm=float(1000*np.linalg.norm(tr['post_COM'][tick]-target[:3])),
                shaft_deg=float(np.rad2deg(np.arctan2(np.hypot(shaft[1],shaft[2]),shaft[0]))))
            pred=selected['predicted'][i]
            diff={k:metrics[k]-pred[k] for k in METRICS}
            forced.append(dict(phase=step,executed=True,passed=step<=case['passing'],
                actual=metrics,frozen_cpu_prediction=pred,w1_minus_frozen_cpu=diff,
                live_shadow_prediction=d['evaluations'][d['choice']],max_pd_error_rad=d['max_pd_error_rad']))
        entry=None
        if case['controls']>=p:
            with np.load(verify_file_pin(baseline['inputs'])) as old:
                entry={k:float(np.max(abs(z[k][p-1]-old[k][p-1]))) for k in
                       ('qpos','qvel','previous_raw','previous_ema','warmstart','refq','target_pose','mocap_pos','mocap_quat')}
        for j,d in enumerate(details,1):
            if j not in (p,p+1) and d['choice'] not in online.af.LABELS:
                raise ValueError('Control outside the intervention did not use the original action pool')
    case.update(forced=forced,passed_both=case['passing']>=p+1,
        start_gap_vs_selected_source_max_abs=entry,policy_resumed=case['controls']>=p+2,
        passing_controls_after_resume=max(0,case['passing']-p-1),physical_and_task_s_after_resume=.02*max(0,case['passing']-p-1))
    return case


def run(selection,out):
    start=time.perf_counter()
    root=b.ROOT/'runs'
    selected_pin=file_pin(selection)
    winners=read(selected_pin)['clips']
    previous=file_pin(root/'pen_online_contact_filter_v1/report.json')
    old=read(previous)
    registered=read(old['intent'])
    for pin in registered['code']:
        verify_file_pin(pin)
    actor=registered['actor']
    anchors=registered['anchors']
    cfgs=read(registered['configs'])['clips']
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(anchors['model'])))
    if set(winners)!=set(STARTS):
        raise ValueError('Need sealed results on both registered clips before W1')
    for clip,p in STARTS.items():
        winner=winners[clip]
        if winner['phase']!=p:
            raise ValueError('Wrong intervention phase')
        if winner['raw_sequence'] is not None:
            raw=np.asarray(winner['raw_sequence'])
            if raw.shape!=(2,20) or not np.isfinite(raw).all() or np.any(abs(raw)>1):
                raise ValueError('Invalid frozen two-control RAW sequence')
            if len(winner['predicted'])!=2 or any(set(METRICS)-set(pred) for pred in winner['predicted']):
                raise ValueError('Missing six per-step CPU predictions')
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'intent.json',dict(cpu_selection=selected_pin,baseline=previous,actor=actor,anchors=anchors,
        starts=STARTS,repeats=3,code=[file_pin(__file__),*registered['code']],
        scope='Development-phase intervention after failure localization. Fresh native W1 starts each clip at phase one, then runs the original CPU filter policy except for two frozen RAW commands. Shadow forecasts of the frozen actions are recorded only. No reset or state replacement at intervention. Skip a clip when its preregistered search has no admissible pair.'))
    results={}
    for clip,p in STARTS.items():
        cfg=cfgs[clip]
        winner=winners[clip]
        prior=old['clips'][clip]['cases']['cpu_dynamics']
        baseline=[{k:r[k] for k in ('rep','passing','passing_s','stop','reasons','passed_halfway','complete','strict','native_case')} for r in prior]
        if winner['raw_sequence'] is None:
            results[clip]=dict(status='no_CPU_feasible_pair',baseline=baseline,cases=[])
            continue
        raw=np.asarray(winner['raw_sequence'],dtype=np.float32)
        folder=out/clip
        folder.mkdir()
        shutil.copyfile(verify_file_pin(cfg['evaluation']),folder/'controlled_evaluation_v4.toml')
        write_json(folder/'intent.json',dict(master=file_pin(out/'intent.json'),reference=cfg['item']['series'],
            projected=cfg['projected'],control=cfg['control'],phase=p))
        with np.load(verify_file_pin(cfg['item']['series'])) as ref, scheduled_policy(folder,model,anchors,ref,p,raw) as cases:
            rows=evaluate_cases(folder,verify_file_pin(cfg['objective']),cfg['item'],{'cpu_dynamics':actor},
                {'cpu_dynamics':0.},3,refit_export=True)
        summaries=[case_details(online.summarize(folder,r,c,cfg['item']['frames']),winner,prior[0])
                   for r,c in zip(rows,cases,strict=True)]
        results[clip]=dict(status='W1_observed',baseline=baseline,cases=summaries,
            passing_median_difference_controls=float(np.median([c['passing'] for c in summaries])-np.median([c['passing'] for c in prior])),
            both_count=sum(c['passed_both'] for c in summaries),complete_count=sum(c['complete'] for c in summaries),
            strict_count=sum(c['strict'] for c in summaries),native_report=file_pin(folder/'report.json'))
        write_json(out/'progress.json',dict(clips=results))
    completed=[r for clip in results.values() for r in clip['cases']]
    injected=sum(sum(f['executed'] for f in r['forced']) for r in completed)
    total=sum(r['cpu_planning_ticks'] for r in completed)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),clips=results,native_ticks=8*sum(r['controls'] for r in completed),
        cpu_filter_ticks_outside_intervention=total-injected*8,shadow_cpu_ticks=injected*8,wall_s=time.perf_counter()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    run(args.selection.resolve(),args.out.resolve())
