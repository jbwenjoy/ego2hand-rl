"""Compare frozen and state-conditioned second commands over the same 32 actions."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import shutil
from unittest.mock import patch

import mujoco
import numpy as np
from evaluate_pen_two_step import online, b, evaluate_cases, read, file_pin, verify_file_pin, write_json
from audit_pen_filter_action_coverage import GATES, METRICS, simulate, transmitted


ARMS = ('frozen','live')


def order(second, finite, first_metrics, action_cost):
    scaled = second/GATES
    peak = np.maximum(scaled.max(1), (first_metrics/GATES).max())
    reference = .5*((scaled[:,4:]**2).mean(1) + ((first_metrics[4:]/GATES[4:])**2).mean())
    score = peak + .25*reference + .05*action_cost
    idx = np.flatnonzero(finite & np.isfinite(second).all(1))
    if not len(idx):
        raise ValueError('No finite candidate')
    return min(idx,key=lambda j:(max(1.,peak[j]),score[j],peak[j],j))


@contextmanager
def policy(folder, model, anchors, reference, winner, pool, first_metrics, cost):
    original = online.af.PenContactActionFilter
    fixed = np.asarray(winner['raw_sequence'],np.float32)
    class SecondControl(original):
        def __init__(self, model, method, pose_checkpoint=None):
            super().__init__(model,'cpu_dynamics')
            self.arm=method
            self.phase=0

        def select_action(self,**x):
            self.phase+=1
            step=self.phase-winner['phase']
            if step not in (0,1):
                return super().select_action(**x)
            if step==0:
                raw=fixed[0]
                return raw,dict(choice='first',triggered=False,pd=transmitted(x,raw).tolist(),
                                evaluations={},planning_ticks=0,predicted_poses=0)
            x['target_pose']=np.r_[reference['COM'][self.phase],reference['quat_wxyz'][self.phase]]
            pd=transmitted(x,pool)
            metrics,finite,ticks=simulate(model,x,self.phase,pd,self)
            selected=int(order(metrics,finite,first_metrics,cost))
            used=selected if self.arm=='live' else winner['second_id']
            return pool[used],dict(choice=f'second_{used}',triggered=True,pd=pd[used].tolist(),
                live_id=selected,used=used,frozen_id=winner['second_id'],scores=metrics.tolist(),
                feasible=int((finite & (metrics<=GATES).all(1)).sum()),evaluations={},
                planning_ticks=ticks,predicted_poses=0)
    with patch.object(online,'ARMS',ARMS),patch.object(online.af,'PenContactActionFilter',SecondControl):
        with online.filter_policy(folder,model,anchors,reference,None) as cases:
            yield cases


def main(out):
    runs=b.ROOT/'runs'
    selected=file_pin(runs/'pen_two_step_recovery_v1/search/selection.json')
    search=read(selected)
    registration=read(search['registration'])
    baseline=file_pin(runs/'pen_online_contact_filter_v1/report.json')
    contract=read(read(baseline)['intent'])
    configs=read(contract['configs'])['clips']
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(contract['anchors']['model'])))
    out.mkdir(exist_ok=False,parents=True)
    write_json(out/'intent.json',dict(selection=selected,baseline=baseline,code=file_pin(__file__),
        scope='Fresh alternating arms x 3. Identical first RAW, second-stage pool, rank and score constants. Both arms evaluate all 32 second actions; only live uses measurements to select. Examine reach, paired controls passed, time after handback, halfway and complete/strict from reference start.'))
    results={}
    for clip,w in search['clips'].items():
        cfg=configs[clip]
        parent=w['first_id']
        with np.load(verify_file_pin(w['branches'])) as branches:
            row,=np.flatnonzero(branches['first_ids']==parent)
            pool=branches['second_raw'][row]
            old=branches['second_metrics'][row]
        frozen=np.asarray(w['raw_sequence'],np.float32)
        np.testing.assert_array_equal(pool[w['second_id']],frozen[1])
        with np.load(verify_file_pin(registration['clips'][clip]['live'])) as z:
            actor=z['actor_raw'][w['phase']-1].clip(-1,1)
        cost=.5*(np.mean((pool-actor)**2,axis=1)+np.mean((frozen[0]-actor)**2))
        first=np.array([w['predicted'][0][k] for k in METRICS])
        assert order(old,np.isfinite(old).all(1),first,cost)==w['second_id']
        folder=out/clip
        folder.mkdir()
        shutil.copyfile(verify_file_pin(cfg['evaluation']),folder/'controlled_evaluation_v4.toml')
        write_json(folder/'intent.json',dict(master=file_pin(out/'intent.json'),phase=w['phase']))
        with np.load(verify_file_pin(cfg['item']['series'])) as ref, policy(folder,model,contract['anchors'],ref,w,pool,first,cost) as cases:
            rows=evaluate_cases(folder,verify_file_pin(cfg['objective']),cfg['item'],
                dict.fromkeys(ARMS,contract['actor']),dict.fromkeys(ARMS,0.),3,refit_export=True)
        stats=[]
        for c,r in zip(cases,rows,strict=True):
            s=online.summarize(folder,r,c,cfg['item']['frames'])
            dec=read(s['decisions'])
            second=dec[w['phase']] if len(dec)>w['phase'] else None
            stats.append(dict(arm=s['arm'],rep=s['rep'],passing=s['passing'],passing_s=s['passing_s'],
                reached_second=second is not None,passed_second=s['passing']>=w['phase']+1,
                post_controls=max(0,s['passing']-w['phase']-1),second=second,
                passed_halfway=s['passed_halfway'],complete=s['complete'],strict=s['strict'],stop=s['stop'],
                max_pd_error_rad=s['max_pd_error_rad'],case=s['native_case'],decisions=s['decisions']))
        results[clip]=stats
        write_json(out/'progress.json',results)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),clips=results))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    main(p.parse_args().out.resolve())
