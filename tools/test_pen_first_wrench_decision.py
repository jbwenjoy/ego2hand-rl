"""Cross the cost used at the first control with the cost used at every later control."""
import argparse
from pathlib import Path
import shutil
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.stats import qmc
import evaluate_pen_receding_support as exp
from evaluate_pen_dev_clip import b,evaluate_cases,file_pin,read,verify_file_pin,write_json
from supermimic_research.mechanics.pen_task_contracts import pose_components


COSTS={'PP':('pose','pose'),'WP':('pose_support','pose'),'PW':('pose','pose_support'),'WW':('pose_support','pose_support')}


def run(out):
    root=b.ROOT/'runs'
    source=file_pin(root/'pen_receding_support_v1/report.json')
    protocol=read(read(source)['intent'])
    for pin in protocol['code']:
        verify_file_pin(pin)
    contract=read(read(protocol['baseline'])['intent'])
    cfg=read(contract['configs'])['clips']['clip_003']
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(contract['anchors']['model'])))
    out.mkdir(exist_ok=False)
    write_json(out/'intent.json',dict(source=source,actor=contract['actor'],code=file_pin(__file__),
        costs=COSTS,seeds=[112025,112026,112027],reps=2,clip='clip_003',
        scope='Post-hoc development factorial. Only switch which of the two already computed ranks is used at control one vs later controls. Same current native observations and phase-indexed pool; from-start numerical repetitions.'))
    with np.load(verify_file_pin(cfg['item']['source'])) as f:
        q=f['reference_qpos'].astype(np.float32)
    summaries=[]
    for seed in (112025,112026,112027):
        folder=out/str(seed)
        folder.mkdir()
        shutil.copyfile(verify_file_pin(cfg['evaluation']),folder/'controlled_evaluation_v4.toml')
        write_json(folder/'intent.json',dict(registration=file_pin(out/'intent.json'),seed=seed))
        sample=qmc.Sobol(20,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(len(q)*8))))
        dirs=(2*sample[:8*len(q)]-1).reshape(-1,8,20)
        with np.load(verify_file_pin(cfg['item']['series'])) as ref,exp.receding_policy(folder,model,contract['anchors'],ref,q,dirs) as cases:
            upstream=exp.online.af.PenContactActionFilter
            class Switch(upstream):
                def __init__(self,model,method,pose_checkpoint=None):
                    self.first,self.later=COSTS[method]
                    super().__init__(model,self.first)
                def select_action(self,**x):
                    self.arm=self.first if self.phase==0 else self.later
                    return super().select_action(**x)
            with patch.object(exp.online.af,'PenContactActionFilter',Switch),patch.object(exp.online,'ARMS',tuple(COSTS)):
                rows=evaluate_cases(folder,verify_file_pin(cfg['objective']),cfg['item'],dict.fromkeys(COSTS,contract['actor']),
                                    dict.fromkeys(COSTS,0.),2,refit_export=True)
        for case,row in zip(cases,rows,strict=True):
            s=exp.online.summarize(folder,row,case,len(q))
            decision=read(s['decisions'])[0]
            assert decision['choice']==str(decision['picks'][COSTS[s['arm']][0]])
            selected=next(x for x in decision['evaluations'] if str(x['id'])==decision['choice'])
            with np.load(verify_file_pin(read(s['native_case'])['trace'])) as tr:
                c=pose_components(tr['post_quaternion_wxyz'][7],tr['quaternion_wxyz'][1])
                first=dict(COM_mm=1000*np.linalg.norm(tr['post_COM'][7]-tr['COM_world_m'][1]),
                    shaft_deg=np.rad2deg(c['shaft_angle_rad']),twist_deg=np.rad2deg(c['reference_axis_twist_rad']),
                    peak_pen_mm=1000*tr['pre_penetration_m'][:8].max(),
                    mean_normal_N=tr['pre_hand_normal_N'][:8].mean())
            summaries.append(dict(seed=seed,case=s,first_choice=int(decision['choice']),first_native=first,first_forecast=selected))
        write_json(out/'progress.json',summaries)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=summaries))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    run(p.parse_args().out.resolve())
