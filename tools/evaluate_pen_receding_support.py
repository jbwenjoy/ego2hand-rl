"""Fixed-budget moving-horizon PD proposals with matched pose and wrench costs."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import shutil
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc
import evaluate_pen_online_contact_filter as online
from evaluate_pen_dev_clip import b,evaluate_cases,file_pin,read,verify_file_pin,write_json
from audit_pen_filter_action_coverage import original_raw,shooting,GATES


ARMS=('pose','pose_support')


@contextmanager
def receding_policy(folder,model,anchors,ref,refq,directions):
    original=online.af.PenContactActionFilter
    rot=Rotation.from_quat(ref['quat_wxyz'][:,[1,2,3,0]])
    omega=(rot[1:]*rot[:-1].inv()).as_rotvec()/.02
    v=np.diff(ref['COM'],axis=0)/.02
    class Controller(original):
        def __init__(self, model, method, pose_checkpoint=None):
            super().__init__(model,'cpu_dynamics')
            self.arm=method
            self.phase=0

        def select_action(self,**x):
            self.phase+=1
            t=self.phase
            horizon=min(2,len(refq)-t)
            actor=x['actor_raw'].clip(-1,1)
            np.testing.assert_array_equal(refq[t],x['refq'])
            pool=original_raw(actor,x['previous_raw'],self.ring)
            perturb=.35*directions[t-1]
            raw=np.vstack([pool,np.clip(actor+perturb,-1,1),np.clip(actor-perturb,-1,1)]).astype(np.float32)
            targets=[]
            ema=x['previous_ema']
            for ref_frame in refq[t:t+horizon]:
                ema=.5*ema+.25*raw
                targets.append(np.clip(ref_frame+ema-x['encoder_bias'],x['lower'],x['upper']))
            d=self.data
            mujoco.mj_resetData(model,d)
            d.qpos[:],d.qvel[:],d.qacc_warmstart[:]=x['qpos'],x['qvel'],x['warmstart']
            d.mocap_pos[:],d.mocap_quat[:]=x['mocap_pos'],x['mocap_quat']
            d.time=(t-1)*.02
            state=np.empty(mujoco.mj_stateSize(model,shooting.STATE))
            mujoco.mj_getState(model,d,state,shooting.STATE)
            estimates=shooting.score_candidates(model,state,np.stack(targets,axis=1),ref['COM'][t:t+horizon],
                ref['quat_wxyz'][t:t+horizon],body=self.body,qa=self.qa,da=self.da,ref_v=v[t-1:t-1+horizon],
                ref_omega=omega[t-1:t-1+horizon])
            rank={arm:[] for arm in ARMS}
            entries=[]
            for j,row in enumerate(estimates):
                if not row['finite']:
                    continue
                ratios=np.asarray([[1000*i['max_object_penetration_m'],1000*i['max_self_penetration_m'],
                    i['max_joint_excursion_rad'],i['max_nonhand_normal_N'],i['mm'],i['shaft_deg']] for i in row['intervals']])/GATES
                worst=float(ratios.max())
                first=float(ratios[0].max())
                stage_cost=row['pose_cost']+.05*float(np.mean((raw[j]-actor)**2))
                entries.append(dict(id=j,peak=worst,first=first,pose=row['pose_cost'],support=row['support_cost']))
                for arm in ARMS:
                    rank[arm].append((max(1.,first),max(1.,worst),stage_cost+(.2*row['support_cost'] if arm=='pose_support' else 0.),j))
            if not entries:
                raise ValueError('No finite forecast')
            pick={a:min(rank[a])[-1] for a in ARMS}
            k=pick[self.arm]
            return raw[k],dict(choice=str(k),triggered=k!=0,pd=targets[0][k].tolist(),planning_ticks=sum(e['native_steps'] for e in estimates),
                predicted_poses=0,evaluations=entries,picks=pick,feasible=sum(e['peak']<=1 for e in entries))
    with patch.object(online,'ARMS',ARMS),patch.object(online.af,'PenContactActionFilter',Controller):
        with online.filter_policy(folder,model,anchors,ref,None) as cases:
            yield cases


def run(out):
    source=file_pin(b.ROOT/'runs/pen_online_contact_filter_v1/report.json')
    contract=read(read(source)['intent'])
    configs=read(contract['configs'])['clips']
    anchors=contract['anchors']
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(anchors['model'])))
    out.mkdir(exist_ok=False)
    write_json(out/'intent.json',dict(baseline=source,code=[file_pin(__file__),file_pin(shooting.__file__)],seed=112025,
        horizon=2,rolling=True,candidates=22,repetitions=3,support_weight=.2,
        scope='Same phase-indexed random directions, CPU physics, actor, and per-decision budget. Recompute every control from the current native state. Proposals hold RAW constant over two forecast controls. Lexicographic first-window then full-window worst normalized six-gate excess, then pose and action cost with optional bounded contact wrench mismatch. Only first action executed. Development clips 002 and 003.'))
    results={}
    for clip in ('clip_002','clip_003'):
        cfg=configs[clip]
        folder=out/clip
        folder.mkdir()
        shutil.copyfile(verify_file_pin(cfg['evaluation']),folder/'controlled_evaluation_v4.toml')
        write_json(folder/'intent.json',dict(master=file_pin(out/'intent.json'),actor=contract['actor']))
        with np.load(verify_file_pin(cfg['item']['source'])) as f:
            q=f['reference_qpos'].astype(np.float32)
        sample=qmc.Sobol(20,scramble=True,seed=112025).random_base2(int(np.ceil(np.log2(len(q)*8))))
        dirs=(2*sample[:8*len(q)]-1).reshape(-1,8,20)
        with np.load(verify_file_pin(cfg['item']['series'])) as ref,receding_policy(folder,model,anchors,ref,q,dirs) as cases:
            rows=evaluate_cases(folder,verify_file_pin(cfg['objective']),cfg['item'],dict.fromkeys(ARMS,contract['actor']),
                                dict.fromkeys(ARMS,0.),3,refit_export=True)
        results[clip]=[]
        for r,c in zip(rows,cases,strict=True):
            s=online.summarize(folder,r,c,cfg['item']['frames'])
            decisions=read(s['decisions'])
            s['ranking_disagreements']=sum(d['picks']['pose']!=d['picks']['pose_support'] for d in decisions)
            results[clip].append(s)
        write_json(out/'progress.json',results)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),clips=results))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    run(p.parse_args().out.resolve())
