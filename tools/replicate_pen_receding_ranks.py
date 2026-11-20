"""Change only the Sobol candidate seed; compare matched costs and an assignment control."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import shutil
from unittest.mock import patch
import mujoco
import numpy as np
from scipy.stats import qmc
import evaluate_pen_receding_support as experiment
from evaluate_pen_dev_clip import b,evaluate_cases,file_pin,read,verify_file_pin,write_json


def run(out):
    base=file_pin(b.ROOT/'runs/pen_receding_support_v1/report.json')
    original=read(read(base)['intent'])
    for p in original['code']:
        verify_file_pin(p)
    contract=read(read(original['baseline'])['intent'])
    anchors=contract['anchors']
    cfgs=read(contract['configs'])['clips']
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(anchors['model'])))
    scorer=experiment.shooting.score_candidates
    def permuted(*args,**kwargs):
        rows=scorer(*args,**kwargs)
        if len(rows)!=22 or not all(r['finite'] for r in rows):
            raise ValueError('Incomplete candidate score vector')
        return [dict(row,support_cost=rows[(i+7)%22]['support_cost']) for i,row in enumerate(rows)]
    out.mkdir(exist_ok=False)
    seeds=[112026,112027]
    write_json(out/'intent.json',dict(source=base,code=file_pin(__file__),seeds=seeds,clips=list(cfgs),
        scope='Two fresh proposal seeds, one native run per condition and clip. Same fitted actor and known clips. Registered after first seed; compare all 3 conditions on both seeds.'))
    results=[]
    for seed in seeds:
        for clip in ('clip_002','clip_003'):
            cfg=cfgs[clip]
            with np.load(verify_file_pin(cfg['item']['source'])) as f:
                q=f['reference_qpos'].astype(np.float32)
            samples=qmc.Sobol(20,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(8*len(q)))))
            dirs=(2*samples[:8*len(q)]-1).reshape(-1,8,20)
            for mode in ('pose','pose_support','shifted'):
                folder=out/f'{clip}_{seed}_{mode}'
                folder.mkdir()
                shutil.copyfile(verify_file_pin(cfg['evaluation']),folder/'controlled_evaluation_v4.toml')
                write_json(folder/'intent.json',dict(master=file_pin(out/'intent.json'),mode=mode,clip=clip,seed=seed))
                arm='pose_support' if mode=='shifted' else mode
                with np.load(verify_file_pin(cfg['item']['series'])) as ref, (patch.object(experiment.shooting,'score_candidates',permuted) if mode=='shifted' else nullcontext()),experiment.receding_policy(folder,model,anchors,ref,q,dirs) as cases,patch.object(experiment.online,'ARMS',(arm,)):
                    rows=evaluate_cases(folder,verify_file_pin(cfg['objective']),cfg['item'],{arm:contract['actor']},{arm:0.},1,refit_export=True)
                summary=experiment.online.summarize(folder,rows[0],cases[0],cfg['item']['frames'])
                results.append(dict(seed=seed,clip=clip,mode=mode,case=summary))
                write_json(out/'progress.json',results)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=results))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    run(p.parse_args().out.resolve())
