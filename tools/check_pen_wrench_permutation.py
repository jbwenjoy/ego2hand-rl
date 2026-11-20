"""Post-hoc control preserving candidate wrench scores but shifting their assignment."""
import argparse
from pathlib import Path
import shutil
from unittest.mock import patch
import mujoco
import numpy as np
from scipy.stats import qmc
import evaluate_pen_receding_support as experiment
from evaluate_pen_dev_clip import b,evaluate_cases,file_pin,read,verify_file_pin,write_json


def run(out):
    source=file_pin(b.ROOT/'runs/pen_receding_support_v1/report.json')
    reference=read(read(source)['intent'])
    previous=read(read(reference['baseline'])['intent'])
    for pin in reference['code']:
        verify_file_pin(pin)
    configs=read(previous['configs'])['clips']
    anchors=previous['anchors']
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(anchors['model'])))
    out.mkdir(exist_ok=False)
    write_json(out/'intent.json',dict(original=source,code=file_pin(__file__),cyclic_shift=7,repetitions=3,
               scope='Post-hoc matched-cost control. Candidate i receives predicted support cost of (i+7) mod 22; pose, admissibility, and action distance stay attached to i. Same fixed Sobol directions by phase. Internal native policy label pose_support calls the identical scoring path.'))
    scorer=experiment.shooting.score_candidates
    def shuffled(*args,**kwargs):
        rows=scorer(*args,**kwargs)
        if len(rows)!=22 or not all(r['finite'] for r in rows):
            raise ValueError('Expected 22 finite matched support costs')
        costs=[r['support_cost'] for r in rows]
        return [dict(row,support_cost=costs[(i+7)%22]) for i,row in enumerate(rows)]
    results={}
    for clip in ('clip_002','clip_003'):
        cfg=configs[clip]
        folder=out/clip
        folder.mkdir()
        shutil.copyfile(verify_file_pin(cfg['evaluation']),folder/'controlled_evaluation_v4.toml')
        write_json(folder/'intent.json',dict(master=file_pin(out/'intent.json')))
        with np.load(verify_file_pin(cfg['item']['source'])) as f:
            q=f['reference_qpos'].astype(np.float32)
        sample=qmc.Sobol(20,scramble=True,seed=112025).random_base2(int(np.ceil(np.log2(len(q)*8))))
        dirs=(2*sample[:8*len(q)]-1).reshape(-1,8,20)
        # Keep both costs computed by the original ranker. Only instantiate one live arm.
        with np.load(verify_file_pin(cfg['item']['series'])) as ref,patch.object(experiment.shooting,'score_candidates',shuffled),experiment.receding_policy(folder,model,anchors,ref,q,dirs) as cases,patch.object(experiment.online,'ARMS',('pose_support',)):
            rows=evaluate_cases(folder,verify_file_pin(cfg['objective']),cfg['item'],{'pose_support':previous['actor']},
                                {'pose_support':0.},3,refit_export=True)
        results[clip]=[experiment.online.summarize(folder,r,c,cfg['item']['frames']) for r,c in zip(rows,cases,strict=True)]
        write_json(out/'progress.json',results)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),arm='shifted_support',clips=results))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    run(p.parse_args().out.resolve())
