"""Replay each frozen three-action witness and resume the original live H3 policy."""
import argparse
from pathlib import Path
from unittest.mock import patch
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import test_hot3d_model_horizon as h3


def run(out):
    root=Path(__file__).resolve().parents[1]/'runs'
    source=h3.file_pin(root/'mug_future_recoveries_v1/report.json')
    recover=h3.physics.read_json(source)
    baseline=h3.file_pin(root/'mug_model_horizon_trials_v1/report.json');trials=h3.physics.read_json(baseline)
    fit=h3.physics.read_json(h3.physics.read_json(trials['intent'])['fit'])
    inputs=h3.physics.read_json(fit['intent'])['inputs'];old=h3.physics.load_arrays(inputs['reference'])
    bind=h3.physics.read_json(inputs['binding']);path=h3.verify_file_pin(inputs['model']);m=mujoco.MjModel.from_binary_path(str(path))
    ck=torch.load(h3.verify_file_pin(fit['model']),weights_only=True,map_location='cpu')
    torch.set_num_threads(1);net=h3.MugHorizonHead(ck['n']);net.load_state_dict(ck['state']);net.eval()
    out.mkdir(exist_ok=False)
    h3.write_json(out/'intent.json',dict(source=source,baseline=baseline,code=h3.file_pin(__file__),
        scope='Each geometrically validated witness is tested from reference start with normal H3 controller except the registered three actions. Run original policy in same process as a byte-checked baseline. After phase+2 return immediately to H3; no state reset. Development failure times selected retrospectively.'))
    rows=[];original=h3.propose
    for group in ('mug_annotated_windows_original_v1','mug_unilateral_interval_v1'):
        item,=h3.physics.read_json(h3.file_pin(root/group/'report.json'))['rows']
        ref=h3.physics.load_arrays(item['reference']);T=ref['body'];clip=item['clip']
        z={**old,'elapsed_s':ref['t'],'COM_world_m':T[:,:3,3]+T[:,:3,:3]@old['COM_BODY_m'],
           'quaternion_wxyz':Rotation.from_matrix(T[:,:3,:3]).as_quat()[:,[3,0,1,2]]}
        base=z['feedforward_PD_target20'][0];ik=np.clip(ref['q'][1:]+base-old['initial_qpos27'][:20],*m.actuator_ctrlrange.T)
        for w in recover['rows']:
            if w['clip']!=clip or not w.get('branch_validated'):continue
            seed=w['seed'];arc=h3.physics.load_arrays(w['witness'])
            prior=next(r for r in trials['rows'] if r['clip']==clip and r['seed']==seed and r['mode']=='h3')
            prior_trace=h3.physics.load_arrays(prior['trace'])
            for treatment in (False,True):
                phase=0
                def schedule(*args,**kwargs):
                    nonlocal phase
                    phase+=1;options=original(*args,**kwargs)
                    if treatment and w['phase']<=phase<=w['phase']+2:
                        options[:]=arc['ctrl'][phase-1]
                    return options
                with patch.object(h3,'propose',schedule):
                    case,trace=h3.run_case(m,path,bind,z,ik,base,net,seed,'h3')
                if treatment:
                    np.testing.assert_array_equal(trace['qpos'][:len(arc['qpos'])],arc['qpos'])
                else:np.testing.assert_array_equal(trace['qpos'],prior_trace['qpos'])
                f=out/f"{clip}_{seed}_{int(treatment)}.npz";np.savez_compressed(f,**trace)
                rows.append(dict(clip=clip,seed=seed,treatment=treatment,phase=w['phase'],valid_s=case['valid_s'],
                    valid_controls_after_witness=max(0,int(round(case['valid_s']/.02))-w['phase']-2),
                    full=case['full'],first_failure=case['first_failure'],trace=h3.file_pin(f)))
    h3.write_json(out/'report.json',dict(intent=h3.file_pin(out/'intent.json'),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);run(p.parse_args().out.resolve())
