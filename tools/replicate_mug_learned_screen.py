"""Paired additional seeds for comparing eight verified candidates to the full 64."""
import argparse
from pathlib import Path
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from test_mug_learned_planning import physics,file_pin,verify_file_pin,write_json,trial,MugStepHead


def main(out):
    root=Path(__file__).resolve().parents[1]/'runs'
    earlier=file_pin(root/'mug_response_planning_v1/report.json')
    fit=physics.read_json(physics.read_json(physics.read_json(earlier)['intent'])['fit'])
    inputs=physics.read_json(fit['intent'])['inputs']
    frozen=physics.read_json(fit['frozen'])['pins']['shared']
    bundle=torch.load(verify_file_pin(frozen),map_location='cpu',weights_only=True)
    torch.set_num_threads(1)
    net=MugStepHead(bundle['n']);net.load_state_dict(bundle['state']);net.eval()
    path=verify_file_pin(inputs['model']);model=mujoco.MjModel.from_binary_path(str(path))
    binding=physics.read_json(inputs['binding']);old=physics.load_arrays(inputs['reference'])
    seeds=tuple(range(8393,8401))
    out.mkdir(exist_ok=False,parents=True)
    write_json(out/'intent.json',dict(previous=earlier,model=frozen,code=file_pin(__file__),seeds=seeds,
        scope='Follow-up after inspecting seeds8391/8392; parameters and checkpoint frozen. Report every new seed separately. Compare per-clip median paired difference in passing controls (shared8 minus native64; tolerance -1), actual total and per-decision costs, versus a random8 control. Numerical simulator, two known clips, one object.'))
    rows=[]
    for name in ('mug_annotated_windows_original_v1','mug_unilateral_interval_v1'):
        item,=physics.read_json(file_pin(root/name/'report.json'))['rows']
        ref=physics.load_arrays(item['reference']);T=ref['body']
        z={**old,'elapsed_s':ref['t'],'COM_world_m':T[:,:3,3]+T[:,:3,:3]@old['COM_BODY_m'],
            'quaternion_wxyz':Rotation.from_matrix(T[:,:3,:3]).as_quat()[:,[3,0,1,2]]}
        base=old['feedforward_PD_target20'][0];direct=np.clip(ref['q'][1:]+base-z['initial_qpos27'][:20],*model.actuator_ctrlrange.T)
        for seed in seeds:
            for mode in ('native64','shared_verify8','random_verify8'):
                summary,trace=trial(model,path,binding,z,direct,base,seed,mode,{'shared':net},4)
                p=out/f"{item['clip']}_{seed}_{mode}.npz";np.savez_compressed(p,**trace)
                rows.append(dict(clip=item['clip'],seed=seed,mode=mode,trace=file_pin(p),**summary))
            write_json(out/'progress.json',rows)
            print(item['clip'],seed,[r['valid_s'] for r in rows[-3:]],flush=True)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);main(p.parse_args().out.resolve())
