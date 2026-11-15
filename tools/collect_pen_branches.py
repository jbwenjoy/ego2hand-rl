"""Fresh original-clock positive/negative action branches at four source times."""
import argparse
import json
import sys
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import ArtifactRegistry,file_pin,verify_file_pin
from supermimic_research.runtime.controlled_evaluation import evaluate_case,write
from supermimic_research.runtime.pen_gpu_initialization import initialize_and_query_pen
from supermimic_research.learning.pen_interventions import first_raw_action
from supermimic_research.learning.pen_branch_profiles import branch_profiles


def main(a):
    import torch
    import warp as wp
    from scipy.spatial.transform import Rotation
    cp = a.config.resolve();cfg = tomllib.loads(cp.read_text());directions,profiles = branch_profiles(cfg)
    bp = (cp.parent/cfg['base_config']).resolve();base = tomllib.loads(bp.read_text())
    pp = (cp.parent/cfg['proposal']).resolve();proposal = json.loads(pp.read_text())
    selected = np.load(verify_file_pin(proposal['artifacts']['selected']))
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'runs') or not cfg['numerical_device_override']:
        raise ValueError('Fresh unified no-aid branch collection')
    torch.set_num_threads(4);wp.set_device('cuda:0');a.output.mkdir()
    np.savez_compressed(a.output/'directions.npz',**directions)
    sources = [file_pin(__file__)]+[file_pin(ROOT/p) for p in [
        'src/supermimic_research/learning/pen_branch_profiles.py','src/supermimic_research/learning/pen_interventions.py',
        'src/supermimic_research/runtime/controlled_evaluation.py','src/supermimic_research/runtime/pen_gpu_initialization.py',
        'src/supermimic_research/runtime/numerical_options.py','src/supermimic_research/mechanics/warp_contact_labels.py']]
    intent = dict(schema='controlled_pen_source_branch_collection:1',config=file_pin(cp),base_config=file_pin(bp),
        proposal=file_pin(pp),directions=file_pin(a.output/'directions.npz'),declarations=profiles,sources=sources,
        environment_seed=base['seed'],neural_updates=0,
        contract='256 fresh full actual episodes. Only the20ms interval at each declared source anchor may enter new TRAIN/CHECK; common prefixes and later states are NOT extra fitting rows. Same initial pose/clip/seed, unchanged clock/native terminal/no aid. Later-source repeat variation is measured, not reset away. No reuse of previous CHECK/diagnostics as TRAIN.')
    write(a.output/'intent.json',intent);registry = ArtifactRegistry(bp.parent/base['artifact_manifest']);cases=[]
    for dt in cfg['dt']:
        for profile in profiles:
            folder = a.output/f'{profile["name"]}_{dt}'
            raw = first_raw_action(selected['delta20'],directions[profile['role']][profile['direction']],profile['epsilon'],profile['sign'])
            branch = torch.as_tensor(raw,device='cuda')[None]
            baseline = torch.as_tensor((2*selected['delta20']).astype(np.float32),device='cuda')[None]
            physics = {}
            def initialize(env):
                receipt = initialize_and_query_pen(env,folder,selected['source_qpos'],selected['qpos'],selected['delta20'],selected['target_BODY'],support_all_robot_contacts=True)
                body=env.sim.mj_model.body('object/cube').id;m=env.sim.wp_model
                I,q=m.body_inertia.numpy()[0,body],m.body_iquat.numpy()[0,body];R=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
                physics.update(mass=float(m.body_mass.numpy()[0,body]),inertia_BODY=(R@np.diag(I)@R.T).tolist(),gravity=m.opt.gravity.numpy()[0].tolist())
                return receipt
            c = evaluate_case('hand2-pen-spin','SOURCE_BRANCH',None,
                {**base,'physics_dt':dt,'numerical_options':cfg['numerical_options'],'numerical_device_override':True},
                bp.parent,registry,folder,initializer=initialize,
                action_provider=lambda env,obs,k:branch if k==profile['anchor'] else baseline)
            c.update(**profile,physics=physics,supervision='BRANCH_WM_DYNAMICS_NONEXPERT_NOT_IMITATION')
            cases.append(c);write(folder/'report.json',c);write(a.output/'progress.json',dict(cases=cases))
            print(json.dumps(dict(name=c['name'],dt=dt,intervals=c['actual_intervals'],anchor_reached=c['actual_intervals']>profile['anchor'],qualified=c['summary']['qualified'])),flush=True)
    for pin in sources:verify_file_pin(pin)
    task_sources=[]
    for name,module in list(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if name.startswith('dex_supermimic') and path:
            if not Path(path).resolve().is_relative_to(ROOT/'src'):raise ValueError('Historical task import')
            task_sources.append(file_pin(path))
    write(a.output/'report.json',dict(**intent,cases=cases,task_sources=task_sources))


if __name__=='__main__':
    import torch
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    with torch.jit.optimized_execution(False):main(p.parse_args())
