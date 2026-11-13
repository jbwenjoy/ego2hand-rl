"""Fresh predeclared TRAIN/CHECK physical episodes for pen WM dynamics only."""
from pathlib import Path
import argparse,json,sys,tomllib
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import ArtifactRegistry,file_pin,verify_file_pin
from supermimic_research.runtime.controlled_evaluation import evaluate_case,write
from supermimic_research.runtime.pen_gpu_initialization import initialize_and_query_pen


def main(args):
    import torch,warp as wp
    if args.output.exists():raise ValueError('Fresh actual WM collection')
    cp=args.config.resolve();cfg=tomllib.loads(cp.read_text());bp=(cp.parent/cfg['base_config']).resolve();base=tomllib.loads(bp.read_text())
    pp=(cp.parent/cfg['proposal']).resolve();p=json.loads(pp.read_text());s=np.load(verify_file_pin(p['artifacts']['selected']))
    if set(cfg['train_action_seeds'])&set(cfg['check_action_seeds']):raise ValueError('Independent episode action streams')
    torch.set_num_threads(4);wp.set_device('cuda:0');registry=ArtifactRegistry(bp.parent/base['artifact_manifest']);args.output.mkdir()
    sources=[file_pin(__file__)]+[file_pin(ROOT/q) for q in ['src/supermimic_research/runtime/controlled_evaluation.py','src/supermimic_research/runtime/pen_gpu_initialization.py','src/supermimic_research/runtime/numerical_options.py','src/supermimic_research/mechanics/warp_contact_labels.py']]
    intent=dict(schema='controlled_pen_WM_collection:1',config=file_pin(cp),base_config=file_pin(bp),proposal=file_pin(pp),sources=sources,
                neural_updates=0,role='FRESH_WM_DYNAMICS_TRAIN_CHECK_NONEXPERT',environment_seed=base['seed'])
    write(args.output/'intent.json',intent);cases=[]
    for role,seeds in [('TRAIN',cfg['train_action_seeds']),('CHECK',cfg['check_action_seeds'])]:
        for seed in seeds:
            generator=np.random.default_rng(seed);noise=np.zeros(20);plan=[]
            for _ in range(368):
                noise=cfg['noise_memory']*noise+(1-cfg['noise_memory'])*generator.normal(0,cfg['raw_noise_std'],20)
                plan.append(2*s['delta20']+np.clip(noise,-cfg['raw_noise_clip'],cfg['raw_noise_clip']))
            plan=np.array(plan,dtype=np.float32)
            if not np.isfinite(plan).all() or np.max(abs(plan))>1:raise ValueError('Raw legal action plans')
            for dt in cfg['dt']:
                folder=args.output/f'{role}_{seed}_{dt}';actions=torch.as_tensor(plan,device='cuda')
                physics={}
                def initialize(env):
                    receipt=initialize_and_query_pen(env,folder,s['source_qpos'],s['qpos'],s['delta20'],s['target_BODY'],support_all_robot_contacts=True)
                    m=env.sim.mj_model;b=m.body('object/cube').id
                    from scipy.spatial.transform import Rotation
                    I=env.sim.wp_model.body_inertia.numpy()[0,b];q=env.sim.wp_model.body_iquat.numpy()[0,b];R=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
                    physics.update(mass=float(env.sim.wp_model.body_mass.numpy()[0,b]),inertia_BODY=(R@np.diag(I)@R.T).tolist(),
                                   gravity=env.sim.wp_model.opt.gravity.numpy()[0].tolist())
                    return receipt
                c=evaluate_case('hand2-pen-spin','WM_EXPLORATION',None,{**base,'physics_dt':dt,'numerical_options':cfg['numerical_options'],'numerical_device_override':True},bp.parent,registry,folder,
                    initializer=initialize,action_provider=lambda env,observation,k:actions[k:k+1])
                np.savez_compressed(folder/'action_plan.npz',raw_actions=plan)
                c.update(name=f'{role}_{seed}',role=role,action_seed=seed,physics=physics,action_plan=file_pin(folder/'action_plan.npz'),supervision='WM_DYNAMICS_NONEXPERT_NOT_IMITATION')
                cases.append(c);write(folder/'report.json',c);write(args.output/'progress.json',dict(cases=cases))
                print(json.dumps(dict(role=role,seed=seed,dt=dt,intervals=c['actual_intervals'],duration=c['summary']['metrics']['actual_duration_s'],qualified=c['summary']['qualified'])),flush=True)
    for pin in sources:verify_file_pin(pin)
    used=[]
    for name,module in list(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if name.startswith('dex_supermimic') and path:
            if not Path(path).resolve().is_relative_to(ROOT/'src'):raise ValueError('Historical source import')
            used.append(file_pin(path))
    write(args.output/'report.json',dict(**intent,cases=cases,task_sources=used,
        scope='Fresh actual original-clock selected-initial-pose physical dynamics under independent predeclared role/action streams. Failures remain nonexpert; no Actor fitting or ego generalization.'))


if __name__=='__main__':
    import torch
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    with torch.jit.optimized_execution(False):main(p.parse_args())
