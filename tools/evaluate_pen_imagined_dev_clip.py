"""Evaluate frozen paired-imagination endpoints on development clip 003."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dex_supermimic.rl.dex_supermimic.networks import GaussianPPOActor
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.strict_json import write_json


def read(pin):
    return json.loads(verify_file_pin(pin).read_bytes())


def check_actors(study):
    source = file_pin(study/'trained.json')
    intent = file_pin(study/'intent.json')
    manifest = read(intent)
    names = ['parent', 'A1_7411', 'A2_7411', 'A2_7412', 'A1_7412', 'A1_7413', 'A2_7413']
    entries = read(source)['actors']
    if [key for key, _ in entries] != names or manifest['seeds'] != [7411,7412,7413] or manifest['updates'] != 24:
        raise ValueError('Expected parent and the six paired 24-update checkpoints')
    actors = dict(entries)
    if actors['parent'] != manifest['pins']['actor']:
        raise ValueError('Warm-start parent differs from training registration')
    base = torch.load(verify_file_pin(actors['parent']), map_location='cpu',weights_only=False)
    config = base['config']
    policy = GaussianPPOActor(161,20,init_std=config['ppo_init_std'],std_range=(config['ppo_std_min'],config['ppo_std_max'])).eval()
    tail = [k for k, module in policy.named_modules() if module is policy.net[-1]][0]
    controls = (tail+'.weight', tail+'.bias')
    rows = {}
    with torch.no_grad():
        for name,pin in entries:
            path = verify_file_pin(pin)
            state = torch.load(path,map_location='cpu',weights_only=False)
            if state['config'] != config or state['batch_backend'] != base['batch_backend'] or state['iteration'] != 2781:
                raise ValueError('Mismatched parent configuration, backend, or PPO iteration')
            policy.load_state_dict(state['actor'],strict=True)
            if any(not torch.isfinite(t).all() for t in policy.state_dict().values()):
                raise ValueError('Nonfinite Actor checkpoint')
            changed = sorted(k for k,v in state['actor'].items() if not torch.equal(v,base['actor'][k]))
            if name == 'parent':
                training = dict(imagined_arm='none',imagined_seed=None,imagined_updates=0)
            else:
                arm,seed = name.split('_')
                receipt_pin=file_pin(study/(name+'.json'))
                receipt=read(receipt_pin)
                if path.parent != study or receipt['actor']!=pin or receipt['label']!=arm or receipt['seed']!=int(seed) or len(receipt['history'])!=24:
                    raise ValueError('Mismatched imagined endpoint receipt')
                if not changed or not set(changed).issubset(controls):
                    raise ValueError('Imagined endpoint must change only permitted mean head')
                for key in controls:
                    indices=[i for i in range(len(state['actor'][key])) if i not in range(4,12)]
                    if not torch.equal(state['actor'][key][indices],base['actor'][key][indices]):
                        raise ValueError('Endpoint modified a row outside the 8 permitted mean actions')
                training=dict(imagined_arm=arm,imagined_seed=int(seed),imagined_updates=24,training_receipt=receipt_pin)
            rows[name]=dict(**training,ppo_parent_iteration=state['iteration'])
    return actors,rows,dict(training=source,training_intent=intent)


def checked_config(folder):
    import evaluate_pen_dev_clip as native
    obj, item = native.config(folder,'clip_003')
    objective = tomllib.loads(obj.read_text())
    adapter = tomllib.loads(Path(objective['adapter_config']).read_text())
    saved_dir = ROOT/'runs/pen_dev_clip3_eval_v1'
    old = tomllib.loads((saved_dir/'pen_full_pose_native_v1.toml').read_text())
    # Output-specific file paths differ. Check the frozen data pins and all physical settings.
    for key in ('clip_pin','fixed_reference_pin','projected_pin','control_pin'):
        verify_file_pin(adapter[key])
        if adapter[key] != old[key]:raise ValueError(f'Changed clip_003 dependency: {key}')
    for key in ('frame_count','physics_dt','numerical_options','collision_numerics','reward'):
        if adapter[key] != old[key]:raise ValueError(f'Changed clip_003 adapter setting: {key}')
    if item['frames']!=373 or adapter['frame_count']!=373 or objective['fixed_frame_features']!='CACHED':
        raise ValueError('Unexpected native construction contract')
    gates=tomllib.loads(Path(adapter['base_config']).read_text())
    orig_gates=tomllib.loads((saved_dir/'controlled_evaluation_v4.toml').read_text())
    if {k:v for k,v in gates.items() if k!='pen_recipe'} != {k:v for k,v in orig_gates.items() if k!='pen_recipe'}:
        raise ValueError('Changed evaluation configuration')
    recipe=json.loads(Path(gates['pen_recipe']).read_bytes())
    if recipe['clips']!=['01_single_axis/clip_003']:raise ValueError('Unexpected evaluation clip')
    with np.load(verify_file_pin(adapter['projected_pin'])) as projected,np.load(verify_file_pin(adapter['control_pin'])) as command:
        if projected['initial_hand_q'].shape!=(20,) or command['ctrl'].shape!=(20,):raise ValueError('Bad initializer shape')
    return obj,item


def execute(study,out,guard_pid):
    actors,labels,inputs=check_actors(study)
    if guard_pid is not None:
        # Read-only query on Windows; never sends the running worker a signal.
        command=f'if (Get-Process -Id {int(guard_pid)} -ErrorAction SilentlyContinue) {{ exit 17 }} else {{ exit 0 }}'
        if subprocess.run(['powershell','-NoProfile','-NonInteractive','-Command',command]).returncode:
            raise RuntimeError('The guarded PID is still live or the OS query failed')
    final=file_pin(study/'report.json')
    terminal=read(final)
    if terminal['training']!=inputs['training'] or terminal['intent']!=inputs['training_intent'] or [r['key'] for r in terminal['evaluated']]!=list(actors):
        raise ValueError('The registered clip_001 study must have a terminal report before executing the new evaluation')
    for row in terminal['evaluated']:
        if row['actor']!=actors[row['key']] or read(row['evaluation'])['total']!=2:
            raise ValueError('Original study has incomplete or inconsistent endpoints')
    out.mkdir(parents=True,exist_ok=False)
    obj,item=checked_config(out)
    weights=dict.fromkeys(actors,0.)
    import evaluate_pen_dev_clip as native
    write_json(out/'intent.json',dict(reference=item['source'],actors=actors,training_labels=labels,environment_support_weights=weights,
        repetitions=2,order=list(actors),terminal_clip001_report=final,**inputs,
        sources=[file_pin(__file__),file_pin(native.__file__)],prior_clip003=file_pin(ROOT/'runs/pen_dev_clip3_eval_v1/report.json'),
        scope='Clip 003 development transfer. Two fresh guarded LIVE W1 numerical repeats per checkpoint, environment support weight zero. Frozen parent and A1/A2 training labels. Same preload, phase 1 start, physical audit and original strict trajectory gates.'))
    rows=native.evaluate_cases(out,obj,item,actors,weights,2,labels)
    if len(rows)!=14 or any(row['environment_support_weight']!=0. or row['training']!=labels[row['arm']] for row in rows):
        raise ValueError('Incorrect paired evaluation matrix')
    endpoints={}
    for key in actors:
        cases=[row for row in rows if row['arm']==key]
        if [r['rep'] for r in cases]!=[0,1]:raise ValueError('Missing repeat')
        endpoints[key]=dict(training=labels[key],environment_support_weight=0.,actor=actors[key],strict_successes=sum(r['strict_success'] for r in cases),
            total=len(cases),valid_s=[r['valid_s'] for r in cases],stop=[r['stop'] for r in cases],cases=[r['case'] for r in cases])
    write_json(out/'comparison.json',dict(evaluation=file_pin(out/'report.json'),endpoints=endpoints))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'runs/pen_paired_imagination_v1')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/pen_paired_imagination_clip003_v1')
    parser.add_argument('--execute',action='store_true',help='Run GPU evaluation after the source study exits; default is CPU preflight only')
    parser.add_argument('--guard-pid',type=int,help='Refuse --execute while the specified Windows process still exists')
    args=parser.parse_args();study=args.study.resolve()
    if args.execute:execute(study,args.output.resolve(),args.guard_pid)
    else:
        torch.set_num_threads(1)
        actors,labels,inputs=check_actors(study)
        with tempfile.TemporaryDirectory() as temp:
            _,clip=checked_config(Path(temp))
        print(json.dumps(dict(checked_actors=list(actors),support_weight=0.,native_cases=14,clip=clip['source'],frames=clip['frames'],native_integrations=0)))
