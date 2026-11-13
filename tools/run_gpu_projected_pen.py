"""Fixed same-pose GPU contact/LP and moving-reference two-dt comparison."""
from pathlib import Path
import argparse,json,sys,tomllib,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import ArtifactRegistry,file_pin,verify_file_pin
from supermimic_research.runtime.controlled_evaluation import evaluate_case,write
from supermimic_research.runtime.pen_gpu_initialization import initialize_and_query_pen


def main(args):
    import torch,warp as wp
    if args.output.exists():raise ValueError('Fresh GPU projection diagnostic')
    cp=args.config.resolve();config=tomllib.loads(cp.read_text());base_path=(cp.parent/config['base_config']).resolve();base=tomllib.loads(base_path.read_text())
    if config['variants']!=['ORIGINAL_ZERO_RESIDUAL','PROJECTED_CONSTANT_RESIDUAL'] or config['dt']!=[.005,.0025]:raise ValueError('Fixed four-case protocol')
    pp=(cp.parent/config['projection']).resolve();projection=json.loads(pp.read_text());proposal=np.load(verify_file_pin(projection['proposal']));source=np.load(verify_file_pin(projection['case']['trace']))
    wp_path=(cp.parent/config['dynamic_wrench']).resolve();wr=json.loads(wp_path.read_text());target=np.load(verify_file_pin(wr['targets']))['wrench_BODY_COM'][0]
    if not projection['geometry_qualified']:raise ValueError('Original bounded geometry prerequisite')
    torch.set_num_threads(4);wp.set_device('cuda:0');registry=ArtifactRegistry(base_path.parent/base['artifact_manifest'])
    sources=[file_pin(__file__)]+[file_pin(ROOT/p) for p in [
        'src/supermimic_research/runtime/controlled_evaluation.py','src/supermimic_research/runtime/pen_gpu_initialization.py',
        'src/supermimic_research/mechanics/warp_contact_labels.py','src/supermimic_research/mechanics/indexed_contact_support.py',
        'src/supermimic_research/mechanics/support.py','src/supermimic_research/mechanics/wrench_separation.py']]
    args.output.mkdir();intent=dict(schema='controlled_projected_pen_GPU:1',config=file_pin(cp),base_config=file_pin(base_path),projection=file_pin(pp),dynamic_wrench=file_pin(wp_path),sources=sources,neural_updates=0,role='DIAGNOSTIC_ONLY_NOT_EXPERT_NOT_FIT')
    write(args.output/'intent.json',intent);cases=[];start=time.perf_counter()
    for name in config['variants']:
        for dt in config['dt']:
            folder=args.output/f'{name}_{dt}';delta=np.zeros(20) if name=='ORIGINAL_ZERO_RESIDUAL' else proposal['delta20'];qpos=source['initial_qpos'] if name=='ORIGINAL_ZERO_RESIDUAL' else proposal['qpos']
            result=evaluate_case('hand2-pen-spin','CONSTANT_RESIDUAL',None,{**base,'physics_dt':dt},base_path.parent,registry,folder,
                initializer=lambda env:initialize_and_query_pen(env,folder,source['initial_qpos'],qpos,delta,target),constant_action=2*delta)
            result['name']=name;cases.append(result)
            write(folder/'report.json',result);write(args.output/'progress.json',dict(cases=cases,wall_s=time.perf_counter()-start))
            print(json.dumps(dict(name=name,dt=dt,duration=result['summary']['metrics']['actual_duration_s'],qualified=result['summary']['qualified'],static_support=result['initialization']['static_support_with_PD_bounds'],dynamic_cone=result['initialization']['dynamic_wrench_cone_feasible'],contacts=len(result['initialization']['actual_active_support_contacts']))),flush=True)
    for p in sources:verify_file_pin(p)
    used=[]
    for name,module in list(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if name.startswith('dex_supermimic') and path:
            if not Path(path).resolve().is_relative_to(ROOT/'src'):raise ValueError('Unexpected historical source import')
            used.append(file_pin(path))
    write(args.output/'report.json',dict(**intent,cases=cases,wall_s=time.perf_counter()-start,GPU=torch.cuda.get_device_name(),
        task_sources=used,
        scope='Same projected/original pose and source velocity/clock, actual GPU contact and control execution, two dt; static contact queries distinct, no Actor fitting or force assistance.'))


if __name__=='__main__':
    import torch
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    with torch.jit.optimized_execution(False):main(p.parse_args())
