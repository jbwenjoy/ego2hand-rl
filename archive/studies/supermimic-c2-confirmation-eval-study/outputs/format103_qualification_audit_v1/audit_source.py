"""Artifact-only CPU audit of the separately authorized single qualification."""
import importlib.util
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
import numpy as np

ROOT=Path('D:/workspace/supermimic-c2-confirmation-eval-study')
spec=importlib.util.spec_from_file_location('analysis',ROOT/'scripts/analyze_confirmation.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
pfile=ROOT/'configs/format103-qualification-v1.json';p=a.read(pfile);state=a.read(Path(p['output_root'])/'state.json')
protocol=a.read(p['source_pins']['prospective_training_protocol']['path']);protocol['task_protocol_sha256']=protocol['unchanged_inputs']['task_protocol_sha256']
job=p['jobs'][0];models={m['model_id']:m for m in p['models']}
import mujoco
model=mujoco.MjModel.from_binary_path(str(Path(models[job['model_id']]['training_run'])/'compiled/nominal_model.mjb'))
mj={**job,'status':'completed','process':state['process'],'health':state['health']}
run=a.load_job(job,mj,p,models,protocol,None,model,a.npz(p['contact_targets']['path']),a.read(p['contact_target_manifest']['path']),a.sha(pfile))
oldp=a.read('D:/workspace/supermimic-c2-020-portfolio-study/configs/c2-portfolio-v1.json')
anchor=Path(oldp['jobs'][0]['output_dir'])/'control_targets_50hz.npz'
prior=a.npz(anchor);current=run['target_arrays']
initial={k:bool(np.array_equal(current[k],prior[k])) for k in ('initial_qpos','initial_qvel','initial_ctrl','u_prev_rad','initial_cached_target_rad')}
result={'status':'passed' if run['row']['status']=='VALID_CAPACITY_ZERO' and all(initial.values()) else 'failed',
 'scope':'One engineering qualification only; excluded from prospective27 and all method-level N summaries',
 'plan_sha256':a.sha(pfile),'analysis_source_sha256':a.sha(ROOT/'scripts/analyze_confirmation.py'),
 'qualification_state_sha256':a.sha(Path(p['output_root'])/'state.json'),'row':run['row'],
 'original_t0_exact':initial,'t0_anchor':{'path':str(anchor),'sha256':a.sha(anchor),'scope':'Initial physical state and control only; no old policy quality or outcome pooled'},
 'cost':{'whole_supervisor_wall_seconds':state['whole_wall_seconds'],'child_process_wall_seconds':state['process']['elapsed_s'],
   'wrapper_wall_seconds':a.read(job['wrapper_metadata_dir']+'/runtime.json')['wall_seconds'],
   'evaluator_report_wall_seconds':run['evaluation']['wall_seconds'],'rollout_report_wall_seconds':run['metadata']['wall_seconds'],
   'GPU_kernel_busy_seconds':None,'cache':'All Warp module loads shown as cached in retained stdout; fresh OS Python/CUDA process',
   'interpretation':'Nested wall scopes, never sum as disjoint GPU compute; qualification is extra engineering cost'},
 'GPU_used_for_audit':False}
a.write(Path(__file__).parent/'report.json',result)
print(result['status']);print(json.dumps({'metrics':run['row']['metrics'],'t0':initial,'cost':result['cost']},indent=2))
raise SystemExit(0 if result['status']=='passed' else 1)
