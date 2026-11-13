"""Pack fresh predeclared pen roles, retaining numerical-quality masks."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.controlled_pen_records import pack_case
from supermimic_research.learning.pen_response import fit_pen_scales,INPUT_SCHEMA,OUTPUT_SCHEMA,HEAD_SCHEMA
from supermimic_research.mechanics.impulse_codec import close_velocities


def main(args):
    import torch
    if args.output.exists():raise ValueError('Fresh pen WM preparation')
    r=json.loads(args.report.read_text());efc=json.loads(args.efc_report.read_text())
    if verify_file_pin(efc['run']).resolve()!=args.report.resolve():raise ValueError('Exact actual contact receipt cohort')
    for pin in r['sources']+r['task_sources']:verify_file_pin(pin)
    p=json.loads(verify_file_pin(r['proposal']).read_text());selected=np.load(verify_file_pin(p['artifacts']['selected']))
    roles={'TRAIN':[],'CHECK':[]};physics=r['cases'][0]['physics'];counts=[]
    for caseid,c in enumerate(r['cases']):
        if c['physics']!=physics or not c['numerical_options']['device_override'] or c['numerical_options']['GPU']['tolerance']!=float(np.float32(1e-12)):raise ValueError('One explicit strict device physics profile')
        z=np.load(verify_file_pin(c['trace']));model=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])))
        data=mujoco.MjData(model);mujoco.mj_setState(model,data,z['initial_native_state'],mujoco.mjtState.mjSTATE_INTEGRATION)
        e=next(e for e in efc['cases'] if e['name']==c['name'] and e['dt']==c['dt'])
        if not e['checks']['valid_source_indices'] or not e['checks']['local_labels_exact']:raise ValueError('Actual EFC source decode required')
        w=np.load(verify_file_pin(e['wrenches']))
        packet=pack_case(model,c,z,w,selected['qids'],selected['dofs'],selected['aids'],data.qacc_warmstart.copy(),physics)
        packet['case_id']=np.full(len(packet['x']),caseid,dtype=int)
        roles[c['role']].append(packet);counts.append(dict(case_id=caseid,role=c['role'],seed=c['action_seed'],dt=c['dt'],rows=len(packet['x']),valid=int(packet['valid'].sum()),task_qualified=c['summary']['qualified']))
    packed={role:{k:np.concatenate([v[k] for v in rows]) for k in rows[0]} for role,rows in roles.items()}
    train=packed['TRAIN'];scales,head_scales=fit_pen_scales(train['x'][train['valid']],train['y'][train['valid']],train['heads'][train['valid']])
    reconstructed={};checks={}
    for role,z in packed.items():
        D=lambda value:torch.tensor(value,dtype=torch.float64)
        decoded=close_velocities(D(z['x']),D(z['heads']),D(physics['mass']),D(physics['inertia_BODY']),D(np.linalg.inv(physics['inertia_BODY'])),D(physics['gravity']),D(z['heads'][:,46:52])).numpy()
        error=float(abs(decoded-z['y']).max());reconstructed[role]=error
        checks[role+'_finite']=bool(all(np.isfinite(v).all() for v in z.values()))
        checks[role+'_observed_defect_closure']=error<1e-8
        checks[role+'_nonnegative_Fn6']=bool(np.all(z['y'][:,84:]>=0))
    checks.update(minimum_TRAIN=int(packed['TRAIN']['valid'].sum())>=64,minimum_CHECK=int(packed['CHECK']['valid'].sum())>=16,
                  episode_roles_disjoint=not bool(set(packed['TRAIN']['case_id'])&set(packed['CHECK']['case_id'])))
    args.output.mkdir();outputs={}
    for role,z in packed.items():
        path=args.output/(role+'.npz');np.savez_compressed(path,**z);outputs[role]=file_pin(path)
    np.savez_compressed(args.output/'scales.npz',**scales);np.savez_compressed(args.output/'head_scales.npz',**head_scales)
    sources=[file_pin(__file__)]+[file_pin(ROOT/q) for q in ['src/supermimic_research/mechanics/controlled_pen_records.py','src/supermimic_research/mechanics/transition_records.py','src/supermimic_research/mechanics/response_decode.py','src/supermimic_research/mechanics/impulse_codec.py','src/supermimic_research/learning/pen_response.py','src/supermimic_research/learning/compound_response.py','src/supermimic_research/learning/impulse_response.py']]
    result=dict(schema='pen_response_preparation:1',run=file_pin(args.report),EFC_review=file_pin(args.efc_report),source=file_pin(__file__),sources=sources,outputs=outputs,physics=physics,
        scales=file_pin(args.output/'scales.npz'),head_scales=file_pin(args.output/'head_scales.npz'),input_schema=INPUT_SCHEMA,output_schema=OUTPUT_SCHEMA,head_schema=HEAD_SCHEMA,
        checks=checks,passed=sum(checks.values()),total=len(checks),rows=counts,closure_errors=reconstructed,neural_updates=0,
        scope='Fresh predeclared episode/action roles, observed H20 dynamics and measured source-BODY wrenches/Fn6. Numerical-quality masks retain all failed rows. Valid TRAIN only fits scales; no expert demonstrations, successful policies, real ego validation or physical-gradient correctness claim.')
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(checks=checks,rows={r:[len(z['x']),int(z['valid'].sum())] for r,z in packed.items()},closure_errors=reconstructed)))
    if not all(checks.values()):raise ValueError('Preserve failed preparation; no training')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--efc-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
