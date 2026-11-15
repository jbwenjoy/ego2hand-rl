"""Independent saved-receipt first20ms physical response/WM derivative check."""
import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
import mujoco
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.warp_saved_review import review_saved_warp_contacts
from supermimic_research.mechanics.controlled_pen_records import pack_case
from supermimic_research.learning.pen_interventions import intervention_profiles,first_raw_action
from supermimic_research.learning.pen_response import PenResponseWM
from supermimic_research.learning.pen_native_response import compare_native_response,vector_comparison
from supermimic_research.learning.pen_stage import write


def main(a):
    r=json.loads(a.report.read_text()); directions,profiles=intervention_profiles()
    expected={(p['name'],dt) for p in profiles for dt in [.005,.0025]}
    if ({(c['name'],c['dt']) for c in r['cases']}!=expected or len(r['cases'])!=54
            or a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'runs')):
        raise ValueError('Exact predeclared54 cases and fresh unified output')
    for pin in r['sources']+r['task_sources']+[r['config'],r['base_config'],r['checkpoint_health']]: verify_file_pin(pin)
    proposal=json.loads(verify_file_pin(r['proposal']).read_text()); selected=np.load(verify_file_pin(proposal['artifacts']['selected']))
    a.output.mkdir()
    try:
        with (a.output/'EFC_console.json').open('w',encoding='utf8') as stream, contextlib.redirect_stdout(stream):
            efc=review_saved_warp_contacts(a.report,a.output/'EFC',expected)
    except ValueError:
        # Full-episode quality failures remain failures; first20ms validity below
        # is separately declared, not a relaxation of those full-episode gates.
        if not (a.output/'EFC'/'report.json').exists(): raise
        efc=json.loads((a.output/'EFC'/'report.json').read_text())
    ck=torch.load(verify_file_pin(r['checkpoint']),map_location='cpu',weights_only=False)
    cases=[]; rows=[]; checks={}; contexts=[]
    for case_id,c in enumerate(r['cases']):
        z=np.load(verify_file_pin(c['trace'])); m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])))
        native=mujoco.MjData(m); mujoco.mj_setState(m,native,z['initial_native_state'],mujoco.mjtState.mjSTATE_INTEGRATION)
        e=next(e for e in efc['cases'] if e['name']==c['name'] and e['dt']==c['dt'])
        packet=pack_case(m,c,z,np.load(verify_file_pin(e['wrenches'])),selected['qids'],selected['dofs'],selected['aids'],native.qacc_warmstart,c['physics'])
        direction=np.zeros(20) if c['direction']<0 else directions[c['direction']]
        expected_raw=first_raw_action(selected['delta20'],direction,c['epsilon'],c['sign'])
        local=dict(first_interval_quality=bool(packet['valid'][0]),
            first_action=np.array_equal(z['actions20'][0],expected_raw),
            remaining_raw_baseline=np.array_equal(z['actions20'][1:],np.tile((2*selected['delta20']).astype(np.float32),(len(z['actions20'])-1,1))),
            source_phase=int(z['pre_action_phase'][0])==1,
            correct_nonfit_role=c['role']=='CHECK_PHYSICAL_RESPONSE_NONEXPERT_NEVER_FIT',
            physics_matches_frozen_WM=c['physics']==ck['arms']['DIRECT']['physics'],
            EFC_indices_and_labels=bool(e['checks']['valid_source_indices'] and e['checks']['local_labels_exact']))
        checks.update({f'{case_id}_{k}':bool(v) for k,v in local.items()})
        cases.append(dict(case_id=case_id,name=c['name'],dt=c['dt'],direction=c['direction'],epsilon=c['epsilon'],sign=c['sign'],repeat=c['repeat'],checks=local))
        rows.append({k:packet[k][0] for k in ['x','y','heads','valid']})
        contexts.append(dict(np.load(verify_file_pin(c['GPU_physical_context']))))
    data={k:np.array([row[k] for row in rows]) for k in rows[0]}
    checks['all_physical_tables_same']=all(all(np.array_equal(v,contexts[0][k]) for k,v in t.items()) for t in contexts)
    for dt in [.005,.0025]:
        ids=[i for i,c in enumerate(cases) if c['dt']==dt]; baseline=[i for i in ids if cases[i]['direction']==-1]
        source=data['x'][baseline[0],:81]; target=data['x'][baseline[0],81:101]
        checks[f'{dt}_identical_initial_state_warm']=all(np.array_equal(data['x'][i,:81],source) for i in ids)
        checks[f'{dt}_unclipped_actual_PD_perturbations']=all(np.max(abs(data['x'][i,81:101]-target-(
            np.zeros(20) if cases[i]['direction']<0 else cases[i]['sign']*cases[i]['epsilon']*directions[cases[i]['direction']])))<2e-7 for i in ids)
    np.savez_compressed(a.output/'first20ms.npz',**data)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True); torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False; device=torch.device('cuda:0')
    models={}; comparisons=[]; derivatives={}
    for mode,saved in ck['arms'].items():
        model=PenResponseWM(saved['scales'],saved['head_scales'],saved['physics'],width=saved['width'],coupled=saved['coupled']).to(device)
        model.load_state_dict(saved['model'],strict=True); model.eval(); models[mode]=model
        scale=model.y_scale.detach().cpu().numpy()
        for dt in [.005,.0025]:
            baseline=next(i for i,c in enumerate(cases) if c['dt']==dt and c['direction']==-1)
            x=torch.tensor(data['x'][baseline:baseline+1],device=device,dtype=torch.float32)
            for direction in range(2):
                tangent=torch.zeros_like(x); tangent[:,81:101]=torch.tensor(directions[direction],device=device,dtype=x.dtype)
                _,jvp=torch.autograd.functional.jvp(model.prediction,x,tangent,create_graph=False)
                derivative=jvp[0].detach().cpu().numpy()
                for epsilon in [.002,.005]:
                    chosen=lambda sign:[i for i,c in enumerate(cases) if c['dt']==dt and c['direction']==direction and c['epsilon']==epsilon and c['sign']==sign]
                    plus,minus=data['y'][chosen(1)],data['y'][chosen(-1)]
                    item=dict(mode=mode,dt=dt,direction=direction,epsilon=epsilon)
                    for name,sl in [('object12',slice(40,52)),('wrench6',slice(78,84))]:
                        value,fd=compare_native_response(plus[:,sl],minus[:,sl],derivative[sl],scale[sl],epsilon)
                        item[name]=value; derivatives[(mode,dt,direction,epsilon,name)]=fd
                    comparisons.append(item)
    amplitude=[]
    for dt in [.005,.0025]:
        for direction in range(2):
            row=dict(dt=dt,direction=direction)
            for name in ['object12','wrench6']:
                row[name]=vector_comparison(derivatives[('DIRECT',dt,direction,.002,name)],derivatives[('DIRECT',dt,direction,.005,name)])
            amplitude.append(row)
    result=dict(schema='pen_native_gradient_saved_review:1',run=file_pin(a.report),checkpoint=r['checkpoint'],
        sources=[file_pin(__file__)]+[file_pin(ROOT/p) for p in ['src/supermimic_research/learning/pen_native_response.py','src/supermimic_research/mechanics/controlled_pen_records.py']],
        EFC_review=file_pin(a.output/'EFC'/'report.json'),full_episode_EFC_passed=efc['passed'],full_episode_EFC_total=efc['total'],
        first20ms=file_pin(a.output/'first20ms.npz'),cases=cases,checks=checks,passed=sum(checks.values()),total=len(checks),
        response_comparisons=comparisons,native_amplitude_comparison=amplitude,neural_updates=0,native_integration_steps=0,
        scope='New physical first20ms finite-amplitude response diagnostics ONLY, never TRAIN. Same initial state/clip/seed,2 directions,3 execution repeats. Data/control/quality checks do not certify model derivative accuracy or differentiability. No Actor/task/ego success.')
    write(a.output/'report.json',result)
    print(json.dumps(dict(report=file_pin(a.output/'report.json'),passed=result['passed'],total=result['total'],full_EFC=[efc['passed'],efc['total']],response_comparisons=comparisons,native_amplitude_comparison=amplitude)))
    if not all(checks.values()): raise ValueError('Retain failed response contract; no derivative accuracy claim')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
