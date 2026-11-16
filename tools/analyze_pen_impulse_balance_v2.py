"""Reanalyze sealed partitions with actual integrator and explicit cache timing."""
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import mujoco
import numpy as np
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.shared_snapshot_transport import load_shared
from supermimic_research.strict_json import write_json


def read(p):return json.loads(Path(p).read_text())


def main():
    started=time.perf_counter();bp=ROOT/'runs/pen_impulse_balance_v1/report.json';base=read(bp);out=ROOT/'runs/pen_impulse_balance_v2'
    if out.exists() or base['passed']!=3 or base['total']!=6:raise ValueError('Fresh explicit correction of preserved V1 diagnostic required')
    # V1 algebra/decoder/gate results were valid;its Euler-only guard and
    # comparison of a post-forward endpoint cache to pre-forward tick8 failed.
    if any(not all(v for k,v in r['checks'].items() if k!='model_conditions') for r in base['rows']):raise ValueError('V1 contains another discrepancy;reuse forbidden')
    if any([k for k,v in d.items() if not v]!=['Euler'] for d in base['model_conditions'].values()):raise ValueError('V1 model failures are not solely the declared integrator assumption')
    verify_source_snapshot(read(verify_file_pin(base['source_receipt'])),live=True);a=dict(np.load(verify_file_pin(base['arrays'])));life=read(verify_file_pin(base['inputs']['lifecycle']));node=read(verify_file_pin(base['inputs']['node']));cases=node['branches']+life['rows']
    for pin in base['inputs']['first3_cases']:
        r=read(verify_file_pin(pin))
        for n in r['nodes']:
            for branch in ['BASE','ALT']:
                if n[branch] is not None:cases.append(read(verify_file_pin(n[branch])))
    if [c['name'] for c in cases]!=[r['name'] for r in base['rows']]:raise ValueError('All676 original row identities/order required')
    first={}
    for c in cases:first.setdefault(c['model']['path'],c)
    models=[]
    for c in first.values():
        m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['model'])));s=load_shared(c['end']) if c['end']['schema'].startswith('owned_native4_shared_') else None;native_integrator=int(s['native']['model']['scalars']['opt.integrator']) if s is not None else None
        models.append(dict(model=c['model'],actual_integrator=int(m.opt.integrator),name=mujoco.mjtIntegrator(m.opt.integrator).name,native_integrator=native_integrator,source_conditions_except_integrator=all(v for k,v in base['model_conditions'][c['model']['path']].items() if k!='Euler')))
    endpoints=[]
    for c in cases:
        if c['name'] not in {x['name'] for x in base['bad_final_tick_endpoints']}:continue
        s=load_shared(c['end']);q=s['native']['data']['arrays'];z=dict(np.load(verify_file_pin(c['trace'])));da=c['model_layout']['object_dofadr'];mass=c['physics']['mass'];g=np.asarray(c['physics']['gravity'],np.float64);acc=q['qacc'][0,da:da+3].astype(np.float64);f=q['qfrc_constraint'][0,da:da+3].astype(np.float64);ma=q['efc.Ma'][0,da:da+3].astype(np.float64);smooth=q['qfrc_smooth'][0,da:da+3].astype(np.float64)
        parts=dict(mass_times_qacc_minus_stored_Ma=mass*acc-ma,inferred_Ma_minus_smooth_constraint=ma-smooth-f,smooth_minus_native_gravity=smooth-mass*g);total=mass*acc-f-mass*g;recomp=sum(parts.values());alignment=dict(qpos=np.array_equal(q['qpos'][0],z['post_qpos'][-1]),qvel=np.array_equal(q['qvel'][0],z['post_qvel'][-1]),qacc=np.array_equal(q['qacc'][0],z['post_qacc'][-1]),constraint=np.array_equal(q['qfrc_constraint'][0],z['post_qfrc_constraint'][-1]))
        endpoints.append(dict(name=c['name'],alignment=alignment,cache_context='AFTER_CONTROL_ENV_FORWARD_NOT_LAST_NATIVE_TICK' if not (alignment['qacc'] and alignment['constraint']) else 'MATCHES_LAST_NATIVE_TICK',endpoint_components_N={k:v.tolist() for k,v in parts.items()},endpoint_acceleration_force_N=total.tolist(),endpoint_recomposition=bool(np.all(abs(recomp-total)<=32*np.finfo(np.float64).eps*(1+abs(total)))),
            qacc_change_max=float(abs(q['qacc'][0]-z['post_qacc'][-1]).max()),constraint_change_max=float(abs(q['qfrc_constraint'][0]-z['post_qfrc_constraint'][-1]).max()),solver_niter=q['solver_niter'].tolist(),limitation='Endpoint inferred residual is not the missing per-tick SolverContext gradient;never substitute this cache for tick8.'))
    failed=[r for r in base['rows'] if not r['original_impulse_pass']];reconstructed=sum(a[k] for k in ['velocity_update','acceleration_force','constraint_contact','gravity_representation']);budget=32*np.finfo(np.float64).eps*(1+sum(abs(a[k]) for k in ['velocity_update','acceleration_force','constraint_contact','gravity_representation']))
    checks=dict(all676_rows=len(cases)==len(base['rows'])==676,all_original_math_and_masks=all(all(v for k,v in r['checks'].items() if k!='model_conditions') for r in base['rows']),all_saved_recompositions=bool(np.all(abs(reconstructed-a['total'])<=budget)),actual_integrators_recorded=len(models)==9 and all(m['actual_integrator']==3 and (m['native_integrator'] is None or m['native_integrator']==3) and m['source_conditions_except_integrator'] for m in models),endpoint_contexts_explicit=all(x['endpoint_recomposition'] and x['alignment']['qpos'] and x['alignment']['qvel'] for x in endpoints),original5_failures_preserved=len(failed)==5 and [r['name'] for r in failed]==base['failed_impulse_rows'])
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__)],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt)
    installed=Path(mujoco.__file__).parent.parent;source_pins=dict(solver=base['inputs']['installed_solver'],environment=file_pin(installed/'mjlab/envs/manager_based_rl_env.py'),simulation=file_pin(installed/'mjlab/sim/sim.py'),data_copy=file_pin(installed/'mujoco_warp/_src/io.py'))
    for pin in source_pins.values():verify_file_pin(pin)
    result=dict(schema='pen_recorded_impulse_partition:2',prior_failed_diagnostic=file_pin(bp),reused_numeric_arrays=base['arrays'],inputs=base['inputs'],source_receipt=file_pin(out/'source_receipt.json'),installed_sources=source_pins,checks=checks,passed=sum(checks.values()),total=len(checks),models=models,endpoints=endpoints,
        failed_peak_rows=[dict(name=r['name'],impulse_Ns=r['peak_impulse_Ns'],components_Ns=r['peak_component_norms_Ns'],signed_projections=r['peak_signed_projections']) for r in failed],signed_acceleration_force_projection_range=[min(r['peak_signed_projections']['acceleration_force'] for r in failed),max(r['peak_signed_projections']['acceleration_force'] for r in failed)],
        actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,wall_s=time.perf_counter()-started,primary_reference=base['primary_reference'],
        scope='Reuses all676 sealed numeric partitions only after every original algebra/decoder/mask subcheck passes. V1 retained3/6,not relabelled as success. Corrects Euler assumption to observed implicitfast3 and separates post-control forward endpoint caches from last native tick. Five original5e-7Ns failures unchanged;dominant acceleration-force term localizes them but does not prove optimizer convergence versus Ma accumulation/other engine arithmetic. Native solver gradient at failing ticks remains uncaptured;full54 producer unchanged.')
    verify_source_snapshot(receipt,live=True);write_json(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],reused_rows=676,impulse_failures=5,projection_range=result['signed_acceleration_force_projection_range'])),flush=True)
    if not all(checks.values()):raise ValueError('Corrected diagnostic discrepancy retained')


if __name__=='__main__':main()
