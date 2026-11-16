"""Locate recorded impulse gaps without replay,threshold edits or solver claims."""
import json
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import mujoco
import numpy as np
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.mechanics.warp_packet_decode import decode_saved_case
from supermimic_research.mechanics.pen_impulse_balance import partition
from supermimic_research.runtime.shared_snapshot_transport import load_shared,SCHEMA
from supermimic_research.runtime.snapshot_transport import load_owned
from supermimic_research.strict_json import write_json


def read(p):return json.loads(Path(p).read_text())


def main():
    started=time.perf_counter();out=ROOT/'runs/pen_impulse_balance_v1';npth=ROOT/'runs/pen_live_node_qualification_v1/report.json';lp=ROOT/'runs/pen_terminal_lifecycle_v1/report.json';regp=ROOT/'runs/pen_native_screened_rollout_registration_v1/registration.json';node,life,reg=[read(p) for p in [npth,lp,regp]]
    if out.exists() or node['passed']!=node['total'] or life['passed']!=life['total']:raise ValueError('Fresh recorded impulse partition required')
    cases=[('NODE',c) for c in node['branches']]+[('LIFECYCLE',c) for c in life['rows']];casepins=[]
    for declared in reg['cases'][:3]:
        p=ROOT/'runs/pen_native_screened_rollout_v1'/declared['name']/'report.json';casepins.append(file_pin(p));r=read(p)
        for n in r['nodes']:
            for branch in ['BASE','ALT']:
                if n[branch] is not None:cases.append(('FIRST3_FULL54_CONTROLS',read(verify_file_pin(n[branch]))))
    inputs=dict(node=file_pin(npth),lifecycle=file_pin(lp),registration=file_pin(regp),first3_cases=casepins,installed_solver=file_pin(Path(mujoco.__file__).parents[0].parent/'mujoco_warp/_src/solver.py'))
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__)],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt);rows=[];arrays={k:[] for k in ['total','velocity_update','acceleration_force','constraint_contact','gravity_representation']};models={};physical={};model_checks={};endpoints=[]
    with patch.object(mujoco,'mj_step',side_effect=AssertionError('No native replay')),patch.object(mujoco,'mj_forward',side_effect=AssertionError('No forward solve')):
        for role,c in cases:
            key=c['model']['path']
            if key not in models:
                m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['model'])));p=dict(np.load(verify_file_pin(c['physical_context'])));models[key]=m;physical[key]=p;b=c['model_layout']['object_body'];da=c['model_layout']['object_dofadr'];jid=int(m.body_jntadr[b]);native_mass=float(p['body_mass'][0,b])
                model_checks[key]=dict(free_world_root=int(m.jnt_type[jid])==int(mujoco.mjtJoint.mjJNT_FREE) and int(m.body_parentid[b])==0,no_descendants=not bool(np.any(m.body_parentid==b)),COM_at_origin=bool(np.all(m.body_ipos[b]==0)),no_armature=not bool(np.any(p['dof_armature'][0,da:da+6])),no_damping=not bool(np.any(p['dof_damping'][0,da:da+6])),no_frictionloss=not bool(np.any(m.dof_frictionloss[da:da+6])),no_stiffness=float(m.jnt_stiffness[jid])==0,no_fluid=float(m.opt.density)==float(m.opt.viscosity)==0,
                    no_direct_actuator=bool(np.all(m.actuator_trntype==int(mujoco.mjtTrn.mjTRN_JOINT)) and not np.any(m.actuator_trnid[:,0]==jid)),no_gravcomp=float(m.body_gravcomp[b])==0,free_not_mocap=int(m.body_mocapid[b])<0,model_mass_equal=float(m.body_mass[b])==native_mass,Euler=int(m.opt.integrator)==int(mujoco.mjtIntegrator.mjINT_EULER))
            m=models[key];p=physical[key];z=dict(np.load(verify_file_pin(c['trace'])));metrics,d=decode_saved_case(c,z,m,p);b=c['model_layout']['object_body'];da=c['model_layout']['object_dofadr'];mass=float(p['body_mass'][0,b]);dt=c['dt'];v0=z['pre_qvel'][:,da:da+3];v1=z['post_qvel'][:,da:da+3];a=z['post_qacc'][:,da:da+3];fc=z['post_qfrc_constraint'][:,da:da+3];fl=d['world_COM_wrench'][:,:3];gn=p['gravity'][0].astype(np.float64);parts=partition(mass,dt,v0,v1,a,fc,fl,gn,m.opt.gravity);norm=np.linalg.norm(parts['total'],axis=1);peak=int(norm.argmax());projections={k:float(np.dot(parts[k][peak],parts['total'][peak])/norm[peak]**2) if norm[peak] else None for k in arrays if k!='total'}
            checks=dict(model_conditions=all(model_checks[key].values()),no_object_aid=not bool(np.any(z['post_qfrc_applied']) or np.any(z['post_xfrc_applied'])),decoder_total=np.array_equal(parts['total'],d['linear_impulse_residual_N_s']),original_impulse_gate=metrics['checks']['linear_impulse']==c['physical_audit']['checks']['EFC_linear_impulse'],recomposition=parts['recomposition_within_float64'])
            row=dict(role=role,name=c['name'],model=c['model'],trace=c['trace'],checks=checks,original_impulse_pass=metrics['checks']['linear_impulse'],original_physical_pass=c['physical_audit']['valid'],peak_tick_zero_based=peak,peak_impulse_Ns=float(norm[peak]),peak_component_norms_Ns={k:float(np.linalg.norm(parts[k][peak])) for k in arrays if k!='total'},peak_signed_projections=projections,velocity_update_max_m_per_s=float(abs(v1-v0-dt*a).max()),acceleration_force_max_N=float(np.linalg.norm(parts['acceleration_force']/dt,axis=1).max()),label_force_max_N=metrics['max_force_residual_N'],recomposition_max_abs=parts['recomposition_max_abs'])
            rows.append(row)
            for k in arrays:arrays[k].append(parts[k])
            if not metrics['checks']['linear_impulse'] and peak==7:
                end=load_shared(c['end']) if c['end']['schema']==SCHEMA else load_owned(c['end']);q=end['native']['data']['arrays'];ma=q['efc.Ma'][0,da:da+3].astype(np.float64);smooth=q['qfrc_smooth'][0,da:da+3].astype(np.float64);f=q['qfrc_constraint'][0,da:da+3].astype(np.float64);acc=q['qacc'][0,da:da+3].astype(np.float64);gap=mass*acc-ma;inferred=ma-smooth-f;extra=smooth-mass*gn
                reference=parts['acceleration_force'][-1]/dt;equal=bool(np.all(abs(gap+inferred+extra-reference)<=32*np.finfo(np.float64).eps*(1+abs(reference))))
                endpoints.append(dict(name=c['name'],solver_niter=q['solver_niter'].tolist(),mass_times_qacc_minus_stored_Ma_N=gap.tolist(),stored_Ma_minus_smooth_and_constraint_N=inferred.tolist(),smooth_minus_native_gravity_N=extra.tolist(),acceleration_force_sum_N=(gap+inferred+extra).tolist(),trace_acceleration_force_N=reference.tolist(),sum_matches_trace=equal))
    checks=dict(all676=len(rows)==676,all_model_conditions=all(all(v.values()) for v in model_checks.values()),all_partitions=all(all(r['checks'].values()) for r in rows),all8_ticks=all(np.asarray(v).shape==(676,8,3) for v in arrays.values()),all_available_endpoint_partitions=all(x['sum_matches_trace'] for x in endpoints),no_native_optimizer=True)
    np.savez_compressed(out/'arrays.npz',**{k:np.asarray(v) for k,v in arrays.items()});verify_source_snapshot(receipt,live=True)
    result=dict(schema='pen_recorded_impulse_partition:1',inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),arrays=file_pin(out/'arrays.npz'),checks=checks,passed=sum(checks.values()),total=len(checks),rows=rows,model_conditions=model_checks,failed_impulse_rows=[r['name'] for r in rows if not r['original_impulse_pass']],bad_final_tick_endpoints=endpoints,actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,wall_s=time.perf_counter()-started,
        primary_reference='https://mujoco.readthedocs.io/en/stable/computation/',scope='Recorded world-linear impulse residual partition for16node+281lifecycle+allqueries of first3 declared full54 controls. Original5e-7Ns gate/masks unchanged. Acceleration-force imbalance is a localization,not proof of its optimizer/rounding cause. Endpoint Ma/smooth decomposition only for failures peaking at final tick;internal SolverContext gradient not recorded. Full54 producer remains separate/unmodified/live;no physics/engine edit.')
    write_json(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],rows=len(rows),impulse_failures=len(result['failed_impulse_rows']),final_tick_failures=len(endpoints))),flush=True)
    if not all(checks.values()):raise ValueError('Recorded impulse partition discrepancy retained')


if __name__=='__main__':main()
