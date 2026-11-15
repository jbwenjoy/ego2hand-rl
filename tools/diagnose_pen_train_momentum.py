"""Post-hoc offline decomposition of retained strict-path impulse failures."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import mujoco
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.mechanics.warp_packet_decode import decode_saved_case


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    prep=json.loads(a.preparation.read_text());probe=json.loads(verify_file_pin(prep['probe']).read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or probe['restore_mode']!='STRICT' or prep['role']!='TRAIN_CONTROLLED_PHYSICAL_RESPONSE':raise ValueError('Fresh registered strict TRAIN diagnosis')
    out.mkdir();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve()]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    details=[];checks={};packets={};models={};worst_v=worst_identity=0.;all_bad=[]
    for row in prep['metadata']:
        c=probe['cases'][row['case_index']];z=dict(np.load(verify_file_pin(c['trace'])));key=c['model']['path']
        if key not in models:models[key]=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['model'])))
        m=models[key];p=dict(np.load(verify_file_pin(c['physical_context'])));metrics,decoded=decode_saved_case(c,z,m,p)
        d=c['model_layout']['object_dofadr'];mass=c['physics']['mass'];gravity=np.asarray(c['physics']['gravity']);dt=c['dt'];b=c['model_layout']['object_body']
        # Preserve the decoder's native-dtype velocity subtraction/mass product,
        # then explicitly account for that rounding and host/GPU gravity values.
        # Neither tiny correction changes the original physical gate or mask.
        native_dv=z['post_qvel'][:,d:d+3]-z['pre_qvel'][:,d:d+3];dv=native_dv.astype(np.float64);integration=dv-dt*z['post_qacc'][:,d:d+3].astype(np.float64)
        generalized=z['post_qfrc_constraint'][:,d:d+3].astype(np.float64);balance=mass*z['post_qacc'][:,d:d+3].astype(np.float64)-generalized-mass*gravity
        force_error=generalized-decoded['world_COM_wrench'][:,:3]
        decoder_rounding=(mass*native_dv).astype(np.float64)-mass*dv+dt*mass*(gravity-m.opt.gravity)
        impulse=decoded['linear_impulse_residual_N_s'];identity=mass*integration+dt*(balance+force_error)+decoder_rounding-impulse
        worst_v=max(worst_v,float(np.linalg.norm(integration,axis=1).max()));worst_identity=max(worst_identity,float(np.linalg.norm(identity,axis=1).max()))
        checks[c['name']+'_decomposition_identity']=bool(np.max(abs(identity))<1e-12)
        if metrics['checks']['linear_impulse']:continue
        all_bad.append(row['row_index']);ticks=np.flatnonzero(np.linalg.norm(impulse,axis=1)>=5e-7)
        name=c['name'];packets[name+'_velocity_update_residual']=integration;packets[name+'_force_balance_residual']=balance;packets[name+'_EFC_force_discrepancy']=force_error;packets[name+'_impulse_residual']=impulse;packets[name+'_decoder_rounding_correction']=decoder_rounding
        no_linear_passive=bool(np.all(p['dof_damping'][:,d:d+3]==0) and np.all(p['dof_armature'][:,d:d+3]==0) and m.body_gravcomp[b]==0 and not np.any(m.body_ipos[b]) and m.opt.density==0 and m.opt.viscosity==0)
        checks[name+'_no_recorded_linear_passive_terms']=no_linear_passive
        details.append(dict(row_index=row['row_index'],name=name,source_name=c['source_name'],profile=c['profile'],ticks=ticks.tolist(),metrics=metrics,
            no_recorded_linear_passive_terms=no_linear_passive,solver_options=dict(solver=int(m.opt.solver),iterations=int(m.opt.iterations),tolerance=float(m.opt.tolerance),ls_iterations=int(m.opt.ls_iterations),ls_tolerance=float(m.opt.ls_tolerance)),
            per_tick=[dict(tick=int(t),impulse_Ns=impulse[t].tolist(),qacc_force_balance_N=balance[t].tolist(),velocity_update_residual_m_s=integration[t].tolist(),EFC_force_discrepancy_N=force_error[t].tolist(),decoder_rounding_correction_Ns=decoder_rounding[t].tolist()) for t in ticks],trace=c['trace']))
    repeats=[]
    for r in details:
        if r['profile']['repeat']!=0:continue
        others=[s for s in details if s['source_name']==r['source_name'] and s['profile']['direction']==r['profile']['direction'] and s['profile']['epsilon']==r['profile']['epsilon'] and s['profile']['sign']==r['profile']['sign'] and s['profile']['repeat']==1]
        for s in others:
            identical=all(np.array_equal(packets[r['name']+'_'+suffix],packets[s['name']+'_'+suffix]) for suffix in ['velocity_update_residual','force_balance_residual','EFC_force_discrepancy','impulse_residual'])
            repeats.append(dict(names=[r['name'],s['name']],identical_recorded_residuals=identical))
    checks['all_impulse_failures_masked']=set(all_bad)<=set(prep['rejected_rows'])
    np.savez_compressed(out/'diagnostic_arrays.npz',**packets);verify_source_snapshot(receipt,live=True)
    result=dict(schema='strict_TRAIN_impulse_offline_diagnosis:1',preparation=file_pin(a.preparation),probe=prep['probe'],source_receipt=file_pin(out/'source_receipt.json'),arrays=file_pin(out/'diagnostic_arrays.npz'),
        cases=details,repeated_conditions=repeats,all_impulse_failure_rows=all_bad,max_velocity_update_residual_m_s=worst_v,max_identity_error_Ns=worst_identity,
        checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,
        scope='Post-hoc recorded-data algebra,not new replay or solver causality. Same-condition repetitions are the two original STRICT collection repeats. No physical threshold relaxation,parameter changes or discarded counterexamples. Solver termination/hidden state requires targeted observation before attributing root cause.')
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],impulse_failure_rows=all_bad,repeated_conditions=repeats,max_velocity_update_residual_m_s=worst_v,max_identity_error_Ns=worst_identity)),flush=True)
    if not all(checks.values()):raise ValueError('Retain offline decomposition discrepancy')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preparation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
