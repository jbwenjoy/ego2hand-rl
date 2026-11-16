"""Closed-replica capsule review with separately implemented numeric balance."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
import tomllib
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.strict_json import write_json


def read(p):return json.loads(Path(p).read_text())


def independent_arrays(z,o,ma,c,gravity_decoder,decoded,cap):
    """Separate NumPy implementation; no collector partition/epoch helpers."""
    f={k[5:]:np.asarray(v)[:,0].astype(np.float64) for k,v in o.items() if k.startswith('data.')}
    g={k[8:]:np.asarray(v)[:,0].astype(np.float64) for k,v in o.items() if k.startswith('context.')}
    acc=f['qacc'];ticks,nv=acc.shape;stored=f['efc.Ma'];smooth=f['qfrc_smooth'];constraint=f['qfrc_constraint'];matrix=f['M']
    if matrix.ndim==2:
        dense=np.zeros((ticks,nv,nv))
        for row in range(nv):
            for k in range(int(ma['M_mulm_rowadr'][row]),int(ma['M_mulm_rowadr'][row+1])):
                dense[:,row,int(ma['M_mulm_col'][k])]+=matrix[:,int(ma['M_mulm_madr'][k])]
    elif matrix.shape==(ticks,nv,nv):dense=matrix
    else:raise ValueError('Explicit recorded packed or dense M required')
    direct=np.einsum('tij,tj->ti',dense,acc);gradient=g['grad']*g['grad_scale'][:,None]
    dt=c['dt'];mass=c['physics']['mass'];gravity=np.asarray(c['physics']['gravity']);adr=c['model_layout']['object_dofadr'];sl=slice(adr,adr+3)
    arrays=dict(free_mass_chart=dt*(mass*acc[:,sl]-direct[:,sl]),Ma_accumulation=dt*(direct-stored)[:,sl],actual_scaled_gradient=dt*gradient[:,sl],gradient_force_recovery=dt*(stored-smooth-constraint-gradient)[:,sl],smooth_minus_gravity=dt*(smooth[:,sl]-mass*gravity),total_acceleration_force_Ns=dt*(mass*acc[:,sl]-constraint[:,sl]-mass*gravity),scaled_gradient_N=gradient,direct_mass_balance_N=direct-smooth-constraint,stored_Ma_balance_N=stored-smooth-constraint,gradient_force_recovery_N=stored-smooth-constraint-gradient)
    terms=['free_mass_chart','Ma_accumulation','actual_scaled_gradient','gradient_force_recovery','smooth_minus_gravity'];arrays['reconstructed_Ns']=sum(arrays[k] for k in terms)
    dv=z['post_qvel'][:,sl]-z['pre_qvel'][:,sl];contact=decoded['world_COM_wrench'][:,:3]
    arrays.update(original_total=mass*dv-dt*(contact+mass*np.asarray(gravity_decoder)),original_velocity_update=mass*(dv-dt*acc[:,sl]),original_acceleration_force=arrays['total_acceleration_force_Ns'],original_constraint_contact=dt*(constraint[:,sl]-contact),original_gravity_representation=np.broadcast_to(dt*mass*(gravity-np.asarray(gravity_decoder)),(ticks,3)).copy())
    arrays['original_reconstructed']=sum(arrays['original_'+k] for k in ['velocity_update','acceleration_force','constraint_contact','gravity_representation'])
    denominator=float(ma['stat.meaninertia'][0])*nv;tolerance=float(ma['opt.tolerance'][0])
    arrays.update(terminal_improvement_below_tolerance=g['improvement']/denominator<tolerance,terminal_gradient_below_tolerance=np.sqrt(g['grad_dot'])/denominator<tolerance,terminal_newton_decrement_below_tolerance=.5*g['newton_decrement']/denominator<tolerance,terminal_at_iteration_cap=f['solver_niter']==cap)
    return arrays


def near(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape:return False
    if a.dtype.kind=='b' or b.dtype.kind=='b':return np.array_equal(a,b)
    return bool(np.isfinite(a).all() and np.isfinite(b).all() and np.all(abs(a-b)<=128*np.finfo(np.float64).eps*(1+abs(b))))


def projection_near(component,vector,recorded):
    norm2=float(np.dot(vector,vector))
    if norm2==0:return recorded is None
    expected=float(np.dot(component,vector)/norm2)
    # Propagate the component-array comparison bound through the dot product.
    # This avoids demanding the same absolute tolerance after dividing by a
    # small impulse; it does not change any physical residual threshold.
    component_bound=128*np.finfo(np.float64).eps*(1+abs(component))
    bound=float(np.dot(component_bound,abs(vector))/norm2)+128*np.finfo(np.float64).eps*(1+abs(expected))
    return bool(np.isfinite(recorded) and abs(expected-recorded)<=bound)


def analytic_checks():
    n=8;a=np.tile([1.,0,0],(n,1));zero=np.zeros_like(a);matrix=np.tile([.03,0,.03,0,0,.03],(n,1));maps=dict(M_mulm_rowadr=[0,3,6,9],M_mulm_col=[0,1,2]*3,M_mulm_madr=[0,1,3,1,2,4,3,4,5],**{'stat.meaninertia':[1.],'opt.tolerance':[1e-12]})
    case=dict(dt=.0025,physics=dict(mass=.03,gravity=[0,0,0]),model_layout=dict(object_dofadr=0));z=dict(pre_qvel=zero.copy(),post_qvel=.0025*a);decoded=dict(world_COM_wrench=np.c_[.02*a,zero])
    fields=dict(qacc=a,M=matrix,**{'efc.Ma':.03*a},qfrc_smooth=zero,qfrc_constraint=.02*a,solver_niter=np.ones(n));ctx=dict(grad=.002*a,grad_scale=np.full(n,5.),grad_dot=np.ones(n),newton_decrement=np.ones(n),improvement=np.ones(n))
    obs={'data.'+k:v[:,None] for k,v in fields.items()};obs.update({'context.'+k:v[:,None] for k,v in ctx.items()});r=independent_arrays(z,obs,maps,case,[0,0,0],decoded,50)
    checks=dict(actual_scale=near(r['actual_scaled_gradient'],.000025*a),packed_direct=near(r['free_mass_chart'],zero),all_terms_close=near(r['reconstructed_Ns'],r['total_acceleration_force_Ns']))
    obs['data.efc.Ma']=.13*a[:,None];obs['data.qfrc_constraint']=.13*a[:,None];obs['context.grad']=zero[:,None];decoded['world_COM_wrench'][:,:3]=.13*a;r=independent_arrays(z,obs,maps,case,[0,0,0],decoded,50)
    checks.update(zero_grad_can_hide_drift=near(r['actual_scaled_gradient'],zero) and near(r['Ma_accumulation'],-.00025*a),impulse_recomposition=near(r['original_reconstructed'],r['original_total']))
    return checks


def differences(z,reference):
    return {k:dict(bytes_equal=z[k].dtype==reference[k].dtype and z[k].shape==reference[k].shape and z[k].tobytes()==reference[k].tobytes(),max_abs=float(np.max(abs(z[k].astype(np.float64)-reference[k].astype(np.float64)))) if z[k].shape==reference[k].shape and z[k].size else None) for k in z}


def main(a):
    started=time.perf_counter();cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());paths={k:(cp.parent/cfg[k]).resolve() for k in ['worker_config','worker_registration','registration','output','protocol']}
    if (cfg['schema'],cfg['mode'],cfg['replicas'],cfg['queries_per_replica'],cfg['native_steps_per_replica'])!=(1,'READ_ONLY_SOLVE_EXIT_SAVED_REVIEW',['A','B'],60,480):raise ValueError('Fixed closed-replica review required')
    reg=read(paths['worker_registration']);worker_cfg=tomllib.loads(paths['worker_config'].read_text());sources=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,paths['protocol']];selfchecks=analytic_checks()
    if not all(selfchecks.values()):raise ValueError('Independent analytic self-check failed')
    def freeze(out):
        receipt=freeze_sources(sources,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt);return receipt
    inputs=dict(worker_config=file_pin(paths['worker_config']),worker_registration=file_pin(paths['worker_registration']),protocol=file_pin(paths['protocol']))
    if a.register:
        out=paths['registration'].parent
        if out.exists():raise ValueError('Fresh reader registration required')
        out.mkdir();freeze(out);write_json(paths['registration'],dict(schema='pen_solve_exit_saved_review_registration:1',config=file_pin(cp),inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),analytic_checks=selfchecks,passed=sum(selfchecks.values()),total=len(selfchecks),actual_native_integrations=0,optimizer_steps=0));print(json.dumps(dict(registration=file_pin(paths['registration']),analytic_checks=[sum(selfchecks.values()),len(selfchecks)],native=0)),flush=True);return
    registration=read(paths['registration'])
    if registration['inputs']!=inputs or registration['config']!=file_pin(cp):raise ValueError('Reader registration changed')
    verify_source_snapshot(read(verify_file_pin(registration['source_receipt'])),live=True)
    if a.replica not in cfg['replicas']:raise ValueError('Explicit replica required')
    folder=(paths['worker_config'].parent/worker_cfg['output']).resolve()/a.replica;rp=folder/'report.json'
    if not rp.exists() or (folder/'failure.json').exists():raise ValueError('Completed successful replica required;preserve failed/partial artifacts')
    producer=read(rp)
    if (producer['schema']!='pen_solve_exit_replay:1' or producer['replica']!=a.replica or producer['registration']!=file_pin(paths['worker_registration']) or producer['passed']!=producer['total'] or producer['total']!=6 or not all(producer['checks'].values()) or producer['actual_native_integrations']!=480 or producer['attempted_native_integrations']!=480 or producer['optimizer_steps']!=0 or producer['normalizer_fits']!=0 or producer['setup_extra_native'] or producer['setup_native_unknown']):raise ValueError('Exact terminal replica accounting required')
    from supermimic_research.runtime.pen_solve_exit_observer import require_closed_full54
    observation_config=(paths['worker_config'].parent/worker_cfg['observation_config']).resolve()
    if producer['prerequisites']!=require_closed_full54(observation_config):raise ValueError('Native producer prerequisite ownership changed')
    before=[file_pin(p) for p in sorted(folder.rglob('*')) if p.is_file()];out=paths['output']/a.replica
    if out.exists():raise ValueError('Fresh closed-replica review required')
    import mujoco
    import torch
    from scipy.spatial.transform import Rotation
    from supermimic_research.runtime.shared_snapshot_transport import load_shared
    from supermimic_research.learning.world_model import raw_tree_equal
    from supermimic_research.mechanics.pen_screened_review import ScreenedCaseReviewer
    from supermimic_research.mechanics.warp_packet_decode import decode_saved_case
    torch.set_num_threads(4);verify_source_snapshot(read(verify_file_pin(producer['source_receipt'])),live=True);objective=tomllib.loads(verify_file_pin(reg['inputs']['objective_config']).read_text());adapter=(Path(reg['inputs']['objective_config']['path']).parent/objective['adapter_config']).resolve();math_reader=SimpleNamespace(reward=tomllib.loads(adapter.read_text())['reward'],objective=objective);prepared=read(verify_file_pin(reg['inputs']['preparation']));original_cases={r['name']:r['case'] for r in prepared['rows']}
    checks={};records=[];models={};physicals={};original_sources={};observations={};prior_traces={};array_checks=0
    plan=reg['schedules'][a.replica];checks['exact_schedule']=[r['declaration'] for r in producer['rows']]==plan and len(plan)==60
    with patch.object(mujoco,'mj_step',side_effect=AssertionError('Saved review has no native steps')),patch.object(mujoco,'mj_forward',side_effect=AssertionError('No forward in saved review')):
        for build in producer['builds']:
            x,y=[load_shared(build[k]) for k in ['before','after']];checks[build['teacher']+'_capture_state']=all(raw_tree_equal(x[k],y[k]) for k in ['native','python','rng']) and build['copy_events']==build['clock_native_estimate']==0 and build['counter_before']==build['counter_after']
        for rowpin,decl in zip(producer['rows'],plan):
            row=read(verify_file_pin(rowpin['report']));c=read(verify_file_pin(row['case']));source=load_shared(c['source']);old=row['original_case'];key=json.dumps(old['source'],sort_keys=True)
            if key not in original_sources:original_sources[key]=load_shared(old['source'])
            same_source=all(raw_tree_equal(source[k],original_sources[key][k]) for k in ['native','python','rng']);mk=c['model']['path']
            if mk not in models:models[mk]=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['model'])));physicals[mk]=dict(np.load(verify_file_pin(c['physical_context'])))
            # Reuse only the frozen canonical saved-packet decoder; no neural
            # reviewer/model is initialized or admitted by this adapter.
            packet,end,decision,raw=ScreenedCaseReviewer.packet(math_reader,c,source,models[mk],physicals[mk]);z=dict(np.load(verify_file_pin(c['trace'])));oldz=dict(np.load(verify_file_pin(old['trace'])));obs=torch.load(verify_file_pin(row['regenerated_observation']),map_location='cpu',weights_only=False);name=decl['name'];qchecks=dict(all_packet_checks=all(packet['checks'].values()),full_source=same_source,actual_PD=np.array_equal(z['post_ctrl'][:,c['aids']],oldz['post_ctrl'][:,old['aids']]),raw_old=np.array_equal(raw,oldz['actions20'][0]),declaration=row['declaration']==decl,original_condition=old==original_cases[decl['original_name']],old_mask=row['original_physical_valid']==old['physical_audit']['valid'],old_impulse=row['original_impulse_valid']==old['physical_audit']['checks']['EFC_linear_impulse'],recorded_checks=row['passed']==row['total'] and all(row['checks'].values()),restore=all(row['source_restore']['complete_state_equal'].values()) and row['source_restore']['external_counter_unchanged'],observation=old['name'] not in observations or raw_tree_equal(observations[old['name']],obs),original_differences=differences(z,oldz)==row['differences_from_original'],new_mask=packet['physical']['valid']==row['new_physical_valid'],new_impulse=packet['physical']['checks']['EFC_linear_impulse']==row['new_impulse_valid'])
            observations.setdefault(old['name'],obs);pair=(decl['repeat'],old['name'])
            if pair in prior_traces:qchecks['scope_differences']=differences(z,prior_traces[pair])==row['differences_between_scopes']
            else:prior_traces[pair]=z
            if decl['scope']=='PACKET_PLUS_SOLVE_EXIT':
                o=dict(np.load(verify_file_pin(row['solve_exit'])));f={k[5:]:v[:,0] for k,v in o.items() if k.startswith('data.')};qchecks.update(copies8=all(len(v)==8 for v in o.values()),finite_copies=all(np.isfinite(v).all() for v in o.values()),epochs=np.array_equal(f['qpos'],z['pre_qpos']) and np.array_equal(f['qvel'],z['pre_qvel']) and np.array_equal(f['qacc'],z['post_qacc']) and np.array_equal(f['qfrc_constraint'],z['post_qfrc_constraint']),applied=np.array_equal(f['qfrc_applied'],z['post_qfrc_applied']) and np.array_equal(f['xfrc_applied'],z['post_xfrc_applied']),clock=bool(np.all(abs(z['post_GPU_time']-f['time']-np.float32(c['dt']))<=4*np.finfo(np.float32).eps*(1+abs(f['time'])))),copy_events=row['copy_events']==8)
                _,decoded=decode_saved_case(c,z,models[mk],physicals[mk]);arrays=independent_arrays(z,o,source['native']['model']['arrays'],c,models[mk].opt.gravity,decoded,int(source['native']['model']['scalars']['opt.iterations']));saved=dict(np.load(verify_file_pin(row['balance'])));qchecks['array_inventory']=arrays.keys()==saved.keys();qchecks['all_independent_arrays']=all(near(v,saved[k]) for k,v in arrays.items());qchecks['original_impulse']=near(arrays['original_total'],decoded['linear_impulse_residual_N_s']);array_checks+=sum(v.size for v in arrays.values());epoch=read(verify_file_pin(row['epoch_review']));qchecks['epoch_summary']=epoch['passed']==epoch['total'] and all(epoch['checks'].values())
                terms=['free_mass_chart','Ma_accumulation','actual_scaled_gradient','gradient_force_recovery','smooth_minus_gravity'];peak=int(np.linalg.norm(arrays['original_total'],axis=1).argmax());norm=float(np.linalg.norm(arrays['original_total'][peak]));qchecks['peak_summary']=epoch['peak_tick']==peak and near(norm,epoch['peak_original_impulse_Ns']) and all(near(np.linalg.norm(arrays[k][peak]),epoch['peak_component_norms_Ns'][k]) and projection_near(arrays[k][peak],arrays['original_total'][peak],epoch['peak_signed_projections'][k]) for k in terms);qchecks['terminal_summary']=all(np.array_equal(arrays['terminal_'+k],v) for k,v in epoch['terminal_predicates'].items())
            else:qchecks['no_copy_events']=row['copy_events']==0 and 'solve_exit' not in row
            checks.update({name+'_'+k:bool(v) for k,v in qchecks.items()});records.append(dict(name=name,original_name=old['name'],scope=decl['scope'],repeat=decl['repeat'],original_valid=row['original_physical_valid'],new_valid=row['new_physical_valid'],new_impulse_valid=row['new_impulse_valid'],checks=qchecks))
    for pin in before:verify_file_pin(pin)
    checks['same_file_inventory']=[p['path'] for p in before]==[str(p.resolve()) for p in sorted(folder.rglob('*')) if p.is_file()]
    checks['all60_rows']=len(records)==60;checks['all30_observed']=sum(r['scope']=='PACKET_PLUS_SOLVE_EXIT' for r in records)==30
    out.mkdir(parents=True);receipt=freeze(out);write_json(out/'report.json',dict(schema='pen_solve_exit_saved_review:1',registration=file_pin(paths['registration']),producer=file_pin(rp),source_receipt=file_pin(out/'source_receipt.json'),input_file_capsule=dict(records_before_read=before,all_bytes_verified_after=True),rows=records,checks=checks,passed=sum(checks.values()),total=len(checks),recomputed_array_elements=array_checks,actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,wall_s=time.perf_counter()-started,scope='Separate arithmetic implementation plus reused canonical raw/EFC reader. Complete current replica only;not an external physics oracle or root-cause/Actor/ego admission.'))
    verify_source_snapshot(receipt,live=True);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[sum(checks.values()),len(checks)],array_elements=array_checks,native=0)),flush=True)
    if not all(checks.values()):raise ValueError('Saved replica review discrepancy retained')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--replica',choices=['A','B']);p.add_argument('--register',action='store_true');main(p.parse_args())
