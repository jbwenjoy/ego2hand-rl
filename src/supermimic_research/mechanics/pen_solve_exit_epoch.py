"""Align eight solve-exit copies with the unchanged original raw tick packet."""
import numpy as np
from .pen_solver_tick_balance import mass_product,partition,terminal_predicates
from .pen_impulse_balance import partition as impulse_partition


def review(packet,observations,model_arrays,case,decoded,decoder_gravity,iteration_cap):
    fields={k[5:]:np.asarray(v)[:,0] for k,v in observations.items() if k.startswith('data.')}
    context={k[8:]:np.asarray(v)[:,0] for k,v in observations.items() if k.startswith('context.')}
    n=len(packet['post_qpos']);adr=case['model_layout']['object_dofadr'];dt=case['dt'];mass=case['physics']['mass'];gravity=np.asarray(case['physics']['gravity'],np.float64)
    checks=dict(eight_copies=n==8 and all(len(v)==8 for v in fields.values()) and all(len(v)==8 for v in context.values()),
        finite_copies=all(np.isfinite(v).all() for v in list(fields.values())+list(context.values())),
        pre_qpos=np.array_equal(fields['qpos'],packet['pre_qpos']),pre_qvel=np.array_equal(fields['qvel'],packet['pre_qvel']),
        post_qacc=np.array_equal(fields['qacc'],packet['post_qacc']),post_constraint=np.array_equal(fields['qfrc_constraint'],packet['post_qfrc_constraint']),
        applied_generalized=np.array_equal(fields['qfrc_applied'],packet['post_qfrc_applied']),
        applied_body=np.array_equal(fields['xfrc_applied'],packet['post_xfrc_applied']),
        tick_time=bool(np.all(abs(packet['post_GPU_time']-fields['time']-np.float32(dt))<=4*np.finfo(np.float32).eps*(1+abs(fields['time'])))))
    if not all(checks.values()):return dict(checks=checks,passed=sum(checks.values()),total=len(checks)),{}
    product=mass_product(fields['M'],fields['qacc'],model_arrays['M_mulm_rowadr'],model_arrays['M_mulm_col'],model_arrays['M_mulm_madr'])
    balance=partition(mass,dt,adr,gravity,fields,context,product)
    sl=slice(adr,adr+3)
    original=impulse_partition(mass,dt,packet['pre_qvel'][:,sl],packet['post_qvel'][:,sl],packet['post_qacc'][:,sl],
        packet['post_qfrc_constraint'][:,sl],decoded['world_COM_wrench'][:,:3],gravity,decoder_gravity)
    # Keep the original velocity/contact terms; replace only its force-balance
    # term with the new five-term decomposition, without changing its audit.
    combined=original['velocity_update']+balance['reconstructed_Ns']+original['constraint_contact']+original['gravity_representation']
    error=abs(combined-decoded['linear_impulse_residual_N_s'])
    budget=128*np.finfo(np.float64).eps*(1+abs(combined)+abs(original['total']))
    checks.update(balance_recomposition=balance['recomposition_within_float64'],
        original_force_part=bool(np.all(abs(original['acceleration_force']-balance['total_acceleration_force_Ns'])<=budget)),
        original_complete_residual=bool(np.all(error<=budget)))
    predicates=terminal_predicates(fields['qacc'].shape[1],float(model_arrays['stat.meaninertia'][0]),
        float(model_arrays['opt.tolerance'][0]),iteration_cap,dict(context,solver_niter=fields['solver_niter']))
    peak=int(np.linalg.norm(original['total'],axis=1).argmax());norm=float(np.linalg.norm(original['total'][peak]));vector=original['total'][peak]
    terms=['free_mass_chart','Ma_accumulation','actual_scaled_gradient','gradient_force_recovery','smooth_minus_gravity']
    arrays={k:v for k,v in balance.items() if isinstance(v,np.ndarray)}
    arrays.update({'original_'+k:v for k,v in original.items() if isinstance(v,np.ndarray)})
    arrays.update({'terminal_'+k:v for k,v in predicates.items()})
    result=dict(checks=checks,passed=sum(checks.values()),total=len(checks),peak_tick=peak,peak_original_impulse_Ns=norm,
        peak_component_norms_Ns={k:float(np.linalg.norm(balance[k][peak])) for k in terms},
        peak_signed_projections={k:float(np.dot(balance[k][peak],vector)/norm**2) if norm else None for k in terms},
        terminal_predicates={k:v.tolist() for k,v in predicates.items()},
        scope='Same copied solve-exit epochs only;final OR predicates descriptive,not unique termination cause. Original physical gate unchanged.')
    return result,arrays
