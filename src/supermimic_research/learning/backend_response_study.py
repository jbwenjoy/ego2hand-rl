"""Fresh final-policy response cohorts, distinct from old-backend datasets."""
from .backend_control_study import require_final_training
from .backend_pen_ppo import require_saved_backend
from .pen_interventions import validate_full_rank_cohort


def response_spec(cfg):
    expected=dict(schema=1,study='BACKEND_FULLRANK_PHYSICAL_RESPONSE',actors=['RESIDUAL','SUPPORT'],
        PD_eps_rad=[.002,.005],native_dt_s=.0025,control_horizon_s=.02,source_mean_hand_load_N=.1,
        object_penetration_m=.002,self_penetration_m=.0002,joint_excursion_rad=.001,nonhand_N=.1)
    if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError('Explicit unchanged physical-response gates required')
    directions,profiles=validate_full_rank_cohort(dict(cfg,schema=3))
    producer=len(cfg['actors'])*len(cfg['noise_seeds'])*(max(cfg['source_phases'])-1)*8
    return directions,profiles,dict(max_producer_native=producer,max_branch_native=cfg['expected_cases']*8,
        max_total_native=producer+cfg['expected_cases']*8,optimizer_steps=0)


def require_final_response_actor(state,contract,arm):
    require_saved_backend(state,contract)
    if state.get('schema')!='pen_geometry_support_PPO_training:4' or (state['iteration'],state['optimizer_steps'],state['training_transitions'])!=(2000,64000,4096000) or state['native']['worlds']!=32 or not state['normalizers_ready']:
        raise ValueError('Exact completed final2000 actor source required')
    if arm not in ['RESIDUAL','SUPPORT'] or state['metadata']['arm']!=arm or state['metadata']['support_weight']!=(0. if arm=='RESIDUAL' else .2):
        raise ValueError('Response actor arm/weight differs')


def require_evaluated_cohort(training,evaluation,config_pin,backend_id,registration_pin,training_pin):
    require_final_training(training,config_pin,backend_id,registration_pin)
    if evaluation.get('schema')!='fresh_backend_long_control_evaluation:1' or evaluation['config']!=config_pin or evaluation['registration']!=registration_pin or evaluation['backend_id']!=backend_id or evaluation['training']!=training_pin:
        raise ValueError('Exact completed new-backend trial evaluation required')
    cases=evaluation['cases'];expected=[(r,label) for r in range(3) for label in ['INITIAL','RESIDUAL','SUPPORT']]
    if [(c['repetition'],c['policy']) for c in cases]!=expected or evaluation['optimizer_steps']!=0 or evaluation['actual_native_integrations']!=sum(c['native_integrations'] for c in cases):
        raise ValueError('All nine final-policy executions and exact work required')
    pins={'RESIDUAL':training['arms'][0]['checkpoint'],'SUPPORT':training['arms'][1]['checkpoint']}
    for case in cases:
        audit=case['actor_initialization'];label=case['policy']
        if not audit['backend_binding_verified'] or audit['collision_backend_id']!=backend_id or audit['old_native_backend_imported']:
            raise ValueError('Evaluation backend identity differs')
        if label!='INITIAL' and case['checkpoint']!=pins[label]:raise ValueError('Evaluation used another final actor')
    # A failed strict task is data, not exclusion of the entire actor cohort.
    # Numerical/source/row/gradient gates remain independently mandatory.
