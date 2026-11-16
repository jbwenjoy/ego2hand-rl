"""Strict fresh-backend trial/evaluation budgets, independent of legacy reports."""

def study_budget(cfg):
    if (cfg['study'],cfg['num_envs'],cfg['rollout_controls'],cfg['iterations'],cfg['checkpoint_interval'],cfg['arms'],cfg['calibration_controls'],cfg['repetitions'],cfg['case_order'],cfg['video_repetition'])!=('FRESH_BACKEND_LONG_CONTROL',32,64,2000,100,['RESIDUAL','SUPPORT'],8,3,['INITIAL','RESIDUAL','SUPPORT'],0):
        raise ValueError('Registered fresh32-world equal2000-round nine-case study required')
    return dict(training_native_per_arm=32768000,training_Adam_per_arm=64000,training_transitions_per_arm=4096000,
        training_native_total=65536000,training_Adam_total=128000,preparation_native=2048,cohort_native_before_evaluation=65538048,
        cohort_Adam_before_evaluation=128000,max_final_evaluation_native=26496)


def require_fresh_common(state,contract):
    if (state['schema'],state['iteration'],state['optimizer_steps'],state['training_transitions'],state['calibration_transitions'],state['native']['worlds'])!=('pen_geometry_support_PPO_training:4',0,0,0,256,32):
        raise ValueError('Fresh calibrated common state of this backend required')
    if state['optimizer']['state'] or state['memory'] is not None or state['metadata'].get('collision_backend')!=contract or state['native']['metadata'].get('collision_backend')!=contract:
        raise ValueError('No old/native trained common source or backend rebinding allowed')


def require_final_training(report,config_pin,backend_id,registration_pin):
    if report.get('schema')!='fresh_backend_long_control_training:1' or report['config']!=config_pin or report['registration']!=registration_pin or report['backend_id']!=backend_id:
        raise ValueError('Explicit completed new-backend trial required;legacy results ineligible')
    if report['actual_native_integrations']!=65536000 or report['optimizer_steps']!=128000 or report['passed']!=report['total'] or not all(report['checks'].values()):
        raise ValueError('Final matched budget/checks incomplete')
    if [r['arm'] for r in report['arms']]!=['RESIDUAL','SUPPORT']:raise ValueError('Both registered arms required')
    for row in report['arms']:
        if (row['iterations'],row['actual_native_integrations'],row['optimizer_steps'],row['training_transitions'])!=(2000,32768000,64000,4096000):raise ValueError('Per-arm final budget differs')
    if report['cohort_native_before_evaluation']!=65538048 or report['preparation_native_integrations']!=2048:raise ValueError('Shared preparation cost must be counted once')
