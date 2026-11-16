"""Admission of actual new-backend raw data; old probe schemas stay separate."""
from .backend_response_study import response_spec


def require_response_probe(run,cfg,contract,registration_pin,config_pin):
    _,profiles,budget=response_spec(cfg)
    if run.get('schema')!='backend_controlled_response_probe:1' or run['config']!=config_pin or run['registration']!=registration_pin or run['collision_backend']!=contract or run['role']!=cfg['role']:
        raise ValueError('Exact registered new-backend raw cohort required')
    if run['passed']!=run['total'] or not all(run['checks'].values()) or run['restore_mode']!='STRICT' or run['fixed_frame_features']!='LIVE' or run['scales_fitted'] or run['CHECK_response_rows_used'] or run['validation_rows_used_for_TRAIN']:
        raise ValueError('Completed strict unnormalized cohort/roles required')
    identities=[(a,n,p) for a in cfg['actors'] for n in cfg['noise_seeds'] for p in cfg['source_phases']]
    if [(s['actor'],s['noise_seed'],s['phase']) for s in run['sources']]!=identities or len({s['name'] for s in run['sources']})!=len(identities):raise ValueError('All declared source identities required')
    if len(run['sources'])!=cfg['expected_sources'] or len(run['cases'])!=cfg['expected_cases']:raise ValueError('Declared exclusions cannot shrink denominators')
    names={s['name'] for s in run['sources']}
    if any(c['source_name'] not in names or c['role']!=cfg['role'] or c['collision_backend_id']!=contract['backend_id'] for c in run['cases']):raise ValueError('Mixed source/backend/role in raw cases')
    for source in run['sources']:
        cases=[c for c in run['cases'] if c['source_name']==source['name']]
        if [c['profile'] for c in cases]!=profiles:raise ValueError('All declared profiles must remain,including exclusions')
        if any(c['native_integrations']!=(8 if c['available'] else 0) for c in cases):raise ValueError('One control or explicit unavailable branch required')
    actual=run['producer_native_integrations']+sum(c['native_integrations'] for c in run['cases'])
    if run['actual_native_integrations']!=actual or run['attempted_native_integrations']!=actual or actual>budget['max_total_native'] or run['optimizer_steps'] or run['attempted_optimizer_steps'] or not run['completed_call_accounting_exact']:
        raise ValueError('Exact completed/attempted source/branch work required')
