env_cfg.terminations['object_position_error'].params['threshold'] = 0.05
env_cfg.terminations['object_orientation_error'] = TerminationTermCfg(func=mdp.object_shaft_direction_error_exceeded, params={'robot_cfg': SceneEntityCfg('robot', body_names=('right_palm_link',)), 'object_cfg': SceneEntityCfg('object'), 'threshold_deg': 60.0}, time_out=False)
if is_a0:
    env_cfg.terminations.pop('object_position_error')
    env_cfg.terminations.pop('object_orientation_error')
