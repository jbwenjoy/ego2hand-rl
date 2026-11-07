"""Source-only A0 orchestration review. No production import, Env or execution."""
import argparse,ast,hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def pin(p):
    p=Path(p).resolve(strict=True);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def calls(tree,name):return [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==name]
def line(text,needle):return next(i for i,s in enumerate(text.splitlines(),1) if needle in s)


def main(output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);tick=time.perf_counter();checks={}
    paths={
      'native':ROOT/'src/full151_native/native.py','probe':ROOT/'src/full151_native/probe.py',
      'runtime':ROOT/'src/full151_native/runtime.py','controller':ROOT/'src/full151_native/controller.py',
      'capacity':Path('D:/workspace/supermimic-rough-prefix-learning-study/src/rough_prefix/capacity.py'),
      'PenReader':Path('D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/pen.py'),
      'SimReader':Path('D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/sim.py'),
      'Manager':Path('D:/workspace/dex-supermimic-research/.venv/Lib/site-packages/mjlab/envs/manager_based_rl_env.py')}
    sources={k:pin(v) for k,v in paths.items()};texts={k:v.read_text(encoding='utf-8-sig') for k,v in paths.items()};n,p,r,c=texts['native'],texts['probe'],texts['runtime'],texts['capacity'];nt=ast.parse(n);pt=ast.parse(p)
    def check(k,v):checks[k]=bool(v)
    check('native_exact_requested_snapshot',sources['native']['sha256']=='d9177adf578db78fb999c7cd4a67883e0cefb5d3967033c892ad7319070929ab')
    check('probe_exact_requested_snapshot',sources['probe']['sha256']=='19aff6ca2afffd9a649eca23a4c2e50b789d2deaa15ab006a7f843407c06ee48')
    check('only_Manager_and_PenReader_instances',len(calls(nt,'ManagerBasedRlEnv'))==1 and len(calls(nt,'PenSimReader'))==1 and not any(calls(nt,k) for k in ('DexSuperMimicRunner','Actor','WorldModel','Critic','FailureHead','Adam','AdamW')))
    check('no_direct_network_or_checkpoint_load',not any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr in ('load','load_model_only','load_state_dict') for x in ast.walk(nt)))
    check('A0_role_one_world_before_framework_imports',"config.get('role') != 'native_a0'" in n and "config.get('method') != 'A0'" in n and "config.get('training') != {'num_envs': 1}" in n and n.index('validate_a0_inputs(config, bundle)',n.index('def run_native'))<n.index('import torch',n.index('def run_native')))
    check('explicit_authorization_before_framework_imports',n.index("config.get('GPU_run_authorized') is not True",n.index('def run_native'))<n.index('import torch',n.index('def run_native')))
    check('wrong_horizon_rejected_not_padded',"u.shape != (150, 20)" in n and "'mapped_object_pose_w': (151, 7)" in n and "'planned_time601_s': (601,)" in n)
    check('factory_play_termination_exact_guard_and_position_only_removed',"if set(cfg.terminations) != {'time_out', 'unstable', 'object_position_error', 'reference_end'}" in n and "cfg.terminations.pop('object_position_error')" in n and "cfg.terminations.pop('object_orientation_error')" not in n)
    check('manual_reset_noise_DR_and_noRSI_guards',all(x in n for x in ('cfg.auto_reset = False','motion.first_frame_hold_prob = 0.','motion.trajectory_sampling_mode = \'uniform\'','cfg.actions[\'joint_pos\'].mask_prob = 0.','any(g.enable_corruption for g in cfg.observations.values())')))
    check('native_4x5ms_and_original_capacities',all(x in n for x in ('cfg.decimation != 4','cfg.sim.mujoco.timestep != .005','cfg.sim.nconmax = 192','cfg.sim.njmax = 4096')))
    check('actual8_double_readback_and_restoreTrue',"result = {'after_copy': verify('after_copy')}" in n and "result['after_set_const_0'] = verify('after_set_const_0')" in n and 'mjwarp.set_const_0(env.sim.wp_model, env.sim.wp_data, restore=True)' in n)
    check('actual8_before_explicit_native_reset',n.index('install_actual_context(env, bundle, output)',n.index('def run_native'))<n.index('env.reset(seed=seed)',n.index('def run_native')))
    check('nominal_MJB_hash_and_xi46_qualified',"sha(output / 'nominal_model.mjb') != bundle['model']['sha256']" in n and "xi.shape != (1, 46)" in n and 'np.array_equal(xi, np.broadcast_to(expected, xi.shape))' in n)
    check('capacity_existing_frozen_source',sources['capacity']['sha256']=='7380bef76a4c6af5c65204d0bf9f5921d68826072a71a77656f1b4b2fda3844a' and "observer._read('after_constructor_before_actual8')" in n)
    check('capacity_each_reset_before_native_reset',c.index("observer.before_reset(kwargs.get('env_ids'))")<c.index('return native(*args,**kwargs)'))
    check('capacity_hostOR_sticky_no_clear_and_idempotent_final','self.cumulative_bitmask|=mask' in c and 'if not self.finalized:' in c and "'flags_cleared_by_observer':False" in c)
    check('capacity_probe_final_and_all_native_exit_cleanup',"observer.final_read()" in p and "if not observer.finalized:" in n and "write(output / 'capacity_observer_final.json', observer.snapshot())" in n)
    check('capacity_overflow_or_unknown_cannot_qualityPASS',"native_A0_pass': all(checks.values()) if capacity.get('status') == 'OK' else None" in p and "capacity['status'] = 'UNKNOWN_AT_ALLOCATION_BOUNDARY'" in p)
    check('P0_q_not_projected_and_actual_bodyomega_separate',"robot.write_joint_state_to_sim(q,qd" in r and "obj.write_root_state_to_sim(torch.cat((pose,v,omega),-1)" in r and "expected_body=_tensor(i['object_angular_requested_FP32_body_rad_s']" in r and 'atol=1e-6' in r)
    check('startup_u0_once_and_action_reset_zero',"env.action_manager.reset(env_ids=ids)" in r and "term._targets[ids]=u0" in r and "bool(actual['EMA_residual'].any())" in r)
    check('150_real_intervals_no151st_action',len([x for x in ast.walk(pt) if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='k' and ast.dump(x.iter)==ast.dump(ast.parse('range(150)',mode='eval').body)])==1 and 'if len(states) > 600:' in p)
    check('step_readback_before_original_Manager_forward','original_step(); sample(len(states))' in p and 'env.sim.step = step; env.sim.forward = forward' in p)
    check('original_Manager_cadence_source_order',texts['Manager'].index('self.sim.step()',texts['Manager'].index('def step('))<texts['Manager'].index('self.reward_manager.compute(dt=self.step_dt)',texts['Manager'].index('def step('))<texts['Manager'].index('self.sim.forward()',texts['Manager'].index('def step(')))
    check('601states600solver150forward_exact_coverage','len(states) != 601 or len(solver) != 600 or forward_calls != 150' in p)
    check('preactionk_postactionkplus1_terminal150',"np.array_equal(phase, np.array([k]))" in p and "np.array_equal(postphase, np.array([k+1]))" in p and 'phase==LAST_INTERVAL' in r)
    check('terminal149_timeout_no_quality_early_exit',"if k < 149 and bool(truncated.any())" in p and "if bool(terminated.any()): raise FloatingPointError" in p and "reference_end'][0]) for e in events[:-1]" in p)
    check('151primary601diagnostic_distinct','primary = np.arange(0, 601, 4)' in p and "'max_object_position_error_151_m': pos[primary].max()" in p and "'max_object_position_error_601_m': pos.max()" in p)
    check('phase_step4u0_step5u1_u149_terminal','return 0 if index == 0 else (index - 1) // 4' in p and "indices = [0, *range(5, 598, 4)]" in p)
    check('reward150_capture_no_extra_query_or_forward',"cap = reader.reward_snapshot" in p and "physics.shape != (1, 58) or self15.shape != (1, 15)" in p and "result = self._original_reward(dt)" in texts['PenReader'] and "env.reward_manager.compute = reward" in texts['PenReader'])
    check('reward_contact_truth_is_self15_not_HO',"selfcollision15_is_not_handobject=True" in p and "'native_selfcollision15': self15[0]" in p and 'found = self.env.scene["finger_collision"].data.found > 0' in texts['PenReader'])
    check('force_contact_solver_preintegration_poststate_named',"'solver_t_s': states[-2]['raw_time_s']" in p and "'poststate_t_s': state['raw_time_s']" in p and "'solver_object_COM_w': host.xipos[obj].copy()" in p and 'just_completed_solver_preintegration_before_original_Manager_forward' in p)
    check('clock_bittrace_FP32_original3s_drift_contract','values[k + 1] = np.float32(values[k] + dt)' in p and 'expected.view(np.uint32)' in p and 'drift.max() <= 5e-5' in p and 'legacy_2us_diagnostic_pass' in p)
    check('fixed_mocap_all601_and_no_extra_objectwrites',"Fixed wrist moved during native episode" in p and not any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr.startswith('write_root') for x in ast.walk(pt)))
    check('all13_original_gate_names_present',all("'"+x+"'" in p for x in ('object_position_151','directed_shaft_151','actual_q_all601','penetration_solver600','no_nonhand_object_contacts','no_exogenous','all_actual_u_exact','actual_target_rate','clock','load_ratio','load_charge','drop','capacity')))
    check('partial_and_raw_preservation_before_rethrow','INCOMPLETE_OR_INVALID_ENGINEERING_UNKNOWN' in p and "retain('raw_states'" in p and "retain('native_reward_transitions'" in p and 'if error is not None: raise error' in p)
    check('native_finally_closes_and_persists_errors',"write(output / 'result.json', result)" in n and 'env.close()' in n and "FAILED_ENGINEERING_CLEANUP_UNQUALIFIED" in n)
    check('knownrough_not_promoted',"'physical_reference_qualified': False" in n and 'physical_reference_qualified=False' in p and "'old_checkpoint_migration': False" in n)
    check('source_postreview_unchanged',all(pin(paths[k])==v for k,v in sources.items()))
    anchors={
      'A0_no_Runner_creation':{'source':'native','line':line(n,"env = ManagerBasedRlEnv(cfg")},
      'actual8_before_reset':{'source':'native','line':line(n,'                    install_actual_context(env, bundle, output)')},
      'capacity_latch_constructor':{'source':'native','line':line(n,"observer._read('after_constructor_before_actual8')")},
      'poststep_sampling':{'source':'probe','line':line(p,'original_step(); sample(len(states))')},
      'real150_loop':{'source':'probe','line':line(p,'for k in range(150):')},
      'reward_capture':{'source':'probe','line':line(p,'cap = reader.reward_snapshot')},
      'terminal_coverage':{'source':'probe','line':line(p,'len(states) != 601 or len(solver) != 600 or forward_calls != 150')},
      'clock_bittrace':{'source':'probe','line':line(p,'def clock_audit(')}}
    report={'schema':'full151_native_A0_static_independent:1','status':'PASS_SOURCE_BOUND_STATIC_REVIEW' if all(checks.values()) else 'STATIC_CHECK_FAILURE_RETAINED','reviewer':pin(__file__),'sources':sources,'checks':checks,'passed':sum(checks.values()),'total':len(checks),'anchors':anchors,'wall_s':time.perf_counter()-tick,
      'blocking':[] if all(checks.values()) else [k for k,v in checks.items() if not v],
      'scope':['No production module imported or called. Source/AST inspection is not actual native runtime evidence.','A0 imports Runner class solely to install scoped hooks; no Runner/Actor/WM/optimizer instance is created. Hook construction is not training.','601 raw q/qvel poststates and600 solver contacts/force have different phases. At each20ms reward, physics x58 uses postq/qvel while self15 and torque use the last preintegration solver state, before original Manager.forward.','150 original Manager.forward calls are part of this new native cadence; constructor/actual8/reset forward work is separate. No claim of global zero physics when a device entry later runs.','Clock gate is predeclared FP32 recurrence equality plus50us drift for3s. Legacy2us is diagnostic only.','Qualityfail is retained across all150 intervals; only engineering/hard instability/unexpected early timeout interrupt. Overflow/unknown yields UNKNOWN, never valid taskFAIL/PASS.'],
      'production_imported':False,'Env_created':False,'new_FK_collision_query':False,'physics_steps':0,'GPU_used':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print({'status':report['status'],'passed':report['passed'],'total':report['total'],'report':pin(out/'report.json')},flush=True)
    return 0 if all(checks.values()) else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();raise SystemExit(main(a.output))
