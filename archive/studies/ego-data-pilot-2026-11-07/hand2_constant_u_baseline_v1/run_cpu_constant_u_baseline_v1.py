"""Matched constant-u baseline CPU diagnostic; no training or object aid.

All native forces/geometry/energy belong to PRE-step state; qpos/qvel and the
derived object COM belong to POST-step state. No extra forward inside rollout.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parent
DTS = [.0003125, .00015625]
DURATION = 2.
GATES = {
    'duration_s': 2., 'final_COM_drift_m': .005, 'final_angle_rad': .08726646259971647,
    'tail_duration_s': .5, 'tail_linear_velocity_RMS_m_s': .01,
    'tail_angular_velocity_RMS_rad_s': .2, 'max_native_penetration_m': .002,
    'tail_each_pad_normal_force_N': .1, 'tail_dual_pad_force_coverage': .95,
    'max_phone_nonpad_normal_force_N': .1, 'max_self_penetration_m': .0002,
    'max_joint_limit_excursion_rad': .001, 'max_table_normal_force_N': 1e-8,
    'warning_count': 0, 'abort_COM_drift_m': .05, 'abort_angle_rad': .7853981633974483,
}


def pin(p):
    p = Path(p).resolve()
    return {'path': str(p), 'bytes': p.stat().st_size,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def checked(p):
    q = Path(p['path'])
    if pin(q) != p:
        raise ValueError('Changed input: '+str(q))
    return q


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def write(p, x):
    with Path(p).open('x', encoding='utf8') as f:
        json.dump(x, f, indent=2, allow_nan=False)


SCHEMA = 'hand2_phone_matched_constant_u_baseline_CPU_protocol:1'
ACCEL_GATE_KEYS = {'COM_acceleration_infinity_m_s2','angular_acceleration_infinity_rad_s2','joint_acceleration_infinity_rad_s2'}
PAIR_GATES = {
    'final_COM_difference_m': .002,
    'final_orientation_difference_rad': .03490658503988659,
    'require_both_duration_complete': True,
}


def finite_vector(x, size):
    return (isinstance(x, list) and len(x) == size
            and all(type(v) in (int, float) and math.isfinite(v) for v in x))


NATIVE_GATE_KEYS = {
    'COM_acceleration_infinity_m_s2','angular_acceleration_infinity_rad_s2',
    'joint_acceleration_infinity_rad_s2','native_other_penetration_m',
    'native_other_normal_force_N','native_finite_no_warnings',
    'native_two_pads_loaded','native_pad_budget','native_pad_compression',
    'requested_ctrl_unclamped',
}
REPEAT_KEYS = {'qpos27','ctrl20','qacc26','qfrc_actuator26','qfrc_constraint26',
               'a_COM_world_m_s2','alpha_world_rad_s2','contact_rows24'}


def all_true_exact(value, keys):
    return isinstance(value,dict) and set(value)==keys and all(v is True for v in value.values())


def required_native_gates(value):
    return (isinstance(value,dict) and set(value)==NATIVE_GATE_KEYS
            and all(type(v) is bool for v in value.values())
            and all(value[k] is True for k in NATIVE_GATE_KEYS-ACCEL_GATE_KEYS))


def metadata_preflight(protocol_path):
    protocol=read(protocol_path)
    if protocol['baseline_selection']!={'seed':0,'rank':0} or set(protocol['acceleration_descriptors_only'])!=ACCEL_GATE_KEYS:
        raise ValueError('Fixed baseline and exactly three descriptive acceleration metrics')
    if protocol['schema']!=SCHEMA:
        raise ValueError('New native-equilibrium holding protocol namespace')
    if protocol['gates']!=GATES or protocol['dt_s']!=DTS or protocol['duration_s']!=DURATION:
        raise ValueError('Unchanged original physical budget/gates')
    if protocol['dt_pair_gates']!=PAIR_GATES:
        raise ValueError('Unchanged2mm/2degree saved endpoint dt comparison')
    if (type(protocol['CPU_run_authorized']) is not bool
            or protocol['training_or_GPU_authorized'] is not False):
        raise ValueError('Explicit CPU-only boundary')
    if set(protocol['inputs'])!={'equilibrium_protocol','equilibrium_report','model_binding','model_mjb'}:
        raise ValueError('Exact four native-equilibrium semantic inputs')
    for rec in [*protocol['inputs'].values(),*protocol['source_pins'].values()]:
        checked(rec)
    if protocol['source_pins']['runner']!=pin(__file__):
        raise ValueError('Current runner')
    if protocol['source_pins']['old_holding_runner']!=pin(ROOT.parent/'hand2_holding_v1/run_cpu_holding_v1.py'):
        raise ValueError('Original physics loop authority')
    output=Path(protocol['output'])
    if not output.is_absolute() or output.exists() or output.parent.resolve()!=ROOT.resolve():
        raise ValueError('Absolute fresh holding_v4 output')
    pose=read(checked(protocol['inputs']['equilibrium_report']))
    cp=read(checked(protocol['inputs']['equilibrium_protocol']))
    if (pose.get('schema')!='native_soft_contact_equilibrium_result:6'
            or pose.get('status')!='COMPLETED_BOUNDED_NATIVE_STATIC_SEARCH'
            or pose.get('protocol')!=protocol['inputs']['equilibrium_protocol']
            or pose.get('integration_training_GPU')!=0
            or cp.get('schema')!='native_soft_contact_equilibrium_protocol:6'
            or cp.get('training_GPU_integration_authorized') is not False):
        raise ValueError('Completed matching native-equilibrium static search only')
    # The actual producer spelling is model_audit, not globalmodel_audit.
    audit=pose.get('model_audit',{})
    if audit.get('passed') is not True or not isinstance(audit.get('model_digest'),str):
        raise ValueError('Global endpoint model audit must pass')
    for rec in [*cp['inputs'].values(),*cp['source_pins'].values()]:
        checked(rec)
    binding=read(checked(protocol['inputs']['model_binding']))
    if (binding['model']!=protocol['inputs']['model_mjb']
            or binding['source']!=protocol['source_pins']['scene']
            or protocol['inputs']['model_binding'] not in cp['inputs'].values()
            or protocol['inputs']['model_mjb'] not in cp['inputs'].values()):
        raise ValueError('Same original explicit-pair compiled model, directly bound by native protocol')
    if cp.get('pad_compression_limit_m')!=.0001:
        raise ValueError('New designated-pad initial compression assumption exactly0.1mm')
    seeds=pose['seed_results']
    if len(seeds)!=2 or sorted(v['seed'] for v in seeds)!=[0,1]:
        raise ValueError('Both predefined native search seeds retained')
    variants,screening=[],[]
    total_qualified=0
    for seed in seeds:
        shortlist=seed['shortlist']
        ranks=[c['rank'] for c in shortlist]
        if any(type(v) is not int or v<0 for v in ranks) or len(ranks)!=len(set(ranks)):
            raise ValueError('Distinct nonnegative frozen shortlist ranks')
        candidates=[];selected_rank=None
        for candidate in sorted(shortlist,key=lambda c:c['rank']):
            reasons=[]
            if (seed['seed'],candidate['rank'])!=(0,0):reasons.append('Not preregistered seed0rank0 constant-u baseline; no rank substitution')
            if candidate.get('unknown')!=[]:reasons.append('candidate unknown not empty')
            if not all_true_exact(candidate.get('fresh_repeat_bit_equal'),REPEAT_KEYS):
                reasons.append('complete eight-field fresh repeat not bit-equal')
            if not required_native_gates(candidate.get('native_entry_gates')):
                reasons.append('required nonacceleration primary entry gates not allTrue')
            if not required_native_gates(candidate.get('fresh_repeat_entry_gates')):
                reasons.append('required nonacceleration repeat entry gates not allTrue')
            native=candidate.get('native',{});raw=native.get('raw',{})
            repeat=candidate.get('fresh_repeat_native',{})
            if native.get('unknown')!=[] or repeat.get('unknown')!=[]:
                reasons.append('native or repeated evidence unknown')
            if native.get('model_parameter_digest')!=audit['model_digest'] or repeat.get('model_parameter_digest')!=audit['model_digest']:
                reasons.append('native/endpoint model digest identity mismatch')
            geometry=candidate.get('geometry',{});state=geometry.get('state',{})
            if (geometry.get('geometry_gate_pass') is not True
                    or geometry.get('complete22_189') is not True
                    or geometry.get('new_pad_gap_hypothesis_m')!=[-.0001,.0002]
                    or geometry.get('constraints',{}).get('all_satisfied') is not True
                    or state.get('unknown')!=[]
                    or len(state.get('phone_geometry_rows',[]))!=22
                    or len(state.get('self_geometry_rows',[]))!=189):
                reasons.append('complete independent22/189/new-compression geometry not qualified')
            x=candidate.get('x46')
            if not finite_vector(x,46):reasons.append('finite x46 missing')
            if not finite_vector(raw.get('qpos27'),27):reasons.append('finite native qpos27 missing')
            if not finite_vector(raw.get('ctrl20'),20):reasons.append('finite native ctrl20 missing')
            if not finite_vector(raw.get('qvel26'),26) or any(raw.get('qvel26',[1])):
                reasons.append('native velocity must be zero26')
            if finite_vector(x,46) and raw.get('ctrl20')!=x[26:]:
                reasons.append('x46 ctrl versus native raw ctrl mismatch')
            selected=candidate.get('state')
            if not isinstance(selected,dict):reasons.append('candidate state pin missing')
            else:
                checked(selected)
                if Path(selected['path']).parent.resolve()!=Path(protocol['inputs']['equilibrium_report']['path']).parent.resolve():
                    raise ValueError('Candidate state outside matching native search output')
            qualified=not reasons
            if candidate.get('candidate_pass') is True:total_qualified+=1
            chosen=qualified and selected_rank is None
            if chosen:selected_rank=candidate['rank']
            candidates.append({'rank':candidate['rank'],'enrolled_for_matched_constant_u_baseline':qualified,
                'original_static_candidate_pass':candidate.get('candidate_pass'),
                'acceleration_descriptors_only':{tag:{k:candidate.get(key,{}).get(k) for k in sorted(ACCEL_GATE_KEYS)} for tag,key in [('primary','native_entry_gates'),('repeat','fresh_repeat_entry_gates')]},
                'selected':chosen,'reasons':reasons,
                'selection_status':'FIRST_PASSING_RANK_SELECTED' if chosen else 'QUALIFIED_LATER_RANK_NOT_SELECTED_PREDECLARED_RULE' if qualified else 'REJECTED_EVIDENCE',
                'state':selected})
            if chosen:
                variants.append({'name':'matched_constant_u_seed0_rank0',
                    'seed':seed['seed'],'rank':candidate['rank'],'reachable':True,'ctrl20':x[26:],
                    'candidate_state':selected,'verified_qpos27':raw['qpos27'],
                    'verified_qvel26':raw['qvel26'],'verified_ctrl20':raw['ctrl20'],'verified_x46':x,
                    'moment_infinity_Nm':None,'strict_support_proposal_pass':None})
        screening.append({'seed':seed['seed'],'selected_rank':selected_rank,'enrolled':selected_rank is not None,
            'shortlist_screening':candidates,'empty_shortlist':not shortlist})
        if seed.get('qualified_candidate_count')!=sum(c.get('candidate_pass') is True for c in shortlist):
            raise ValueError('Full per-seed producer qualification count consistent')
    if pose.get('qualified_candidate_count')!=total_qualified:
        raise ValueError('Full global producer qualification count consistent')
    return protocol,output,binding,variants,screening


def dt_pair_summary(variants, cases, endpoints, np):
    pairs = []
    for variant in variants:
        both = [v for v in cases if v['name'] == variant['name']]
        if len(both) != 2 or sorted(v['dt_s'] for v in both) != sorted(DTS):
            raise ValueError('Each enrolled seed must retain both dt results')
        e = sorted(endpoints[variant['name']], key=lambda v: v['dt_s'], reverse=True)
        quat = [np.asarray(v['final_quaternion_wxyz'], dtype=float) for v in e]
        norms = [float(np.linalg.norm(v)) for v in quat]
        if not all(math.isfinite(v) and v > 0 for v in norms):
            raise ValueError('Finite nonzero terminal quaternion')
        unit = [v/n for v,n in zip(quat,norms)]
        angle = float(2*np.arccos(np.clip(abs(unit[0]@unit[1]), 0, 1)))
        drift = float(np.linalg.norm(np.asarray(e[0]['final_COM_world_m'])-e[1]['final_COM_world_m']))
        complete = all(v['gates']['duration_complete'] for v in both)
        gates = {'both_duration_complete': complete,
                 'final_COM_difference_m': drift <= PAIR_GATES['final_COM_difference_m'],
                 'final_orientation_difference_rad': angle <= PAIR_GATES['final_orientation_difference_rad']}
        pairs.append({'name': variant['name'], 'seed': variant['seed'],
            'comparison_status': 'TWO_COMPLETE_2S_ENDPOINTS' if complete else 'INCOMPLETE_DURATION_NOT_2S_CONSISTENCY',
            'final_COM_difference_m': drift, 'final_orientation_difference_rad': angle,
            'terminal_quaternions_raw_wxyz': [v.tolist() for v in quat],
            'terminal_quaternion_norms_F64': norms,
            'orientation_observer': 'F64 endpoint normalization for pair comparison only; no state repair',
            'endpoints': e, 'gates': gates, 'dt_pair_consistency_pass': all(gates.values()),
            'both_original_case_gates_pass': all(v['nominal_holding_dynamics_pass'] for v in both)})
    return pairs


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--protocol', type=Path, required=True)
    a.add_argument('--run', action='store_true')
    args = a.parse_args()
    protocol, output, binding, variants, screening = metadata_preflight(args.protocol)
    if not args.run:
        print(json.dumps({'status': 'METADATA_VALID_NO_MODEL_OR_STEPS', 'screening': screening,
                          'variants': [v['name'] for v in variants],
                          'maximum_steps': int(sum(DURATION/dt for dt in DTS)*len(variants))}))
        return
    if protocol['CPU_run_authorized'] is not True:
        raise ValueError('Root explicit CPU run authorization required')
    output.mkdir()
    write(output/'protocol.json', protocol)
    shutil.copyfile(__file__, output/'execution_source.py')
    started = time.perf_counter()
    counts = {'MJB_reads': 0, 'MjData': 0, 'initial_forward': 0, 'integration_steps': 0,
              'contact_force_reads': 0, 'neural_training_GPU': 0}
    if not variants:
        write(output/'report.json', {'schema': 'hand2_phone_matched_constant_u_baseline_CPU_result:1',
            'status': 'BLOCKED_FIXED_BASELINE_EVIDENCE_NO_MODEL_OR_STEPS', 'protocol': pin(args.protocol),
            'screening': screening, 'counts': counts, 'cases': [], 'dt_pair_consistency': [],
            'all_original_cases_and_dt_pairs_pass': False, 'holding_or_ego_policy_qualified': False})
        return
    phase = 'backend_import'
    try:
        import mujoco
        import numpy as np
        if mujoco.__version__ != '3.11.0':
            raise ValueError('Binary/source version')
        model = mujoco.MjModel.from_binary_path(str(checked(protocol['inputs']['model_mjb'])))
        counts['MJB_reads'] += 1
        if (model.nq, model.nv, model.nu, model.neq, model.nmocap) != (27, 26, 20, 0, 0):
            raise ValueError('Exact fixed-wrist,20joints,free-phone,no equality/mocap')
        if not np.array_equal(model.body_gravcomp, np.zeros(model.nbody)):
            raise ValueError('No body gravity compensation')
        phone = model.body('hot3d_phone24').id
        phone_geom = model.geom('phone24_collision_convex').id
        table = model.geom('stage1_single_table').id
        pads = [model.geom(f'right_finger{i}_tip_sensor_col').id for i in (1,2)]
        qadr = int(model.jnt_qposadr[model.body_jntadr[phone]])
        vadr = int(model.jnt_dofadr[model.body_jntadr[phone]])
        if (qadr, vadr) != (20, 20) or model.body_mass[phone] != .2:
            raise ValueError('Phone layout/nominal mass')
        if len(binding['explicit_pairs']) != 22 or binding['mu_phone_hand'] != .7 or binding['mu_phone_table'] != .6:
            raise ValueError('Nominal actual materials')
        if (model.opt.tolerance, model.opt.ls_tolerance, model.opt.iterations, model.opt.ls_iterations) != (1e-6, .01, 100, 20):
            raise ValueError('Frozen effective solver settings')
        if (int(model.opt.integrator), int(model.opt.solver), int(model.opt.cone)) != (
                int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST),int(mujoco.mjtSolver.mjSOL_NEWTON),int(mujoco.mjtCone.mjCONE_PYRAMIDAL)):
            raise ValueError('Frozen algorithm settings')
        if not np.array_equal(model.actuator_trnid[:,0], np.arange(20)) or not np.array_equal(model.actuator_gear[:,0], np.ones(20)):
            raise ValueError('Direct unit-gear20-joint actuators')
        qlimits = model.jnt_range[:20].copy()
        R = np.empty(9)
        def COM(q):
            mujoco.mju_quat2Mat(R, q[23:27])
            return q[20:23]+R.reshape(3,3)@model.body_ipos[phone]
        def angle(q):
            return float(2*np.arccos(np.clip(abs(q[23:27]@q0[23:27]),0,1)))
        cases = []
        endpoints = {v['name']: [] for v in variants}
        for variant in variants:
            # Each eligible seed has its own pose, COM0 and quaternion0.
            with np.load(checked(variant['candidate_state']), allow_pickle=False) as z:
                q0 = z['qpos27'].copy()
                qv0 = z['qvel26'].copy()
                state_ctrl = z['ctrl20'].copy()
                state_x46 = z['x46'].copy()
            expected={'qpos27':variant['verified_qpos27'],'qvel26':variant['verified_qvel26'],
                      'ctrl20':variant['verified_ctrl20'],'x46':variant['verified_x46']}
            actual={'qpos27':q0,'qvel26':qv0,'ctrl20':state_ctrl,'x46':state_x46}
            for key,raw in actual.items():
                ref=np.asarray(expected[key],dtype=np.float64)
                if raw.dtype!=np.dtype('float64') or raw.shape!=ref.shape or raw.tobytes()!=ref.tobytes():
                    raise ValueError('NPZ/native/x46 exact raw binding: '+key)
            if not np.array_equal(state_x46[:20],q0[:20]) or not np.array_equal(state_x46[26:],state_ctrl):
                raise ValueError('NPZ x46 q/control ordering')
            if (q0.shape != (27,) or qv0.shape != (26,)
                    or not np.isfinite(q0).all() or not np.isfinite(qv0).all()
                    or np.any(qv0) or not np.array_equal(q0, np.asarray(variant['verified_qpos27'], dtype=float))):
                raise ValueError('Exact finite original rank0 pose and zero speed')
            if (np.any(q0[:20] < qlimits[:,0]) or np.any(q0[:20] > qlimits[:,1])
                    or abs(float(q0[23:27]@q0[23:27])-1.) > 1e-12):
                raise ValueError('Original real joint ranges and valid native quaternion')
            COM0 = COM(q0).copy()
            ctrl = np.asarray(variant['ctrl20'], dtype=float)
            if ctrl.shape != (20,) or not np.isfinite(ctrl).all():
                raise ValueError('Finite control20')
            limited = model.actuator_ctrllimited.astype(bool)
            if np.any(ctrl[limited] < model.actuator_ctrlrange[limited,0]) or np.any(ctrl[limited] > model.actuator_ctrlrange[limited,1]):
                raise ValueError('Never silently clamp unreachable controller')
            for dt in DTS:
                case_name = variant['name']+'__dt'+str(round(dt*1e9))+'ns'
                case_dir = output/case_name
                case_dir.mkdir()
                phase = case_name
                model.opt.timestep = dt
                data = mujoco.MjData(model)
                counts['MjData'] += 1
                data.qpos[:] = q0
                data.qvel[:] = 0
                data.ctrl[:] = ctrl
                mujoco.mj_forward(model, data)
                counts['initial_forward'] += 1
                if np.any(data.xfrc_applied) or np.any(data.qfrc_applied):
                    raise ValueError('No applied forces')
                steps = round(DURATION/dt)
                sample_stride = round(.01/dt)
                rows = {k: [] for k in ['post_time','pre_time_from_integer_tick','post_COM','post_angle','post_qpos','post_qvel',
                    'pre_native_energy','pre_ncon','pre_pad_normal_N','pre_phone_nonpad_normal_N',
                    'pre_table_normal_N','pre_penetration_m','pre_self_penetration_m',
                    'pre_qfrc_actuator','pre_actuator_force','pre_solver_niter']}
                contact_rows = []
                geometry_samples = []
                warnings_initial = np.asarray(data.warning.number).copy()
                if np.any(warnings_initial):
                    raise ValueError('Initial native warning')
                terminate = 'COMPLETE'
                attempted_steps = 0
                for tick in range(steps):
                    mujoco.mj_step(model, data)
                    counts['integration_steps'] += 1
                    attempted_steps += 1
                    values = [data.time, data.qpos, data.qvel, data.qacc, data.energy, data.qfrc_actuator, data.actuator_force]
                    if not all(np.isfinite(x).all() for x in values):
                        terminate = 'ABORT_NONFINITE'
                        break
                    if np.any(data.xfrc_applied) or np.any(data.qfrc_applied) or not np.array_equal(data.ctrl,ctrl):
                        raise ValueError('Object aid or controller mutation')
                    pad_force = np.zeros(2)
                    other_force = table_force = penetration = self_pen = 0.
                    for ci in range(data.ncon):
                        con = data.contact[ci]
                        if not np.isfinite(np.r_[con.dist,con.pos,con.frame]).all():
                            raise ValueError('Nonfinite native contact distance/position/frame')
                        g1,g2 = map(int,con.geom)
                        force = np.empty(6)
                        mujoco.mj_contactForce(model,data,ci,force)
                        counts['contact_force_reads'] += 1
                        if not np.isfinite(force).all():
                            raise ValueError('Nonfinite native local contact wrench')
                        penetration = max(penetration,-float(con.dist))
                        if phone_geom in (g1,g2):
                            other = g2 if g1==phone_geom else g1
                            if other in pads:
                                pad_force[pads.index(other)] += max(0.,float(force[0]))
                            elif other == table:
                                table_force += max(0.,float(force[0]))
                            else:
                                other_force += max(0.,float(force[0]))
                        elif table not in (g1,g2):
                            self_pen = max(self_pen,-float(con.dist))
                        contact_rows.append(np.r_[tick,g1,g2,int(con.dim),int(con.efc_address),con.dist,con.pos,con.frame,force])
                    c = COM(data.qpos).copy()
                    a = angle(data.qpos)
                    for k,v in zip(rows,[float(data.time),tick*dt,c,a,data.qpos.copy(),data.qvel.copy(),data.energy.copy(),
                            int(data.ncon),pad_force,other_force,table_force,penetration,self_pen,
                            data.qfrc_actuator.copy(),data.actuator_force.copy(),np.asarray(data.solver_niter).copy()]):
                        rows[k].append(v)
                    if tick%sample_stride == 0:
                        geometry_samples.append({'pre_tick':tick,'geom_xpos':data.geom_xpos.copy(),
                            'geom_xmat':data.geom_xmat.copy(),'pre_qpos':q0.copy() if tick==0 else rows['post_qpos'][-2].copy()})
                    if np.any(data.warning.number):
                        terminate = 'ABORT_NATIVE_WARNING'
                        break
                    if np.linalg.norm(c-COM0)>GATES['abort_COM_drift_m'] or a>GATES['abort_angle_rad']:
                        terminate = 'ABORT_LOSS_OF_HOLD'
                        break
                r = {k: np.asarray(v) for k,v in rows.items()}
                completed = len(r['post_time'])
                if not completed:
                    raise ValueError('No finite completed record')
                tail_start_tick = round((DURATION-GATES['tail_duration_s'])/dt)
                tail = np.arange(1,completed+1) >= tail_start_tick
                pre_tail = np.arange(completed) >= tail_start_tick
                qvio = np.maximum(qlimits[:,0]-r['post_qpos'][:,:20],0)+np.maximum(r['post_qpos'][:,:20]-qlimits[:,1],0)
                lin = r['post_qvel'][tail,20:23]
                omega = r['post_qvel'][tail,23:26]
                metrics = {
                    'final_COM_drift_m': float(np.linalg.norm(r['post_COM'][-1]-COM0)),
                    'final_angle_rad': float(r['post_angle'][-1]),
                    'tail_linear_velocity_RMS_m_s': float(np.sqrt(np.mean(np.sum(lin*lin,axis=1)))) if len(lin) else None,
                    'tail_angular_velocity_RMS_rad_s': float(np.sqrt(np.mean(np.sum(omega*omega,axis=1)))) if len(omega) else None,
                    'tail_dual_pad_force_coverage': float(np.mean(np.all(r['pre_pad_normal_N'][pre_tail]>.1,axis=1))) if np.any(pre_tail) else None,
                    'max_native_penetration_m': float(r['pre_penetration_m'].max()),
                    'max_self_penetration_m': float(r['pre_self_penetration_m'].max()),
                    'max_phone_nonpad_normal_force_N': float(r['pre_phone_nonpad_normal_N'].max()),
                    'max_table_normal_force_N': float(r['pre_table_normal_N'].max()),
                    'max_joint_limit_excursion_rad': float(qvio.max()),
                    'warning_count': int(np.sum(data.warning.number)),
                }
                gates = {'duration_complete': terminate=='COMPLETE' and completed==steps}
                for k,v in metrics.items():
                    gates[k] = v is not None and (v>=GATES[k] if k=='tail_dual_pad_force_coverage' else v<=GATES[k])
                np.savez_compressed(case_dir/'trace.npz',**r,ctrl20=ctrl,initial_qpos27=q0,
                    initial_COM_world3=COM0.copy(),
                    contact_rows=np.asarray(contact_rows,dtype=float).reshape(-1,24),
                    sample_pre_tick=np.asarray([v['pre_tick'] for v in geometry_samples]),
                    sample_pre_qpos=np.asarray([v['pre_qpos'] for v in geometry_samples]),
                    sample_pre_geom_xpos=np.asarray([v['geom_xpos'] for v in geometry_samples]),
                    sample_pre_geom_xmat=np.asarray([v['geom_xmat'] for v in geometry_samples]))
                summary = {'name':variant['name'],'dt_s':dt,'termination':terminate,'integration_steps':attempted_steps,
                    'finite_steps_saved':completed,
                    'actual_final_time_s':float(data.time),'metrics':metrics,'gates':gates,
                    'nominal_holding_dynamics_pass':all(gates.values()),'warnings':np.asarray(data.warning.number).tolist(),
                    'trace':pin(case_dir/'trace.npz'),'energy_force_geometry_phase':'PRE(step tick*dt); qpos/qvel/COM POST((tick+1)*dt)',
                    'contact_row_columns':['pre_tick','geom1','geom2','dim','efc_address','distance_m','position3','frame9','local_wrench6'],
                    'no_external_forces_equality_mocap_or_pose_reset':True,'training_GPU':False}
                summary.update(seed=variant['seed'], source_candidate_state=variant['candidate_state'],
                    initial_COM_world_m=COM0.tolist(), initial_phone_quaternion_wxyz=q0[23:27].tolist(),
                    native_equilibrium_rank=variant['rank'],
                    enrollment='Registered constant-u baseline: fixedrank0, all nonacceleration entry/repeat/audit/22+189 gates; three static accelerations descriptive only')
                endpoints[variant['name']].append({'dt_s':dt, 'final_COM_world_m':r['post_COM'][-1].tolist(),
                    'final_quaternion_wxyz':r['post_qpos'][-1,23:27].tolist(),
                    'actual_final_time_s':float(data.time), 'termination':terminate,
                    'finite_steps_saved':completed, 'original_case_pass':summary['nominal_holding_dynamics_pass']})
                write(case_dir/'summary.json',summary)
                cases.append(summary)
                print(json.dumps({'case':case_name,'termination':terminate,'steps':completed,'metrics':metrics,'pass':summary['nominal_holding_dynamics_pass']}),flush=True)
        pairs = dt_pair_summary(variants, cases, endpoints, np)
        report = {'schema':'hand2_phone_matched_constant_u_baseline_CPU_result:1',
            'status':'COMPLETED_MATCHED_CONSTANT_U_BASELINE_BOTH_DT',
            'protocol':pin(args.protocol),'counts':counts,'screening':screening,
            'cases':cases,'dt_pair_consistency':pairs,'elapsed_seconds':time.perf_counter()-started,
            'all_original_cases_and_dt_pairs_pass':bool(cases and all(v['nominal_holding_dynamics_pass'] for v in cases)
                and all(v['dt_pair_consistency_pass'] for v in pairs)),
            'mass_or_friction_calibrated':False,'ego_trajectory_or_policy_qualified':False,
            'scope':'Fixed original seed0rank0 constant-u baseline, both fixed dt, unchanged original2s case gates. '
                'New designated-pad initial compression up to0.1mm is a hypothesis, not measured compliance; no old point LP or positive-gap condition. '
                'No object aid, pose reset, materials/gains/solver tuning, sample selection or retries. '
                'Dt endpoint consistency is an additional saved comparison and never replaces individual case gates.'}
        write(output/'report.json',report)
    except BaseException as error:
        write(output/'failure.json',{'phase':phase,'error':repr(error),'counts':counts,'automatic_retry':False})
        raise


if __name__ == '__main__':
    main()
