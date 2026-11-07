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


SCHEMA = 'hand2_source_mapping_constant_u_diagnostic_protocol:11'
ACCEL_GATE_KEYS = {'COM_acceleration_infinity_m_s2','angular_acceleration_infinity_rad_s2','joint_acceleration_infinity_rad_s2'}
PAIR_GATES = {
    'final_COM_difference_m': .002,
    'final_orientation_difference_rad': .03490658503988659,
    'require_both_duration_complete': True,
}


def finite_vector(x, size):
    return (isinstance(x, list) and len(x) == size
            and all(type(v) in (int, float) and math.isfinite(v) for v in x))


BRANCH_INPUTS = {
    'exact_source_original_hand': ('exact_source_protocol','exact_source_report'),
    'source1mm_opposed_side_original_hand': ('sensitivity_protocol','sensitivity_report'),
}
REPEAT_FIELDS = ('qpos27','ctrl20','qacc26','qfrc_actuator26','qfrc_constraint26',
                 'a_COM_world_m_s2','alpha_world_rad_s2')

def raw_bytes(value):
    import struct
    if isinstance(value,list):return b''.join(raw_bytes(v) for v in value)
    return struct.pack('<d',float(value))

def valid_matrix(x,rows,cols):
    return isinstance(x,list) and len(x)==rows and all(finite_vector(row,cols) for row in x)

def source_choices(protocol):
    expected={'model_binding','model_mjb','exact_source_protocol','exact_source_report',
              'sensitivity_protocol','sensitivity_report'}
    if set(protocol['inputs'])!=expected:raise ValueError('Exact six source/model inputs')
    for rec in [*protocol['inputs'].values(),*protocol['source_pins'].values()]:checked(rec)
    if protocol['source_pins']['runner']!=pin(__file__):raise ValueError('Current diagnostic runner')
    binding=read(checked(protocol['inputs']['model_binding']))
    if binding['model']!=protocol['inputs']['model_mjb'] or binding['source']!=protocol['source_pins']['scene']:
        raise ValueError('Original fixed compiled physical model and scene')
    choices=[];screening=[]
    for branch,(pk,rk) in BRANCH_INPUTS.items():
        cp=read(checked(protocol['inputs'][pk]));report=read(checked(protocol['inputs'][rk]))
        if (cp.get('schema')!='assembly_then_hard_geometry_support_protocol:11'
                or cp.get('anchor_branch')!=branch
                or report.get('schema')!='assembly_then_hard_geometry_support_result:11'
                or report.get('status')!='COMPLETED_BOUNDED_TWO_SEED_STAGED_SEARCH'
                or report.get('anchor_branch')!=branch or report.get('protocol')!=protocol['inputs'][pk]
                or report.get('new_steps_training_GPU')!=0):
            raise ValueError('Complete matching source-mapping result, never partial live output')
        for rec in [*cp['inputs'].values(),*cp['source_pins'].values()]:checked(rec)
        if (cp['inputs']['model_binding']!=protocol['inputs']['model_binding']
                or cp['inputs']['model_mjb']!=protocol['inputs']['model_mjb']):
            raise ValueError('Matching source model binding')
        anchors=read(checked(cp['inputs']['surface_anchors']))
        if anchors.get('branch')!=branch:raise ValueError('Named source mapping anchor branch')
        audit=report.get('model_audit',{})
        if audit.get('passed') is not True or report.get('cheap_model_audit',{}).get('passed') is not True:
            raise ValueError('Complete source model audits')
        records=[];eligible=[]
        seeds=report['seed_results']
        if sorted(s['seed'] for s in seeds)!=[0,1]:raise ValueError('Both predefined seeds must complete')
        for seed in seeds:
            for final in seed['finals']:
                candidate_pin=final['result'];record=read(checked(candidate_pin));v=record['candidate'];k=v['kinematics']
                reasons=[];rank=record['rank']
                if record['seed']!=seed['seed'] or rank!=final['rank']:raise ValueError('Source candidate identity')
                if rank!=0:reasons.append('Only each source seed rank0 is preregistered for selection')
                geo=record.get('final_geometry',{});gs=geo.get('state',{});cons=geo.get('constraints',{})
                if (v.get('assembly_gate') is not True or final.get('assembly_gate') is not True
                        or not finite_vector(v.get('eq10_scaled'),10)
                        or max(abs(x) for x in v['eq10_scaled'])>1e-5):
                    reasons.append('Source assembly/hard equation evidence did not pass')
                if (geo.get('geometry_gate_pass') is not True or geo.get('complete22_189') is not True
                        or geo.get('new_pad_gap_hypothesis_m')!=[-.0001,.0002]
                        or len(gs.get('phone_geometry_rows',[]))!=22 or len(gs.get('self_geometry_rows',[]))!=189
                        or gs.get('unknown')!=[] or cons.get('unknown') is not False
                        or cons.get('all_satisfied') is not True
                        or not isinstance(cons.get('values_m'),list) or len(cons['values_m'])!=213
                        or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in cons['values_m'])):
                    reasons.append('Complete source22/189 compression-window geometry did not pass')
                strict=record.get('strict_native_support_LP')
                strict_pass=bool(isinstance(strict,dict) and strict.get('strict_force_moment_pass') is True)
                original_holding_flag=record.get('holding_diagnostic_eligible')
                probes=record.get('fresh_command_probes',[])
                control_origin='saved_native_probe_control'
                native=record.get('native_probe')
                if original_holding_flag is True:
                    control_origin='strict_witness_fresh_command0'
                    if (len(probes)!=2 or not strict_pass or strict.get('hard_bounds_pass') is not True
                            or not record.get('native_entry') or any(v is not True for v in record['native_entry'].values())):
                        reasons.append('Qualified strict control lacks its complete two-fresh source evidence')
                    else:
                        native=probes[0]
                        for name in REPEAT_FIELDS:
                            if raw_bytes(probes[0]['raw'][name])!=raw_bytes(probes[1]['raw'][name]):
                                reasons.append('Qualified command fresh raw repeat mismatch '+name)
                        if (raw_bytes([row[1:] for row in probes[0]['raw']['contact_rows24']])
                                !=raw_bytes([row[1:] for row in probes[1]['raw']['contact_rows24']])):
                            reasons.append('Qualified command fresh physical contact repeat mismatch')
                        if raw_bytes(native['raw']['ctrl20'])!=raw_bytes(strict['proposed_ctrl20']):
                            reasons.append('Strict witness and source command control differ')
                        if any(p.get('unknown')!=[] or any(p.get('warnings_number',[1])) for p in probes):
                            reasons.append('Qualified command repeat unknown or warning')
                if not isinstance(native,dict):reasons.append('Native source probe missing');native={}
                raw=native.get('raw',{});pd=native.get('PD',{});context=native.get('contact_design',{}).get('PD_raw',{})
                if native.get('unknown')!=[] or any(native.get('warnings_number',[1])):
                    reasons.append('Native evidence unknown or warning')
                if native.get('model_parameter_digest')!=audit.get('model_digest'):
                    reasons.append('Native versus global source model digest mismatch')
                for name,size in [('qpos27',27),('qvel26',26),('ctrl20',20)]:
                    if not finite_vector(raw.get(name),size):reasons.append('Finite source raw '+name+' missing')
                if finite_vector(raw.get('qvel26'),26) and any(raw['qvel26']):reasons.append('Source velocity is not zero26')
                if (not finite_vector(k.get('qpos27'),27) or (finite_vector(raw.get('qpos27'),27)
                        and raw_bytes(raw['qpos27'])!=raw_bytes(k['qpos27']))):
                    reasons.append('Chosen native full qpos27 versus fixed assembly/geometry pose mismatch')
                ranges=context.get('actuator_ctrlrange20x2')
                if not valid_matrix(ranges,20,2) or context.get('actuator_ctrllimited20')!=[True]*20:
                    reasons.append('Explicit native control-range evidence missing')
                elif finite_vector(raw.get('ctrl20'),20) and any(u<r[0] or u>r[1] for u,r in zip(raw['ctrl20'],ranges)):
                    reasons.append('Source control outside actual native ranges; no clamp')
                if (not finite_vector(pd.get('requested_ctrl20'),20) or not finite_vector(pd.get('effective_ctrl20'),20)
                        or raw_bytes(pd['requested_ctrl20'])!=raw_bytes(pd['effective_ctrl20'])
                        or (finite_vector(raw.get('ctrl20'),20) and raw_bytes(raw['ctrl20'])!=raw_bytes(pd['requested_ctrl20']))):
                    reasons.append('Requested/raw/effective control mismatch')
                for name,values_name,limited_name in [('actuator_forcerange20x2','actual_actuator_force20','actuator_forcelimited20'),
                        ('joint_actfrcrange20x2','actual_generalized_actuator_force26','joint_actfrclimited20')]:
                    rg=context.get(name);vv=context.get(values_name)
                    if (not valid_matrix(rg,20,2) or context.get(limited_name)!=[True]*20
                            or not isinstance(vv,list) or len(vv)<20
                            or any(not math.isfinite(float(x)) or x<r[0]-1e-12 or x>r[1]+1e-12 for x,r in zip(vv[:20],rg or []))):
                        reasons.append('Actual native torque/force cap evidence '+name)
                ag=native.get('native_contact_aggregates',{})
                def fn(name):return ag.get(name,{}).get('normal_force_signed_N')
                if any(type(fn(name)) not in (int,float) or not math.isfinite(fn(name)) or fn(name)<=.1 for name in ['thumb_pad','index_pad']):
                    reasons.append('Both native source pads must be loaded >0.1N')
                if any(type(fn(name)) not in (int,float) or not math.isfinite(fn(name)) or fn(name)>1e-8 for name in ['phone_nonpad','phone_table']):
                    reasons.append('Native source phone nonpad/table force exceeds original no-aid threshold')
                native_lp=record.get('current_native_LP');score=native_lp.get('scaled_loss') if isinstance(native_lp,dict) else None
                if type(score) not in (int,float) or not math.isfinite(score):reasons.append('Finite saved native-LP selection score missing')
                failures={'source_ideal_native_support_candidate':record.get('ideal_native_support_candidate'),
                    'source_holding_diagnostic_eligible':original_holding_flag,
                    'strict_support_status':None if strict is None else strict.get('status'),
                    'strict_support_solver_status':None if strict is None else strict.get('solver_status'),
                    'strict_support_pass':strict_pass,'native_accelerations_are_admission_gates':False}
                row={'branch':branch,'seed':seed['seed'],'rank':rank,'record':candidate_pin,
                    'qualified_for_dynamic_diagnostic':not reasons,'reasons':reasons,'native_LP_selection_loss':score,
                    'control_origin':control_origin,'source_failure_flags_preserved':failures,
                    'initial_pad_normal_N':[fn('thumb_pad'),fn('index_pad')],
                    'initial3N_design_budget_pass':all(type(fn(n)) in (int,float) and fn(n)<=3 for n in ['thumb_pad','index_pad']),
                    'initial3N_budget_hard_admission_gate':False}
                records.append(row)
                if not reasons:
                    q=k['q20'];t=v['x14'][8:11];rv=v['x14'][11:14];ctrl=raw['ctrl20']
                    if not finite_vector(q,20) or not finite_vector(t,3) or not finite_vector(rv,3):raise ValueError('Finite source pose parts')
                    eligible.append(dict(name='source_mapping_'+branch,branch=branch,seed=seed['seed'],rank=rank,record=candidate_pin,
                        ctrl20=ctrl,verified_qpos27=raw['qpos27'],verified_qvel26=raw['qvel26'],verified_ctrl20=ctrl,
                        verified_x46=q+t+rv+ctrl,control_origin=control_origin,native_LP_selection_loss=score,
                        source_failure_flags_preserved=failures,moment_infinity_Nm=None,strict_support_proposal_pass=strict_pass,reachable=True))
        eligible.sort(key=lambda v:(v['native_LP_selection_loss'],v['seed']))
        winner=eligible[0] if eligible else None
        if winner:choices.append(winner)
        screening.append({'branch':branch,'source_protocol':protocol['inputs'][pk],'source_report':protocol['inputs'][rk],
            'candidates':records,'selected_record':None if winner is None else winner['record'],
            'selected_seed':None if winner is None else winner['seed'],'enrolled':winner is not None,
            'selection_rule':'Minimum saved nativeLP loss among evidence-qualified rank0 seeds; tie seed. Strict LP/zero acceleration/initial3N are not necessary gates.'})
    return binding,choices,screening

def metadata_preflight(protocol_path):
    protocol=read(protocol_path)
    if protocol['schema']!=SCHEMA or protocol['gates']!=GATES or protocol['dt_s']!=DTS or protocol['duration_s']!=DURATION or protocol['dt_pair_gates']!=PAIR_GATES:
        raise ValueError('New namespace, unchanged original physical gates/budget/dt pair')
    if type(protocol['CPU_run_authorized']) is not bool or protocol['training_or_GPU_authorized'] is not False:
        raise ValueError('Explicit CPU-only boundary')
    output=Path(protocol['output'])
    if not output.is_absolute() or output.exists() or output.parent.resolve()!=ROOT.resolve():raise ValueError('Fresh output in new namespace')
    if protocol['source_pins']['old_holding_runner']!=pin(ROOT.parent/'hand2_holding_v1/run_cpu_holding_v1.py'):
        raise ValueError('Original physics authority')
    binding,variants,screening=source_choices(protocol)
    selections=protocol['selections']
    if not isinstance(selections,list) or len(selections)!=len(variants):raise ValueError('One explicit pinned selected state per enrolled branch')
    for v in variants:
        rows=[s for s in selections if s['branch']==v['branch']]
        if len(rows)!=1 or rows[0]['record']!=v['record']:raise ValueError('Explicit selection differs from saved minimum rank0 rule')
        selected=rows[0];checked(selected['state']);v['candidate_state']=selected['state']
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
        write(output/'report.json', {'schema': 'hand2_source_mapping_constant_u_diagnostic_result:11',
            'status': 'BLOCKED_NO_SOURCE_MAPPING_DYNAMIC_CANDIDATE_NO_MODEL_OR_STEPS', 'protocol': pin(args.protocol),
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
                raise ValueError('Exact finite pinned source-mapping pose and zero speed')
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
                    enrollment='Pinned source mapping: assembly/fullgeometry/known/loaded/legal control; strictLP and zeroacc not admission; initial3N descriptive')
                endpoints[variant['name']].append({'dt_s':dt, 'final_COM_world_m':r['post_COM'][-1].tolist(),
                    'final_quaternion_wxyz':r['post_qpos'][-1,23:27].tolist(),
                    'actual_final_time_s':float(data.time), 'termination':terminate,
                    'finite_steps_saved':completed, 'original_case_pass':summary['nominal_holding_dynamics_pass']})
                write(case_dir/'summary.json',summary)
                cases.append(summary)
                print(json.dumps({'case':case_name,'termination':terminate,'steps':completed,'metrics':metrics,'pass':summary['nominal_holding_dynamics_pass']}),flush=True)
        pairs = dt_pair_summary(variants, cases, endpoints, np)
        report = {'schema':'hand2_source_mapping_constant_u_diagnostic_result:11',
            'status':'COMPLETED_SOURCE_MAPPING_CONSTANT_U_DYNAMIC_DIAGNOSTIC',
            'protocol':pin(args.protocol),'counts':counts,'screening':screening,
            'cases':cases,'dt_pair_consistency':pairs,'elapsed_seconds':time.perf_counter()-started,
            'all_original_cases_and_dt_pairs_pass':bool(cases and all(v['nominal_holding_dynamics_pass'] for v in cases)
                and all(v['dt_pair_consistency_pass'] for v in pairs)),
            'mass_or_friction_calibrated':False,'ego_trajectory_or_policy_qualified':False,
            'scope':'Two pinned source-normal mapping branches, same original model, both dt and unchanged2s gates. '
                'Source designated-pad initial compression up to0.1mm remains a hypothesis. Strict ideal LP is not necessary for soft-contact dynamic diagnosis. '
                'No object aid, pose reset or materials/gains/solver tuning. Preserved failed source flags; no faithful ego, learning or isolated geometry causal claim. '
                'Dt endpoint consistency is an additional saved comparison and never replaces individual case gates.'}
        write(output/'report.json',report)
    except BaseException as error:
        write(output/'failure.json',{'phase':phase,'error':repr(error),'counts':counts,'automatic_retry':False})
        raise


if __name__ == '__main__':
    main()
