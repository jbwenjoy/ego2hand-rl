"""Bounded, explicit-contact CPU holding experiments; no training or object aid.

All native forces/geometry/energy belong to PRE-step state; qpos/qvel and the
derived object COM belong to POST-step state. No extra forward inside rollout.
"""
import argparse
import hashlib
import json
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


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--protocol', type=Path, required=True)
    a.add_argument('--run', action='store_true')
    args = a.parse_args()
    protocol = read(args.protocol)
    if protocol['schema'] != 'hand2_phone_nominal_CPU_holding_protocol:1':
        raise ValueError('Protocol schema')
    if protocol['gates'] != GATES or protocol['dt_s'] != DTS or protocol['duration_s'] != DURATION:
        raise ValueError('Frozen physical budget/gates')
    if protocol['CPU_run_authorized'] is not True or protocol['training_or_GPU_authorized'] is not False:
        raise ValueError('CPU experiment only')
    for p in [*protocol['inputs'].values(), *protocol['source_pins'].values()]:
        checked(p)
    if protocol['source_pins']['runner'] != pin(__file__):
        raise ValueError('Current source')
    output = Path(protocol['output'])
    if output.exists() or output.parent != ROOT:
        raise ValueError('Fresh output inside holding folder')
    geom = read(checked(protocol['inputs']['geometry_report']))
    if not geom['seed_results'][2]['geometry_candidate_qualified']:
        raise ValueError('Actual geometry qualification required')
    if geom['seed_results'][2]['selected_state'] != protocol['inputs']['candidate_state']:
        raise ValueError('Exact qualified seed2 state')
    prep = read(checked(protocol['inputs']['support_report']))
    if (prep['input_pins']['state'] != protocol['inputs']['candidate_state']
            or prep['input_pins']['binding'] != protocol['inputs']['model_binding']
            or prep['geometry_recheck_passed'] is not True
            or prep['counts']['integration_steps'] != 0):
        raise ValueError('Actual matching nonintegrating geometry/control preparation')
    variants = prep['controller_variants']
    if not variants or len({v['name'] for v in variants}) != len(variants):
        raise ValueError('Explicit distinct preregistered controls')
    binding = read(checked(protocol['inputs']['model_binding']))
    if binding['model'] != protocol['inputs']['model_mjb'] or binding['source'] != protocol['source_pins']['scene']:
        raise ValueError('Exact explicit-pair model')
    if not args.run:
        print(json.dumps({'status': 'METADATA_VALID_NO_MODEL_OR_STEPS', 'variants': [v['name'] for v in variants],
                          'maximum_steps': int(sum(DURATION/dt for dt in DTS)*sum(v['reachable'] for v in variants))}))
        return
    output.mkdir()
    write(output/'protocol.json', protocol)
    shutil.copyfile(__file__, output/'execution_source.py')
    started = time.perf_counter()
    counts = {'MJB_reads': 0, 'MjData': 0, 'initial_forward': 0, 'integration_steps': 0,
              'contact_force_reads': 0, 'neural_training_GPU': 0}
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
        with np.load(checked(protocol['inputs']['candidate_state']), allow_pickle=False) as z:
            q0 = z['qpos27'].copy()
            if np.any(z['qvel26']):
                raise ValueError('Static zero-speed start')
        qlimits = model.jnt_range[:20].copy()
        R = np.empty(9)
        def COM(q):
            mujoco.mju_quat2Mat(R, q[23:27])
            return q[20:23]+R.reshape(3,3)@model.body_ipos[phone]
        COM0 = COM(q0).copy()
        def angle(q):
            return float(2*np.arccos(np.clip(abs(q[23:27]@q0[23:27]),0,1)))
        cases = []
        for variant in variants:
            if not variant['reachable']:
                cases.append({'name': variant['name'], 'status': 'BLOCKED_CONTROL_UNREACHABLE_NO_STEPS', 'integration_steps': 0})
                continue
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
                write(case_dir/'summary.json',summary)
                cases.append(summary)
                print(json.dumps({'case':case_name,'termination':terminate,'steps':completed,'metrics':metrics,'pass':summary['nominal_holding_dynamics_pass']}),flush=True)
        report = {'schema':'hand2_phone_nominal_CPU_holding_result:1','protocol':pin(args.protocol),'counts':counts,
            'cases':cases,'elapsed_seconds':time.perf_counter()-started,'mass_or_friction_calibrated':False,
            'ego_trajectory_or_policy_qualified':False,'scope':'Two-second fixed-command nominal engineering experiment only. Actual free-object mesh contacts, no object aid. Actual effective explicit pair model. Gates precede all trajectories. Condim3 has no torsion/rolling. No parameter selection or automatic retries.'}
        write(output/'report.json',report)
    except BaseException as error:
        write(output/'failure.json',{'phase':phase,'error':repr(error),'counts':counts,'automatic_retry':False})
        raise


if __name__ == '__main__':
    main()
