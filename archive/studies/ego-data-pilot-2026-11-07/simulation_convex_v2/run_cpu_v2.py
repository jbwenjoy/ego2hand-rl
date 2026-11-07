"""Preregistered convex-contact CPU screening / fresh grid confirmation."""
import argparse
import datetime
import itertools
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import phone_convex_common_v2 as common

ROOT = common.ROOT
TESTS = ('drop', 'table_settling', 'force_push_release')

def run_case(model, geometry, test):
    data = common.initial_state(model, geometry, test)
    bid = model.body(common.BODY).id
    dt, steps = float(model.opt.timestep), round(3 / float(model.opt.timestep))
    fields = {name: [] for name in ('time_s', 'qpos', 'qvel', 'qacc', 'com_world_m', 'com_velocity_world_m_s',
                                   'angular_velocity_world_rad_s', 'physical_energy_U_T_J',
                                   'native_energy_potential_kinetic_J', 'penetration_m', 'contact_count',
                                   'previous_interval_applied_force_world_x_N')}
    contacts, warnings = [], []
    last_warning = [0] * len(data.warning)
    jacp, jacr = np.empty((3, model.nv)), np.empty((3, model.nv))
    completed = 0
    error = None

    def record(force):
        mujoco.mj_jacBodyCom(model, data, jacp, jacr, bid)
        vcom, omega = jacp @ data.qvel, jacr @ data.qvel
        omega_i = data.ximat[bid].reshape(3, 3).T @ omega
        kinetic = .5 * (model.body_mass[bid]*np.dot(vcom, vcom) + np.dot(model.body_inertia[bid]*omega_i, omega_i))
        values = (float(data.time), data.qpos.copy(), data.qvel.copy(), data.qacc.copy(), data.xipos[bid].copy(), vcom, omega,
                  [float(model.body_mass[bid]*9.81*data.xipos[bid, 2]), float(kinetic)], data.energy.copy(),
                  max(0., -min((float(data.contact[i].dist) for i in range(data.ncon)), default=0.)), int(data.ncon), force)
        for key, value in zip(fields, values):
            fields[key].append(value)
        for slot in range(data.ncon):
            c = data.contact[slot]
            native = np.zeros(6)
            mujoco.mj_contactForce(model, data, slot, native)
            contacts.append([completed, float(data.time), slot, int(c.geom[0]), int(c.geom[1]), float(c.dist),
                             *c.pos.tolist(), *c.frame.tolist(), *native.tolist()])
        for i, warning in enumerate(data.warning):
            if int(warning.number) != last_warning[i]:
                warnings.append([completed, i, int(warning.number), int(warning.lastinfo)])
                last_warning[i] = int(warning.number)

    record(0.)
    for tick in range(steps):
        force = common.force_at_time(test, tick*dt)
        data.xfrc_applied[bid] = 0.
        data.xfrc_applied[bid, 0] = force
        mujoco.mj_step(model, data)
        completed += 1
        mujoco.mj_forward(model, data)
        record(force)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or abs(data.time-completed*dt) > 1e-9:
            error = 'nonfinite_or_reset_clock'
            break
    a = {name: np.array(values) for name, values in fields.items()}
    a.update(tick=np.arange(completed+1, dtype=np.int64), tick_time_s=np.arange(completed+1)*dt,
             contact_long_table=np.asarray(contacts, dtype=np.float64).reshape((-1, 24)))
    com, vcom, omega = common.kinematics_from_state(a['qpos'], a['qvel'], geometry['center_m'])
    metrics = {'completed_steps': completed, 'expected_steps': steps, 'error': error,
               'finite': all(np.isfinite(x).all() for x in a.values()),
               'max_penetration_m': float(a['penetration_m'].max()),
               'maximum_COM_formula_difference_m': float(np.abs(com-a['com_world_m']).max()),
               'maximum_COM_velocity_formula_difference_m_s': float(np.abs(vcom-a['com_velocity_world_m_s']).max()),
               'maximum_world_omega_formula_difference_rad_s': float(np.abs(omega-a['angular_velocity_world_rad_s']).max()),
               'warning_terminal_counts': [int(x.number) for x in data.warning], 'warning_history': warnings,
               'final_time_s': float(data.time), 'extra_nonintegrating_forward_calls': completed+1}
    return a, metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('screen', 'confirm'), required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT/'protocols/phone_convex_contact_v2.json')
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    if protocol['schema'] != 'phone_convex_contact_protocol:2':
        raise ValueError('Unexpected frozen protocol')
    screen = protocol['screening']
    confirm = protocol['confirmation']
    if (screen['mass_kg'], screen['sliding_friction'], screen['timeconst_s'], screen['timesteps_s'], screen['test_names']) != (.2, .6, [.02, .005, .0025], [.00125, .000625], list(TESTS)):
        raise ValueError('Frozen screening grid mismatch')
    if confirm['cartesian_grid'] != {'mass_kg': [.1, .2, .3], 'sliding_friction': [.3, .6, .9]} or confirm['timesteps_s'] != [.00125, .000625] or confirm['test_names'] != list(TESTS):
        raise ValueError('Frozen confirmation grid mismatch')
    if protocol['shared_physics']['tolerance'] != 1e-6 or protocol['shared_physics']['ls_tolerance'] != .01:
        raise ValueError('CPU/GPU effective solver contract mismatch')
    if args.mode == 'screen':
        parameters = [(.2, .6, tau, dt, test) for tau in (.02, .005, .0025)
                      for dt in (.00125, .000625) for test in TESTS]
        selection_pin = None
    else:
        if args.selection is None:
            raise ValueError('Fresh confirmation requires saved screening selection')
        selection = json.loads(args.selection.read_text())
        tau = selection['selected_timeconst_s']
        if tau not in (.02, .005, .0025) or not selection['screening_complete'] or not selection['selection_complete']:
            raise ValueError('No complete eligible screening selection')
        if selection['protocol_pin']['sha256'] != common.pin(args.protocol)['sha256']:
            raise ValueError('Screening selection came from another protocol')
        pairs = [(.2, .6)] + [p for p in itertools.product((.1, .2, .3), (.3, .6, .9)) if p != (.2, .6)]
        parameters = [(mass, mu, tau, dt, test) for mass, mu in pairs for dt in (.00125, .000625) for test in TESTS]
        selection_pin = common.pin(args.selection)
    if args.output.exists():
        raise FileExistsError('Fresh output required')
    args.output.mkdir(parents=True)
    before = {'schema': 'phone_convex_CPU_execution_binding:2', 'mode': args.mode,
              'protocol': common.pin(args.protocol), 'selection': selection_pin,
              'runner': common.pin(__file__), 'common': common.pin(common.__file__),
              'legacy_helpers': common.pin(ROOT/'simulation_stage1/run_phone_cpu_bench_v1.py'),
              'asset': common.pin(common.ASSET),
              'geometry_report': common.pin(common.ASSET.parent/'report.json'),
              'source_GLB': common.pin(ROOT/'hot3d/object_models/obj_000024.glb'),
              'collision_OBJ': common.pin(common.ASSET.parent/'meshes/phone24_collision_convex_m.obj'),
              'visual_OBJ': common.pin(common.ASSET.parent/'meshes/phone24_visual_m.obj'),
              'mujoco_version': mujoco.__version__, 'parameters': parameters,
              'integration_steps_before_binding': 0, 'parameters_measured': False}
    common.write(args.output/'execution_binding.json', before)
    started, rows = time.perf_counter(), []
    for mass, mu, tau, dt, test in parameters:
        model, geometry = common.make_model(common.ASSET, mass, mu, dt, tau)
        identity = f'{test}_mass{mass:g}_mu{mu:g}_tau{tau:g}_dt{dt:g}'
        xml_path = args.output/(identity+'.xml')
        with xml_path.open('x') as stream:
            stream.write(geometry['xml'])
        compiled = common.compiled_options(model)
        support = float(((geometry['local_collision_vertices_m']-geometry['center_m']) @ common.ROT.T)[:, 2].min())
        desired_com = np.array([0., 0., (.1 if test == 'drop' else .0005)-support])
        common.write(args.output/(identity+'_before_steps.json'), {'compiled_options': compiled,
                     'model_xml': common.pin(xml_path), 'inertia_kg_m2': geometry['inertia_kg_m2'].tolist(),
                     'COM_body_m': geometry['center_m'].tolist(),
                     'local_collision_vertices_m': geometry['local_collision_vertices_m'].tolist(),
                     'initial_minimum_collision_height_relative_to_COM_m': support,
                     'initial_requested_qpos': np.concatenate((desired_com-common.ROT@geometry['center_m'], common.QUAT)).tolist(),
                     'initialization': 'Rx90, clearance based on compiled collision hull support'})
        trajectory, metrics = run_case(model, geometry, test)
        trace_path = args.output/(identity+'.npz')
        np.savez_compressed(trace_path, **trajectory)
        rows.append({'case': identity, 'mass_kg': mass, 'sliding_friction': mu, 'timeconst_s': tau,
                     'dt_s': dt, 'test': test, 'metrics': metrics, 'trajectory': common.pin(trace_path),
                     'compiled_options': compiled, 'model_xml': common.pin(xml_path)})
        print(json.dumps({'status': 'CPU_CONVEX_RECORDED', 'case': identity,
                          'max_penetration_mm': 1000*metrics['max_penetration_m']}), flush=True)
    common.write(args.output/'result.json', {'schema': ('phone_convex_contact_CPU_screening:2' if args.mode == 'screen' else 'phone_convex_contact_CPU_confirmation:2'), 'mode': args.mode,
                 'created_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'binding': common.pin(args.output/'execution_binding.json'),
                 'protocol': common.pin(args.protocol), 'cases': rows, 'case_count': len(rows),
                 'integration_steps': sum(r['metrics']['completed_steps'] for r in rows),
                 'elapsed_seconds': time.perf_counter()-started,
                 'contact_columns': ['tick','time_s','index','geom1','geom2','dist_m','px','py','pz']+
                                    ['frame'+str(i) for i in range(9)]+['native_force'+str(i) for i in range(6)],
                 'force_row_phase': 'row i stores force used for interval tick i-1 to i; row0 is zero',
                 'contact_phase': 'post-state native contacts via explicitly counted nonintegrating forward',
                 'thresholds_assessed': False, 'Hand2_or_training_executed': False,
                 'real_parameters_calibrated': False})
    print(json.dumps({'status': 'CPU_CONVEX_COMPLETE_NOT_ASSESSED', 'cases': len(rows), 'output': str(args.output)}), flush=True)

if __name__ == '__main__':
    main()
