"""Free-object CPU bench: assumed parameters, no hand, policy, or trajectory forcing."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import itertools
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BODY = 'hot3d_phone24'
QUAT = np.array([np.sqrt(0.5), np.sqrt(0.5), 0., 0.])
ROT = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
PHYSICS = {'integrator': 'implicitfast', 'solver': 'Newton', 'iterations': 100,
           'ls_iterations': 20, 'tolerance': 1e-10, 'gravity_m_s2': [0., 0., -9.81],
           'condim': 3, 'solref': [0.02, 1.], 'solimp': [0.9, 0.95, 0.001],
           'friction': 'sliding_mu,0.005,0.0001 (last two inactive with condim=3)',
           'margin_m': 0., 'gap_m': 0., 'support': 'infinite plane z=0; not reconstructed table'}

def pin(path):
    path = Path(path)
    content = path.read_bytes()
    return {'path': str(path.resolve()), 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}

def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)

def make_model(asset_path, mass, mu, dt):
    """Shared by the Warp runner: construct the exact same explicitly configured model."""
    asset_path = Path(asset_path).resolve()
    tree = ET.parse(asset_path)
    root = tree.getroot()
    compiler = root.find('compiler')
    meshdir = compiler.get('meshdir', '') if compiler is not None else ''
    for mesh in root.findall('./asset/mesh'):
        if mesh.get('file'):
            mesh.set('file', str((asset_path.parent / meshdir / mesh.get('file')).resolve()))
    if compiler is not None:
        compiler.attrib.pop('meshdir', None)
    option = root.find('option')
    if option is None:
        option = ET.SubElement(root, 'option')
    option.attrib.update(timestep=str(dt), gravity='0 0 -9.81', integrator='implicitfast', cone='pyramidal',
                         solver='Newton', iterations='100', ls_iterations='20', tolerance='1e-10')
    world = root.find('worldbody')
    body = world.find(f"body[@name='{BODY}']")
    if body is None or body.find('freejoint') is None:
        raise ValueError('Expected one free phone body')
    collision = [g for g in body.findall('geom') if g.get('type') == 'box' and g.get('contype', '1') != '0']
    if len(collision) != 1:
        raise ValueError('Expected one explicit box collider')
    box = collision[0]
    half = np.fromstring(box.get('size'), sep=' ')
    center = np.fromstring(box.get('pos', '0 0 0'), sep=' ')
    if half.shape != (3,) or center.shape != (3,) or np.any(half <= 0):
        raise ValueError('Invalid box dimensions')
    if not (half[1] < half[0] < half[2]):
        raise ValueError('Unexpected phone axes; review Rx90 initial orientation')
    full = 2 * half
    inertia = mass / 12 * np.array([full[1]**2 + full[2]**2,
                                    full[0]**2 + full[2]**2,
                                    full[0]**2 + full[1]**2])
    inertial = body.find('inertial')
    if inertial is None:
        inertial = ET.SubElement(body, 'inertial')
    inertial.attrib.clear()
    inertial.attrib.update(pos=' '.join(map(str, center)), mass=str(mass),
                           diaginertia=' '.join(map(str, inertia)))
    contact = {'friction': f'{mu} 0.005 0.0001', 'condim': '3', 'solref': '0.02 1',
               'solimp': '0.9 0.95 0.001', 'margin': '0', 'gap': '0', 'priority': '0',
               'contype': '1', 'conaffinity': '1'}
    box.attrib.update(contact)
    box.set('density', '0')
    box.attrib.pop('mass', None)
    for visual in body.findall('geom'):
        if visual is not box:
            visual.attrib.update(contype='0', conaffinity='0', density='0')
            visual.attrib.pop('mass', None)
    ET.SubElement(world, 'geom', name='bench_support', type='plane', size='5 5 0.1',
                  pos='0 0 0', rgba='0.6 0.65 0.7 1', **contact)
    xml = ET.tostring(root, encoding='unicode')
    model = mujoco.MjModel.from_xml_string(xml)
    if model.nq != 7 or model.nv != 6 or model.nu != 0:
        raise ValueError('Bench must contain only one free rigid body and no actuators')
    bid = model.body(BODY).id
    if not np.isclose(model.body_mass[bid], mass, rtol=1e-12):
        raise ValueError('Compiled mass mismatch')
    if not np.allclose(model.body_ipos[bid], center, atol=1e-12):
        raise ValueError('Compiled COM offset mismatch')
    model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_ENERGY)
    return model, {'half_m': half, 'center_m': center, 'inertia_kg_m2': inertia,
                   'model_xml_sha256': hashlib.sha256(xml.encode()).hexdigest(), 'xml': xml}

def compiled_options(model):
    option_names = ('integrator', 'solver', 'iterations', 'ls_iterations', 'tolerance',
                    'ls_tolerance', 'cone', 'disableflags', 'enableflags', 'noslip_iterations',
                    'ccd_iterations', 'impratio', 'density', 'viscosity', 'timestep', 'gravity')
    return {'option': {name: np.asarray(getattr(model.opt, name)).tolist() for name in option_names},
            'geom_names': [model.geom(i).name for i in range(model.ngeom)],
            'geom_condim': model.geom_condim.tolist(), 'geom_friction': model.geom_friction.tolist(),
            'geom_priority': model.geom_priority.tolist(), 'geom_solmix': model.geom_solmix.tolist(),
            'geom_solref': model.geom_solref.tolist(), 'geom_solimp': model.geom_solimp.tolist(),
            'geom_margin': model.geom_margin.tolist(), 'geom_gap': model.geom_gap.tolist(),
            'body_gravcomp': model.body_gravcomp.tolist(), 'body_mass': model.body_mass.tolist(),
            'body_ipos': model.body_ipos.tolist(), 'body_inertia': model.body_inertia.tolist(),
            'body_iquat': model.body_iquat.tolist(), 'nu': model.nu, 'neq': model.neq,
            'ntendon': model.ntendon, 'nconmax_model': model.nconmax, 'njmax_model': model.njmax}

def kinematics_from_state(qpos, qvel, center):
    """Single free joint: world root v, body-local omega; preserve saved raw state."""
    qpos, qvel = np.asarray(qpos), np.asarray(qvel)
    w, x, y, z = np.moveaxis(qpos[..., 3:], -1, 0)
    rotation = np.stack((1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                         2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w),
                         2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)), axis=-1)
    rotation = rotation.reshape(qpos.shape[:-1] + (3, 3))
    offset_world = np.einsum('...ij,j->...i', rotation, center)
    omega_world = np.einsum('...ij,...j->...i', rotation, qvel[..., 3:])
    com = qpos[..., :3] + offset_world
    com_velocity = qvel[..., :3] + np.cross(omega_world, offset_world)
    return com, com_velocity, omega_world

def initial_state(model, geometry, test):
    data = mujoco.MjData(model)
    clearance = 0.1 if test == 'drop' else 0.0005
    com = np.array([0., 0., clearance + geometry['half_m'][1]])
    data.qpos[:3] = com - ROT @ geometry['center_m']
    data.qpos[3:] = QUAT
    data.qvel[:] = 0
    data.xfrc_applied[:] = 0
    data.qfrc_applied[:] = 0
    mujoco.mj_forward(model, data)
    return data

def force_at_time(test, time_s):
    return 2.0 if test in ('push_release', 'force_push_release') and 1.0 <= time_s < 1.2 else 0.0

def run_case(model, geometry, test):
    dt = float(model.opt.timestep)
    steps = round(3.0 / dt)
    if not np.isclose(steps * dt, 3.0):
        raise ValueError('Nonintegral benchmark duration')
    data = initial_state(model, geometry, test)
    bid = model.body(BODY).id
    times, qposes, qvels, coms, energies, penetration, contacts, forces = [], [], [], [], [], [], [], []
    com_velocities, world_omegas, physical_energies, contact_rows, warning_history = [], [], [], [], []
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    previous_warnings = [0] * len(data.warning)
    completed = 0
    error = None

    def record(force):
        times.append(float(data.time))
        qposes.append(data.qpos.copy())
        qvels.append(data.qvel.copy())
        coms.append(data.xipos[bid].copy())
        energies.append(data.energy.copy())
        mujoco.mj_jacBodyCom(model, data, jacp, jacr, bid)
        com_velocity = jacp @ data.qvel
        omega_world = jacr @ data.qvel
        com_velocities.append(com_velocity)
        world_omegas.append(omega_world)
        omega_inertial = data.ximat[bid].reshape(3, 3).T @ omega_world
        kinetic = 0.5 * (model.body_mass[bid] * np.dot(com_velocity, com_velocity)
                         + np.dot(model.body_inertia[bid] * omega_inertial, omega_inertial))
        physical_energies.append([float(model.body_mass[bid] * 9.81 * data.xipos[bid, 2]), float(kinetic)])
        penetration.append(max(0., -min((float(data.contact[i].dist) for i in range(data.ncon)), default=0.)))
        contacts.append(int(data.ncon))
        forces.append(force)
        for i in range(data.ncon):
            contact = data.contact[i]
            native_force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, native_force)
            contact_rows.append([completed, float(data.time), i, int(contact.geom[0]), int(contact.geom[1]),
                                 float(contact.dist), *contact.pos.tolist(), *native_force.tolist()])
        for i, warning in enumerate(data.warning):
            if int(warning.number) != previous_warnings[i]:
                warning_history.append([completed, i, int(warning.number), int(warning.lastinfo)])
                previous_warnings[i] = int(warning.number)

    record(0.)
    for step in range(steps):
        # Integer-grid schedule avoids cumulative floating-clock boundary drift.
        force = force_at_time(test, step * dt)
        data.xfrc_applied[bid, :] = 0.
        data.xfrc_applied[bid, 0] = force
        mujoco.mj_step(model, data)
        completed += 1
        mujoco.mj_forward(model, data)  # sample all derived quantities at the post-step state
        record(force)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            error = 'nonfinite state'
            break
        if not np.isclose(data.time, completed * dt, rtol=0, atol=1e-9):
            error = 'simulation clock reset or drift'
            break
    arrays = {'time_s': np.array(times), 'qpos': np.array(qposes), 'qvel': np.array(qvels),
              'com_world_m': np.array(coms), 'energy_potential_kinetic_J': np.array(energies),
              'physical_energy_U_T_J': np.array(physical_energies),
              'com_velocity_world_m_s': np.array(com_velocities),
              'angular_velocity_world_rad_s': np.array(world_omegas),
              'penetration_m': np.array(penetration), 'contact_count': np.array(contacts),
              'applied_com_force_world_x_N': np.array(forces),
              'tick': np.arange(len(times), dtype=np.int64), 'tick_time_s': np.arange(len(times)) * dt,
              'contact_long_table': np.asarray(contact_rows).reshape((-1, 15))}
    rest = arrays['time_s'] >= 2.5 - dt / 4
    velocities = arrays['com_velocity_world_m_s'][rest]
    omegas = arrays['angular_velocity_world_rad_s'][rest]
    com = arrays['com_world_m']
    warning_counts = [int(x.number) for x in data.warning]
    total_energy = arrays['physical_energy_U_T_J'].sum(axis=1)
    derived_com, derived_com_v, derived_omega = kinematics_from_state(arrays['qpos'], arrays['qvel'], geometry['center_m'])
    # External force is constant over each integration interval. COM displacement
    # gives its discrete work; internal/contact dissipation is not identified here.
    external_work = float(np.sum(arrays['applied_com_force_world_x_N'][1:] * np.diff(com[:, 0])))
    metrics = {'completed_steps': completed, 'expected_steps': steps, 'final_time_s': float(data.time),
               'error': error, 'finite': all(np.isfinite(a).all() for a in arrays.values()),
               'warning_counts': warning_counts, 'max_penetration_m': float(max(penetration)),
               'warning_history': warning_history,
               'rest_linear_speed_rms_m_s': float(np.sqrt(np.mean(np.sum(velocities**2, axis=1)))) if len(velocities) else None,
               'rest_angular_speed_rms_rad_s': float(np.sqrt(np.mean(np.sum(omegas**2, axis=1)))) if len(omegas) else None,
               'rest_linear_speed_max_m_s': float(np.linalg.norm(velocities, axis=1).max()) if len(velocities) else None,
               'rest_angular_speed_max_rad_s': float(np.linalg.norm(omegas, axis=1).max()) if len(omegas) else None,
               'rest_max_penetration_m': float(arrays['penetration_m'][rest].max()) if rest.any() else None,
               'rest_contact_observed': bool(arrays['contact_count'][rest].max() > 0) if rest.any() else False,
               'kinematic_formula_max_error_m': float(np.abs(derived_com - arrays['com_world_m']).max()),
               'kinematic_COM_velocity_max_error_m_s': float(np.abs(derived_com_v - arrays['com_velocity_world_m_s']).max()),
               'kinematic_world_omega_max_error_rad_s': float(np.abs(derived_omega - arrays['angular_velocity_world_rad_s']).max()),
               'max_speed_m_s': float(np.linalg.norm(arrays['qvel'][:, :3], axis=1).max()),
               'max_contacts': int(max(contacts)), 'contact_observed': max(contacts) > 0,
               'final_root_qpos': arrays['qpos'][-1].tolist(), 'final_com_world_m': com[-1].tolist(),
               'com_displacement_x_m': float(com[-1, 0] - com[0, 0]),
               'max_energy_J': float(total_energy.max()), 'initial_energy_J': float(total_energy[0]),
               'final_energy_J': float(total_energy[-1]), 'known_external_work_J': external_work,
               'energy_gain_minus_known_external_work_J': float(total_energy[-1] - total_energy[0] - external_work),
               'force_impulse_N_s': float(np.sum(arrays['applied_com_force_world_x_N'][1:]) * dt)}
    return metrics, arrays

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'simulation_stage1/cpu_v1')
    parser.add_argument('--protocol', type=Path, default=ROOT / 'protocols/phone_physics_stage1_v1.json')
    parser.add_argument('--asset', type=Path, default=ROOT / 'asset_stage1/phone_asset.xml')
    args = parser.parse_args()
    protocol_pin = pin(args.protocol)
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    if protocol['schema'] != 'phone_physics_stage1_protocol:1':
        raise ValueError('Unexpected protocol schema')
    masses = protocol['cases']['cartesian_grid']['mass_kg']
    mus = protocol['cases']['cartesian_grid']['sliding_friction']
    dts = protocol['cases']['timesteps_s']
    if (masses, mus, dts) != ([0.1, 0.2, 0.3], [0.3, 0.6, 0.9], [0.0025, 0.00125]):
        raise ValueError('Unexpected frozen benchmark parameter grid')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows = []
    pairs = [(0.2, 0.6)] + [pair for pair in itertools.product(masses, mus) if pair != (0.2, 0.6)]
    for mass, mu in pairs:
        for dt in dts:
            model, geometry = make_model(args.asset, mass, mu, dt)
            for test in ('drop', 'table_settling', 'force_push_release'):
                case = f'{test}_mass{mass:g}_mu{mu:g}_dt{dt:g}'
                metrics, trajectory = run_case(model, geometry, test)
                trace_path = args.output / (case + '.npz')
                np.savez_compressed(trace_path, **trajectory)
                rows.append({'case': case, 'test': test, 'mass_kg': mass, 'sliding_friction': mu,
                             'dt_s': dt, 'metrics': metrics, 'trajectory': pin(trace_path),
                             'model_xml_sha256': geometry['model_xml_sha256'],
                             'compiled_model_options': compiled_options(model),
                             'uniform_box_inertia_kg_m2': geometry['inertia_kg_m2'].tolist()})
        print(json.dumps({'status': 'CPU_GROUP_COMPLETE', 'mass_kg': mass, 'mu': mu}), flush=True)
    mesh = args.asset.parent / 'meshes/phone24_visual_m.obj'
    write(args.output / 'result.json', {'schema': 'phone_free_object_cpu_bench:1',
          'created_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'physics': PHYSICS, 'protocol': protocol_pin, 'source': pin(__file__),
          'asset': pin(args.asset), 'visual_mesh': pin(mesh), 'mujoco_version': mujoco.__version__,
          'cases': rows, 'case_count': len(rows), 'elapsed_seconds': time.perf_counter() - started,
          'step_count': sum(row['metrics']['completed_steps'] for row in rows),
          'extra_forward_calls': sum(row['metrics']['completed_steps'] for row in rows),
          'initial_state_forward_calls': len(rows),
          'contact_phase': 'post-state contact and native forces from explicit nonintegrating mj_forward at ticks0..N',
          'contact_long_table_columns': ['tick', 'time_s', 'contact_index', 'geom1', 'geom2', 'distance_m',
                                        'pos_x_m', 'pos_y_m', 'pos_z_m', 'native_force_normal_N',
                                        'native_force_tangent1_N', 'native_force_tangent2_N',
                                        'native_torque_normal_Nm', 'native_torque_tangent1_Nm', 'native_torque_tangent2_Nm'],
          'thresholds_assessed': False, 'physical_parameters_measured': False,
          'human_contact_dynamics_validated': False, 'hand2_or_learning_executed': False,
          'old_experiment_state_modified': False})
    print(json.dumps({'status': 'CPU_BENCH_RECORDED_NOT_YET_ASSESSED', 'case_count': len(rows),
                      'output': str(args.output), 'elapsed_seconds': time.perf_counter() - started}), flush=True)

if __name__ == '__main__':
    main()
