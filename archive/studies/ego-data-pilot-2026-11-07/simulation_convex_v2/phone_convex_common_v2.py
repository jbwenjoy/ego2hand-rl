"""Shared physical model and coordinate definitions for the convex-phone study."""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'simulation_stage1'))
from run_phone_cpu_bench_v1 import compiled_options, kinematics_from_state, force_at_time, pin

BODY = 'hot3d_phone24'
ROT = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
QUAT = np.array([np.sqrt(.5), np.sqrt(.5), 0., 0.])
ASSET = ROOT / 'asset_stage2_convex/completed_v2/phone_asset.xml'
LEGACY_SOURCE_SHA256 = '3e3303f09d710c5d9de9e61324fcfc84776cb648b5d8cf3a8d32972953aa858c'

def write(path, obj):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)

def make_model(asset_path, mass, mu, dt, timeconst):
    if pin(ROOT / 'simulation_stage1/run_phone_cpu_bench_v1.py')['sha256'] != LEGACY_SOURCE_SHA256:
        raise ValueError('Pinned reusable v1 helpers changed')
    if min(mass, mu, dt, timeconst) <= 0 or timeconst < 2*dt - 1e-15:
        raise ValueError('Positive parameters and timeconst >= 2dt required')
    asset_path = Path(asset_path).resolve()
    root = ET.parse(asset_path).getroot()
    compiler = root.find('compiler')
    meshdir = compiler.get('meshdir', '')
    for mesh in root.findall('./asset/mesh'):
        mesh.set('file', str((asset_path.parent / meshdir / mesh.get('file')).resolve()))
    compiler.attrib.pop('meshdir', None)
    info = json.loads((ROOT / 'hot3d/object_models/models_info.json').read_text())['24']
    full = np.array([info['size_' + axis] for axis in 'xyz'])
    low = np.array([info['min_' + axis] for axis in 'xyz'])
    half, center = full / 2, low + full / 2
    inertia = mass / 12 * np.array([full[1]**2+full[2]**2,
                                    full[0]**2+full[2]**2,
                                    full[0]**2+full[1]**2])
    body = root.find(f"./worldbody/body[@name='{BODY}']")
    inertial = body.find('inertial')
    inertial.attrib.clear()
    inertial.attrib.update(pos=' '.join(map(str, center)), mass=str(mass),
                           diaginertia=' '.join(map(str, inertia)))
    body.set('gravcomp', '0')
    collision = body.find("geom[@name='phone24_collision_convex']")
    if collision is None or collision.get('type') != 'mesh':
        raise ValueError('Convex mesh collider required; no primitive substitution')
    collision_fields = dict(friction=f'{mu} 0.005 0.0001', condim='3',
                            solref=f'{timeconst} 1', solimp='0.9 0.95 0.001 0.5 2',
                            priority='0', solmix='1', margin='0', gap='0',
                            contype='1', conaffinity='1', density='0')
    collision.attrib.update(collision_fields)
    collision.attrib.pop('mass', None)
    for visual in body.findall('geom'):
        if visual is not collision:
            visual.attrib.update(contype='0', conaffinity='0', density='0')
            visual.attrib.pop('mass', None)
    ET.SubElement(root.find('worldbody'), 'geom', name='bench_support', type='plane',
                  pos='0 0 0', size='5 5 .1', rgba='.6 .65 .7 1', **collision_fields)
    option = root.find('option')
    option.attrib.update(timestep=str(dt), integrator='implicitfast', solver='Newton',
                         iterations='100', ls_iterations='20', tolerance='0.000001',
                         ls_tolerance='.01', cone='pyramidal', gravity='0 0 -9.81')
    ET.SubElement(option, 'flag', energy='enable')
    xml = ET.tostring(root, encoding='unicode')
    model = mujoco.MjModel.from_xml_string(xml)
    if (model.nq, model.nv, model.nu, model.neq, model.ntendon) != (7, 6, 0, 0, 0):
        raise ValueError('Only one free rigid body allowed in this bench')
    bid = model.body(BODY).id
    gid = model.geom('phone24_collision_convex').id
    mid = int(model.geom_dataid[gid])
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, model.geom_quat[gid])
    start, count = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    vertices = model.mesh_vert[start:start+count].astype(np.float64) @ matrix.reshape(3, 3).T + model.geom_pos[gid]
    if not np.allclose(vertices.min(0), low, rtol=0, atol=1e-7) or not np.allclose(vertices.max(0), low+full, rtol=0, atol=1e-7):
        raise ValueError('Compiled hull no longer matches the original object frame')
    if not np.allclose(model.body_ipos[bid], center, atol=1e-12) or not np.isclose(model.body_mass[bid], mass):
        raise ValueError('Compiled inertial parameters mismatch')
    return model, {'half_m': half, 'center_m': center, 'inertia_kg_m2': inertia,
                   'local_collision_vertices_m': vertices, 'xml': xml,
                   'model_xml_sha256': hashlib.sha256(xml.encode()).hexdigest(),
                   'mass_kg': mass, 'mu': mu, 'timeconst_s': timeconst}

def initial_state(model, geometry, test):
    clearance = .1 if test == 'drop' else .0005
    relative = geometry['local_collision_vertices_m'] - geometry['center_m']
    minimum_relative_height = float((relative @ ROT.T)[:, 2].min())
    desired_com = np.array([0., 0., clearance - minimum_relative_height])
    data = mujoco.MjData(model)
    data.qpos[:3] = desired_com - ROT @ geometry['center_m']
    data.qpos[3:] = QUAT
    data.qvel[:] = 0.
    data.xfrc_applied[:] = 0.
    data.qfrc_applied[:] = 0.
    mujoco.mj_forward(model, data)
    return data
