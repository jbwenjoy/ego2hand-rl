"""Exact Hand2 model with explicit phone contact pairs; no rollout on import."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'hand2_stage1'))
import scene_loader_v1 as loader

def pin(path):
    path = Path(path).resolve()
    content = path.read_bytes()
    return {'path': str(path), 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}

def build_model(*, dt=.00015625, mass=.2, mu_hand=.7, mu_table=.6, timeconst=.0025):
    if min(dt, mass, mu_hand, mu_table, timeconst) <= 0 or timeconst < 2*dt:
        raise ValueError('Positive physical assumptions and resolved contact time scale required')
    original = loader.load_model()
    spec = loader.load_spec()
    spec.option.timestep = dt
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.iterations, spec.option.ls_iterations = 100, 20
    spec.option.tolerance, spec.option.ls_tolerance = 1e-6, .01
    spec.option.gravity = [0, 0, -9.81]
    spec.option.enableflags |= int(mujoco.mjtEnableBit.mjENBL_ENERGY)
    phone_body = spec.body('hot3d_phone24')
    old_mass = float(phone_body.mass)
    phone_body.mass = mass
    phone_body.inertia = np.asarray(phone_body.inertia)*(mass/old_mass)
    phone_body.gravcomp = 0
    oid = original.body('hot3d_phone24').id
    phone_name = 'phone24_collision_convex'
    table_gid = original.geom('stage1_single_table').id
    hand_gids = [gid for gid in range(original.ngeom)
                 if (original.geom_contype[gid] or original.geom_conaffinity[gid])
                 and original.geom_bodyid[gid] != oid and gid != table_gid]
    for gid in hand_gids + [table_gid]:
        other_name = original.geom(gid).name
        mu = mu_table if gid == table_gid else mu_hand
        spec.add_pair(name='holding_phone_to_'+other_name,
                      geomname1=phone_name, geomname2=other_name, condim=3,
                      friction=[mu, mu, .005, .0001, .0001],
                      solref=[timeconst, 1], solreffriction=[0, 0],
                      solimp=[.9, .95, .001, .5, 2], margin=0, gap=0)
    model = spec.compile()
    if (model.nq, model.nv, model.nu, model.neq, model.nmocap) != (27, 26, 20, 0, 0):
        raise ValueError('Fixed wrist,20 joints, one free phone and no support equalities required')
    for name in ('jnt_range','actuator_gainprm','actuator_biasprm','actuator_ctrlrange',
                 'actuator_ctrllimited','actuator_forcerange','actuator_forcelimited',
                 'jnt_actuatorfrcrange','jnt_actuatorfrclimited','actuator_gear',
                 'dof_armature','dof_damping','dof_frictionloss','body_pos','body_quat',
                 'geom_type','geom_size','geom_pos','geom_quat','exclude_signature'):
        if not np.array_equal(getattr(model, name), getattr(original, name)):
            raise ValueError('Source robot or geometry changed: '+name)
    for name in ('body_mass','body_ipos','body_inertia','body_iquat'):
        ids = np.arange(model.nbody) != oid
        if not np.array_equal(getattr(model, name)[ids], getattr(original, name)[ids]):
            raise ValueError('Source robot inertial properties changed: '+name)
    pair_rows = []
    for i in range(model.npair):
        a, b = int(model.pair_geom1[i]), int(model.pair_geom2[i])
        if phone_name not in (model.geom(a).name, model.geom(b).name):
            continue
        pair_rows.append({'name': model.pair(i).name, 'geom1': model.geom(a).name,
                          'geom2': model.geom(b).name, 'dim': int(model.pair_dim[i]),
                          'friction_5': model.pair_friction[i].tolist(),
                          'solref': model.pair_solref[i].tolist(),
                          'solreffriction': model.pair_solreffriction[i].tolist(),
                          'solimp': model.pair_solimp[i].tolist(),
                          'margin': float(model.pair_margin[i]), 'gap': float(model.pair_gap[i])})
    if len(pair_rows) != len(hand_gids)+1:
        raise ValueError('Incomplete explicit phone collision material coverage')
    binding = {'schema': 'fixed_hand2_phone_explicit_contact_model:1',
               'source': pin(__file__), 'exact_loader': pin(loader.__file__),
               'source_scene_report': pin(loader.OUTPUT/'report.json'),
               'mass_kg': mass, 'mu_phone_hand': mu_hand, 'mu_phone_table': mu_table,
               'parameters_measured': False, 'dt_s': dt, 'timeconst_s': timeconst,
               'tolerance': float(model.opt.tolerance), 'ls_tolerance': float(model.opt.ls_tolerance),
               'condim': 3, 'cone': 'pyramidal', 'explicit_pairs': pair_rows,
               'robot_gains_limits_geometry_wrist_unchanged': True,
               'no_mocap_equalities_object_assistance': True,
               'integration_steps': 0, 'hand_physics_or_holding_qualified': False}
    return model, binding

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model, binding = build_model()
    mujoco.mj_saveModel(model, str(args.output/'model.mjb'), None)
    binding['model'] = pin(args.output/'model.mjb')
    with (args.output/'binding.json').open('x') as stream:
        json.dump(binding, stream, indent=2, allow_nan=False)
    print(json.dumps({'status': 'EXPLICIT_CONTACT_MODEL_COMPILED_NO_STEPS', 'pairs': len(binding['explicit_pairs']),
                      'output': str(args.output)}))

if __name__ == '__main__':
    main()
