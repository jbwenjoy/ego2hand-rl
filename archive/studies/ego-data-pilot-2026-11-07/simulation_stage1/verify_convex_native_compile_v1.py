"""Compile the new hull asset and verify its compiled frame; zero integration steps."""
import json
from pathlib import Path
import mujoco
import numpy as np
from run_phone_cpu_bench_v1 import ROOT, pin, write

def main():
    asset = ROOT / 'asset_stage2_convex/completed_v2/phone_asset.xml'
    mesh = asset.parent / 'meshes/phone24_collision_convex_m.obj'
    original = np.array([list(map(float, line.split()[1:4])) for line in mesh.read_text().splitlines() if line.startswith('v ')])
    model = mujoco.MjModel.from_xml_path(str(asset))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geom_id = model.geom('phone24_collision_convex').id
    mesh_id = int(model.geom_dataid[geom_id])
    start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    compiled = model.mesh_vert[start:start + count] @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
    distances = np.linalg.norm(compiled[:, None] - original[None, :], axis=-1)
    maximum = float(max(distances.min(axis=0).max(), distances.min(axis=1).max()))
    if maximum > 1e-6:
        raise ValueError('Compiled hull no longer matches original object frame')
    bid = model.body('hot3d_phone24').id
    result = {'schema': 'phone_convex_native_compile_and_frame:1', 'status': 'PASS_COMPILE_AND_COORDINATE_ONLY',
              'source': pin(__file__), 'asset': pin(asset), 'collision_mesh': pin(mesh),
              'mujoco_version': mujoco.__version__, 'nq': model.nq, 'nv': model.nv, 'nu': model.nu,
              'mesh_vertices_compiled': count, 'compiled_to_original_bidirectional_vertex_max_distance_m': maximum,
              'compiled_mass_kg': float(model.body_mass[bid]), 'compiled_COM_body_m': model.body_ipos[bid].tolist(),
              'compiled_inertia_principal_kg_m2': model.body_inertia[bid].tolist(),
              'collision_contype': int(model.geom_contype[geom_id]),
              'visual_contype': int(model.geom_contype[model.geom('phone24_visual_only').id]),
              'nonintegrating_forward_calls': 1, 'physics_integration_steps': 0,
              'physical_parameters_calibrated': False, 'dynamics_or_Warp_mesh_qualified': False}
    destination = ROOT / 'outputs/convex_asset_native_compile_v1.json'
    write(destination, result)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
