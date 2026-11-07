"""Check real-MJB material point coordinates/Jacobians without physics."""
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
PARENT = Path('D:/workspace/supermimic-observation-retarget-study/scripts')
sys.path.insert(0, str(PARENT))
from native_filtered_contact_geometry import NativeFilteredContactGeometry
from material_anchors import MaterialAnchorGeometry


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = ROOT / 'outputs/material_anchor_cpu_v1'
    output.mkdir(parents=True, exist_ok=False)
    model = Path('D:/workspace/supermimic-chord-study/outputs/rotating_reference_warp_a0/model.mjb')
    context = Path('D:/workspace/supermimic-chord-study/outputs/a1l_paired_v2_clip007_2000iter_seed1/compiled/fixed_physics_context.npz')
    candidate = Path('D:/workspace/supermimic-observation-retarget-study/outputs/native_contact_refined_geometry_v1/geometric_retarget.npz')
    forbidden_counts = {}
    def forbid(name):
        def call(*a, **kw):
            forbidden_counts[name] = forbidden_counts.get(name, 0) + 1
            raise AssertionError('No physics, force or gap query in material-point check: ' + name)
        return call
    for name in ('mj_step', 'mj_step1', 'mj_step2', 'mj_forward', 'mj_inverse', 'mj_geomDistance', 'mj_contactForce'):
        setattr(mujoco, name, forbid(name))
    g = NativeFilteredContactGeometry(model, context)
    with np.load(candidate, allow_pickle=False) as z:
        q = z['joint_pos_rad'][0].copy()
        pose = z['mapped_object_pose_w'][0].copy()
    q_before, pose_before = q.copy(), pose.copy()
    g.set_object_pose(pose)
    anchors = MaterialAnchorGeometry(g)
    objR = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    # Synthetic object-surface direction for derivative qualification only;
    # it does not select a data-contact hypothesis or produce a new candidate.
    normal = objR[:, 1]
    point = pose[:3] + g.model.geom_size[g.model.geom('object/cube').id, 0] * normal
    cases = []
    for finger in range(1, 6):
        anchor = anchors.select(q, finger, point, normal)
        p, jp, n, jn = anchors.oriented_features(q, anchor)
        finite_p, finite_n = np.empty((3, 20)), np.empty((3, 20))
        for joint in range(20):
            plus, minus = q.copy(), q.copy()
            plus[joint] += 1e-6
            minus[joint] -= 1e-6
            pp, _, np_, _ = anchors.oriented_features(plus, anchor)
            pm, _, nm, _ = anchors.oriented_features(minus, anchor)
            finite_p[:, joint] = (pp - pm) / 2e-6
            finite_n[:, joint] = (np_ - nm) / 2e-6
        pe, ne = float(np.max(np.abs(jp - finite_p))), float(np.max(np.abs(jn - finite_n)))
        assert pe < 1e-6 and ne < 1e-6
        assert abs(np.linalg.norm(n) - 1) < 1e-10
        support_error = None
        if anchor.vertex_index is not None:
            mesh = int(g.model.geom_dataid[anchor.geom_id])
            start, count = int(g.model.mesh_vertadr[mesh]), int(g.model.mesh_vertnum[mesh])
            verts = np.asarray(g.model.mesh_vert[start:start + count], dtype=np.float64)
            assert np.array_equal(verts[anchor.vertex_index], anchor.point_geom_local)
            support_error = float(np.max(verts @ anchor.support_normal_geom_local) -
                                  anchor.point_geom_local @ anchor.support_normal_geom_local)
            assert abs(support_error) < 1e-12
        cases.append({'finger': finger, 'geom_name': anchor.geom_name,
                      'construction': anchor.construction, 'mesh_vertex_index': anchor.vertex_index,
                      'position_Jacobian_FD_max_abs_m_per_rad': pe,
                      'normal_Jacobian_FD_max_abs_per_rad': ne,
                      'mesh_support_plane_error_m': support_error})
    assert np.array_equal(q_before, q) and np.array_equal(pose_before, pose)
    assert not forbidden_counts and 'warp' not in sys.modules and 'mujoco_warp' not in sys.modules
    report = {'status': 'PASS_LOCAL_MATERIAL_POINT_AND_NORMAL_JACOBIANS',
              'inventory': anchors.inventory(), 'cases': cases,
              'model_sha256': sha(model), 'context_sha256': sha(context),
              'candidate_sha256': sha(candidate), 'checker_sha256': sha(__file__),
              'material_module_sha256': sha(Path(__file__).with_name('material_anchors.py')),
              'forbidden_call_counts': forbidden_counts, 'GPU_used': False, 'physics_steps': 0,
              'scope': 'Five material-point branches on one candidate pose; no data-contact selection, optimization, grasp or global geometry qualification.'}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'cases': cases,
                      'enabled_geom_types': sorted(set(x['type'] for x in report['inventory']))}, indent=2))


if __name__ == '__main__':
    main()
