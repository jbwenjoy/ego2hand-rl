"""All source hand vertices to actual mug triangles, no contact-truth labels."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.assets.proximity import SurfaceQuery
from supermimic_research.assets.validation import nearest_surface


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')


def main(args):
    import trimesh
    import mujoco
    started = time.perf_counter(); source = args.reference.resolve(); out = args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):
        raise ValueError('Fresh unified runs output')
    report = json.loads((source/'report.json').read_text())
    for pin in report['outputs'].values():
        verify_file_pin(pin)
    mesh_report_path = ROOT/'runs/mug_mesh_v1/report.json'
    mesh_report = json.loads(mesh_report_path.read_text())
    mesh_path = verify_file_pin(mesh_report['raw'])
    z = np.load(source/'reference61.npz'); raw = json.loads((source/'raw_profile_and_poses.json').read_text())
    mesh_data = np.load(mesh_path); v, f = mesh_data['vertices_model_m'], mesh_data['faces']
    query = SurfaceQuery(v, f)
    vertices = z['hand_vertices_object_BODY_F64']; frames = z['frame_ids']
    if vertices.shape != (61, 788, 3) or not np.array_equal(frames, np.arange(47, 108)):
        raise ValueError('Require the predeclared full61-frame articulation window')
    source_pins = [file_pin(p) for p in [Path(__file__), ROOT/'src/supermimic_research/assets/proximity.py',
        ROOT/'src/supermimic_research/assets/validation.py', mesh_report_path, source/'report.json']]
    out.mkdir(parents=True)
    write(out/'intent.json', dict(schema='right_mug_source_surface_intent:1', source_pins=source_pins,
        input_reference=file_pin(source/'reference61.npz'), all_frames=frames.tolist(), all_vertices=788,
        descriptor_thresholds_m=[.001, .003], handle_exterior_side_BODY_x_lt_m=-.045,
        predeclared_bruteforce_distance_tolerance_m=1e-10,
        contact_truth=False, role='GEOMETRIC_INTAKE_ONLY_NO_FITTING', neural_updates=0, integration_steps=0))
    closest, distance, indices = [], [], []; checked = 0
    for row, points in enumerate(vertices):
        c, d, ids, count = query.nearest(points, chunk=128)
        closest.append(c); distance.append(d); indices.append(ids); checked += count
        if row % 10 == 0 or row == 60:
            print(json.dumps(dict(phase='unsigned_triangle_distance', frame=int(frames[row]), rows_complete=row+1,
                                 elapsed_s=time.perf_counter()-started)), flush=True)
    closest, distance, indices = np.array(closest), np.array(distance), np.array(indices)
    control_rows = np.array([0, 17, 30, 45, 60]); control_vertices = np.arange(0, 788, 53)
    control_points = vertices[control_rows[:, None], control_vertices[None, :]].reshape(-1, 3)
    _, brute = nearest_surface(trimesh.Trimesh(v, f, process=False), control_points, chunk=8)
    pruned = distance[control_rows[:, None], control_vertices[None, :]].ravel()
    brute_error = float(np.abs(brute-pruned).max())
    weights = np.array(raw['umetrack_profile']['dense_bone_weights'])
    dominant = np.argmax(weights, axis=1)
    groups = np.where(dominant < 2, 0, 1+(dominant-2)//3)
    names = ['root_wrist_dominant', 'thumb', 'index', 'middle', 'ring', 'little']
    stats = []
    for group, name in enumerate(names):
        mask = groups == group; d = distance[:, mask]; c = closest[:, mask]
        perframe = d.min(1)
        stats.append(dict(group=name, dominant_skinning_group=group, sampled_vertices=int(mask.sum()),
            minimum_unsigned_gap_m=float(d.min()), maximum_over_frames_of_minimum_gap_m=float(perframe.max()),
            median_over_frames_of_minimum_gap_m=float(np.median(perframe)),
            frames_with_vertex_within_1mm=int(np.any(d <= .001, axis=1).sum()),
            frames_with_vertex_within_3mm=int(np.any(d <= .003, axis=1).sum()),
            frames_with_near_handle_exterior_side_vertex_3mm=int(np.any((d <= .003) & (c[..., 0] < -.045), axis=1).sum())))
    model = mujoco.MjModel.from_binary_path(str(source/'hand2_mug.mjb'))
    robot_parts = []
    for digit in range(1, 6):
        for suffix in ('link4_col', 'tip_sensor_col'):
            geom = model.geom(f'right_finger{digit}_{suffix}'); gid = geom.id
            kind = int(model.geom_type[gid])
            item = dict(name=geom.name, type=kind, geom_size_m=model.geom_size[gid].tolist(),
                        size_is_not_a_finger_fit_certificate=True)
            if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
                mid = int(model.geom_dataid[gid]); start = int(model.mesh_vertadr[mid]); count = int(model.mesh_vertnum[mid])
                p = model.mesh_vert[start:start+count]
                item.update(compiled_geom_principal_frame_bounds_m=[p.min(0).tolist(), p.max(0).tolist()],
                            compiled_geom_principal_frame_spans_m=np.ptp(p, axis=0).tolist())
            robot_parts.append(item)
    np.savez_compressed(out/'source_proximity61.npz', frame_ids=frames, timestamps_ns=z['timestamps_ns'],
        distances_m=distance, closest_source_points_BODY_m=closest, source_triangle_ids=indices,
        dominant_skinning_group=groups)
    result = dict(schema='supermimic_research.right_mug_source_proximity:1',
        status='PASS_UNSIGNED_QUERY_VALIDATION_NOT_CONTACT_QUALIFICATION' if brute_error <= 1e-10 else 'QUERY_CHECK_FAILED',
        source_pins=source_pins, input_reference=file_pin(source/'reference61.npz'),
        output=file_pin(out/'source_proximity61.npz'), frames=61, queries=int(distance.size),
        brute_control_queries=int(len(brute)), brute_max_distance_error_m=brute_error,
        pruned_triangle_projections=checked, brute_all_triangle_projections=int(distance.size*len(f)),
        cached_surface_acceleration_built_once=True, surface_triangles=len(f), maximum_triangle_centroid_radius_m=query.maximum_radius,
        overall_minimum_unsigned_vertex_gap_m=float(distance.min()),
        maximum_over_frames_of_minimum_unsigned_gap_m=float(distance.min(1).max()),
        dominant_skinning_groups=stats, robot_fingertip_parts=robot_parts,
        grouping='argmax dense LBS bone weight, bones0/1 root+wrist and three frames per digit; not semantic surface segmentation',
        handle_descriptor='closest source triangle point BODY x<-.045m excludes most cup wall but is not a complete semantic handle mask',
        native_hand_fit_or_contact_truth=False, source_inside_material_sign_known=False,
        neural_updates=0, integration_steps=0, wall_s=time.perf_counter()-started,
        scope='Unsigned proximity of all788 skinned human vertices in61 fixed source frames to original31606 mug triangles. Vertex proximity is not continuous hand-surface separation, measured contact, force, robot fit or physical holding. Positive gaps do not establish absence of triangle intersections; near-zero gaps do not establish contact truth.')
    for pin in source_pins:
        verify_file_pin(pin)
    write(out/'report.json', result)
    print(json.dumps(dict(report=file_pin(out/'report.json'), status=result['status'], wall_s=result['wall_s'],
        brute_error_m=brute_error, queries=result['queries'], groups=stats)), flush=True)
    if brute_error > 1e-10:
        raise RuntimeError('Retain failed query validation')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); main(p.parse_args())
