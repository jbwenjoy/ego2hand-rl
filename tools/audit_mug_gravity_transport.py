"""Recheck existing real-ego asset bindings and conditional gravity distortion."""
import argparse
import datetime
import json
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.ego.gravity_transport import gravity_transport_diagnostic


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def gravity_fields(value, prefix=''):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = prefix + '/' + key
            if 'gravity' in key.lower():
                found.append(path)
            found.extend(gravity_fields(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(gravity_fields(child, prefix + '/' + str(index)))
    return found


def main(a):
    out = a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT / 'runs'):
        raise ValueError('Fresh research output required')
    frames_path = ROOT / 'runs/right_mug_frames_v4/report.json'
    frames = json.loads(frames_path.read_text())
    asset_path = ROOT / 'runs/mug_asset_validation_v2/report.json'
    asset = json.loads(asset_path.read_text())
    if (frames['passed'], frames['total'], asset['passed'], asset['total']) != (15, 15, 9, 9):
        raise ValueError('Declared saved asset/frame contracts differ')
    for pin in frames['outputs'].values():
        verify_file_pin(pin)
    clip = verify_file_pin(frames['clip'])
    with np.load(verify_file_pin(frames['outputs']['reference61.npz'])) as saved:
        z = dict(saved)
    raw_frame_rows = []
    all_gravity_fields = []
    with tarfile.open(clip, 'r') as tar:
        names = tar.getnames()
        for frame in z['frame_ids']:
            members = {kind: json.load(tar.extractfile(f'{frame:06d}.{kind}.json')) for kind in ['info', 'cameras', 'hands', 'objects']}
            fields = {kind: gravity_fields(data) for kind, data in members.items()}
            all_gravity_fields.extend(path for paths in fields.values() for path in paths)
            raw_frame_rows.append(dict(frame=int(frame), timestamp_ns=members['info']['ref_timestamp_ns'],
                                       sequence_id=members['info']['sequence_id'], gravity_named_fields=fields))
    if all_gravity_fields:
        raise ValueError('New gravity fields found;review their provenance before assuming unknown')
    H = z['T_source_world_from_rest_palm']
    R = z['T_robot_world_from_robot_palm']
    O = z['T_source_world_from_object_BODY_F64']
    proper_inverse = np.linalg.inv(H)
    mapped = R @ proper_inverse @ O
    transported = gravity_transport_diagnostic(H, R, [0, 0, -9.81], [0, 0, -9.81])
    m = transported['metrics']
    checks = dict(
        source_frame_ids_all61=np.array_equal(z['frame_ids'], np.arange(47, 108)),
        raw_timestamp_bytes=np.array_equal(z['timestamps_ns'], np.array([row['timestamp_ns'] for row in raw_frame_rows], dtype=np.int64)),
        fixed_palm_mapping_recomputed=bool(np.max(np.abs(mapped - z['T_robot_world_from_object_BODY'])) <= 1e-12),
        gravity_transport_all_proper=bool(np.max(np.abs(np.linalg.det(transported['source_to_robot_rotation']) - 1)) <= 1e-10),
        eigenvalue_SVD_agreement=abs(m['minimum_RMS_unit_gravity_change_from_initial'] - m['independent_SVD_minimum']) <= 1e-12,
        minimizing_direction_attains_bound=abs(m['minimum_RMS_unit_gravity_change_from_initial'] - m['minimizing_direction_RMS_change']) <= 1e-12,
        source_gravity_remains_unknown=not transported['source_gravity_verified'],
        no_source_force_labels=not transported['source_force_ground_truth_created'])
    if not all(checks.values()):
        raise ValueError('Saved-frame or mathematical diagnostic mismatch')
    out.mkdir()
    paths = [Path(__file__), ROOT / 'src/supermimic_research/ego/gravity_transport.py',
             ROOT / 'src/supermimic_research/runtime/ego_reference.py', ROOT / 'src/supermimic_research/runtime/state_codec.py']
    receipt = freeze_sources(paths, out / 'source_snapshot', ROOT,
                             subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write(out / 'source_receipt.json', receipt)
    write(out / 'raw_metadata_scan.json', dict(rows=raw_frame_rows, tar_members=names,
          scope='All four per-frame JSON types in the retained61-frame window;absence of gravity-named fields is not proof no external calibration exists.'))
    arrays = {k: v for k, v in transported.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(out / 'gravity_transport_arrays.npz', **arrays, timestamps_ns=z['timestamps_ns'])
    web = []
    (out / 'primary_sources').mkdir()
    urls = dict(hot3d_paper='https://arxiv.org/html/2411.19167v2',
                MPS_trajectory='https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/slam/mps_trajectory',
                clips_README='https://raw.githubusercontent.com/facebookresearch/hot3d/main/hot3d/clips/README.md')
    for name, url in urls.items():
        entry = dict(url=url, observed_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'SuperMimic-Research-Metadata-Audit/1.0'})
            with urllib.request.urlopen(request, timeout=15) as response:
                content = response.read(2_000_001)
                entry['resolved_url'] = response.url
            if len(content) > 2_000_000:
                raise ValueError('Primary-source snapshot exceeds bounded2MB')
            path = out / 'primary_sources' / (name + '.source')
            path.write_bytes(content)
            entry['snapshot'] = file_pin(path)
        except Exception as error:
            entry['retrieval_error'] = type(error).__name__ + ': ' + str(error)
        web.append(entry)
    prior_path = ROOT / 'runs/mug_asset_validation_v2/physical_prior.json'
    physical = json.loads(prior_path.read_text())
    report = dict(schema='real_ego_fixed_palm_gravity_audit:1',
        source_receipt=file_pin(out / 'source_receipt.json'), frames_report=file_pin(frames_path), asset_report=file_pin(asset_path),
        inputs=dict(clip=frames['clip'], reference=frames['outputs']['reference61.npz'],
                    model=frames['outputs']['hand2_mug.mjb'], physical_prior=file_pin(prior_path)),
        checks=checks, passed=sum(checks.values()), total=len(checks), metrics=m,
        gravity_assumption=dict(source_world_m_s2=[0, 0, -9.81], status='CONDITIONAL_UNVERIFIED_NOT_CALIBRATION', robot_world_m_s2=[0, 0, -9.81]),
        minimum_RMS_gravity_change_for_any_source_direction_at_9p81_m_s2=m['minimum_RMS_unit_gravity_change_from_initial'] * 9.81,
        physical_status=dict(mass_inertia_friction_measured=physical['measured'],
                             nominal_mass_kg=physical['mass_kg'], nominal_only=True,
                             can_use_explicit_nominal_robot_simulation=True,
                             can_claim_source_load_or_force_ground_truth=False,
                             source_gravity_registered=False, independent_RGB_reconstruction=False),
        metadata_scan=file_pin(out / 'raw_metadata_scan.json'), arrays=file_pin(out / 'gravity_transport_arrays.npz'),
        primary_sources=web, actual_native_integrations=0, optimizer_steps=0,
        scope=transported['scope'] + ' Existing prior-based cup asset is usable for nominal exploration,not source-physics certification or successful ego retargeting. Current pen training and its physics/reference are untouched.')
    verify_source_snapshot(receipt, live=True)
    write(out / 'report.json', report)
    print(json.dumps(dict(report=file_pin(out / 'report.json'), checks=[report['passed'], report['total']], metrics=m,
                         primary_sources_downloaded=sum('snapshot' in item for item in web))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
