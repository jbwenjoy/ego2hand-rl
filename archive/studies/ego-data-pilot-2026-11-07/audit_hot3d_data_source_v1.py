"""Read-only JSON/tar audit; no extraction, image decode, geometry or models."""
import collections
import datetime
import hashlib
import itertools
import json
import math
import pathlib
import re
import struct
import tarfile

BASE = pathlib.Path(__file__).resolve().parent


def pin(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': h.hexdigest()}


def numeric_scan(x, at='', bad=None, nulls=None):
    if bad is None:
        bad, nulls = [], []
    if x is None:
        nulls.append(at)
    elif isinstance(x, float) and not math.isfinite(x):
        bad.append(at)
    elif isinstance(x, dict):
        for k, v in x.items():
            numeric_scan(v, at + '/' + str(k), bad, nulls)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            numeric_scan(v, at + '/' + str(i), bad, nulls)
    return bad, nulls


def transform(d, norms):
    q, t = d['quaternion_wxyz'], d['translation_xyz']
    assert len(q) == 4 and len(t) == 3
    assert all(isinstance(x, (int, float)) and math.isfinite(x) for x in q + t)
    n = math.sqrt(math.fsum(x*x for x in q))
    assert n > 0
    norms.append(n)


def summary(v):
    return {'min': min(v), 'max': max(v)} if v else None


def jpeg_header(f):
    """Read SOF dimensions/component count only; never decode image pixels."""
    assert f.read(2) == b'\xff\xd8'
    while True:
        assert f.read(1) == b'\xff'
        c = f.read(1)
        while c == b'\xff':
            c = f.read(1)
        m = c[0]
        assert m not in (0xd9, 0xda), 'no SOF before image data/end'
        if m == 0x01 or 0xd0 <= m <= 0xd7:
            continue
        n = struct.unpack('>H', f.read(2))[0]
        if m in {0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf}:
            p, height, width, components = struct.unpack('>BHHB', f.read(6))
            return [width, height, components, p]
        f.seek(n - 2, 1)


def inspect_clip(p, expected):
    pp = pin(p)
    assert pp['sha256'] == expected['sha256'] == expected['expected_LFS_SHA256']
    assert pp['bytes'] == expected['bytes']
    suffixes = ['cameras.json', 'hand_crops.json', 'hands.json', 'image_1201-1.jpg',
                'image_1201-2.jpg', 'image_214-1.jpg', 'info.json', 'objects.json']
    wanted = {f'{i:06d}.{s}' for i in range(150) for s in suffixes} | {'__hand_shapes.json__'}
    streams = ['1201-1', '1201-2', '214-1']
    times = {s: [] for s in streams}
    hand_counts = {s: collections.Counter() for s in ['left', 'right']}
    norms = {s: [] for s in ['camera_world', 'device_camera', 'hand_world', 'object_world']}
    angle_lengths, angle_tail_nonzero = collections.Counter(), 0
    objids, participants, sequences, devices, projection_models = set(), set(), set(), set(), set()
    object_records, json_count, null_count = 0, 0, 0
    object_names, member_type_counts, jpeg_shapes = {}, collections.Counter(), {s: collections.Counter() for s in streams}
    timestamps, json_hash = [], hashlib.sha256()
    shape_summary, ref_equal_rgb = {}, True
    with tarfile.open(p, 'r:') as tf:
        members = tf.getmembers()
        assert len(members) == len(wanted) and len({m.name for m in members}) == len(wanted)
        assert {m.name for m in members} == wanted and all(m.isfile() and m.size > 0 for m in members)
        for m in members:
            member_type_counts['shared_hand_shape' if m.name == '__hand_shapes.json__' else m.name.split('.', 1)[1]] += 1
        def read(name):
            nonlocal json_count, null_count
            raw = tf.extractfile(name).read()
            json_hash.update(name.encode() + b'\0' + raw)
            obj = json.loads(raw)
            bad, nulls = numeric_scan(obj)
            assert not bad, (name, bad[:3])
            json_count += 1
            null_count += len(nulls)
            return obj
        shape = read('__hand_shapes.json__')
        assert {'mano', 'umetrack'} <= shape.keys()
        shape_summary = {'keys': sorted(shape), 'mano_beta_count': len(shape['mano']),
                         'umetrack_keys': sorted(shape['umetrack']),
                         'joint_rest_position_count': len(shape['umetrack']['joint_rest_positions'])}
        for i in range(150):
            pref = f'{i:06d}.'
            info, cameras, hands, objects = [read(pref + s) for s in ['info.json', 'cameras.json', 'hands.json', 'objects.json']]
            read(pref + 'hand_crops.json')
            participants.add(info['participant_id']); sequences.add(info['sequence_id']); devices.add(info['device'])
            assert set(info['image_timestamps_ns']) == set(cameras) == set(streams)
            assert type(info['ref_timestamp_ns']) is int
            timestamps.append(info['ref_timestamp_ns'])
            ref_equal_rgb &= info['ref_timestamp_ns'] == info['image_timestamps_ns']['214-1']
            for s in streams:
                ts = info['image_timestamps_ns'][s]
                assert type(ts) is int
                times[s].append(ts)
                c = cameras[s]
                transform(c['T_world_from_camera'], norms['camera_world'])
                transform(c['calibration']['T_device_from_camera'], norms['device_camera'])
                assert c['calibration']['stream_id'] == s
                projection_models.add(c['calibration']['projection_model_type'])
                dims = jpeg_header(tf.extractfile(pref + f'image_{s}.jpg'))
                jpeg_shapes[s][tuple(dims)] += 1
                assert dims[:2] == [c['calibration']['image_width'], c['calibration']['image_height']]
                assert dims[2:] == [3 if s == '214-1' else 1, 8]
            for side in hand_counts:
                d = hands.get(side)
                if d is None:
                    hand_counts[side]['missing_or_null_hand'] += 1
                    continue
                hand_counts[side]['present'] += 1
                if 'umetrack_pose' not in d or d['umetrack_pose'] is None:
                    hand_counts[side]['missing_umetrack'] += 1
                    continue
                u = d['umetrack_pose']
                transform(u['T_world_from_wrist'], norms['hand_world'])
                angle_lengths[len(u['joint_angles'])] += 1
                angle_tail_nonzero += int(any(x != 0 for x in u['joint_angles'][20:]))
                hand_counts[side]['umetrack_present'] += 1
                hand_counts[side]['mano_present'] += int(d.get('mano_pose') is not None)
                hand_counts[side]['no_amodal_boxes'] += int(not d.get('boxes_amodal'))
            for key, instances in objects.items():
                assert isinstance(instances, list)
                for ob in instances:
                    assert int(key) == int(ob['object_bop_id'])
                    transform(ob['T_world_from_object'], norms['object_world'])
                    oid = int(ob['object_bop_id']); objids.add(oid); object_records += 1
                    object_names.setdefault(str(oid), set()).add(ob['object_name'])
    assert len(participants) == len(sequences) == len(devices) == 1 and devices == {'Aria'}
    assert all(all(b > a for a, b in zip(v, v[1:])) for v in times.values())
    assert all(b > a for a, b in zip(timestamps, timestamps[1:]))
    return {'tar': pp, 'frames': 150, 'members': len(members), 'eight_member_kinds_per_frame_complete': True,
            'member_kind_counts': dict(member_type_counts), 'json_records_checked': json_count,
            'json_read_order_payload_sha256': json_hash.hexdigest(), 'nonfinite_numeric_values': 0,
            'null_values_all_JSON': null_count, 'participant_ids': sorted(participants), 'sequence_ids': sorted(sequences),
            'device': sorted(devices), 'object_bop_ids': sorted(objids), 'object_instance_records': object_records,
            'object_names': {k: sorted(v) for k, v in object_names.items()}, 'hands': {k: dict(v) for k, v in hand_counts.items()},
            'umetrack_joint_angle_length_counts': dict(angle_lengths), 'umetrack_tail_after20_nonzero_records': angle_tail_nonzero,
            'quaternion_raw_norm_ranges': {k: summary(v) for k, v in norms.items()}, 'hand_shape': shape_summary,
            'projection_models': sorted(projection_models), 'JPEG_SOF_width_height_components_bits': {s: [{'shape': list(k), 'frames': v} for k, v in d.items()] for s, d in jpeg_shapes.items()},
            'timestamp_unit': 'ns (source field names and pinned source constants)', 'ref_equals_RGB_timestamp_all': ref_equal_rgb,
            'ref_range_ns': [timestamps[0], timestamps[-1]], 'ref_span_s': (timestamps[-1] - timestamps[0]) / 1e9,
            'ref_step_ns_range': summary([b-a for a, b in zip(timestamps, timestamps[1:])]),
            'camera_timestamp_minus_ref_ns_range': {s: summary([t-r for t,r in zip(v,timestamps)]) for s,v in times.items()}}


def main():
    manifest_path = BASE / 'download_manifest_v1.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    clips = [inspect_clip(pathlib.Path(e['path']), e) for e in manifest['clips']]
    src_path = BASE / 'sources/independent_parser_source_v1/source_manifest.json'
    source_refs = json.loads(src_path.read_text())
    for e in source_refs:
        assert pin(pathlib.Path(e['path']))['sha256'] == e['sha256']
    report = {'schema': 'hot3d_three_clip_data_source_independent:1', 'status': 'PASS_READ_ONLY_TAR_JSON_SCHEMA_AND_SOURCE_IDENTITY',
              'created_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'auditor': pin(pathlib.Path(__file__).resolve()),
              'download_manifest': pin(manifest_path), 'dataset_revision': manifest['revision'], 'clips': clips,
              'total_frames': 450, 'total_tar_bytes': sum(c['tar']['bytes'] for c in clips),
              'unique_participants': sorted({p for c in clips for p in c['participant_ids']}),
              'unique_source_sequences': sorted({p for c in clips for p in c['sequence_ids']}),
              'source_evidence': source_refs, 'source_manifest': pin(src_path),
              'parser_notes': [
                  'Actual UmeTrack T_world_from_wrist is a quaternion_wxyz/translation_xyz dict, not README wrist_xform 4x4.',
                  'Observed joint_angles length is preserved exactly; README 20 must not determine parser slicing.',
                  'Object T_world_from_object is object/model-to-world: official vis_clips.py129-137; README prose world-to-object is reversed.',
                  'Camera T_world_from_camera is camera-to-world; world-to-camera rendering uses its inverse, official vis_clips.py135-137.',
                  'Pose translations are meters per original HOT3D trajectory source; UmeTrack profile joint/landmark/mesh rest positions are millimeters converted x0.001 by official from_json54-61. Do not apply that conversion to world translations.',
                  'Timestamps are per-image ns; retain actual offsets and time steps, do not replace by uniform 1/30s.',
                  'Right-hand geometry is mirrored by the official FK wrist first-column operation; no FK or geometry was run here.',
                  'Missing boxes/visibility outside image are not missing hand pose. Frame-level annotations are not proof of contact/force.'],
              'scope': {'data_tar_JSON_and_JPEG_headers_only': True, 'pixel_decodes': 0, 'full_extractions': 0,
                        'Torch_NN_FK_models_pickle_retargeting_physics': 0, 'old_study_state_modified': False,
                        'independent_trajectories_claimed': False, 'contact_dynamics_qualified': False}}
    out = BASE / 'outputs/data_source_independent_v1.json'
    out.parent.mkdir(exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({'report': pin(out), 'frames': 450, 'participants': report['unique_participants'],
                      'sequences': report['unique_source_sequences'], 'clips': [{'path': c['tar']['path'], 'objects': c['object_bop_ids'],
                      'hands': c['hands'], 'angles': c['umetrack_joint_angle_length_counts'], 'ref_step_ns': c['ref_step_ns_range'],
                      'camera_offsets_ns': c['camera_timestamp_minus_ref_ns_range']} for c in clips]}, indent=2))


if __name__ == '__main__':
    main()
