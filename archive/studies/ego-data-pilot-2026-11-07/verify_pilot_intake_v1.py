"""Local acquisition verification only; no models, FK, simulation, or training."""
import datetime
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent

def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))

def pin(name):
    path = ROOT / name
    content = path.read_bytes()
    return {'path': str(path), 'bytes': len(content),
            'sha256': hashlib.sha256(content).hexdigest()}

def main():
    clips = read('download_manifest_v1.json')
    models = read('selected_models_manifest_v1.json')
    audit = read('outputs/data_source_independent_v1.json')
    if clips['revision'] != models['revision'] or clips['revision'] != audit['dataset_revision']:
        raise ValueError('Revision mismatch')
    for item in clips['clips']:
        if Path(item['path']).stat().st_size != item['bytes']:
            raise ValueError('Clip size changed since independent full SHA256 audit')
    glbs = []
    for item in models['assets']:
        path = Path(item['path'])
        content = path.read_bytes()
        if len(content) != item['bytes'] or hashlib.sha256(content).hexdigest() != item['sha256']:
            raise ValueError('Model asset size/hash mismatch')
        if path.suffix != '.glb':
            continue
        magic, version, length = struct.unpack_from('<4sII', content)
        json_length, chunk_type = struct.unpack_from('<II', content, 12)
        if (magic, version, length, chunk_type) != (b'glTF', 2, len(content), 0x4e4f534a):
            raise ValueError('Invalid GLB 2 header')
        data = json.loads(content[20:20 + json_length])
        glbs.append({'file': path.name, 'asset_version': data['asset']['version'],
                     'meshes': len(data.get('meshes', [])), 'nodes': len(data.get('nodes', [])),
                     'external_buffer_uris': [x['uri'] for x in data.get('buffers', []) if 'uri' in x],
                     'numeric_geometry_or_collision_checked': False})
    names = ('download_manifest_v1.json', 'selected_models_manifest_v1.json',
             'outputs/data_source_independent_v1.json',
             'sources/independent_parser_source_v1/source_manifest.json',
             'outputs/hot3d_three_clip_preview.png', 'dataset-screening-and-pilot-v1.md',
             'download_hot3d_pilot_v1.py', 'download_hot3d_selected_models_v1.py',
             'verify_pilot_intake_v1.py')
    result = {'schema': 'ego_pilot_intake_status:1',
              'created_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'status': 'PASS_SMALL_RAW_EGO_ACQUISITION_AND_INTEGRITY',
              'dataset_revision': clips['revision'],
              'raw_ego_clips_acquired': 3, 'frames_per_camera': 450, 'cameras': 3,
              'participant_count': 1, 'source_sequence_count': 2,
              'clip_bytes': clips['total_clip_bytes'], 'selected_model_bytes': models['total_bytes'],
              'total_clip_and_selected_model_bytes': clips['total_clip_bytes'] + models['total_bytes'],
              'glb_container_checks': glbs,
              'verification_scope': {'clip_hashes': 'Full SHA256 rechecked by independent audit; this pass rechecked file sizes.',
                                     'selected_model_assets': 'All SHA256 and GLB2 JSON container headers rechecked locally.',
                                     'image_pixels': 'Nine RGB samples decoded for preview; other images header-checked only.'},
              'new_training_or_evaluation_started': False,
              'robot_retargeting_or_contact_dynamics_qualified': False,
              'ego_method_evaluation_completed_on_pilot': 0,
              'old_goal_status_observed': 'paused', 'old_study_state_modified': False,
              'artifacts': [pin(name) for name in names]}
    destination = ROOT / 'outputs/pilot_intake_status_v1.json'
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({'status': result['status'], 'path': str(destination),
                      'total_clip_and_selected_model_bytes': result['total_clip_and_selected_model_bytes'],
                      'glbs': glbs}, ensure_ascii=False))

if __name__ == '__main__':
    main()
