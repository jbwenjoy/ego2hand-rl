"""Download three original geometry assets at the same revision as the pilot clips."""
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
REV = '30fe9674782f32e1e5edba98476b6ff4300132c5'
NAMES = ('models_info.json', 'obj_000009.glb', 'obj_000024.glb', 'obj_000032.glb')
LIMIT = 20_000_000

def main():
    tree_path = ROOT / 'sources/hot3d_original_models_tree.json'
    tree_bytes = tree_path.read_bytes()
    lookup = {x['path']: x for x in json.loads(tree_bytes)}
    selected = [lookup['object_models/' + name] for name in NAMES]
    if sum(x['size'] for x in selected) > LIMIT:
        raise ValueError('Selected model size exceeds budget')
    dest = ROOT / 'hot3d/object_models'
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in selected:
        path = dest / Path(item['path']).name
        temporary = path.with_suffix(path.suffix + '.partial')
        if path.exists() or temporary.exists():
            raise ValueError('Fresh destination required; preserving existing assets')
        url = f'https://huggingface.co/datasets/bop-benchmark/hot3d/resolve/{REV}/{item["path"]}'
        request = urllib.request.Request(url, headers={'User-Agent': 'ego-data-pilot/1.0'})
        digest = hashlib.sha256()
        count = 0
        with urllib.request.urlopen(request, timeout=30) as response, temporary.open('xb') as stream:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                count += len(block)
                if count > item['size']:
                    raise ValueError('Unexpected larger content')
                stream.write(block)
                digest.update(block)
        if count != item['size']:
            raise ValueError('Asset size mismatch')
        expected = item.get('lfs', {}).get('oid')
        if expected and digest.hexdigest() != expected:
            raise ValueError('LFS SHA256 mismatch')
        path_sha1 = item.get('oid')
        git_sha1 = None
        if not expected and path_sha1 and len(path_sha1) == 40:
            content = temporary.read_bytes()
            git_sha1 = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
            if git_sha1 != path_sha1:
                raise ValueError('Git blob SHA1 mismatch')
        temporary.rename(path)
        rows.append({'requested_url': url, 'path': str(path), 'bytes': count,
                     'sha256': digest.hexdigest(), 'expected_lfs_sha256': expected,
                     'verified_git_blob_sha1': git_sha1})
        print(json.dumps({'status': 'VERIFIED', **rows[-1]}), flush=True)
    result = {'schema': 'hot3d_selected_original_mesh_download:1', 'revision': REV,
              'created_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'selection': {'9': 'mug_white', '24': 'cellphone', '32': 'whiteboard_marker'},
              'source_tree_sha256': hashlib.sha256(tree_bytes).hexdigest(),
              'download_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'assets': rows, 'total_bytes': sum(x['bytes'] for x in rows),
              'all_scene_objects_downloaded': False,
              'mesh_loaded_or_contact_dynamics_verified': False}
    with (ROOT / 'selected_models_manifest_v1.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({'status': 'COMPLETE', 'total_bytes': result['total_bytes']}), flush=True)

if __name__ == '__main__':
    main()
