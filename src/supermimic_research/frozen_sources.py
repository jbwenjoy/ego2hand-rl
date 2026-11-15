"""Explicit source-byte snapshots and exact historical source relocations."""
from pathlib import Path
import shutil

from .artifacts import file_pin, verify_file_pin


def freeze_sources(paths, destination, root, revision):
    root=Path(root).resolve(strict=True);destination=Path(destination).resolve()
    paths=[Path(p).resolve(strict=True) for p in paths]
    if destination.exists() or len(paths)!=len(set(paths)):
        raise ValueError('Fresh source destination and unique source paths required')
    for p in paths:
        if not p.is_relative_to(root) or p.is_relative_to(destination):
            raise ValueError('Only explicit repository source paths may be captured')
    destination.mkdir(parents=True);records=[]
    for p in paths:
        observed=file_pin(p);target=destination/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target);snapshot=file_pin(target)
        if any(observed[k]!=snapshot[k] for k in ['bytes','sha256']):
            raise ValueError('Source changed during capture')
        records.append(dict(observed=observed,snapshot=snapshot))
    return dict(schema='frozen_source_bytes:1',code_revision=revision,records=records,
                scope='Captured source superset, not proof every module executed. Active observed paths must match before/after collection; snapshots stay authoritative for later review.')


def verify_source_snapshot(manifest, *, live=False):
    if manifest['schema']!='frozen_source_bytes:1':raise ValueError('Unknown source snapshot schema')
    seen=set()
    for record in manifest['records']:
        observed,snapshot=record['observed'],record['snapshot']
        if observed['path'] in seen:raise ValueError('Duplicate observed source')
        seen.add(observed['path'])
        if any(observed[k]!=snapshot[k] for k in ['bytes','sha256']):
            raise ValueError('Snapshot is not the observed source bytes')
        verify_file_pin(snapshot)
        if live:verify_file_pin(observed)


def verify_source_pin(pin, relocations=None):
    """No basename/SHA-only fallback; relocation must match full original pin."""
    if relocations is not None:
        if relocations['schema']!='explicit_source_relocations:1':raise ValueError('Unknown source relocations')
        matches=[r for r in relocations['records'] if r['original']==pin]
        if len(matches)>1:raise ValueError('Ambiguous source relocation')
        if matches:
            archived=matches[0]['archived']
            if any(pin[k]!=archived[k] for k in ['bytes','sha256']):raise ValueError('Source relocation changes bytes')
            return verify_file_pin(archived)
    return verify_file_pin(pin)


def frozen_import_sources(manifest, source_root, namespace):
    """Enumerate this live process's namespace imports, never a reconstructed past process."""
    import sys
    verify_source_snapshot(manifest,live=True)
    source_root=Path(source_root).resolve(strict=True)
    mapping={r['observed']['path']:r['snapshot'] for r in manifest['records']}
    result=[]
    for name,module in sorted(list(sys.modules.items())):
        path=getattr(module,'__file__',None)
        if path and (name==namespace or name.startswith(namespace+'.')):
            path=Path(path).resolve(strict=True)
            if not path.is_relative_to(source_root) or str(path) not in mapping:
                raise ValueError('Imported source outside captured namespace sources')
            result.append(mapping[str(path)])
    return result
