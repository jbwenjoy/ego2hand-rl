"""Create an explicit one-allocation capacity artifact, never edit the package."""
from pathlib import Path

ALLOCATION = 'epa_horizon = wp.empty(shape=(d.naccdmax, MJ_MAX_EPAHORIZON), dtype=int)'


def capacity_source(original, size):
    if size not in [48, 96] or original.count(ALLOCATION) != 1:
        raise ValueError('One known EPA allocation and explicit48/96 capacity required')
    return original.replace(ALLOCATION, f'epa_horizon = wp.empty(shape=(d.naccdmax, {size}), dtype=int)')


def write_capacity_artifact(source, target, size):
    source, target = Path(source), Path(target)
    if target.exists(): raise ValueError('No isolated artifact replacement')
    original = source.read_text(encoding='utf-8')
    result = capacity_source(original, size)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result, encoding='utf-8', newline='\n')
    if source.read_text(encoding='utf-8') != original:
        raise ValueError('Installed source changed during copy')
    return result
