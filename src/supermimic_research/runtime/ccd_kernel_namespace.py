"""Avoid upstream cache_kernel's function-name collision between source variants."""
import re


def namespace_builders(source, namespace):
    if not re.fullmatch(r'[a-z][a-z0-9_]*', namespace):
        raise ValueError('Explicit isolated kernel namespace required')
    names = re.findall(r'@cache_kernel\s+def ([a-z0-9_]+)\(', source)
    if set(names) != {'ccd_hfield_kernel_builder', 'ccd_kernel_builder'} or len(names) != 2:
        raise ValueError('Exactly two known cached builders required')
    mapping = {name: name + '_' + namespace for name in names}
    for old, new in mapping.items(): source = re.sub(r'\b' + old + r'\b', new, source)
    return source, mapping
