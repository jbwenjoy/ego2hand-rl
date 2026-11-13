"""Bind explicit external assets to existing task factories without global edits."""
from functools import partial


def bind_controlled_assets(cfg, task, registry):
    names = ['controlled_pen_xml'] if task == 'hand2-pen-spin' else ['controlled_cube_xml', 'controlled_jig_mesh']
    if task not in ['hand2-pen-spin', 'hand2-reorient']:
        raise ValueError('Controlled Hand2 task required')
    pins = {name: registry.pin(name) for name in names}
    xml = registry.resolve(names[0])
    cfg.scene.entities['object'].spec_fn = partial(cfg.scene.entities['object'].spec_fn, xml_path=xml)
    if task == 'hand2-reorient':
        cfg.scene.entities['robot'].spec_fn = partial(cfg.scene.entities['robot'].spec_fn,
                                                     jig_mesh=registry.resolve('controlled_jig_mesh'))
    return pins
