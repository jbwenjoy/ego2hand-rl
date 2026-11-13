"""Compare explicit cube/jig assets against immutable archive loader physics."""
from pathlib import Path
import argparse
import json
import sys
import types
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from supermimic_research.artifacts import ArtifactRegistry, file_pin
from supermimic_research.runtime.native_evaluation import model_digest_except_dt
from dex_supermimic.assets.inhand_cube import _get_spec
from dex_supermimic.assets.wuji_hand2 import get_wuji_hand2_cfg
from dex_supermimic.tasks.dex_reorient.config.wuji_hand2.hand2_constants import get_hand2_reorient_spec


def archived_module(path, virtual_file):
    module = types.ModuleType('archived_controlled_asset_reference')
    module.__file__ = str(virtual_file)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def main(args):
    if args.output.exists(): raise ValueError('Fresh output')
    registry = ArtifactRegistry(args.manifest)
    cube, jig = [registry.resolve(n) for n in ['controlled_cube_xml','controlled_jig_mesh']]
    archive = ROOT/'archive/studies/dex-supermimic-research/src/dex_supermimic'
    # Only archive code is executed, with a declared resource anchor to satisfy
    # the old import-time assertion. No historical study Python is imported.
    oldcube = archived_module(archive/'assets/inhand_cube.py', cube.parents[3]/'inhand_cube.py')
    oldrig = archived_module(archive/'tasks/dex_reorient/config/wuji_hand2/hand2_constants.py',
                             archive/'tasks/dex_reorient/config/wuji_hand2/hand2_constants.py')
    oldrig.WUJI_HAND2_JIG_TABLE_MESH = jig
    cases = []
    for edge in [None, .045, .065]:
        a, b = oldcube._get_spec(edge).compile(), _get_spec(edge, xml_path=cube).compile()
        left, right = [model_digest_except_dt(m,np) for m in [a,b]]
        cases.append(dict(component='cube',edge_m=edge,old_digest=left,new_digest=right,exact=left==right))
    base = get_wuji_hand2_cfg().spec_fn
    a,b = oldrig.get_hand2_reorient_spec(base).compile(), get_hand2_reorient_spec(base,jig_mesh=jig).compile()
    left,right = [model_digest_except_dt(m,np) for m in [a,b]]
    cases.append(dict(component='full_hand_and_rig',old_digest=left,new_digest=right,exact=left==right))
    assets = [file_pin(p) for p in sorted(cube.parents[1].rglob('*')) if p.is_file()]+[file_pin(jig)]
    sources=[file_pin(__file__)]+[file_pin(archive/p) for p in ['assets/inhand_cube.py','tasks/dex_reorient/config/wuji_hand2/hand2_constants.py']]
    sources += [file_pin(ROOT/'src/dex_supermimic'/p) for p in ['assets/inhand_cube.py','tasks/dex_reorient/config/wuji_hand2/hand2_constants.py']]
    result=dict(schema='controlled_cube_asset_equivalence:1',sources=sources,assets=assets,cases=cases,
                passed=sum(c['exact'] for c in cases),total=len(cases),neural_updates=0,native_integration_steps=0)
    args.output.mkdir()
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(passed=result['passed'],total=result['total'])))
    if result['passed']!=result['total']:raise ValueError('Keep failed equivalence')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
