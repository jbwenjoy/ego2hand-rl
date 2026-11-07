"""Fail-closed source/seed/target plans for the separately frozen R/F method."""
from pathlib import Path
from .io import read,sha

ROOT=Path(__file__).resolve().parents[2]
FROZEN=Path('D:/workspace/supermimic-observation-constrained-prefix-study')
PARENT=Path('D:/workspace/supermimic-observation-prefix-study')
PROTOCOL=ROOT/'protocols/restoration-filter-prefix-v1.json'
PROTOCOL_SHA='3c4113a60567321fdfc57fa201ad5edc194475d72493f840d1ac8ce71acb387c'
TARGET_PROTOCOL=FROZEN/'protocols/constrained-prefix-v1.json'
TARGET_SHA='0798ea7aa6da6f56684067992906bbba3136f9e15b413f743674f54159ad2e6d'
PARENT_PROTOCOL=Path('D:/workspace/supermimic-observation-dynamic-wrench-study/proposals/short_dynamic_tracking_v1/protocol.json')
PARENT_SHA='94ea7cbe03f3e2aaa7bb4f7c8860bae8e599cd4e8e823cbe47220f73ffe101a3'
SEED=FROZEN/'outputs/constrained_prefix_geometry_v1/candidate.npz'
SEED_SHA='64aa2cfe44f20fee4546b07c4546916c95d57771404cb5fb4e39156206ad9b43'
RETARGET_VARIANT_ID='H123_cubic_minimax_restoration_filtered_prefix16_v1'


def required_sources(parent):
    return [ROOT/'scripts/build_restoration_prefix_cpu.py',ROOT/'vendor/runtime_io.py',*sorted((ROOT/'src/restoration_prefix').glob('*.py')),
            FROZEN/'src/bspline_control.py',FROZEN/'vendor/runtime_io.py',*sorted((FROZEN/'src/constrained_prefix').glob('*.py')),
            PARENT/'src/prefix_curves.py',PARENT/'src/prefix_control.py',PARENT/'vendor/runtime_io.py',*sorted((PARENT/'src/prefix_study').glob('*.py')),
            *[Path(parent['inputs'][key]['path']) for key in ('native_filtered_geometry','kinematics','material_geometry')]]


def checked_plan(path,*,run=False):
    p=read(path)
    if p.get('schema')!='restoration_prefix_cpu_plan:1':raise ValueError('Wrong new restoration plan schema')
    if run and p.get('optimization_authorized') is not True:raise ValueError('Root optimization authorization absent')
    expected={'R_maxiter':60,'R_max_objective_calls':200,'R_ftol':1e-9,'F_max_major':8,'QP_maxiter':25,'QP_max_objective_calls':100,'QP_ftol':1e-9,'soft_wall_seconds':340,'owned_process_seconds':360,'automatic_retry':False,'retarget_variant_id':RETARGET_VARIANT_ID}
    if any(p.get(k)!=v for k,v in expected.items()):raise ValueError('R/F frozen budgets/variant differ')
    for role,file,digest in [('protocol',PROTOCOL,PROTOCOL_SHA),('target_protocol',TARGET_PROTOCOL,TARGET_SHA),('parent_protocol',PARENT_PROTOCOL,PARENT_SHA),('seed_candidate',SEED,SEED_SHA)]:
        pin=p[role]
        if Path(pin['path']).resolve()!=file.resolve() or pin['sha256']!=digest or sha(pin['path'])!=digest:raise ValueError('Wrong frozen binding: '+role)
    parent=read(p['parent_protocol']['path']);target=read(p['target_protocol']['path']);protocol=read(p['protocol']['path'])
    if protocol.get('schema')!='restoration_prefix_protocol:1' or protocol['retarget_variant_id']!=RETARGET_VARIANT_ID or protocol['parent_protocol']['sha256']!=TARGET_SHA or protocol['input_parent_protocol']['sha256']!=PARENT_SHA:raise ValueError('New method protocol schema/variant/parent differs')
    if p['inputs']!=parent['inputs'] or p['inputs']!=target['inputs'] or protocol['inputs']!=target['inputs']:raise ValueError('Frozen target/input identity differs')
    for name,pin in p['inputs'].items():
        if sha(pin['path'])!=pin['sha256']:raise ValueError('Input/source drift: '+name)
    for role,name in [('linear_contract','linear_contract.npz'),('linear_contract_metadata','linear_contract_metadata.json')]:
        pin=p[role];expected_file=FROZEN/'outputs/constrained_prefix_geometry_v1'/name
        if Path(pin['path']).resolve()!=expected_file.resolve() or sha(pin['path'])!=pin['sha256']:raise ValueError('Wrong original exact linear artifact')
        if pin!=protocol['linear_metadata' if role=='linear_contract_metadata' else role]:raise ValueError('Linear artifact not frozen by new protocol')
    if protocol['seed_candidate']!=p['seed_candidate']:raise ValueError('Protocol seed differs')
    for role in ('seed_manifest','seed_report'):
        if sha(protocol[role]['path'])!=protocol[role]['sha256']:raise ValueError('Frozen seed provenance drift: '+role)
    for name,pin in p['source_pins'].items():
        if sha(pin['path'])!=pin['sha256']:raise ValueError('Source drift: '+name)
    bound={str(Path(v['path']).resolve()).lower():v['sha256'] for v in p['source_pins'].values()}
    for file in required_sources(parent):
        if bound.get(str(file.resolve()).lower())!=sha(file):raise ValueError('Unbound executed dependency: '+str(file))
    return p,protocol,parent
