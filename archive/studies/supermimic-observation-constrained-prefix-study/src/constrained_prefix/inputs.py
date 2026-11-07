"""Pinned plans; the parent protocol supplies targets, never a solver seed."""
from pathlib import Path
import sys
from .io import sha,read

ROOT=Path(__file__).resolve().parents[2]
PARENT=Path('D:/workspace/supermimic-observation-prefix-study')
PROTOCOL=ROOT/'protocols/constrained-prefix-v1.json'
PROTOCOL_SHA='0798ea7aa6da6f56684067992906bbba3136f9e15b413f743674f54159ad2e6d'
RETARGET_VARIANT_ID='H123_cubic_bspline_constrained_prefix16_v1'
sys.path.insert(0,str(PARENT/'src'))
from prefix_study.inputs import PROTOCOL as PARENT_PROTOCOL,PROTOCOL_SHA as PARENT_PROTOCOL_SHA,checked_protocol


def checked_plan(path,*,run=False):
    p=read(path)
    if p.get('schema')!='constrained_prefix_geometry_cpu_plan:1':raise ValueError('Plan schema')
    if run and p.get('optimization_authorized') is not True:raise ValueError('Root optimization authorization absent')
    for key in ('protocol','parent_protocol'):
        if sha(p[key]['path'])!=p[key]['sha256']:raise ValueError('Protocol drift: '+key)
    if p['parent_protocol']['sha256']!=PARENT_PROTOCOL_SHA or Path(p['parent_protocol']['path']).resolve()!=PARENT_PROTOCOL.resolve():raise ValueError('Wrong parent target contract')
    if p['protocol']['sha256']!=PROTOCOL_SHA or Path(p['protocol']['path']).resolve()!=PROTOCOL.resolve():raise ValueError('Wrong constrained protocol')
    parent=checked_protocol(p['parent_protocol']['path'])
    protocol=read(p['protocol']['path'])
    if protocol.get('schema')!='constrained_prefix_protocol:1' or protocol['inputs']!=parent['inputs'] or protocol['parent_protocol']['sha256']!=PARENT_PROTOCOL_SHA:raise ValueError('Constrained protocol input/schema differs')
    if p['inputs']!=parent['inputs']:raise ValueError('All parent target/input identities must remain unchanged')
    expected={'frames':16,'joints':20,'geometry_samples':61,'n_control':18,'free_variables':340,'maxiter':80,'max_evaluations':250,'soft_wall_seconds':220,'owned_process_seconds':240,'automatic_retry':False,
              'planning_depth_bound_m':.002999,'control_margin_rad':.000002,'force_margin_Nm':.000002,'rate_margin_rad_s':.0001,'retarget_variant_id':RETARGET_VARIANT_ID}
    if any(p.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen constrained shape/budget/margin differs')
    for name,pin in p['source_pins'].items():
        if sha(pin['path'])!=pin['sha256']:raise ValueError('Source drift: '+name)
    required=[ROOT/'scripts/build_constrained_prefix_geometry_cpu.py',ROOT/'src/bspline_control.py',ROOT/'vendor/runtime_io.py',*sorted((ROOT/'src/constrained_prefix').glob('*.py')),
              PARENT/'src/prefix_curves.py',PARENT/'src/prefix_control.py',*sorted((PARENT/'src/prefix_study').glob('*.py')),PARENT/'vendor/runtime_io.py']
    required += [Path(parent['inputs'][k]['path']) for k in ('kinematics','native_filtered_geometry','material_geometry')]
    bound={str(Path(v['path']).resolve()).lower():v['sha256'] for v in p['source_pins'].values()}
    for file in required:
        if bound.get(str(file.resolve()).lower())!=sha(file):raise ValueError('Unbound executed dependency: '+str(file))
    return p,parent,protocol
