"""Source-only deterministic one-factor derivation; no model or solver imports."""
from pathlib import Path
import ast
import difflib
import hashlib
import json

ROOT = Path(__file__).resolve().parent
P = ROOT.parent
PARENT = P / 'surface_assembly_v9' / 'solve_assembly_support_v9.py'
PARENT_SHA = 'a320422c5f675748b61579742a0e3089029e16647ca5b5363adb1eb1b4efbacd'

def pin(p):
    p = Path(p).resolve()
    b = p.read_bytes()
    return {'path': str(p), 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}

def save(p, obj):
    with p.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def static_value(n):
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
        return static_value(n.left) * static_value(n.right)
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
        return static_value(n.left) + static_value(n.right)
    if isinstance(n, ast.Dict):
        return {static_value(k): static_value(v) for k, v in zip(n.keys, n.values)}
    if isinstance(n, ast.List):
        return [static_value(x) for x in n.elts]
    return ast.literal_eval(n)

def main():
    original = PARENT.read_bytes()
    assert hashlib.sha256(original).hexdigest() == PARENT_SHA
    edits = [
        (b'from assembly_kinematics_v9 import AssemblyKinematics',
         b"sys.path.insert(0,str(P/'surface_assembly_v9'))\nfrom assembly_kinematics_v9 import AssemblyKinematics"),
        (b"'q_trust_rad':.45", b"'q_trust_rad':None"),
        (b'lo=np.r_[np.maximum(e.qlo[:8]+1e-9,qbase[:8]-.45),xb[8:11]-.025,xb[11:]-.5]',
         b'lo=np.r_[e.qlo[:8]+1e-9,xb[8:11]-.025,xb[11:]-.5]'),
        (b'hi=np.r_[np.minimum(e.qhi[:8]-1e-9,qbase[:8]+.45),xb[8:11]+.025,xb[11:]+.5]',
         b'hi=np.r_[e.qhi[:8]-1e-9,xb[8:11]+.025,xb[11:]+.5]'),
        (b'assembly_then_hard_geometry_support_protocol:9', b'assembly_then_hard_geometry_support_protocol:10'),
        (b'assembly_then_hard_geometry_support_result:9', b'assembly_then_hard_geometry_support_result:10'),
    ]
    derived = original
    for old, new in edits:
        assert derived.count(old) == 1, old
        derived = derived.replace(old, new, 1)
    restored = derived
    for old, new in reversed(edits):
        assert restored.count(new) == 1, new
        restored = restored.replace(new, old, 1)
    assert restored == original
    tree = ast.parse(derived.decode('utf-8'))
    cfg = next(static_value(n.value) for n in tree.body
               if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CFG' for t in n.targets))
    target = ROOT / 'solve_assembly_support_v10.py'
    with target.open('xb') as f:
        f.write(derived)
    diff = ''.join(difflib.unified_diff(original.decode('utf-8').splitlines(True),
                                      derived.decode('utf-8').splitlines(True),
                                      fromfile='frozen_v9_a320422c', tofile='v10_full_hardware_q8'))
    diff_path = ROOT / 'v9_to_v10_exact.diff'
    with diff_path.open('x', encoding='utf-8', newline='') as f:
        f.write(diff)
    parent_protocol = P / 'surface_assembly_v9' / 'protocol_v1.json'
    protocol = json.loads(parent_protocol.read_text(encoding='utf-8'))
    old_cfg = protocol['config'].copy()
    old_cfg['q_trust_rad'] = None
    assert cfg == old_cfg
    old_sources = protocol['source_pins'].copy()
    protocol['schema'] = 'assembly_then_hard_geometry_support_protocol:10'
    protocol['CPU_run_authorized'] = False
    protocol['config'] = cfg
    protocol['output_path'] = str(ROOT / 'completed_v1')
    protocol['source_pins']['runner'] = pin(target)
    assert all(protocol['source_pins'][k] == v for k, v in old_sources.items() if k != 'runner')
    protocol['single_factor_ablation'] = {
        'parent_runner': pin(PARENT), 'parent_protocol': pin(parent_protocol),
        'change': 'Remove only algorithmic q8 +/-0.45 rad trust intersection; use original hardware qlo+1e-9, qhi-1e-9.',
        'unchanged': 'Original two seeds, remaining12q, phone translation/rotvec boxes, material anchors, fixed model/wrist/scale, numerical methods, budgets, geometry, support and native final gates.'}
    save(ROOT / 'protocol_metadata_v1.json', protocol)
    save(ROOT / 'source_derivation_v1.json', {
        'schema': 'surface_assembly_full_hardware_q8_source_derivation:10',
        'parent_source': pin(PARENT), 'derived_source': pin(target), 'unified_diff': pin(diff_path),
        'derivation_script': pin(__file__), 'parent_protocol': pin(parent_protocol),
        'metadata_protocol': pin(ROOT / 'protocol_metadata_v1.json'),
        'active_bound_basis': pin(P / 'surface_assembly_v9' / 'active_bound_review_v1' / 'report.json'),
        'exact_reverse_edits_equal_parent_bytes': restored == original,
        'enumerated_edits': [{'old': a.decode(), 'new': b.decode()} for a, b in edits],
        'AST_parse_passed': True, 'config_changed_keys': ['q_trust_rad'],
        'model_loads': 0, 'FK_forward_LP_steps_training_GPU': 0,
        'scope': 'Bounded full-hardware-q8 search ablation only; not holding, faithful ego or learned-method qualification.'})
    print(json.dumps({'source': pin(target), 'metadata_protocol': pin(ROOT / 'protocol_metadata_v1.json'),
                      'derivation': pin(ROOT / 'source_derivation_v1.json')}))

if __name__ == '__main__':
    main()
