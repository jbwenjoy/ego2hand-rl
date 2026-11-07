"""Draft source derivation and two CPU-False protocols; no production import."""
from pathlib import Path
import ast
import difflib
import hashlib
import json

R = Path(__file__).resolve().parent
P = R.parent

def pin(path):
    path = Path(path).resolve(strict=True)
    b = path.read_bytes()
    return {'path': str(path), 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}

def put(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def derive(source, target, edits, expected_sha):
    raw = source.read_bytes()
    assert pin(source)['sha256'] == expected_sha
    b = raw
    for old, new in edits:
        assert b.count(old.encode()) == 1, old
        b = b.replace(old.encode(), new.encode(), 1)
    back = b
    for old, new in reversed(edits):
        assert back.count(new.encode()) == 1, new
        back = back.replace(new.encode(), old.encode(), 1)
    assert back == raw
    ast.parse(b.decode())
    with target.open('xb') as f:
        f.write(b)
    diff = ''.join(difflib.unified_diff(raw.decode().splitlines(True), b.decode().splitlines(True),
                                      fromfile=str(source.name), tofile=str(target.name)))
    diffpath = target.with_suffix('.diff')
    with diffpath.open('x', encoding='utf-8', newline='') as f:
        f.write(diff)
    return {'parent': pin(source), 'derived': pin(target), 'diff': pin(diffpath),
            'exact_enumerated_reverse_bytes_equal': back == raw,
            'edits': [{'old': a, 'new': b} for a, b in edits]}

def main():
    helper = P / 'surface_assembly_v9/assembly_kinematics_v9.py'
    helper_edits = [
        ('filled-triangle material anchors/normals come from the immutable307e91 source.',
         'filled-triangle anchors/normals come from an explicit pinned schema11 source branch.'),
        ("ANCHOR_PATH=PILOT/'contact_patch_v7/paired_surface_anchors_v1.json'\nANCHOR_SHA='307e91a728c14fc4f799465649bc11f71c1b0be0971a3ec289d2a1acfaf565bb'",
         'from anchor_validation_v11 import validate_anchor_bundle, verify_compiled_anchors'),
        ('def __init__(self,geometry_evaluator,*,anchors_path=ANCHOR_PATH,qrest12=None):',
         'def __init__(self,geometry_evaluator,*,anchors_path,expected_anchor_pin,qrest12=None):'),
        ("        record=pin(anchors_path)\n        if record['sha256']!=ANCHOR_SHA:raise ValueError('Fixed actual307e91 triangle anchors required')\n        a=json.loads(Path(anchors_path).read_text(encoding='utf8'))\n        if (a['schema']!='saved_filled_triangle_paired_surface_anchors:7'\n                or not a['checks'] or any(v is not True for v in a['checks'].values())):\n            raise ValueError('Completed original anchor contract')",
         "        a,self.anchor_json_checks=validate_anchor_bundle(anchors_path,expected_anchor_pin)\n        record=self.anchor_json_checks['anchor_pin']\n        self.compiled_anchor_checks=verify_compiled_anchors(self.m,mujoco,a,self.oid)"),
        ("'fixed_original_surface_assembly_kinematics:9'", "'source_region_surface_assembly_kinematics:11'"),
        ("'anchor_pin':self.anchor_pin,", "'anchor_pin':self.anchor_pin,'anchor_branch':self.anchors['branch'],"),
    ]
    hd = derive(helper, R / 'assembly_kinematics_v11.py', helper_edits,
                '6eb8feaf190a5ae4951122851100ddd713454456db94ca3bf69dbc587a30dbfd')
    runner = P / 'surface_assembly_v10/solve_assembly_support_v10.py'
    runner_edits = [
        ("sys.path.insert(0,str(P/'surface_assembly_v9'))\nfrom assembly_kinematics_v9 import AssemblyKinematics",
         "sys.path.insert(0,str(P/'surface_assembly_v11'))\nfrom assembly_kinematics_v11 import AssemblyKinematics\nfrom anchor_validation_v11 import validate_anchor_bundle"),
        ("'assembly_then_hard_geometry_support_protocol:10'", "'assembly_then_hard_geometry_support_protocol:11'"),
        ("    if protocol['source_pins']['runner']!=pin(__file__):raise ValueError('Exact runner')",
         "    if protocol['source_pins']['runner']!=pin(__file__):raise ValueError('Exact runner')\n    anchor_bundle,anchor_json_checks=validate_anchor_bundle(protocol['inputs']['surface_anchors']['path'],protocol['inputs']['surface_anchors'])\n    if anchor_bundle['branch']!=protocol['anchor_branch']:raise ValueError('Explicit source mapping branch')"),
        ("kin=AssemblyKinematics(e,qrest12=qbase[8:]);",
         "kin=AssemblyKinematics(e,anchors_path=protocol['inputs']['surface_anchors']['path'],expected_anchor_pin=protocol['inputs']['surface_anchors'],qrest12=qbase[8:]);"),
        ("write(out/'algorithm_surface_configuration.json',{'source_anchor_pin':protocol['inputs']['surface_anchors'],",
         "write(out/'algorithm_surface_configuration.json',{'source_anchor_pin':protocol['inputs']['surface_anchors'],\n          'source_mapping_branch':protocol['anchor_branch'],'custom_anchor_JSON_checks':anchor_json_checks,'custom_anchor_compiled_checks':kin.compiled_anchor_checks,"),
        ('Caller v9 engineering triangle material points from frozen307 anchors; not original ego material targets or actual native contact line.',
         'Caller v11 pinned source-region material points; exact source and algorithmic1mm branches remain distinct. Not actual native contact line.'),
        ("'assembly_then_hard_geometry_support_result:10'", "'assembly_then_hard_geometry_support_result:11'"),
        ("'protocol':pin(a.protocol),'seed_results':seeds,",
         "'protocol':pin(a.protocol),'anchor_branch':protocol['anchor_branch'],'seed_results':seeds,"),
    ]
    rd = derive(runner, R / 'solve_assembly_support_v11.py', runner_edits,
                '44abfc1099d0359856003847b1577874c31eeeab5ea0d25badf902452dcd65ca')
    protocols = {}
    for branch in ['exact_source_original_hand', 'source1mm_opposed_side_original_hand']:
        p = json.loads((P / 'surface_assembly_v10/protocol_metadata_v1.json').read_text(encoding='utf-8'))
        p['schema'] = 'assembly_then_hard_geometry_support_protocol:11'
        p['CPU_run_authorized'] = False
        p['anchor_branch'] = branch
        p['output_path'] = str(R / (branch + '_completed_v1'))
        p['source_pins']['runner'] = pin(R / 'solve_assembly_support_v11.py')
        p['source_pins']['cheap_FK'] = pin(R / 'assembly_kinematics_v11.py')
        p['source_pins']['anchor_validation'] = pin(R / 'anchor_validation_v11.py')
        p['inputs']['surface_anchors'] = pin(P / 'source_surface_mapping_v11' / (branch + '_anchors_v1.json'))
        p.pop('single_factor_ablation', None)
        p['matched_source_mapping_scope'] = 'Same v10 full hardware q8, rest12, original two seeds, phone boxes, model, methods and budgets; branch changes source phone B/normal/tangent only, old hand triangles retained. Source1mm is algorithmic sensitivity, not empirical uncertainty.'
        p['prior_contract_scope'] = 'Old14/87/4 are parent provenance. New pinned anchor JSON validation runs before default return; same original model compiled vertex/support/body-frame checks run in custom constructor. Existing same-x A/B/q checks and actual full-state anchor J remain.'
        path = R / ('protocol_' + branch + '_metadata_v1.json')
        put(path, p)
        protocols[branch] = pin(path)
    put(R / 'source_derivation_v11.json', {'schema': 'source_surface_mapping_solver_derivation:11',
        'status': 'DRAFT_SOURCE_AND_METADATA_ONLY_ROOT_EXECUTION_PENDING',
        'helper': hd, 'runner': rd, 'validation': pin(R / 'anchor_validation_v11.py'),
        'derivation_source': pin(__file__), 'protocols': protocols,
        'budget_or_optimization_equation_changes': False, 'model_FK_forward_LP_steps_training_calls': 0})
    print(json.dumps({'runner': pin(R / 'solve_assembly_support_v11.py'),
                      'helper': pin(R / 'assembly_kinematics_v11.py'), 'validation': pin(R / 'anchor_validation_v11.py'),
                      'derivation': pin(R / 'source_derivation_v11.json'), 'protocols': protocols}))

if __name__ == '__main__':
    main()
