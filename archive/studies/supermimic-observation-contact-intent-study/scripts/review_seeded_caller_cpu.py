"""Read-only AST/source comparison of seeded vs frozen distal-v2 caller.

Does not import or invoke either caller, geometry, optimizer, Torch, or Warp.
"""
from pathlib import Path
import ast
import difflib
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/seeded_caller_review_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n):return ast.dump(n,include_attributes=False)
def main():
    old=ROOT/'scripts/refine_initial_contacts_distal_cpu.py'
    new=ROOT/'scripts/refine_initial_contacts_seeded_cpu.py'
    a,b=old.read_text(),new.read_text();ta,tb=ast.parse(a),ast.parse(b)
    def named(t,name):return next(n for n in ast.walk(t) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
    def call(t,name):return next(n for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==name)
    checks={};details={}
    for name in ('geometry_only','contact_records','json_anchor','residual','jacobian'):
        na,nb=named(ta,name),named(tb,name)
        checks[name+'_AST_identical']=dump(na)==dump(nb)
        checks[name+'_source_segment_identical']=ast.get_source_segment(a,na)==ast.get_source_segment(b,nb)
    ca,cb=call(ta,'least_squares'),call(tb,'least_squares')
    checks['least_squares_only_x0_changes']=len(ca.args)==len(cb.args)==2 and dump(ca.args[0])==dump(cb.args[0]) and [dump(k) for k in ca.keywords]==[dump(k) for k in cb.keywords] and ast.unparse(ca.args[1])=='qbase.copy()' and ast.unparse(cb.args[1])=='qinit.copy()'
    details['least_squares_old']=ast.unparse(ca);details['least_squares_new']=ast.unparse(cb)
    # The exact residual equality above includes unchanged qbase regularization,
    # bone/overlap/material position/normal targets and all six old weights.
    checks['qbaseline_prior_stays_qbase']=any(isinstance(n,ast.BinOp) and isinstance(n.op,ast.Sub) and isinstance(n.left,ast.Name) and n.left.id=='q' and isinstance(n.right,ast.Name) and n.right.id=='qbase' for n in ast.walk(named(tb,'residual')))
    oldloops=[n for n in ast.walk(ta) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='h']
    newloops=[n for n in ast.walk(tb) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='h']
    checks['material_anchor_selection_loop_AST_identical']=len(oldloops)==len(newloops)==1 and dump(oldloops[0])==dump(newloops[0])
    def assignments(tree,name):
        return [n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id==name for x in n.targets)]
    for name in ('qbase','pose','target_points','desired','hypotheses','bindings'):
        aa,bb=assignments(ta,name),assignments(tb,name)
        checks[name+'_assignments_AST_identical']=[dump(n) for n in aa]==[dump(n) for n in bb]
    binding_assert=[n for n in ast.walk(tb) if isinstance(n,ast.Assert) and 'parent_bindings' in ast.unparse(n.test)]
    checks['runtime_full_bindings_dict_equality_guard']=len(binding_assert)==1 and ast.unparse(binding_assert[0].test)=="bindings == read(pins['parent_bindings']['path'])"
    mainnode=named(tb,'main')
    statements=list(ast.walk(mainnode))
    source_fields=[]
    for n in statements:
        if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='z' and isinstance(n.slice,ast.Constant):source_fields.append(n.slice.value)
    checks['baseline_npz_field_allowlist_unchanged']=set(source_fields)=={'joint_pos_rad','mapped_object_pose_w','morphology_normalized_target_points_w','geom_friction'}
    details['npz_read_fields']=sorted(set(source_fields))
    checks['source_labels_not_loaded_by_new_reads']=all('robot' not in str(x) for x in source_fields)
    checks['new_seed_has_shape_finite_other_fingers_hardbounds_guards']=all(text in b for text in (
        "qinit.shape == (20,) and np.isfinite(qinit).all()",
        "np.array_equal(qinit[np.r_[0:4,12:20]], qbase[np.r_[0:4,12:20]])",
        "np.all(qinit >= robot.lower) and np.all(qinit <= robot.upper)"))
    checks['same_fixed_object_pose_set']=a.count('g.set_object_pose(pose)')==b.count('g.set_object_pose(pose)')==1
    checks['no_new_mocap_or_object_qpos_writes']=not any(isinstance(n,(ast.Assign,ast.AugAssign)) and ('mocap' in ast.unparse(n) or 'qpos[' in ast.unparse(n)) for n in ast.walk(tb))
    checks['schema3_H23_only']=all(text in b for text in ("recipe['schema'] == 'frame0_material_contact_hypothesis:3'","recipe['fingers'] == [2, 3]"))
    objectshift=ROOT/'configs/frame0-h23-objectshift-review-v1.json'
    shift=json.loads(objectshift.read_text())
    checks['objectshift_review_still_not_authorized']=shift['optimization_authorized'] is False
    checks['objectshift_source_still_frozen']=sha(ROOT/'scripts/refine_initial_contacts_objectshift_cpu.py')=='c4d8cdc9fa4e86660e5d3e8c33adee8d27eebdf8988f46b85708e3b31a61b16f'
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/'caller_diff.patch').write_text(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile=str(old),tofile=str(new))),encoding='utf-8')
    report={'status':'STATIC_SAME_OBJECTIVE_PASS_RUNTIME_INPUT_PENDING' if all(checks.values()) else 'STATIC_REVIEW_FAILED',
        'checks':checks,'details':details,'sources':{'old':{'path':str(old),'sha256':sha(old)},'new':{'path':str(new),'sha256':sha(new)},'reviewer':{'path':str(Path(__file__)),'sha256':sha(__file__)}},
        'pending':{'composite_catalog_and_recipe':'Not available at this static review. Need hash/seed-id/q20/finite/hardbounds/other-finger equality check when frozen inputs exist.',
                   'actual_runtime_bindings_equality':'Caller guard is present; no caller/optimizer executed in this review. Do not report as an observed runtime equality yet.',
                   'native_geometric_outcome':'Not computed; no optimization or native query performed.'},
        'optimization_runs':0,'physics_steps':0,'GPU_used':False,'physical_reference_qualified':False,
        'scope':'The seed is an explicit intervention on the local optimizer initial basin, not a new objective or measured q label. Single-finger point solutions do not establish simultaneous whole-hand feasibility, force closure, or dynamics.'}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2));raise SystemExit(not all(checks.values()))

if __name__=='__main__':main()
