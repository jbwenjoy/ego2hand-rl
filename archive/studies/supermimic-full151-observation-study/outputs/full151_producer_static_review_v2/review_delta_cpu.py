"""Static source/protocol delta only; never import the production producer."""
import ast,copy,difflib,hashlib,json
from pathlib import Path
ROOT=Path('D:/workspace/supermimic-full151-observation-study');OUT=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=lambda p:{'path':str(p),'sha256':sha(p)}
a=ROOT/'scripts/build_full151_input_cpu_v1.py';b=ROOT/'scripts/build_full151_input_cpu_v2.py'
p=ROOT/'protocols/full151-input-cpu-v1.json';q=ROOT/'protocols/full151-input-cpu-v2.json'
s1=a.read_text();s2=b.read_text();j1=json.loads(p.read_text());j2=json.loads(q.read_text())
normalized=s2.replace('json,os,platform','json,platform').replace(",'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')",'')
normalized=normalized.replace("            report['cost']['LP_calls']+=1  # count attempts even if the backend raises\n",'')
normalized=normalized.replace('            total_solver+=elapsed\n',"            total_solver+=elapsed;report['cost']['LP_calls']+=1\n")
jn=copy.deepcopy(j2);jn['sources']['producer']=copy.deepcopy(j1['sources']['producer']);jn.pop('prelaunch_revision',None)
attempt=s2.index("report['cost']['LP_calls']+=1  # count attempts")
call=s2.index('result=linprog(**kw)')
allpins=[]
def walk(v):
    if isinstance(v,dict):
        if set(v)=={'path','sha256'}:allpins.append(v)
        else:
            for x in v.values():walk(x)
    elif isinstance(v,list):
        for x in v:walk(x)
walk(j2)
checks={
 'old_v1_source_preserved':sha(a)=='c7b6f650a9cd577f01b62f4a39fab573b928719d88b89c14d5707f1ebd34a0a7',
 'old_v1_protocol_preserved':sha(p)=='cf1e96898a9cb22c4bd1d53cbc65472756e2c4dc3516ab50c68a97b9cd0e9380',
 'new_v2_source_expected':sha(b)=='47373f210a29cd22407a8d215f6f81a256d19b33a6ef68569f3b719a928df6fd',
 'new_v2_protocol_expected':sha(q)=='3f436240b04d6e428e396dd7a0fbf297469d9e5d59b43519f08e8eb988ec6a37',
 'only_declared_source_replacements_byte_equivalent':normalized==s1,
 'only_declared_source_replacements_AST_equivalent':ast.dump(ast.parse(normalized),include_attributes=False)==ast.dump(ast.parse(s1),include_attributes=False),
 'LP_attempt_counter_precedes_single_linprog_callsite':attempt<call and s2.count('result=linprog(**kw)')==1,
 'only_one_attempt_increment_callsite':s2.count("report['cost']['LP_calls']+=1")==1,
 'protocol_only_producer_pin_revision_changed':jn==j1,
 'same_output_no_retry_constraints_options_inputs':all(j1[k]==j2[k] for k in ['output','execution','inputs','linear_contract','selected_fields','object_numeric_tolerances']),
 'all23_new_protocol_pins_exact':len(allpins)==23 and all(sha(v['path'])==v['sha256'] for v in allpins),
 'CUDA_visibility_recorded_not_new_GPU_operation':"'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')" in s2
}
result={'schema':'full151_input_producer_independent_static_delta_review:1','status':'PASS_STATIC_NO_BLOCKING' if all(checks.values()) else 'FAIL_STATIC_DELTA',
        'passed':sum(checks.values()),'total':len(checks),'checks':checks,'producer':pin(b),'protocol':pin(q),'review_source':pin(__file__),
        'preceding_full_static_review':pin(ROOT/'outputs/full151_producer_static_review_v1/report.json'),
        'required_counter_fix_closed':True,'runtime_scope':'Root will launch CUDA_VISIBLE_DEVICES=-1; v2 records it. Reviewer has not executed the producer or any LP.',
        'scientific_scope':'All P0/P1/tau/u0, actual PD, source allowlist, object law, bounds, objective, tolerances, LP options and post-C2 reference-force checks remain exactly those reviewed in v1. No qualification result follows before actual execution.',
        'source_unified_diff':''.join(difflib.unified_diff(s1.splitlines(True),s2.splitlines(True),fromfile=str(a),tofile=str(b))),
        'production_imports_solver_FK_collision_Env_WM_physics_GPU_calls':0}
(OUT/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'status':result['status'],'passed':result['passed'],'total':result['total'],'report':pin(OUT/'report.json')},indent=2))
