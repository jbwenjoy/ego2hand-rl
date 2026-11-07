"""Source AST only: retain the original47 cross-version physical checks."""
import ast,difflib,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
paths=[ROOT.parent/'hand2_holding_v1/run_cpu_holding_v1.py',
       ROOT.parent/'hand2_holding_v2/run_cpu_support_holding_v2.py',
       ROOT.parent/'hand2_holding_v3/run_cpu_support_holding_v3.py',
       ROOT/'run_source_mapping_dynamics_v11.py']
def pin(p):
    b=p.read_bytes();return dict(path=str(p.resolve()),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def one(tree,cls,predicate):
    result=[n for n in ast.walk(tree) if isinstance(n,cls) and predicate(n)]
    assert len(result)==1,(cls.__name__,len(result))
    return result[0]
def same(a,b):return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
texts=[p.read_text(encoding='utf8') for p in paths];trees=[ast.parse(t) for t in texts]
mains=[one(t,ast.FunctionDef,lambda n:n.name=='main') for t in trees];checks={}
for version,tree,main in zip(('v1','v2','v3'),trees,mains):
    checks[version+'_whole_native_tick_loop_AST_exact']=same(one(main,ast.For,lambda n:ast.unparse(n.target)=='tick'),one(mains[3],ast.For,lambda n:ast.unparse(n.target)=='tick'))
    for name in ('DTS','DURATION','GATES'):
        pred=lambda n:any(isinstance(v,ast.Name) and v.id==name for v in n.targets)
        checks[version+'_'+name+'_exact']=same(one(tree,ast.Assign,pred),one(trees[3],ast.Assign,pred))
    for name in ('COM','angle'):
        checks[version+'_'+name+'_exact']=same(one(main,ast.FunctionDef,lambda n:n.name==name),one(mains[3],ast.FunctionDef,lambda n:n.name==name))
    for name in ('tail_start_tick','tail','pre_tail','qvio','lin','omega','metrics','gates'):
        pred=lambda n:any(isinstance(v,ast.Name) and v.id==name for v in n.targets)
        checks[version+'_'+name+'_exact']=same(one(main,ast.Assign,pred),one(mains[3],ast.Assign,pred))
    pred=lambda n:ast.unparse(n.target)=='(k, v)' and ast.unparse(n.iter)=='metrics.items()'
    checks[version+'_metric_gate_comparisons_exact']=same(one(main,ast.For,pred),one(mains[3],ast.For,pred))
for version,index in (('v2',1),('v3',2)):
    checks[version+'_dt_pair_function_AST_exact']=same(one(trees[index],ast.FunctionDef,lambda n:n.name=='dt_pair_summary'),one(trees[3],ast.FunctionDef,lambda n:n.name=='dt_pair_summary'))
assert len(checks)==47 and all(checks.values()),{k:v for k,v in checks.items() if not v}
diff=ROOT/'original_v3_to_source_mapping_v11.diff'
with diff.open('x',encoding='utf8') as f:f.write(''.join(difflib.unified_diff(texts[2].splitlines(True),texts[3].splitlines(True),fromfile=str(paths[2]),tofile=str(paths[3]))))
report=dict(schema='source_mapping_dynamics_source_preservation:11',status='PASS_SOURCE_AST_ONLY_NO_MODEL_OR_PHYSICS_EXECUTION',
    source_pins={f'v{i+1}':pin(p) for i,p in enumerate(paths)},checker=pin(Path(__file__)),checks=checks,
    passed=sum(checks.values()),total=len(checks),diff=pin(diff),
    enumerated_nonloop_delta=['New six-input source-mapping selector and two predefined mapping branch choices',
        'Minimum saved rank0 nativeLP loss; strict two-fresh control only when original holding flag and evidence all pass, otherwise source native_probe',
        'Full qpos27/zeroqvel/control/x46 source binding; unchanged runtime raw NPZ checks',
        'Strict ideal/three accelerations/initial3N descriptive, no admission test; legal true control and actuator caps remain',
        'New schema/status and scoped enrollment text; main AST exactly the parent after declared literal reversal'],
    actual_model_NPZ_FK_LP_forward_or_step_calls=0)
out=ROOT/'source_preservation_v11.json'
with out.open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
print(json.dumps({'runner':pin(paths[-1]),'report':pin(out),'passed':47,'total':47}))
