"""Source-only cross-version physics preservation; no native/data execution."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
paths=[ROOT.parent/'hand2_holding_v1/run_cpu_holding_v1.py',
       ROOT.parent/'hand2_holding_v2/run_cpu_support_holding_v2.py',
       ROOT.parent/'hand2_holding_v3/run_cpu_support_holding_v3.py',
       ROOT/'run_cpu_native_equilibrium_holding_v4.py']


def pin(p):
    b=p.read_bytes();return {'path':str(p.resolve()),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def one(tree,cls,predicate):
    result=[n for n in ast.walk(tree) if isinstance(n,cls) and predicate(n)]
    assert len(result)==1,(cls.__name__,len(result))
    return result[0]


def same(a,b):return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)


texts=[p.read_text(encoding='utf8') for p in paths]
trees=[ast.parse(t) for t in texts]
mains=[one(t,ast.FunctionDef,lambda n:n.name=='main') for t in trees]
checks={}
for version,tree,main in zip(('v1','v2','v3'),trees,mains):
    checks[version+'_whole_native_tick_loop_AST_exact']=same(
        one(main,ast.For,lambda n:ast.unparse(n.target)=='tick'),
        one(mains[3],ast.For,lambda n:ast.unparse(n.target)=='tick'))
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
    checks[version+'_dt_pair_function_AST_exact']=same(
        one(trees[index],ast.FunctionDef,lambda n:n.name=='dt_pair_summary'),
        one(trees[3],ast.FunctionDef,lambda n:n.name=='dt_pair_summary'))
assert all(checks.values()),{k:v for k,v in checks.items() if not v}
dp=ROOT/'metadata_adapter_v3_to_v4.diff'
with dp.open('x',encoding='utf8') as f:f.write(''.join(difflib.unified_diff(texts[2].splitlines(True),texts[3].splitlines(True),fromfile=str(paths[2]),tofile=str(paths[3]))))
report={'schema':'native_equilibrium_holding_v4_source_preservation:1',
    'status':'PASS_SOURCE_AST_ONLY_NO_SEARCH_RESULT_OR_PHYSICS_EXECUTION',
    'source_pins':{f'v{i+1}':pin(p) for i,p in enumerate(paths)},'checker':pin(Path(__file__)),
    'checks':checks,'passed':sum(checks.values()),'total':len(checks),'diff':pin(dp),
    'intentional_nonloop_delta':[
        'New strict native equilibrium report/protocol:6 adapter and global model_audit.passed',
        'All shortlist records retained; first evidence-qualified rank per original seed only, prior to any holding result',
        'Full primary/repeat native gates and8 repeat fields; full independent22/189 in new[-0.1,+0.2]mm pad hypothesis',
        'No oldLP, positive-gap or strict moment criterion',
        'Before first per-state MjData, NPZqpos/zeroqvel/ctrl/x46 byte-bound to captured primary native and x46',
        'New namespace and initial-compression scope; original dynamic penetration2mm and all2s gates unchanged'],
    'actual_search_result_or_NPZ_reads':0,'MJB_loads':0,'forward_or_steps':0}
rp=ROOT/'source_preservation_v1.json'
with rp.open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
print(json.dumps({'runner':pin(paths[-1]),'report':pin(rp),'passed':report['passed'],'total':report['total']}))
