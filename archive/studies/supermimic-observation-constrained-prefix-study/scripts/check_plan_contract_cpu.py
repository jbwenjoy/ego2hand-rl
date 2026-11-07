"""Plan/source and canonical-output negative checks; no geometry/LP/solver."""
from pathlib import Path
import argparse,ast,copy,json,os,sys
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'vendor'))
from runtime_io import write_json
from constrained_prefix.inputs import checked_plan,PROTOCOL,PROTOCOL_SHA,PARENT,PARENT_PROTOCOL,PARENT_PROTOCOL_SHA,RETARGET_VARIANT_ID
from constrained_prefix.io import sha,read


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    parent=read(PARENT_PROTOCOL);files=[ROOT/'scripts/build_constrained_prefix_geometry_cpu.py',ROOT/'src/bspline_control.py',ROOT/'vendor/runtime_io.py',*sorted((ROOT/'src/constrained_prefix').glob('*.py')),PARENT/'src/prefix_curves.py',PARENT/'src/prefix_control.py',*sorted((PARENT/'src/prefix_study').glob('*.py')),PARENT/'vendor/runtime_io.py']
    files += [Path(parent['inputs'][k]['path']) for k in ('kinematics','native_filtered_geometry','material_geometry')]
    pins={str(i):{'path':str(p),'sha256':sha(p)} for i,p in enumerate(files)}
    p={'schema':'constrained_prefix_geometry_cpu_plan:1','optimization_authorized':False,'retarget_variant_id':RETARGET_VARIANT_ID,'protocol':{'path':str(PROTOCOL),'sha256':PROTOCOL_SHA},'parent_protocol':{'path':str(PARENT_PROTOCOL),'sha256':PARENT_PROTOCOL_SHA},'inputs':parent['inputs'],'source_pins':pins,'frames':16,'joints':20,'geometry_samples':61,'n_control':18,'free_variables':340,'maxiter':80,'max_evaluations':250,'soft_wall_seconds':220,'owned_process_seconds':240,'automatic_retry':False,'planning_depth_bound_m':.002999,'control_margin_rad':.000002,'force_margin_Nm':.000002,'rate_margin_rad_s':.0001}
    checks={};write_json(out/'valid_check_plan.json',p);checked_plan(out/'valid_check_plan.json');checks['full_dependency_plan_valid_without_run']=True
    def reject(label,mutate,run=False):
        altered=copy.deepcopy(p);mutate(altered);path=out/(label+'.json');write_json(path,altered)
        try:checked_plan(path,run=run);checks[label]=False
        except ValueError:checks[label]=True
    reject('run_false_authorization_rejected',lambda x:None,True)
    reject('wrong_protocol_hash_rejected',lambda x:x['protocol'].__setitem__('sha256','0'*64))
    reject('wrong_parent_protocol_hash_rejected',lambda x:x['parent_protocol'].__setitem__('sha256','0'*64))
    reject('changed_input_identity_rejected',lambda x:x['inputs']['canonical_H123_q0'].__setitem__('sha256','0'*64))
    reject('missing_dependency_pin_rejected',lambda x:x['source_pins'].pop('0'))
    reject('stale_entry_source_rejected',lambda x:x['source_pins']['0'].__setitem__('sha256','0'*64))
    reject('wrong_retarget_variant_rejected',lambda x:x.__setitem__('retarget_variant_id','same_source_is_not_same_trajectory'))
    for key,value in [('planning_depth_bound_m',.003),('free_variables',300),('soft_wall_seconds',221),('max_evaluations',251),('automatic_retry',True),('rate_margin_rad_s',0.)]:
        reject('wrong_'+key+'_rejected',lambda x,k=key,v=value:x.__setitem__(k,v))
    tree=ast.parse((ROOT/'scripts/build_constrained_prefix_geometry_cpu.py').read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    getname=lambda n:n.func.id if isinstance(n.func,ast.Name) else n.func.attr if isinstance(n.func,ast.Attribute) else None
    checks['one_linprog_call_site']=sum(getname(n)=='linprog' for n in calls)==1
    checks['one_minimize_call_site']=sum(getname(n)=='minimize' for n in calls)==1
    checks['canonical_writer_only_called_for_normal_result_x']=all(len(n.args)>0 and isinstance(n.args[0],ast.Attribute) and isinstance(n.args[0].value,ast.Name) and n.args[0].value.id=='result' and n.args[0].attr=='x' for n in calls if getname(n)=='finish_candidate')
    checks['no_Torch_Warp_imported']='torch' not in sys.modules and 'warp' not in sys.modules and 'mujoco_warp' not in sys.modules
    report={'status':'PASSED_PLAN_SOURCE_AND_CANONICAL_NEGATIVES' if all(checks.values()) else 'FAILED_CONTRACT_CHECKS','checks':checks,'passed':sum(checks.values()),'total':len(checks),'source_pins':{'checker':{'path':str(Path(__file__)),'sha256':sha(__file__)},'entry':pins['0'],'inputs':{'path':str(ROOT/'src/constrained_prefix/inputs.py'),'sha256':sha(ROOT/'src/constrained_prefix/inputs.py')}},'geometry_queries':0,'LP_calls':0,'SLSQP_calls':0,'physics_steps':0,'GPU_used':False}
    write_json(out/'report.json',report);print(json.dumps(report,indent=2));return 0 if all(checks.values()) else 1
if __name__=='__main__':raise SystemExit(main())
