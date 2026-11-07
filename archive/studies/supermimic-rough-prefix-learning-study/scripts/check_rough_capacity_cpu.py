"""CPU fake-backend tests of reset-aware capacity observation; zero physics."""
from __future__ import annotations
import argparse,copy,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rough_prefix import capacity as c
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
CAPS={'nconmax':192,'njmax':4096}


class Flags:
    def __init__(self,value=None):self.value=np.zeros((1,),np.int32) if value is None else np.array(value,copy=True);self.reads=0;self.error=None
    def numpy(self):
        self.reads+=1
        if self.error:raise self.error
        return self.value


class Env:
    def __init__(self,value=None):self.flags=Flags(value);self.sim=NS(wp_data=NS(overflow=self.flags));self.num_envs=7;self.resets=[];self.capacity=copy.deepcopy(CAPS);self.raise_reset=None;self.forwards=0;self.steps=0
    def reset(self,*,env_ids=None,seed=None,options=None):
        self.resets.append({'env_ids':None if env_ids is None else np.asarray(env_ids).copy().tolist(),'seed':seed,'options':options})
        if self.raise_reset:raise self.raise_reset
        # Mimic the existing native reset's sticky clear. The observer must
        # inspect before this statement and must never itself perform it.
        self.flags.value[:]=0
        return 'native_reset_result'


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'capacity_source.py').write_bytes(Path(c.__file__).read_bytes());(out/'checker_source.py').write_bytes(Path(__file__).read_bytes());checks=[];examples={}
    def check(name,call):
        try:
            if not call():raise AssertionError('False result')
            checks.append({'name':name,'passed':True})
        except Exception as exc:checks.append({'name':name,'passed':False,'error':repr(exc)})
    def observe(e,**kw):return c.capacity_observer(e,stage='CPU_fake_stage',fixed_capacities=CAPS,**kw)
    e=Env();native=Env.reset
    with observe(e) as o:pass
    check('no_reset_one_final_read_OK',lambda:e.flags.reads==1 and o.snapshot()['final_read_count']==1 and o.snapshot()['engineering_capacity_qualified'])
    check('original_class_reset_binding_restored',lambda:e.reset.__func__ is native and '_rough_capacity_observer' not in vars(e) and 'reset' not in vars(e))
    e=Env(np.zeros((2,3),np.uint32))
    with observe(e) as o:
        result=e.reset(env_ids=np.array([1,5],np.int64),seed=12,options={'test':'only_metadata'});o.set_stage('CPU_fake_collect');e.reset(env_ids=np.array([0],np.int64))
    examples['partial_zero_resets']=o.snapshot()
    check('partial_reset_arguments_and_return_preserved',lambda:result=='native_reset_result' and e.resets[0]=={'env_ids':[1,5],'seed':12,'options':{'test':'only_metadata'}})
    check('two_prereset_plus_one_final_reads',lambda:e.flags.reads==3 and o.snapshot()['before_reset_read_count']==2 and o.snapshot()['final_read_count']==1)
    check('dtype_shape_record_not_world_count',lambda:o.records[0]['raw_flags_dtype']=='uint32' and o.records[0]['raw_flags_shape']==[2,3] and o.records[0]['flag_shape_is_world_count'] is False and e.num_envs==7)
    check('stage_and_selected_IDs_recorded',lambda:o.records[0]['stage']=='CPU_fake_stage' and o.records[1]['stage']=='CPU_fake_collect' and o.records[0]['selected_reset']['values']==[1,5])
    e=Env(np.array([4],np.int32));caught=None
    try:
        with observe(e) as o:e.reset(env_ids=np.array([3]))
    except c.CapacityOverflowError as exc:caught=exc
    examples['overflow_before_reset']=o.snapshot()
    check('overflow_caught_before_native_reset_can_clear_flags',lambda:caught is not None and e.resets==[] and e.flags.value.tolist()==[4] and o.cumulative_bitmask==4)
    check('overflow_restores_reset_and_does_not_qualify',lambda:e.reset.__func__ is native and o.snapshot()['engineering_capacity_qualified'] is False)
    e=Env(np.array([2],np.int32));caught=None
    try:
        with observe(e) as o:
            try:e.reset(env_ids=np.array([2]))
            except c.CapacityOverflowError:e.flags.value[:]=0  # fake caller clear, never observer code
    except c.CapacityOverflowError as exc:caught=exc
    examples['latched_even_when_later_backend_zero']=o.snapshot()
    check('historical_OR_survives_later_zero_backend',lambda:caught is not None and o.cumulative_bitmask==2 and o.records[-1]['raw_flags_values']==[0] and o.status=='OVERFLOW')
    e=Env();caught=None
    try:
        with observe(e) as o:e.reset();e.flags.value[0]=8
    except c.CapacityOverflowError as exc:caught=exc
    check('final_read_detects_overflow_since_last_reset',lambda:caught is not None and o.cumulative_bitmask==8 and len(e.resets)==1)
    e=Env();e.flags.error=OSError('fake unreadable backend');caught=None
    try:
        with observe(e) as o:e.reset(env_ids=np.array([1,2]))
    except c.CapacityUnknownError as exc:caught=exc
    examples['unreadable']=o.snapshot()
    check('read_failure_UNKNOWN_never_zero_and_no_native_reset',lambda:caught is not None and o.status=='UNKNOWN' and o.unknown_reads==2 and e.resets==[])
    check('failed_read_still_records_requested_reset_IDs',lambda:o.records[0]['selected_reset']['values']==[1,2] and o.records[0]['raw_flags_values'] is None)
    for name,flags in [('float',np.array([0.],np.float32)),('bool',np.array([False])),('empty',np.array([],np.int32)),('object',np.array([object()],object))]:
        e=Env(flags);caught=None
        try:
            with observe(e) as o:pass
        except c.CapacityUnknownError as exc:caught=exc
        check('malformed_'+name+'_UNKNOWN',lambda caught=caught,o=o:caught is not None and o.status=='UNKNOWN' and o.snapshot()['engineering_capacity_qualified'] is False)
    e=Env(np.array([[1,8],[0,2]],np.int32));caught=None
    try:
        with observe(e) as o:pass
    except c.CapacityOverflowError as exc:caught=exc
    check('all_raw_array_flags_OR_without_world_assumption',lambda:caught is not None and o.cumulative_bitmask==11 and o.records[0]['raw_flags_shape']==[2,2])
    e=Env(np.array([-1],np.int32));caught=None
    try:
        with observe(e) as o:pass
    except c.CapacityOverflowError as exc:caught=exc
    check('nonzero_signed_flags_not_silently_cast_zero',lambda:caught is not None and o.cumulative_bitmask==-1)
    e=Env();e.raise_reset=ValueError('native reset exception');caught=None
    try:
        with observe(e) as o:e.reset()
    except ValueError as exc:caught=exc
    check('native_reset_exception_preserved_and_method_restored',lambda:str(caught)=='native reset exception' and e.reset.__func__ is native and o.snapshot()['final_read_count']==1)
    e=Env();caught=None
    try:
        with observe(e) as o:
            e.flags.error=OSError('secondary unknown');raise ValueError('body failure must win')
    except ValueError as exc:caught=exc
    check('final_unknown_does_not_mask_original_body_exception',lambda:str(caught)=='body failure must win' and o.status=='UNKNOWN' and e.reset.__func__ is native)
    e=Env()
    with observe(e) as o:o.final_read();o.final_read()
    check('explicit_final_read_idempotent_context_no_extra_read',lambda:e.flags.reads==1 and o.snapshot()['final_read_count']==1)
    e=Env();after_final_rejected=False
    with observe(e) as o:
        o.final_read()
        try:e.reset()
        except RuntimeError:after_final_rejected=True
    check('reset_after_final_read_rejected_without_backend_or_reset_call',lambda:after_final_rejected and not e.resets and e.flags.reads==1)
    e=Env();duplicate=False
    with observe(e) as o:
        try:
            with observe(e):pass
        except ValueError:duplicate=True
    check('double_observer_rejected',lambda:duplicate and e.reset.__func__ is native)
    e=Env();custom=lambda **kw:'preexisting_instance_hook';e.reset=custom
    with observe(e) as o:value=e.reset(env_ids=np.array([1]))
    check('preexisting_instance_reset_hook_restored_exactly',lambda:e.reset is custom and value=='preexisting_instance_hook')
    copy_snapshot=o.snapshot();copy_snapshot['records'][0]['raw_flags_values'][0]=99
    check('snapshot_deepcopy_does_not_mutate_latch',lambda:o.cumulative_bitmask==0 and o.records[0]['raw_flags_values']==[0])
    invalid=False
    try:c.CapacityObserver(Env(),stage='x',fixed_capacities={'nconmax':0,'njmax':4096})
    except ValueError:invalid=True
    check('fixed_capacity_metadata_positive_and_unmodified',lambda:invalid and e.capacity==CAPS and all(e.forwards==e.steps==0 for e in [e]))
    check('old_runtime_5736_immutable',lambda:sha(ROOT/'src/rough_prefix/runtime.py')=='5736dd1fd678a39f87867461d30d0710e6242af10e42f6fa2d57f6c9301b1ff9')
    check('no_Torch_Warp_MuJoCo_import_or_GPU',lambda:not any(k in sys.modules for k in ('torch','warp','mujoco_warp','mujoco')))
    result={'status':'PASS' if all(x['passed'] for x in checks) else 'FAIL','check_count':len(checks),'passed':sum(x['passed'] for x in checks),'checks':checks,'capacity_sha256':sha(c.__file__),'checker_sha256':sha(__file__),'examples':examples,'scope':'Fake backend only; no real reset, flag/GPU read or physics. Production hook remains uninstalled in frozen runtime and A0.','GPU_used':False,'physics_steps':0,'training_started':False,'source_fields_fixed':[c.BACKEND_FIELD],'model_capacity_modified':False}
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='examples'},indent=2));raise SystemExit(result['status']!='PASS')
if __name__=='__main__':main()
