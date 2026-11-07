"""CPU fake boundary contracts, not actual native snapshot qualification."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import traceback
from types import SimpleNamespace
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rough_prefix.probe_transaction_v2 import ProbeTransaction,TransactionError,UncoveredSidecarError
from rough_prefix.capacity import CapacityObserver,CapacityOverflowError,CapacityUnknownError


def pin(p):
    p=Path(p).resolve(strict=True)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


class FakeFlags:
    def __init__(self):self.values=np.zeros(1,np.int32);self.unreadable=False
    def numpy(self):
        if self.unreadable:raise RuntimeError('fixture unreadable overflow')
        return self.values.copy()


class NativeLikeFakeSnapshot:
    """Only a fake for CPU contracts: native-style ref containers, no NumPy/RNG."""
    def __init__(self,roots):
        self.entries=[];self.arrays=[];self.tensors=[];self.restores=0;seen=set()
        def walk(value):
            if id(value) in seen:return
            seen.add(id(value))
            if isinstance(value,dict):
                self.entries.append((value,dict(value),'dict'))
                for v in value.values():walk(v)
            elif isinstance(value,list):
                self.entries.append((value,list(value),'list'))
                for v in value:walk(v)
            elif isinstance(value,tuple):
                for v in value:walk(v)
            elif isinstance(value,(SimpleNamespace,FakeFlags)):
                saved=dict(vars(value));self.entries.append((value,saved,'object'))
                for v in saved.values():walk(v)
        walk(roots.env);walk(roots.reader)
    def restore(self):
        self.restores+=1
        for obj,saved,kind in self.entries:
            if kind=='dict':obj.clear();obj.update(saved)
            elif kind=='list':obj[:]=saved
            else:vars(obj).clear();vars(obj).update(saved)


def no_op(*a,**k):pass


def fixture(extra=None,external=None):
    term=SimpleNamespace(_processed_actions=np.zeros((1,20),np.float32),_targets=np.ones((1,20),np.float32),process_actions=no_op)
    motion=SimpleNamespace(time_steps=np.array([4],np.int64))
    am=SimpleNamespace(_terms={'joint_pos':term});am.get_term=lambda name:am._terms[name]
    cm=SimpleNamespace(_terms={'motion':motion});cm.get_term=lambda name:cm._terms[name]
    obs=SimpleNamespace(_obs_buffer={'policy':np.zeros((1,155),np.float32)},history=np.zeros((3,1,155),np.float32))
    flags=FakeFlags();env=SimpleNamespace(sim=SimpleNamespace(wp_data=SimpleNamespace(overflow=flags),step=no_op,forward=no_op),
        action_manager=am,command_manager=cm,observation_manager=obs,reward_manager=SimpleNamespace(compute=no_op),step=no_op,
        pending=[],cache_valid=False)
    reader=SimpleNamespace(env=env)
    reader.memory=lambda:{'c1':term._processed_actions.copy(),'time':motion.time_steps[:,None].copy(),'frozen_target':term._targets.copy()}
    ledger=CapacityObserver(env,stage='CPU_fixture',fixed_capacities={'nconmax':192,'njmax':4096})
    if extra is not None:env.aux=extra
    tx=ProbeTransaction(env,reader,capacity_observer=ledger,source_action=np.zeros((1,20),np.float32),label='fixture',
        extra_sidecars={} if extra is None else {'aux':extra},external_roots=external or {},
        snapshot_factory=NativeLikeFakeSnapshot,strict_native=False)
    return tx,env,reader,ledger,term,motion,flags


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);checks={};source=ROOT/'src/rough_prefix/probe_transaction_v2.py';initial=pin(source)
    def check(name,value):checks[name]=bool(value)
    def reject(name,fn,kind=(TransactionError,)):
        try:fn()
        except kind:checks[name]=True
        else:checks[name]=False
    try:
        tx,env,reader,ledger,term,motion,flags=fixture()
        check('fixture_declares_not_native_qualification',tx.describe()['CPU_fixture_is_not_native_qualification'] is True)
        for k in range(3):
            receipt=tx.begin('baseline'+str(k));check('restore_qualified_'+str(k),receipt['restore_qualified'] and receipt['inventory']['bitexact'])
            check('phase_EMA_cache_sourceaction_restored_'+str(k),motion.time_steps[0]==4 and not term._processed_actions.any() and np.all(term._targets==1)
                and not env.cache_valid and not env.pending and not tx.source_action.any())
            motion.time_steps[:]=8;term._processed_actions[:]=.4;term._targets[:]=2;env.pending.append({'value':9});env.cache_valid=True
            tx.finish('baseline'+str(k))
        check('external_ledger_prefix_not_rolled_back_over3restores',len(ledger.records)==6 and [r['event'] for r in ledger.records]==['before_restore','after_branch']*3
            and [r['sequence'] for r in ledger.records]==list(range(1,7)))
        check('native_like_fake_restored3times',tx.native.restores==3)
        reject('duplicate_branch_rejected',lambda:tx.begin('baseline0'))
        reject('finish_without_matching_active_branch_rejected',lambda:tx.finish('other'))
        tx2,e2,r2,l2,t2,m2,f2=fixture();tx2.begin('overflow');f2.values[:]=8
        reject('afterbranch_overflow_latched_and_rejected',lambda:tx2.finish('overflow'),(CapacityOverflowError,))
        check('overflow_raw_OR_and_prefix_preserved',l2.cumulative_bitmask==8 and len(l2.records)==2 and l2.records[-1]['raw_bitmask']==8)
        # Explicit CPU fake clearing models the danger of a full restore; the
        # ledger must retain the old evidence even when device flags are 0.
        tx2.native.restore();f2.values[:]=0
        check('host_OR_survives_manual_fake_flag_clear',l2.cumulative_bitmask==8 and l2.status=='OVERFLOW')
        reject('failed_transaction_cannot_retry',lambda:tx2.begin('retry'))
        tx3,e3,r3,l3,t3,m3,f3=fixture();f3.values[:]=2
        reject('before_restore_overflow_stops_before_native_restore',lambda:tx3.begin('bad'),(CapacityOverflowError,))
        check('bad_pre_restore_did_not_restore_flags',tx3.native.restores==0 and f3.values[0]==2 and l3.cumulative_bitmask==2)
        tx4,e4,r4,l4,t4,m4,f4=fixture();f4.unreadable=True
        reject('unreadable_capacity_UNKNOWN_stops',lambda:tx4.begin('badread'),(CapacityUnknownError,))
        check('unknown_count_not_filled_zero',l4.unknown_reads==1 and l4.status=='UNKNOWN')
        tx5,e5,r5,l5,t5,m5,f5=fixture();e5.bad_ledger=l5
        reject('ledger_attached_env_rejected',lambda:ProbeTransaction(e5,r5,capacity_observer=l5,source_action=np.zeros((1,20),np.float32),label='bad',snapshot_factory=NativeLikeFakeSnapshot,strict_native=False))
        tx6,e6,r6,l6,t6,m6,f6=fixture();original=t6._targets;t6._targets=original.copy()
        reject('replaced_cached_tensor_identity_detected',tx6.verify_current)
        tx6.begin('restore_identity');check('cached_original_object_identity_restored',t6._targets is original);tx6.finish('restore_identity')
        class Foreign:
            def __init__(self):self.counter=0;self.buffer=np.array([1.,-0.,np.nan],np.float64)
        foreign=Foreign();tx7,e7,r7,l7,t7,m7,f7=fixture();e7.foreign=foreign
        reject('unregistered_foreign_mutable_sidecar_rejected',lambda:ProbeTransaction(e7,r7,capacity_observer=l7,source_action=np.zeros((1,20),np.float32),label='uncovered',snapshot_factory=NativeLikeFakeSnapshot,strict_native=False),(UncoveredSidecarError,))
        foreign=Foreign();tx8,e8,r8,l8,t8,m8,f8=fixture(extra=foreign)
        old=foreign.buffer.tobytes();foreign.buffer.view(np.uint64)[2]=0x7ff8000000000042;foreign.counter=4
        tx8.begin('aux');check('explicit_custom_sidecar_NaN_payload_and_signed0_restored',foreign.counter==0 and foreign.buffer.tobytes()==old);tx8.finish('aux')
        foreign.buffer[1]=0.
        reject('signed0_bitdifference_detected',tx8.verify_current)
        external=Foreign();tx9,e9,r9,l9,t9,m9,f9=fixture(external={'logging':external})
        # A closure-owned external holder must not be restored.
        def hook():external.counter+=1
        e9.sim.step=hook
        tx9=ProbeTransaction(e9,r9,capacity_observer=l9,source_action=np.zeros((1,20),np.float32),label='external',external_roots={'logging':external},snapshot_factory=NativeLikeFakeSnapshot,strict_native=False)
        external.counter=7;tx9.begin('external');check('external_logging_holder_not_rolled_back',external.counter==7);tx9.finish('external')
        def closure_fixture():
            count=4
            def hook():
                nonlocal count
                count+=1
            def get():return count
            return hook,get
        tx10,e10,r10,l10,t10,m10,f10=fixture();hook,get=closure_fixture();e10.sim.step=hook
        tx10=ProbeTransaction(e10,r10,capacity_observer=l10,source_action=np.zeros((1,20),np.float32),label='closure',snapshot_factory=NativeLikeFakeSnapshot,strict_native=False)
        hook();tx10.begin('closure');check('nonlocal_scalar_closure_restored',get()==4);tx10.finish('closure')
        tx11,e11,r11,l11,t11,m11,f11=fixture();py0=random.getstate();np0=np.random.get_state()
        random.random();np.random.random();tx11.begin('rng')
        check('python_global_RNG_restored',random.getstate()==py0)
        after=np.random.get_state();check('NumPy_global_RNG_restored',after[0]==np0[0] and np.array_equal(after[1],np0[1]) and after[2:]==np0[2:]);tx11.finish('rng')
        tx12,e12,r12,l12,t12,m12,f12=fixture();tx12.source_action[0,0]=1
        reject('source_action_mutation_rejected',tx12.verify_current)
        reject('default_strict_rejects_fakeEnv_even_with_native_fields',lambda:ProbeTransaction(e12,r12,capacity_observer=l12,source_action=np.zeros((1,20),np.float32),label='badnative'))
        tx13,e13,r13,l13,t13,m13,f13=fixture()
        def body_failure():
            with tx13.branch('bodyerror'):
                raise ValueError('original fixture branch error')
        reject('branch_context_preserves_body_failure',body_failure,(ValueError,))
        check('body_failure_marks_failed_and_latches_final_capacity',tx13.failed and tx13.active is None and len(l13.records)==2 and tx13.events[-1]['body_completed'] is False)
        reject('body_failure_forbids_nextbranch',lambda:tx13.begin('forbidden'))
        tx14,e14,r14,l14,t14,m14,f14=fixture()
        def body_and_capacity_failure():
            with tx14.branch('dualerror'):
                f14.values[:]=4
                raise ValueError('original remains primary')
        reject('body_error_not_masked_by_finish_capacity_error',body_and_capacity_failure,(ValueError,))
        check('secondary_capacity_error_OR_preserved',l14.cumulative_bitmask==4 and 'error' in tx14.events[-1] and tx14.failed)
        foreign=Foreign();foreign.local_np=np.random.default_rng(44);foreign.local_py=random.Random(33)
        tx15,e15,r15,l15,t15,m15,f15=fixture(extra=foreign)
        npstate=repr(foreign.local_np.bit_generator.state);pystate=foreign.local_py.getstate()
        foreign.local_np.random();foreign.local_py.random();tx15.begin('localrng')
        check('local_numpy_Generator_restored',repr(foreign.local_np.bit_generator.state)==npstate)
        check('local_python_Random_restored',foreign.local_py.getstate()==pystate);tx15.finish('localrng')
        check('no_Torch_Warp_MuJoCo_import_or_actualEnv',not any(x in sys.modules for x in ('torch','warp','mujoco','mujoco_warp')))
        check('source_unchanged_after_contract_checks',pin(source)==initial)
        result={'schema':'probe_transaction_CPU_boundary_contract:1','status':'PASS_FAKE_CPU_CONTRACT_NOT_NATIVE_QUALIFICATION' if all(checks.values()) else 'FAIL_RETAINED','checks':checks,
            'passed':sum(checks.values()),'total':len(checks),'source':initial,'checker':pin(__file__),'example_inventory':tx.describe(),
            'native_source_read_not_executed':pin(Path('D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/snapshot.py')),
            'native_snapshot_qualified':False,'Env_created':False,'physics_steps':0,'WM_calls':0,'GPU_used':False,
            'scope':'Fake stand-ins exercise boundary/latch/sidecar/byte/RNG behavior only. Actual native profile/all-array readback/cost must be checked in separately authorized worker.'}
    except BaseException as exc:
        result={'schema':'probe_transaction_CPU_boundary_contract:1','status':'HARNESS_OR_CONTRACT_FAILURE_RETAINED','checks':checks,'error':repr(exc),'traceback':traceback.format_exc(),'source':initial,'checker':pin(__file__),'native_snapshot_qualified':False,'GPU_used':False,'Env_created':False,'physics_steps':0}
    (a.output/'report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'passed':sum(checks.values()),'total':len(checks),'failed':[k for k,v in checks.items() if not v],'report':pin(a.output/'report.json')},indent=2))
    raise SystemExit(0 if result['status']=='PASS_FAKE_CPU_CONTRACT_NOT_NATIVE_QUALIFICATION' else 2)


if __name__=='__main__':main()
