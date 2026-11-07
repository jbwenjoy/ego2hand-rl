"""Three narrow synthetic H8 contracts; no real inputs/weights/native WM/FW."""
from pathlib import Path
import copy,hashlib,json,os,sys,time
ROOT=Path('D:/workspace/supermimic-full151-observation-study')
FILES={'consumer':ROOT/'src/full151_gradient/component_H8_donor105_v1.py',
       'CLI':ROOT/'scripts/diagnose_full151_component_H8_donor_cpu_v1.py','checker':Path(__file__),
       'independent_math':ROOT/'src/full151_gradient/component_H8_donor_independent_math_v1.py'}


def pin(p):return {'path':str(Path(p).resolve()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}


def main():
    out=ROOT/'outputs/full151105_component_H8_donor_preparation_CPU_v1';out.mkdir()
    pins={k:pin(v) for k,v in FILES.items()}
    for k,v in FILES.items():(out/(k+'_source_before.py')).write_bytes(v.read_bytes())
    os.environ['CUDA_VISIBLE_DEVICES']='-1';sys.path.insert(0,str(ROOT/'src'))
    import numpy as np
    import torch
    from full151_gradient import component_H8_donor105_v1 as h
    from full151_gradient import component_H8_donor_independent_math_v1 as ref
    torch.set_num_threads(4);started=time.perf_counter();checks={}
    def reject(fn):
        try:fn()
        except ValueError:return True
        return False
    # 1: Restored rows deliberately precede donors; fixed path/phase key lookup.
    identities=[]
    for b in range(152):
        k=(0,49,99,149)[b//19%4];path=('A0','A2')[b//76];mode='baseline' if b%19<3 else 'direction'
        identities.append({'group_id':len(identities),'path':path,'k':k,'mode':mode,'branch_id':b,'source_frame':200+k,
                           'terminal_observation':k+1,'valid_next_policy_decision':k<149,'no_u150':True})
    for path in ('A0','A2'):
        for k in range(150):identities.append({'group_id':len(identities),'path':path,'k':k,'mode':'donor','branch_id':None,
            'source_frame':200+k,'terminal_observation':k+1,'valid_next_policy_decision':k<149,'no_u150':True})
    arrays={'pre_x452_FP32':np.zeros((452,58),np.float32),'target_x452_FP32':np.zeros((452,58),np.float32),
            'actual_u452_FP32':np.zeros((452,20),np.float32),'all_substep_u452x4x20_FP32':np.zeros((452,4,20),np.float32)}
    for row in identities:
        if row['mode']=='donor':
            z=row['group_id'];k=row['k'];pi=('A0','A2').index(row['path'])
            arrays['pre_x452_FP32'][z,0]=pi*1000+k;arrays['target_x452_FP32'][z,0]=pi*1000+k+1
            arrays['actual_u452_FP32'][z,0]=np.float32(k)*np.float32(.01)
            arrays['all_substep_u452x4x20_FP32'][z]=arrays['actual_u452_FP32'][z]
    data,starts=h.assemble_donor_windows(arrays,identities,np);mapping=ref.donor_window_mapping(identities)
    ref.source_byte_guards(arrays['pre_x452_FP32'],arrays['target_x452_FP32'],arrays['actual_u452_FP32'],arrays['all_substep_u452x4x20_FP32'],mapping)
    bad_link=copy.deepcopy(arrays);bad_link['target_x452_FP32'][mapping['donor_row_indices2x150'][0,3],56]=np.float32(-0.)
    bad_ctrl=copy.deepcopy(arrays);bad_ctrl['all_substep_u452x4x20_FP32'][mapping['donor_row_indices2x150'][1,6],2,1]=np.float32(-0.)
    checks['donor_keyjoin_286_boundaries_298links_300held_bits']=bool(
        np.array_equal(data['donor_indices2x150'],mapping['donor_row_indices2x150'])
        and np.array_equal(data['source_group_ids286x8'],mapping['window_transition_row_indices286x8'])
        and np.array_equal(data['initial_x286_FP32'],arrays['pre_x452_FP32'][mapping['window_initial_row_indices286']])
        and starts[-1]['start_k']==142 and starts[-1]['action_phases'][-1]==149 and starts[-1]['end_observation']==150
        and reject(lambda:h.assemble_donor_windows(bad_link,identities,np)) and reject(lambda:h.assemble_donor_windows(bad_ctrl,identities,np)))
    # 2: Full xi bytes, not allclose; fake final cannot become initial baseline.
    xi=torch.arange(46,dtype=torch.float32);old=xi.numpy().copy();h.verify_xi(xi,old,np,torch)
    changed=old.copy();changed[0]=np.float32(-0.)
    fake={'schema':h.c.ARTIFACT_SCHEMA,'WM':{},'WM_state':{},'arm':'pointwise','iteration':1000}
    checks['xi_byteguard_and_independent_initial_namespace']=bool(reject(lambda:h.verify_xi(xi,changed,np,torch))
        and reject(lambda:h.verify_xi(xi,old.astype('f8'),np,torch)) and reject(lambda:h.make_initial_eval_component(fake,{},torch)))
    # 3: Tiny callable, not native model. Independent trace and raw-unit reference.
    class Tiny:
        def __init__(self,fail=None):self.calls=0;self.full=[];self.fail=fail
        def __call__(self,x,u,dynamics):
            self.calls+=1;self.full.append(dynamics.numpy().copy());y=x.clone()
            y[:,0]=dynamics[:,0]+u[:,0];y[:,40]=dynamics[:,40]+x[:,0]*.125
            if self.calls==self.fail:y[0,0]=float('nan')
            return y,torch.zeros((286,15),dtype=torch.float32)
    x0=torch.zeros((286,58));x0[:,0]=torch.arange(286,dtype=torch.float32)*.01
    u=torch.ones((286,8,20))*2.;fixed=torch.arange(46,dtype=torch.float32)*.01
    counter={'WM_forward_attempted':0,'WM_forward_returned':0,'predicted_rows':0};model=Tiny()
    result=h.autoregress(model,x0,u,fixed,torch,np,counter,lambda r:None)
    ref.trace_byte_guards(x0.numpy(),result['prediction58_FP32'],result['state_input58_FP32'],result['dynamics_state58_FP32'],fixed.numpy(),np.stack(model.full,axis=1))
    partial={};bad=Tiny(3);bad_count={'WM_forward_attempted':0,'WM_forward_returned':0,'predicted_rows':0}
    def store(r):partial.update({k:v.copy() for k,v in r.items()})
    stopped=reject(lambda:h.autoregress(bad,x0,u,fixed,torch,np,bad_count,store))
    truth=np.zeros((286,8,58),np.float32);synthetic=truth.copy();synthetic[:,:,40]=5.
    error,stats=h.error_statistics(synthetic,truth,starts,np);reference=ref.raw58_error_statistics(synthetic,truth,mapping['window_path_index286'])
    p=[z for z in stats if z['horizon']==1 and z['path']=='ALL' and z['block']=='p'][0]
    checks['autoregress_actual_x_dynamics_noTF_finite_stop_rawunits']=bool(counter=={'WM_forward_attempted':8,'WM_forward_returned':8,'predicted_rows':2288}
        and np.array_equal(result['actual_call_u20_FP32'],u.numpy()) and stopped and bad.calls==3 and bad_count['predicted_rows']==572
        and partial['attempted_mask'][:,:3].all() and not partial['attempted_mask'][:,3:].any() and np.isnan(partial['prediction58_FP32'][:,3:]).all()
        and not partial['completed_mask'][0,2] and abs(p['component_RMSE']-5/np.sqrt(3))<1e-14 and p['vector_or_Frobenius_RMS']==5.
        and np.array_equal(error,reference['signed_error286x8x58_F64']))
    require_unchanged=all(pin(v)==pins[k] for k,v in FILES.items())
    if not require_unchanged or torch.cuda.is_initialized() or 'mujoco' in sys.modules or 'warp' in sys.modules:raise RuntimeError('Source/backend drift')
    report={'schema':'full151105_component_H8_synthetic_CPU_contract:1','status':'PASS_SYNTHETIC_MAPPING_XI_AUTOREGRESS_ONLY' if all(checks.values()) else 'FAIL_SYNTHETIC_CONTRACT',
        'passed':sum(checks.values()),'total':len(checks),'checks':checks,'source_pins':pins,
        'scope':{'synthetic_Tiny_calls':11,'actual_donor_NPZ_reads':0,'actual_window_constructions':0,'real_component_weight_loads':0,
                 'native_WM_constructors_or_actual_H8_FW':0,'AD_optimizer_Env_MJB_physics_GPU':0},'source_unchanged':True,'elapsed_seconds':time.perf_counter()-started}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    ready={'schema':'full151105_component_H8_synthetic_READY:1','status':report['status'],'source_pins':pins,
           'cpu_report':pin(out/'report.json'),'cpu_checks':{'passed':report['passed'],'total':report['total']},'actual_H8_authorized':False}
    (out/'READY.json').write_text(json.dumps(ready,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'READY':pin(out/'READY.json')}))
    return 0 if all(checks.values()) else 1


if __name__=='__main__':raise SystemExit(main())
