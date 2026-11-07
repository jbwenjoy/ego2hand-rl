"""Exactly three synthetic H8 contracts; no real arrays/weights/native model."""
from pathlib import Path
import copy,hashlib,json,math,os,sys,time,traceback
ROOT=Path('D:/workspace/supermimic-full151-observation-study')
OUT=ROOT/'outputs/full151105_public_self15_H8_donor_preparation_CPU_v1'
FILES={'consumer':ROOT/'src/full151_gradient/public_self15_H8_donor105_v1.py','CLI':ROOT/'scripts/diagnose_full151_public_self15_H8_donor_cpu_v1.py','checker':Path(__file__)}
DEPS={'old_H8':ROOT/'src/full151_gradient/component_H8_donor105_v1.py','independent_math':ROOT/'src/full151_gradient/component_H8_donor_independent_math_v1.py','cached_metrics':ROOT/'src/full151_gradient/boundary_self15_cached_saved105_v1.py','source_contract':ROOT/'docs/public_self15_H8_donor_contract_v1.md'}

def pin(path):return {'path':str(Path(path).resolve(strict=True)),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}
def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def fixture(np):
    identities=[];joined=[]
    arrays={k:np.zeros(shape,np.float32) for k,shape in {'pre_x452_FP32':(452,58),'target_x452_FP32':(452,58),'actual_u452_FP32':(452,20),'all_substep_u452x4x20_FP32':(452,4,20)}.items()}
    for pi,path in enumerate(('A0','A2')):
        for k in range(150):
            gid=len(identities);g={'group_id':gid,'path':path,'mode':'donor','k':k,'branch_id':None,'source_frame':200+k,'terminal_observation':k+1,'valid_next_policy_decision':k<149,'no_u150':True,'first_physical_pre_index':4*gid,'last_physical_index':4*gid+3,'post20_GPU_row_index':1808+gid}
            raw=[0]*25;raw[2]=int(path=='A2');target=[0.,float(path=='A2')]+[0.]*13
            identities.append(g);joined.append({**g,'contact15':target,'raw_found25':raw,'label_time':'lastsolver_preintegration~kdt+.015','state_time':'postintegration_kdt+.020'})
            arrays['pre_x452_FP32'][gid,0]=pi*1000+k;arrays['target_x452_FP32'][gid,0]=pi*1000+k+1
            arrays['actual_u452_FP32'][gid,0]=np.float32(k)*np.float32(.01);arrays['all_substep_u452x4x20_FP32'][gid]=arrays['actual_u452_FP32'][gid]
        for off in range(76):
            gid=len(identities);k=(0,49,99,149)[off//19]
            identities.append({'group_id':gid,'path':path,'mode':'baseline' if off%19<3 else 'direction','k':k,'branch_id':pi*76+off,'source_frame':200+k,'terminal_observation':k+1,'valid_next_policy_decision':k<149,'no_u150':True})
    return identities,joined,arrays

def main():
    OUT.mkdir();pins={k:pin(v) for k,v in FILES.items()};deps={k:pin(v) for k,v in DEPS.items()}
    for key,path in FILES.items():(OUT/(key+'_source_before.py')).write_bytes(path.read_bytes())
    os.environ['CUDA_VISIBLE_DEVICES']='-1';sys.path.insert(0,str(ROOT/'src'));started=time.perf_counter();checks={}
    try:
        import numpy as np
        import torch
        from full151_gradient import public_self15_H8_donor105_v1 as h
        from full151_gradient import component_H8_donor_independent_math_v1 as independent
        torch.set_num_threads(4)
        def reject(fn):
            try:fn()
            except ValueError:return True
            return False
        # 1: H1 gaps after A0 donors, terminal obs150, labels by gid dictionary.
        identities,joined,arrays=fixture(np);donor=h.join_donor_labels(identities,joined,np)
        data,starts=h.old_h8.assemble_donor_windows(arrays,identities,np);expanded=h.expand_donor_labels(data['source_group_ids286x8'],donor,np)
        bad_label=copy.deepcopy(joined);bad_label[-1]['contact15'][1]=0.
        bad_order=copy.deepcopy(joined);bad_order[0],bad_order[150]=bad_order[150],bad_order[0]
        bad_control=copy.deepcopy(arrays);bad_control['all_substep_u452x4x20_FP32'][226,2,1]=np.float32(-0.)
        bad_link=copy.deepcopy(arrays);bad_link['target_x452_FP32'][3,57]=np.float32(-0.)
        checks['donor300_gap_gid_terminal_298links_300held_and_binary_label_mapping']=bool(donor['donor_source_group_ids300'][150]==226 and data['source_group_ids286x8'][-1,-1]==375 and starts[-1]['end_observation']==150
            and expanded['target_self15_286x8x15_FP32'][:143,:,1].sum()==0 and expanded['target_self15_286x8x15_FP32'][143:,:,1].sum()==1144
            and expanded['donor_label_repetition_count300'].sum()==2288 and (expanded['donor_label_repetition_count300']>0).all()
            and reject(lambda:h.join_donor_labels(identities,bad_label,np)) and reject(lambda:h.join_donor_labels(identities,bad_order,np))
            and reject(lambda:h.old_h8.assemble_donor_windows(bad_control,identities,np)) and reject(lambda:h.old_h8.assemble_donor_windows(bad_link,identities,np)))
        # 2: Analytic zero logits: BCE ln2, Brier1/4; all ties negative, null recall.
        logits=np.zeros((286,8,15),np.float32);tables=[]
        for role in h.MODEL_ROLES:
            elements,rows=h.cached_self15_statistics(logits,expanded['target_self15_286x8x15_FP32'],starts,role,np);tables.extend(rows)
        all_row=tables[0];empty=tables[1];positive=tables[2]
        checks['all72_cached_tables_complete_elements_strict_ties_nullable_denominators']=bool(len(tables)==72 and elements['BCE15_F64'].shape==(286,8,15)
            and np.all(elements['zero_logit_tie15_bool']) and not elements['decision15_bool'].any() and np.all(elements['Brier15_F64']==.25)
            and abs(all_row['BCE_mean_entry_F64']-math.log(2))<1e-14 and all_row['confusion_total']['zero_logit_tie']==4290
            and empty['positive_recall'] is None and empty['positive_precision'] is None and positive['positive_recall']==0. and positive['positive_precision'] is None
            and positive['per_feature_positive_recall'][0] is None and positive['per_feature_positive_recall'][1]==0.)
        # 3: Tiny callables only, no native ctor: x/dynamics recurrent together.
        class Tiny:
            def __init__(self,wrapped=False,fail=None):self.wrapped=wrapped;self.calls=0;self.fail=fail
            def __call__(self,x,u,dynamics):
                self.calls+=1;y=x.clone();y[:,0]=dynamics[:,0]+u[:,0];y[:,40]=dynamics[:,40]+x[:,0]*.125
                if self.calls==self.fail:y[0,0]=float('nan')
                output=(y,torch.zeros((286,15),dtype=torch.float32))
                return (*output,torch.zeros((286,2,6)),torch.zeros((286,2))) if self.wrapped else output
        x=torch.zeros((286,58));x[:,0]=torch.arange(286,dtype=torch.float32)*.01;u=torch.ones((286,8,20))*2.;xi=torch.arange(46,dtype=torch.float32)*.01
        counter={'native_forward_attempted':0,'native_forward_returned':0,'aux_head_forward_attempted':0,'aux_head_forward_returned':0,'predicted_rows':0}
        trace_ok=True
        for role in h.MODEL_ROLES:
            result=h.autoregress(role,Tiny(role in h.ARMS),x,u,xi,torch,np,counter,lambda r:None)
            independent.trace_byte_guards(x.numpy(),result['prediction58_FP32'],result['state_input58_FP32'],result['dynamics_state58_FP32'],xi.numpy(),result['dynamics278_FP32'])
            trace_ok &= np.array_equal(result['actual_call_u20_FP32'],u.numpy()) and result['completed_mask'].all()
        partial={};bad_counter={key:0 for key in counter}
        def save_partial(value):partial.update({key:v.copy() for key,v in value.items()})
        stopped=reject(lambda:h.autoregress(h.MODEL_ROLES[0],Tiny(fail=3),x,u,xi,torch,np,bad_counter,save_partial))
        checks['three_role_autoregress_x_dyn_u_counts_and_wholebatch_finite_stop']=bool(trace_ok and counter=={'native_forward_attempted':24,'native_forward_returned':24,'aux_head_forward_attempted':16,'aux_head_forward_returned':16,'predicted_rows':6864}
            and stopped and bad_counter['native_forward_returned']==3 and bad_counter['predicted_rows']==572 and not partial['completed_mask'][0,2] and np.isnan(partial['prediction58_FP32'][:,3:]).all() and not partial['attempted_mask'][:,3:].any())
        if not all(checks.values()):raise ValueError('Synthetic contract failed: '+str(checks))
        if torch.cuda.is_initialized() or any(k in sys.modules for k in ('mujoco','warp')):raise ValueError('Backend isolation failed')
        if any(pin(path)!=pins[key] for key,path in FILES.items()) or any(pin(path)!=deps[key] for key,path in DEPS.items()):raise ValueError('Frozen proof sources changed')
        report={'schema':'full151105_public_self15_H8_donor_CPU_contract:1','status':'PASS_SYNTHETIC_GID_LABELS_METRICS_AND_AUTOREGRESS_ONLY','checks':checks,'passed':sum(checks.values()),'total':len(checks),'source_pins':pins,'dependency_pins':deps,
            'scope':{'synthetic_Tiny_callable_FW':27,'actual_NPZ_reads':0,'actual_weights_loads':0,'native_WM_constructors':0,'actual_H8_FW':0,'actual_donor_window_constructions':0,'AD_optimizer_MJB_Env_physics_GPU':0},'CUDA_initialized':False,'source_unchanged':True,'elapsed_seconds':time.perf_counter()-started}
        write(OUT/'report.json',report);ready={'schema':'full151105_public_self15_H8_donor_CPU_READY:1','status':report['status'],'source_pins':pins,'dependency_pins':deps,'cpu_report':pin(OUT/'report.json'),'cpu_checks':{'passed':report['passed'],'total':report['total']},'actual_H8_authorized':False}
        write(OUT/'READY.json',ready);print(json.dumps({'status':ready['status'],'passed':report['passed'],'total':report['total'],'READY':pin(OUT/'READY.json')}));return 0
    except BaseException as error:
        write(OUT/'failure.json',{'error':repr(error),'traceback':traceback.format_exc(),'checks':checks,'source_pins':pins,'automatic_retry':False,'actual_data_weights_nativeModels_MJB_physics':0});raise

if __name__=='__main__':raise SystemExit(main())
