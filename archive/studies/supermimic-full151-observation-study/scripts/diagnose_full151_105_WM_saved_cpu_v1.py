"""One frozen105 WM saved-transition CPU diagnostic, never simulation."""
from pathlib import Path
import argparse,csv,hashlib,json,os,random,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
    p=Path(p).resolve(strict=True)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--run',action='store_true');a=ap.parse_args()
    p=read(a.protocol)
    for v in p['source_pins'].values():
        if pin(v['path'])['sha256']!=v['sha256']:raise ValueError('Pinned input/source changed')
    if p['schema']!='full151105_saved_WM_CPU_diagnostic:1' or p['phases']!=[0,49,99,149] or p['direction_joint_indices']!=[0,4,8,12] or p['eps_rad']!=[.001,.0005]:raise ValueError('Fixed one-batch diagnostic contract changed')
    if not a.run:print(json.dumps({'status':'CPU_METADATA_ONLY','Torch_imported':False,'WM_forwards':0,'GPU_started':False}));return
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'execution_source.py').write_bytes(Path(__file__).read_bytes());save(out/'protocol_snapshot.json',p)
    import torch
    import torch.nn.functional as F
    from full151_native.bundle import load_bundle
    from full151_native.world_model_checkpoint import load_world_model_checkpoint
    from full151_native.checkpoint import _tensor_digest
    from dex_supermimic.rl.dex_supermimic.state import difference
    torch.set_num_threads(4)
    if torch.cuda.is_initialized():raise ValueError('Fresh CPU process required')
    start=time.perf_counter();rng=torch.get_rng_state().clone();pyrng=repr(random.getstate());nprng=repr(np.random.get_state())
    wm,receipt=load_world_model_checkpoint(p['checkpoint']['path'],load_bundle(p['bundle']['path']),expected_method='A1L',training_run=p['training_run'],health_report=p['health']['path'],expected_health_auditor=p['health_auditor'],expected_iteration=2000)
    before=_tensor_digest(wm.state_dict());checks={};arrays={};groups=[];offsets=[]
    def check(n,b):checks[n]=bool(b)
    for path in p['paths']:
        d=Path(path['directory']);rows=read(d/'native_reward_transitions.json');events=read(d/'control_events.json');raw=np.load(d/'raw_states.npz',allow_pickle=False)
        check(path['name']+'_full150_phase',[r['phase'] for r in rows]==list(range(150))==[r['phase'] for r in events])
        x=np.asarray([r['x0'] for r in rows],np.float32);yr=np.asarray([r['x1_reward_snapshot'] for r in rows],np.float32);ym=np.asarray([r['x1_after_Manager'] for r in rows],np.float32);u=np.asarray([r['actualu'] for r in rows],np.float32).reshape(150,20);label=np.asarray([r['native_selfcollision15'] for r in rows],np.float32).reshape(150,15)
        check(path['name']+'_shapes_and_finite',x.shape==yr.shape==ym.shape==(150,58) and all(np.isfinite(v).all() for v in (x,yr,ym,u,label)))
        check(path['name']+'_actualu_event_exact',np.array_equal(u,np.asarray([r['actualu'] for r in events],np.float32).reshape(150,20)))
        check(path['name']+'_native_x0_next_Manager_continuity',np.array_equal(x[1:],ym[:-1]))
        check(path['name']+'_state58_joint_raw_phase',np.array_equal(x[:,:20],raw['joint_rad'][::4][:-1].astype(np.float32)) and np.array_equal(x[:,20:40],raw['joint_velocity_rad_s'][::4][:-1].astype(np.float32)))
        check(path['name']+'_601clock_600solver_150forwards',len(raw['raw_time_s'])==601 and len(read(d/'solver_samples.json'))==600 and read(d/'evaluation.json')['native_postinterval_forward_calls']==150)
        check(path['name']+'_binary_native_self15',np.isin(label,[0,1]).all())
        i=len(groups)*150;groups.append(path['name']);offsets.append((i,i+150));arrays[path['name']+'_x0']=x;arrays[path['name']+'_reward_target']=yr;arrays[path['name']+'_Manager_target']=ym;arrays[path['name']+'_actualu']=u;arrays[path['name']+'_self15_truth']=label
        raw.close()
    x=torch.from_numpy(np.concatenate([arrays[g+'_x0'] for g in groups]));u=torch.from_numpy(np.concatenate([arrays[g+'_actualu'] for g in groups]));xcopy=x.clone();ucopy=u.clone()
    with torch.no_grad():pred,logits=wm(x,u)
    arrays['prediction_x1_300x58']=pred.numpy();arrays['self15_logits_300x15']=logits.numpy();metrics=[];selfmetrics=[]
    for g,(lo,hi) in zip(groups,offsets):
        xt=x[lo:hi];pr=pred[lo:hi]
        for kind,key in [('reward_snapshot','_reward_target'),('after_Manager','_Manager_target')]:
            target=torch.from_numpy(arrays[g+key]);err=difference(pr,target).numpy();dp=difference(xt,pr).numpy();dt=difference(xt,target).numpy()
            arrays[g+'_'+kind+'_error_delta52']=err;arrays[g+'_'+kind+'_predicted_delta52']=dp;arrays[g+'_'+kind+'_true_delta52']=dt
            metrics.append({'path':g,'target_phase':kind,'transitions':150,'q_RMSE_rad':float(np.sqrt(np.mean(err[:,:20]**2))),'qd_RMSE_rad_s':float(np.sqrt(np.mean(err[:,20:40]**2))),'object_p_RMS_m':float(np.sqrt(np.mean(np.sum(err[:,40:43]**2,axis=1)))),'object_R_RMS_deg':float(np.sqrt(np.mean(np.sum(err[:,43:46]**2,axis=1)))*180/np.pi),'object_v_RMS_m_s':float(np.sqrt(np.mean(np.sum(err[:,46:49]**2,axis=1)))),'object_omega_RMS_rad_s':float(np.sqrt(np.mean(np.sum(err[:,49:52]**2,axis=1)))),'delta52_component_RMSE':float(np.sqrt(np.mean((dp-dt)**2)))})
        truth=torch.from_numpy(arrays[g+'_self15_truth']);lg=logits[lo:hi];prob=torch.sigmoid(lg);binary=prob>=.5;pos=int(truth.sum());tp=int((binary&truth.bool()).sum());fp=int((binary&~truth.bool()).sum());fn=int((~binary&truth.bool()).sum());selfmetrics.append({'path':g,'labels':2250,'positive_labels':pos,'BCE':float(F.binary_cross_entropy_with_logits(lg,truth)),'Brier':float(((prob-truth)**2).mean()),'predicted_positive':int(binary.sum()),'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'F1':2*tp/(2*tp+fp+fn) if pos else None,'label_scope':'native selfcollision15 only; no HO/force/human-contact labels'})
    fdrows=[];advalues=[];fdvalues=[]
    for gi,g in enumerate(groups):
        for k in p['phases']:
            xx=x[gi*150+k:gi*150+k+1].clone();uu=u[gi*150+k:gi*150+k+1].clone()
            def delta(v):return difference(xx,wm(xx,v)[0]).reshape(52)
            for j in p['direction_joint_indices']:
                direction=torch.zeros_like(uu);direction[0,j]=1.
                _,ad=torch.autograd.functional.jvp(delta,uu,direction,create_graph=False)
                for eps in p['eps_rad']:
                    with torch.no_grad():fd=(delta(uu+eps*direction).double()-delta(uu-eps*direction).double())/(2*eps)
                    err=ad.double()-fd;advalues.append(ad.numpy());fdvalues.append(fd.numpy());fdrows.append({'path':g,'phase':k,'direction_joint':j,'epsilon_rad':eps,'AD_norm':float(ad.double().norm()),'FD_norm':float(fd.norm()),'max_abs_difference':float(err.abs().max()),'relative_L2_difference':float(err.norm()/torch.maximum(ad.double().norm(),fd.norm()).clamp_min(1e-12))})
    arrays['self_gradient_AD_64x52']=np.asarray(advalues);arrays['self_gradient_FD_64x52']=np.asarray(fdvalues)
    check('prediction_and_gradients_all_finite',all(np.isfinite(v).all() for v in arrays.values()))
    check('WM_complete_state_dict_norm_weights_unchanged',before==_tensor_digest(wm.state_dict()))
    check('all_parameters_frozen',all(not v.requires_grad for v in wm.parameters()))
    check('actual_inputs_unchanged',torch.equal(x,xcopy) and torch.equal(u,ucopy))
    check('Torch_RNG_unchanged',torch.equal(rng,torch.get_rng_state()));check('Python_numpy_RNG_unchanged',pyrng==repr(random.getstate()) and nprng==repr(np.random.get_state()))
    check('checkpoint_bytes_unchanged',pin(p['checkpoint']['path'])==p['checkpoint']);check('GPU_not_initialized',not torch.cuda.is_initialized())
    check('all_pinned_inputs_sources_unchanged',all(pin(v['path'])['sha256']==v['sha256'] for v in p['source_pins'].values()))
    np.savez_compressed(out/'predictions_and_self_gradients.npz',**arrays);save(out/'metrics.json',metrics);save(out/'self15_metrics.json',selfmetrics);save(out/'WM_self_gradient_FD.json',fdrows);save(out/'loader_receipt.json',receipt)
    for name,rows in [('metrics.csv',metrics),('WM_self_gradient_FD.csv',fdrows)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={'schema':'full151105_saved_WM_CPU_diagnostic_report:1','status':'COMPLETED_CPU_DIAGNOSTIC' if all(checks.values()) else 'FAILED_DIAGNOSTIC_CONTRACT','checks':checks,'passed':sum(checks.values()),'total':len(checks),'protocol':pin(a.protocol),'source':pin(__file__),'source_pins':p['source_pins'],'base_predictions':300,'paths':groups,'self_gradient_cases':64,'normalizer_weights_checkpoint_RNG_unchanged':all(checks[n] for n in ('WM_complete_state_dict_norm_weights_unchanged','Torch_RNG_unchanged','Python_numpy_RNG_unchanged','checkpoint_bytes_unchanged')),'wall_seconds':time.perf_counter()-start,'GPU_used':False,'Env_physics_optimizer_calls':0,'C2_or_EMA_reapplication':False,'scope':'Single trained105 A1L seed0 frozen FP32CPU WM;150 dependent transitions/path on2 same-source paths. Primary truth is native reward_snapshot before Managerforward; after_Manager is secondary. DELTA52 native rotation log and self15 native selfcollision only. AD-vs-FD is internal WM numerical differentiability, without a quality threshold or physical/control/contact Jacobian qualification, GPUAMP equivalence, independent ego N or generalization claim.'}
    save(out/'report.json',report);print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'failed':[k for k,v in checks.items() if not v],'metrics':metrics,'report':pin(out/'report.json')},indent=2));raise SystemExit(0 if all(checks.values()) else 1)

if __name__=='__main__':main()
