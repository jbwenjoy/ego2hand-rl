"""Predeclared, saved-only native104 CPU prediction/gradient diagnostic."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,csv,hashlib,json,random,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rough_prefix.io import read,pin,checked,canonical
from rough_prefix.metadata import write


def all_pins(value):
    if isinstance(value,dict):
        if set(value)=={'path','sha256'}:yield value
        else:
            for v in value.values():yield from all_pins(v)
    elif isinstance(value,list):
        for v in value:yield from all_pins(v)


def preflight(plan_path):
    plan=read(plan_path);p=read(checked(plan['protocol']))
    if plan['schema']!='rough_saved_WM_CPU_runplan:1' or p['schema']!='rough_WM_onpolicy_saved_CPU_diagnostic_protocol:1':raise ValueError('Wrong registered CPU diagnostic')
    for value in all_pins(plan):checked(value)
    for value in all_pins(p):checked(value)
    if plan['source_pins']['entry']!=pin(__file__):raise ValueError('Diagnostic source changed')
    if plan.get('CPU_run_authorized') is not True or plan.get('automatic_retry') is not False:raise ValueError('Explicit fixed CPU runplan required')
    if not Path(plan['output']).resolve().is_relative_to((ROOT/'outputs').resolve()):raise ValueError('Owned diagnostic output required')
    for key,name in (('state_codec','saved_state_codec_v3.py'),('WM_loader','world_model_checkpoint_v2.py')):
        if plan['source_pins'][key]!=pin(ROOT/'src/rough_prefix'/name):raise ValueError('Diagnostic dependency binding differs')
    if p['scope']['methods']!=['A1L','P_ONLY','A2'] or p['scope']['trajectory_order']!=['A0','A1L','P_ONLY','A2']:raise ValueError('Unregistered model/path selection')
    if p['transition_table']['total_WM_predictions']!=180 or p['execution']['CPU_only'] is not True:raise ValueError('Wrong prediction scope')
    if any(x in sys.modules for x in ('torch','warp','mujoco_warp')):raise AssertionError('Preflight framework import')
    return plan,p


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-plan',type=Path,required=True)
    ap.add_argument('--run',action='store_true');a=ap.parse_args();plan,p=preflight(a.run_plan)
    if not a.run:print({'status':'CPU_METADATA_PREFLIGHT','GPU_started':False,'Torch_imported':False,'physics_steps':0});return
    out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False)
    for name,value in plan['source_pins'].items():
        path=checked(value);(out/(name+'__'+path.name)).write_bytes(path.read_bytes())
    write(out/'runplan.json',plan);write(out/'protocol.json',p)
    started=time.perf_counter();deadline=started+p['execution']['soft_worker_budget_seconds']
    phases={};summary={'schema':'rough_saved_WM_CPU_diagnostic:1','status':'INITIALIZING','models':[],'groups':[],
        'source_lineage_N':1,'independent_ego_N':0,'physics_steps_added':0,'GPU_used':False,
        'physical_gradient_qualified':False,'selfcollision_truth':'UNKNOWN_not_saved_found_sensor'}
    def stage(name):
        if time.perf_counter()>deadline:raise TimeoutError('Fixed CPU diagnostic budget, no automatic retry')
        write(out/'runtime.json',{'status':'RUNNING','pid':os.getpid(),'phase':name,'elapsed_seconds':time.perf_counter()-started,'physics_steps_added':0,'GPU_used':False})
    try:
        import torch,mujoco
        torch.set_num_threads(4)
        from rough_prefix.bundle_v2 import load_bundle
        from rough_prefix.saved_state_codec_v3 import load_saved_transitions
        from rough_prefix.world_model_checkpoint_v2 import load_world_model_checkpoint
        from rough_prefix.checkpoint_v2 import _tensor_digest
        from dex_supermimic.rl.dex_supermimic.state import unpack,pack,difference
        from dex_supermimic.rl.dex_supermimic.rotations import quat_to_mat,mm,mv,so3_exp
        cpu_rng=torch.get_rng_state().clone();np_rng=np.random.get_state();py_rng=random.getstate()
        bundle=load_bundle(checked(p['inputs']['input_bundle']));model=mujoco.MjModel.from_binary_path(str(checked(bundle['model'])))
        ids=np.array([model.joint(f'robot/right_finger{f}_joint{j}').id for f in range(1,6) for j in range(1,5)])
        lower=torch.tensor(model.jnt_range[ids,0].copy(),dtype=torch.float32);upper=torch.tensor(model.jnt_range[ids,1].copy(),dtype=torch.float32)
        aids=np.array([np.flatnonzero(model.actuator_trnid[:,0]==j).item() for j in ids])
        ulo=np.maximum(model.jnt_range[ids,0],model.actuator_ctrlrange[aids,0]).astype(np.float32)+np.float32(.02)
        uhi=np.minimum(model.jnt_range[ids,1],model.actuator_ctrlrange[aids,1]).astype(np.float32)-np.float32(.02)
        arrays={};datasets={};receipts=[];metric_rows=[];phase_rows=[];reward_rows=[];fd_rows=[]
        def physical_metrics(pred,truth,tag,path):
            delta=difference(pred,truth).detach();qerr=delta[:,:20];qderr=delta[:,20:40]
            groups={'position_m':delta[:,40:43],'rotation_rad':delta[:,43:46],
                    'linear_velocity_m_s':delta[:,46:49],'angular_velocity_rad_s':delta[:,49:52]}
            result={'model':tag,'trajectory':path,'rows':15,
                'q_RMSE_rad':float(qerr.square().mean().sqrt()),'q_max_abs_rad':float(qerr.abs().max()),
                'qd_RMSE_rad_s':float(qderr.square().mean().sqrt()),'qd_max_abs_rad_s':float(qderr.abs().max()),
                'q_per_joint_RMSE_rad':qerr.square().mean(0).sqrt().tolist(),'q_per_joint_max_abs_rad':qerr.abs().max(0).values.tolist(),
                'qd_per_joint_RMSE_rad_s':qderr.square().mean(0).sqrt().tolist(),'qd_per_joint_max_abs_rad_s':qderr.abs().max(0).values.tolist(),
                'predicted_per_joint_min_rad':pred[:,:20].min(0).values.detach().tolist(),
                'predicted_per_joint_max_rad':pred[:,:20].max(0).values.detach().tolist(),
                'true_per_joint_min_rad':truth[:,:20].min(0).values.tolist(),'true_per_joint_max_rad':truth[:,:20].max(0).values.tolist()}
            norms={k:torch.linalg.vector_norm(v,dim=-1) for k,v in groups.items()}
            for key,value in norms.items():
                result.update({key+'_mean':float(value.mean()),key+'_RMS':float(value.square().mean().sqrt()),key+'_max':float(value.max())})
            ex=lambda x:torch.maximum(torch.maximum(lower-x[:,:20],x[:,:20]-upper),torch.zeros_like(x[:,:20]))
            px,tx=ex(pred).detach(),ex(truth)
            result.update(predicted_joint_excess_max_rad=float(px.max()),true_joint_excess_max_rad=float(tx.max()),
                predicted_joint_over1mrad_count=int((px>.001).sum()),true_joint_over1mrad_count=int((tx>.001).sum()))
            for k in range(15):
                phase_rows.append({'model':tag,'trajectory':path,'phase':k,'q_RMSE_rad':float(qerr[k].square().mean().sqrt()),
                    'qd_RMSE_rad_s':float(qderr[k].square().mean().sqrt()),**{key:float(v[k]) for key,v in norms.items()}})
            return result,px,tx
        def object_terms(x,gp,gR):
            _,_,pos,R,_,_=unpack(x);off=.1*R[:,:,0];goff=.1*gR[:,:,0]
            e=((pos+off-gp-goff).norm(dim=-1)+(pos-off-gp+goff).norm(dim=-1))*.5
            return torch.stack((2*torch.exp(-((pos-gp).norm(dim=-1)/.05).square()),8*torch.exp(-(e/.04).square())),-1)
        def reward_table(pred,d,tag,path):
            pred=pred.detach()
            for shift,label in ((0,'k_native'),(1,'kplus1_tracking')):
                gp,gR=d['goal_p'][shift:shift+15],d['goal_R'][shift:shift+15]
                estimated=object_terms(pred,gp,gR);actual=object_terms(d['x_next'],gp,gR)
                for k in range(15):
                    reward_rows.append({'model':tag,'trajectory':path,'phase':k,'reference_phase':label,
                        'pred_position_term':float(estimated[k,0].detach()),'actual_position_term':float(actual[k,0]),
                        'pred_endpoint_term':float(estimated[k,1].detach()),'actual_endpoint_term':float(actual[k,1]),
                        'unscaled_sum_error':float((estimated[k].sum()-actual[k].sum()).detach()),
                        'dt_scaled_sum_error':float((.02*(estimated[k].sum()-actual[k].sum())).detach())})
        tick=time.perf_counter();stage('encode_all_saved_data')
        for path in p['inputs']['trajectories']:
            name=path['name'];spec=plan['datasets'][name]
            d=load_saved_transitions(bundle,path['directory'],checked(spec['audit']),name,spec['codec_input_pins'])
            palm=torch.tensor(d['palm_pose']);Rp=quat_to_mat(palm[3:]);inv=Rp.T
            poses=torch.tensor(bundle['arrays']['reference']['mapped_object_pose_w'],dtype=torch.float32)
            d['goal_p']=mv(inv[None],poses[:,:3]-palm[:3]);d['goal_R']=mm(inv[None],quat_to_mat(poses[:,3:]));datasets[name]=d
            receipts.append(d['receipt'])
            for key in ('x','x_next','u'):arrays[name+'__'+key]=d[key].numpy().copy()
            q,qd,pos,R,v,w=unpack(d['x']);cv=pack(q+.02*qd,qd,pos+.02*v,mm(so3_exp(.02*w),R),v,w)
            for tag,pred in (('identity',d['x']),('constant_velocity',cv)):
                result,px,tx=physical_metrics(pred,d['x_next'],tag,name);metric_rows.append(result)
                arrays[name+'__'+tag]=pred.numpy().copy();reward_table(pred,d,tag,name)
        phases['codec_and_baselines_seconds']=time.perf_counter()-tick
        write(out/'codec_receipts.json',receipts)
        directions=np.random.default_rng(p['internal_gradient_check']['seed'])
        for spec in p['inputs']['models']:
            method=spec['method'];stage('load_'+method);tick=time.perf_counter()
            wm,receipt=load_world_model_checkpoint(checked(spec['checkpoint']),bundle,method,Path(spec['checkpoint']['path']).parents[1],checked(spec['final_health']))
            phases[method+'_load_seconds']=time.perf_counter()-tick;write(out/('loader_'+method+'.json'),receipt)
            before=_tensor_digest(wm.state_dict());tick=time.perf_counter()
            for name in p['scope']['trajectory_order']:
                stage(method+'__'+name);d=datasets[name];x=d['x'].clone();u=d['u'].clone().requires_grad_(True)
                if not bool(((u>=torch.from_numpy(ulo)-plan['input_arithmetic_atol'])&(u<=torch.from_numpy(uhi)+plan['input_arithmetic_atol'])).all()):raise ValueError('Actual target outside registered inset domain')
                pred,contact=wm(x,u);objective=object_terms(pred,d['goal_p'][:15],d['goal_R'][:15]).sum(-1)
                grad=torch.autograd.grad(objective.sum(),u)[0]
                if not bool(torch.isfinite(grad).all()):raise FloatingPointError('Nonfinite model object-only gradient')
                result,px,tx=physical_metrics(pred,d['x_next'],method,name);metric_rows.append(result);reward_table(pred,d,method,name)
                result.update(status='PRIMARY_COMPLETE_PENDING_FD',gradient_finite=True,object_gradient_norm_mean=float(grad.norm(dim=-1).mean()),
                    selfcollision_BCE=None,selfcollision_F1=None,selfcollision_accuracy=None,selfcollision_truth='UNKNOWN_not_saved_found_sensor')
                summary['groups'].append(result)
                prefix=method+'__'+name
                arrays.update({prefix+'__prediction':pred.detach().numpy().copy(),prefix+'__selfcollision_logits':contact.detach().numpy().copy(),
                    prefix+'__selfcollision_probability':contact.detach().sigmoid().numpy().copy(),prefix+'__object_gradient':grad.numpy().copy(),
                    prefix+'__predicted_over1mrad':(px>.001).numpy(),prefix+'__true_over1mrad':(tx>.001).numpy()})
                for k in p['internal_gradient_check']['finite_difference_phases']:
                    dirs=directions.standard_normal((4,20));dirs/=np.linalg.norm(dirs,axis=-1,keepdims=True)
                    for j,vector in enumerate(dirs):
                        slopes=[]
                        for eps in p['internal_gradient_check']['epsilon_rad_L2']:
                            center=d['u'][k].numpy();plus=(center+eps*vector).astype(np.float32);minus=(center-eps*vector).astype(np.float32)
                            crossing=(plus<ulo)|(plus>uhi)|(minus<ulo)|(minus>uhi)
                            row={'model':method,'trajectory':name,'phase':k,'direction':j,'epsilon':eps,'vector':vector.tolist(),
                                'plus_target':plus.tolist(),'minus_target':minus.tolist(),'crossing_mask':crossing.tolist(),
                                'valid_domain':not bool(crossing.any()),'AD':float(np.dot(grad[k].numpy(),vector)),'FD':None,
                                'absolute_discrepancy':None,'relative_discrepancy':None,'AD_effective_FP32_direction':None}
                            if not crossing.any():
                                targets=torch.from_numpy(np.stack((plus,minus)));states=x[k:k+1].expand(2,-1)
                                with torch.no_grad():
                                    y,_=wm(states,targets);values=object_terms(y,d['goal_p'][k:k+1].expand(2,-1),d['goal_R'][k:k+1].expand(2,-1,-1)).sum(-1)
                                fd=float((values[0].double()-values[1].double())/(2*eps));ad=row['AD']
                                row.update(FD=fd,absolute_discrepancy=abs(ad-fd),relative_discrepancy=abs(ad-fd)/max(abs(ad),abs(fd),1e-12),
                                    AD_effective_FP32_direction=float(np.dot(grad[k].numpy().astype(float),(plus.astype(float)-minus.astype(float))/(2*eps))))
                                slopes.append(fd)
                            else:slopes.append(None)
                            fd_rows.append(row)
                        disagreement=None if any(v is None for v in slopes) else abs(slopes[0]-slopes[1])/max(abs(slopes[0]),abs(slopes[1]),1e-12)
                        fd_rows[-1]['two_epsilon_relative_disagreement']=disagreement;fd_rows[-2]['two_epsilon_relative_disagreement']=disagreement
                if not torch.equal(u.detach(),d['u']) or not torch.equal(x,d['x']):raise ValueError('Diagnostic mutated input tensors')
                result['status']='COMPLETED'
            phases[method+'_prediction_and_FD_seconds']=time.perf_counter()-tick
            after=_tensor_digest(wm.state_dict())
            if before!=after:raise ValueError('WM parameters or normalizers changed')
            summary['models'].append({'method':method,'loader_receipt':pin(out/('loader_'+method+'.json')),
                'before_state_sha256':before,'after_state_sha256':after,'unchanged':True})
        summary.update(status='COMPLETED_CPU_DIAGNOSTIC_NOT_PHYSICAL_CERTIFICATE',one_step_prediction_rows=len(summary['groups'])*15,
            FD_pairs=len(fd_rows),FD_domain_skipped=sum(not r['valid_domain'] for r in fd_rows),
            RNG_unchanged=torch.equal(cpu_rng,torch.get_rng_state()) and np.array_equal(np_rng[1],np.random.get_state()[1]) and np_rng[2:]==np.random.get_state()[2:] and py_rng==random.getstate(),
            no_Warp_or_CUDA=not torch.cuda.is_initialized() and 'warp' not in sys.modules and 'mujoco_warp' not in sys.modules)
        if not summary['RNG_unchanged'] or not summary['no_Warp_or_CUDA']:raise ValueError('RNG or CPU boundary violated')
        for value in all_pins(plan):checked(value)
        for value in all_pins(p):checked(value)
        np.savez_compressed(out/'predictions_truth_baselines_gradients.npz',**arrays)
        write(out/'metrics.json',metric_rows);write(out/'reward_rows.json',reward_rows);write(out/'FD_rows.json',fd_rows)
        for filename,rows in (('phase_errors.csv',phase_rows),):
            with (out/filename).open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        summary['phase_seconds']=phases
    except BaseException as exc:
        present={(g['model'],g['trajectory']) for g in summary['groups']}
        for spec in p['inputs']['models']:
            for name in p['scope']['trajectory_order']:
                if (spec['method'],name) not in present:
                    summary['groups'].append({'model':spec['method'],'trajectory':name,'status':'NOT_RUN_OR_INVALID','reason':repr(exc)})
        if 'arrays' in locals():np.savez_compressed(out/'partial_arrays.npz',**arrays)
        if 'fd_rows' in locals():write(out/'partial_FD_rows.json',fd_rows)
        if 'reward_rows' in locals():write(out/'partial_reward_rows.json',reward_rows)
        summary.update(status='INCOMPLETE_OR_INVALID_RETAINED',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        summary.update(wall_seconds=time.perf_counter()-started,runplan=pin(a.run_plan),protocol=plan['protocol'],entry=pin(__file__))
        write(out/'report.json',summary);write(out/'runtime.json',{'status':summary['status'],'pid':os.getpid(),'elapsed_seconds':summary['wall_seconds'],'physics_steps_added':0,'GPU_used':False})
    print({'status':summary['status'],'one_step_predictions':summary['one_step_prediction_rows'],'FD_pairs':summary['FD_pairs'],'FD_skipped':summary['FD_domain_skipped'],'wall_seconds':summary['wall_seconds']})

if __name__=='__main__':main()
