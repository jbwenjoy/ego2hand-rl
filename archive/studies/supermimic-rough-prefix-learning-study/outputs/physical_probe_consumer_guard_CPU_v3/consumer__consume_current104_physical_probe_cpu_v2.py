"""Separate CPU-only consumption of a completed physical probe; default metadata only.

No Env, Warp, physics or new directions. A failed/partial physical attempt is
retained as NOT_RUN, before importing Torch or loading any model.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,csv,hashlib,json,random,sys,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rough_prefix.io import read,pin,checked,loadnp,jsonable
from rough_prefix.physical_probe_consumer_math import (
    PATHS,PHASES,EPS,ARRAYS,need,slots,required_slot,validate_arrays,
    baseline_gate,physical_secants,descriptive_agreement,self15_calibration,
    object_reward,native_rewards,check_terms)

from rough_prefix.physical_probe_manifest_guard_v2 import qualify_saved_evidence

PROPOSAL_PATH=ROOT/'proposals/current104_samewarp_physical_fd_v2/protocol.json'
PROPOSAL_SHA='ce9a9575b6aa7dc9b5b535bfe1cf7ff5be4db6065dc122ba4860561d40096e09'
LOADER_SHA='ab15f23b569d259cbca587e2b6f883c509ae15584567fcc8c27443787f294a20'
MANIFEST_KEYS={'schema','proposal','physical_report','physical_arrays','branch_index',
               'source_pins','capacity_ledgers','lifecycle_evidence','postterminal_identity_qualified'}
PLAN_KEYS={'schema','proposal','physical_manifest','output','source_pins',
           'CPU_run_authorized','soft_worker_seconds','automatic_retry'}


def all_pins(value):
    if isinstance(value,dict):
        if set(value)=={'path','sha256'}:yield value
        else:
            for v in value.values():yield from all_pins(v)
    elif isinstance(value,list):
        for v in value:yield from all_pins(v)


def write(path,value):
    Path(path).write_text(json.dumps(jsonable(value),indent=2,allow_nan=False)+'\n',encoding='utf-8')


def preflight(path):
    plan=read(path)
    need(set(plan)==PLAN_KEYS and plan['schema']=='current104_physical_probe_CPU_consumer_plan:1','Wrong consumer runplan schema/unknown field')
    need(type(plan['CPU_run_authorized']) is bool and plan['automatic_retry'] is False,'Explicit CPU authorization/no retry required')
    need(type(plan['soft_worker_seconds']) is int and plan['soft_worker_seconds']==180,'Fixed separate CPU180 budget required')
    need(plan['proposal']==pin(PROPOSAL_PATH) and plan['proposal']['sha256']==PROPOSAL_SHA,'Wrong frozen proposal2')
    proposal=read(checked(plan['proposal']))
    need(proposal['schema']=='current104_same_Warp_physical_FD_proposal:2','Wrong proposal schema')
    need(proposal['selection']['trajectory_order']==list(PATHS) and proposal['selection']['phase_k']==list(PHASES),'Changed path/phase order')
    need(proposal['counts_maximum']['all_branches']==560 and proposal['counts_maximum']['snapshots']==16,'Changed physical slot count')
    for item in all_pins(plan):checked(item)
    for item in all_pins(proposal):checked(item)
    sources=plan['source_pins']
    need(type(sources) is dict and set(sources)=={'consumer','math','WM_loader','Actor_loader','bundle_loader','io','qualification_guard'},'Exact consumer source closure required')
    for key,file in [('consumer',Path(__file__)),('math',ROOT/'src/rough_prefix/physical_probe_consumer_math.py'),
                     ('WM_loader',ROOT/'src/rough_prefix/world_model_checkpoint_v2.py'),
                     ('Actor_loader',ROOT/'src/rough_prefix/checkpoint_v2.py'),
                     ('bundle_loader',ROOT/'src/rough_prefix/bundle_v2.py'),('io',ROOT/'src/rough_prefix/io.py'),
                     ('qualification_guard',ROOT/'src/rough_prefix/physical_probe_manifest_guard_v2.py')]:
        need(sources[key]==pin(file),'Consumer dependency source differs: '+key)
    need(sources['WM_loader']['sha256']==LOADER_SHA,'Strict CPU WM loader changed')
    manifest=read(checked(plan['physical_manifest']))
    need(set(manifest)==MANIFEST_KEYS and manifest['schema']=='current104_physical_probe_result:1','Unknown/missing finalized physical manifest fields')
    need(manifest['proposal']==plan['proposal'],'Physical attempt used another proposal')
    need(type(manifest['postterminal_identity_qualified']) is bool,'Unknown physical postterminal identity')
    for item in all_pins(manifest):checked(item)
    need(type(manifest['source_pins']) is dict and bool(manifest['source_pins']),'Missing physical source closure')
    report=read(checked(manifest['physical_report']))
    need(report.get('status') in ('COMPLETED_PHYSICAL_PROBE','FAILED_ENGINEERING_PARTIAL'),'Unknown physical terminal status')
    for key in ('engineering_qualified','baseline_all_passed','capacity_finalized','capacity_qualified'):
        need(type(report.get(key)) is bool,'Unknown physical qualification '+key)
    need(type(report.get('source_lineage_N')) is int and report['source_lineage_N']==1 and type(report.get('independent_ego_N')) is int and report['independent_ego_N']==0,'Wrong source independence declaration')
    output=Path(plan['output']).resolve()
    need(output.is_relative_to((ROOT/'outputs').resolve()),'Owned fresh output required')
    need(not any(k in sys.modules for k in ('torch','warp','mujoco_warp')),'Framework imported during metadata preflight')
    ready=(report['status']=='COMPLETED_PHYSICAL_PROBE' and manifest['postterminal_identity_qualified'] and
           all(report[k] for k in ('engineering_qualified','baseline_all_passed','capacity_finalized','capacity_qualified')))
    semantic=qualify_saved_evidence(manifest)
    ready=bool(ready and semantic['qualified'])
    return plan,proposal,manifest,report,ready,semantic


def torch_reward(x,p,R):
    import torch
    pos=x[:,40:43];rot=x[:,43:52].reshape(-1,3,3)
    off=.1*rot[:,:,0];goff=.1*R[:,:,0]
    distance=.5*((pos+off-p-goff).norm(dim=-1)+(pos-off-p+goff).norm(dim=-1))
    return torch.stack((2*torch.exp(-((pos-p).norm(dim=-1)/.05).square()),8*torch.exp(-(distance/.04).square())),-1).sum(-1)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-plan',type=Path,required=True);ap.add_argument('--run',action='store_true')
    args=ap.parse_args();plan,p,manifest,physical_report,ready,semantic=preflight(args.run_plan)
    if not args.run:
        print(json.dumps({'status':'CPU_METADATA_PREFLIGHT_ONLY','physical_gates_reported_and_semantically_pass':ready,'saved_evidence':semantic,
                          'Torch_imported':False,'WM_forwards':0,'physics_steps':0,'GPU_used':False}));return
    need(plan['CPU_run_authorized'] is True,'Separate explicit CPU authorization absent')
    out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();deadline=started+plan['soft_worker_seconds']
    summary={'schema':'current104_physical_probe_CPU_consumption:2','status':'NOT_RUN_PHYSICAL_UNQUALIFIED',
             'entry':pin(__file__),'runplan':pin(args.run_plan),'proposal':plan['proposal'],'physical_manifest':plan['physical_manifest'],
             'saved_evidence_semantic_qualification':semantic,'models':[],'source_lineage_N':1,'independent_ego_N':0,'physics_steps_added':0,'GPU_used':False,
             'model_statuses':{method:'NOT_RUN' for method in ('A1L','P_ONLY','A2')},
             'WM_forwards':0,'physical_gradient_qualified':False,
             'scope':'Single seen source/seed0,16dependent snapshots/eight dependent directions each. Local one-step interior-coordinate sampled causal comparison; no CI/p-values, full-Jacobian/contact-force/hardware/generalization/method-superiority qualification.'}
    def stage(name):
        if time.perf_counter()>deadline:raise TimeoutError('CPU180 consumer budget exceeded, no retry')
        write(out/'runtime.json',{'phase':name,'pid':os.getpid(),'elapsed_seconds':time.perf_counter()-started,'GPU_used':False,'physics_steps':0})
    write(out/'runplan.json',plan);write(out/'physical_manifest.json',manifest)
    for key,item in plan['source_pins'].items():
        file=checked(item);(out/(key+'__'+file.name)).write_bytes(file.read_bytes())
    error=None
    try:
        write(out/'saved_evidence_semantic_qualification.json',semantic)
        if not ready:
            summary['reason']='Physical terminal/engineering/baseline/capacity/postterminal identity prerequisite not qualified'
            return 2
        # Everything below before import Torch is saved NumPy/JSON arithmetic.
        stage('independent_saved_primal_gates')
        a=loadnp(checked(manifest['physical_arrays']));rows=read(checked(manifest['branch_index']))
        direction_pin=p['inputs']['direction_preparation']['targets_and_vectors'];d=loadnp(checked(direction_pin))
        expected=validate_arrays(a,rows,d)
        required=np.array([required_slot(r,d) for r in expected],bool)
        need(a['snapshot_completed'].all() and np.all(a['completed'][required]) and np.all(a['restore_qualified'][required]),'Missing required physical slot/snapshot or failed restore')
        base=baseline_gate(a);write(out/'baseline_recomputed.json',base)
        need(base['all48_completed'] and base['all16_passed'],'Independent48 baseline continuous/reward gate failed')
        secants=physical_secants(a,d,base);write(out/'physical_secants.json',secants)
        term_diagnostics=[]
        for prefix,mask,x,gp,gR in [('donor',a['snapshot_completed'],a['donor_next_x'],a['goal_p'],a['goal_R']),
                                  ('branch',a['completed'],a['branch_x_next'],a['goal_p'][np.repeat(np.arange(16),35)],a['goal_R'][np.repeat(np.arange(16),35)])]:
            for shift in ('k','kplus1'):
                if shift=='kplus1':
                    gp=a['goal_p_next'] if prefix=='donor' else a['goal_p_next'][np.repeat(np.arange(16),35)]
                    gR=a['goal_R_next'] if prefix=='donor' else a['goal_R_next'][np.repeat(np.arange(16),35)]
                diff,bound=check_terms(a[prefix+'_object_terms_'+shift][mask],x[mask],gp[mask],gR[mask])
                term_diagnostics.append({'field':prefix+'_object_terms_'+shift,'native_FP32_minus_independent_F64_terms':diff.tolist(),
                                         'fixed_arithmetic_bounds':bound.tolist(),'native_FP32_primary':True})
        write(out/'reward_arithmetic_diagnostics.json',term_diagnostics)
        summary.update(independent_baseline_all_passed=True,required_branch_slots=int(required.sum()),
                       completed_branch_slots=int(a['completed'].sum()),planned_branch_slots=560,planned_direction_slots=128,
                       completed_direction_slots=sum(r['complete'] for r in secants),
                       physical_classifications={label:sum(r['classification']==label for r in secants) for label in ('NOT_RUN','STABLE_NONFLAT','NONFLAT_UNSTABLE','FLAT_OR_BELOW_NOISE')})
        # This is the first framework/model import in the explicit CPU stage.
        import torch
        need(not torch.cuda.is_initialized(),'Fresh independent CPU process required')
        torch.set_num_threads(4)
        from rough_prefix.bundle_v2 import load_bundle
        from rough_prefix.world_model_checkpoint_v2 import load_world_model_checkpoint
        from rough_prefix.checkpoint_v2 import _tensor_digest
        rng=torch.get_rng_state().clone();np_rng=np.random.get_state();py_rng=random.getstate()
        bundle=load_bundle(checked(p['inputs']['input_bundle']))
        saved={};comparisons=[];state_metrics=[]
        for spec in p['inputs']['models']:
            method=spec['method'];stage('CPU_load_'+method)
            summary['model_statuses'][method]='RUNNING'
            wm,receipt=load_world_model_checkpoint(checked(spec['checkpoint']),bundle,method,
                Path(spec['checkpoint']['path']).parents[1],checked(spec['final_health']))
            write(out/('loader_'+method+'.json'),receipt)
            before=_tensor_digest(wm.state_dict());x=torch.from_numpy(a['snapshot_x'].copy());u=torch.from_numpy(a['snapshot_u'].copy()).requires_grad_(True)
            x_before=x.clone();u_before=u.detach().clone()
            stage('CPU_forward_gradient_'+method)
            pred,logits=wm(x,u);summary['WM_forwards']+=1
            r=torch_reward(pred,torch.from_numpy(a['goal_p']),torch.from_numpy(a['goal_R']))
            g=torch.autograd.grad(r.sum(),u,retain_graph=True)[0]
            rn=torch_reward(pred,torch.from_numpy(a['goal_p_next']),torch.from_numpy(a['goal_R_next']))
            gn=torch.autograd.grad(rn.sum(),u)[0]
            need(bool(torch.isfinite(g).all() and torch.isfinite(gn).all()),'Nonfinite WM objective gradient')
            gradients=g.detach().numpy().astype(float);secondary=gn.detach().numpy().astype(float)
            saved.update({method+'__prediction':pred.detach().numpy(),method+'__self15_logits':logits.detach().numpy(),
                          method+'__gradient_refk':g.numpy(),method+'__gradient_refkplus1':gn.numpy()})
            for sec in secants:
                s,j=sec['snapshot_id'],sec['j']
                for e,eps in enumerate(EPS):
                    minus,plus=sec['branch_minus_plus_ids'][e];done=sec['complete']
                    ad=float(gradients[s]@d['effective_vectors'][s,j,e]);nominal=float(gradients[s]@d['vectors'][s,j])
                    next_fd=float((native_rewards(a['branch_object_terms_kplus1'][plus])-native_rewards(a['branch_object_terms_kplus1'][minus]))/(2*eps)) if done else None
                    row={'model':method,'path':sec['path'],'k':sec['k'],'snapshot_id':s,'j':j,'epsilon_rad':eps,
                         'status':'COMPLETED' if done else 'NOT_RUN','classification':sec['classification'],'stable_nonflat':sec['stable_nonflat'],
                         'physical_slope':sec['slopes'][e],'AD_effective_FP32':ad,'AD_nominal_secondary':nominal,
                         'AD_effective_kplus1_secondary':float(secondary[s]@d['effective_vectors'][s,j,e]),'physical_slope_kplus1_secondary':next_fd,
                         'effective_direction_norm':float(np.linalg.norm(d['effective_vectors'][s,j,e])),
                         'midpoint_shift_L2_rad':float(np.linalg.norm((d['plus'][s,j,e].astype(float)+d['minus'][s,j,e].astype(float))*.5-d['centers'][s])),
                         'sign_match':bool(np.sign(ad)==np.sign(sec['slopes'][e])) if sec['stable_nonflat'] else None}
                    comparisons.append(row)
            calibration=self15_calibration(logits.detach().numpy(),a['donor_self15']);write(out/('self15_'+method+'.json'),calibration)
            # Explicit per-snapshot responses, without introducing independent N.
            for s in range(16):
                ids=np.arange(35*s,35*(s+1));valid=ids[a['completed'][ids]]
                for i in valid:
                    delta=a['branch_x_next'][i].astype(float)-a['donor_next_x'][s].astype(float)
                    state_metrics.append({'model':method,'snapshot_id':s,'row_id':int(i),
                        **{name:float(np.linalg.norm(delta[sl])) for name,sl in [('q_delta_L2_rad',slice(0,20)),('qd_delta_L2_rad_s',slice(20,40)),('p_delta_L2_m',slice(40,43)),('R_entry_delta_Frobenius',slice(43,52)),('v_delta_L2_m_s',slice(52,55)),('omega_delta_L2_rad_s',slice(55,58))]}})
            unchanged=_tensor_digest(wm.state_dict())==before
            need(unchanged and torch.equal(x,x_before) and torch.equal(u.detach(),u_before),'WM state/norm or input mutated')
            summary['models'].append({'method':method,'checkpoint':spec['checkpoint'],'loader_receipt':pin(out/('loader_'+method+'.json')),
                'state_before_after_sha256':before,'weights_and_norm_unchanged':unchanged,'input_unchanged':True,
                'gradient_masked_norm_by_snapshot':np.linalg.norm(gradients*d['free_mask'],axis=-1).tolist(),
                'gradient_full_norm_by_snapshot':np.linalg.norm(gradients,axis=-1).tolist(),
                'self15_calibration':pin(out/('self15_'+method+'.json'))})
            summary['model_statuses'][method]='COMPLETED'
        grouped=[]
        for method in ('A1L','P_ONLY','A2'):
            for e,eps in enumerate(EPS):
                for path in (None,*PATHS):
                    for k in (None,*PHASES):
                        subset=[r for r in comparisons if r['model']==method and r['epsilon_rad']==eps and (path is None or r['path']==path) and (k is None or r['k']==k)]
                        grouped.append({'model':method,'epsilon_rad':eps,'path':path or 'ALL_DEPENDENT_PATHS','k':k,**descriptive_agreement(subset)})
        stage('serialize_saved_comparisons')
        np.savez_compressed(out/'WM_predictions_gradients.npz',**saved)
        write(out/'comparisons.json',comparisons);write(out/'descriptive_agreement.json',grouped);write(out/'physical_response_deltas.json',state_metrics)
        with (out/'comparisons.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
        unchanged_rng=torch.equal(rng,torch.get_rng_state()) and all(np.array_equal(i,j) for i,j in zip(np_rng,np.random.get_state())) and py_rng==random.getstate()
        need(unchanged_rng and not torch.cuda.is_initialized() and not any(k in sys.modules for k in ('warp','mujoco_warp')),'CPU/RNG scope violated')
        for value in all_pins(plan):checked(value)
        for value in all_pins(manifest):checked(value)
        for value in all_pins(p):checked(value)
        summary.update(status='COMPLETED_CPU_SHARED_PHYSICAL_SECANT_COMPARISON',RNG_unchanged=True,
                       normalization_updated=False,control_input_autograd_only=True,comparison_rows=len(comparisons),
                       physical_gradient_qualified=False,local_directional_diagnostic_complete=True)
    except BaseException as exc:
        error=exc;summary.update(status='NOT_RUN_OR_PARTIAL_INVALID_RETAINED',error=repr(exc),traceback=traceback.format_exc())
    finally:
        for method,status in summary['model_statuses'].items():
            if status=='RUNNING':summary['model_statuses'][method]='INCOMPLETE_RETAINED'
        summary.update(wall_seconds=time.perf_counter()-started,all3_WM_completed=len(summary['models'])==3)
        write(out/'report.json',summary);write(out/'runtime.json',{'status':summary['status'],'pid':os.getpid(),'wall_seconds':summary['wall_seconds'],'WM_forwards':summary['WM_forwards'],'GPU_used':False,'physics_steps':0})
    if error:raise error
    return 0


if __name__=='__main__':sys.exit(main())
