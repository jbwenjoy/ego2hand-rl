"""Freeze then test pure formulas and narrow Torch hook fixtures, not real data.

No actual objective is constructed; no full05 NPZ or checkpoint is opened.
Torch is used only for explicit closed-form modules with no learned weights.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from full151_gradient import trained_bonus_analysis105_v1 as a


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output).resolve()
    a.require(not out.exists(),'Fresh proof output required')
    source_pins={k:a.pin(p) for k,p in a.OWN.items()}
    out.mkdir(parents=True);snapshot=out/'executed_sources';snapshot.mkdir()
    for k,v in source_pins.items():(snapshot/(k+'_'+Path(v['path']).name)).write_bytes(Path(v['path']).read_bytes())
    a._write(out,'source_before.json',source_pins)
    checks={}
    def check(name,value):
        checks[name]=bool(value)
        if not checks[name]:raise AssertionError(name)
    def rejects(name,fn,types=(ValueError,KeyError)):
        try:fn()
        except types:check(name,True)
        else:check(name,False)
    try:
        for v in source_pins.values():ast.parse(Path(v['path']).read_text(encoding='utf-8-sig'))
        check('frozen_sources_AST_valid',True)
        check('noTorch_Warp_before_closed_form_fixtures','torch' not in sys.modules and 'warp' not in sys.modules)
        declaration=a.declaration()
        check('declaration_no_model_data_or_backend',declaration['NPZ_opened']==declaration['checkpoint_loads']==declaration['objective_constructions']==0 and declaration['Torch_imported'] is False)
        header={'schema':a.SCHEMA,'role':a.ROLE,'protocol':{'path':'unused','sha256':'unused'},'output':str(ROOT/'outputs/full151105_trained_bonus_CPU_synthetic_NEVER_EXECUTED'),
                'CPU_run_authorized':False,'cpu_threads':1,'source_pins':{'synthetic':{}},'cpu_ready':{}}
        a.validate_header(header);check('positive_exact_pure_header_only',True)
        for key,bad in [('schema','old104'),('role','GPU'),('CPU_run_authorized',1),('cpu_threads',True),('cpu_threads',2),('output',str(ROOT/'source'))]:
            rejects('negative_header_'+key+'_'+str(bad),lambda key=key,bad=bad:a.validate_header({**header,key:bad}))
        rejects('unknown_header_field',lambda:a.validate_header({**header,'implicit_GPU':True}))
        rejects('missing_header_field',lambda:a.validate_header({k:v for k,v in header.items() if k!='cpu_ready'}))
        rejects('authorization_false_rejected_before_science',lambda:a.authorization_gate(header))
        rejects('strict_pin_unknown_keys',lambda:a.checked({**source_pins['consumer'],'extra':False}))
        rejects('strict_pin_relative',lambda:a.checked({'path':'relative.py','sha256':'0'*64}))
        rejects('strict_pin_changed_SHA',lambda:a.checked({**source_pins['consumer'],'sha256':'0'*64}))
        check('alias_samefile_SHA_semantics',a.same_pin(source_pins['consumer'],{'path':str(Path(source_pins['consumer']['path'])).replace('\\','/'),'sha256':source_pins['consumer']['sha256']}))
        ids,ph=a.physical_layout()
        check('layout160_donor8_branch152_all_phases',len(ids)==160 and ids[:8].tolist()==list(range(8)) and ph[:8].tolist()==list(a.PHASES)*2 and (ph==149).sum()==40)
        check('pair_slots_include_last_no_u150',a.pair_slots(7,3,1)==(158,159))
        rejects('pair_index_boolean_rejected',lambda:a.pair_slots(True,0,0))
        zeros={k:0. for k in a.CHANNELS};b0={k:np.zeros(3) for k in a.CHANNELS}
        nf=a.component_floors(zeros,b0)
        check('raw_P_floor_1e4_and_weighted_unit_scaling',nf['floors']['raw_P']==1e-4 and nf['floors']['BP']==.1e-4)
        check('S_floor_and_total_cancellation_sum',nf['floors']['raw_S']==nf['floors']['CS']==1e-5 and nf['floors']['A2']==nf['floors']['BP']+nf['floors']['CS'])
        check('dt_exact_once_scales_existing_floor',nf['dt_weighted_floors']['A2']==.02*nf['floors']['A2'])
        large={**zeros,'raw_P':1e8,'BP':-1e7,'A2':-1e7}
        nb={k:np.full(3,v) for k,v in large.items()};nl=a.component_floors(large,nb)
        check('unbounded_P_scale_aware_guard_frozen_formula',nl['floors']['raw_P']==8*2**-23*1e8 and nl['floors']['BP']==.1*nl['floors']['raw_P'])
        noisy={**b0,'raw_P':np.array([.2,.3,.4]),'BP':np.array([-.02,-.03,-.04])};nn=a.component_floors(zeros,noisy)
        check('noise_uses_max_span_or_donor_error_times10',nn['floors']['raw_P']==10*.4)
        inconsistent={**b0,'BP':np.array([0.,1.,0.])}
        check('BP_unit_inconsistency_not_hidden_by_floor_tuning',a.component_floors(zeros,inconsistent)['BP_representation_consistent'] is False)
        rejects('incomplete_component_set_rejected',lambda:a.component_floors({'raw_P':0},b0))
        rejects('NaN_baseline_rejected',lambda:a.baseline_noise(0.,[0.,np.nan,0.]))
        cl=a.classify_pair([0,0],[.004,.002],1e-5)
        check('closed_linear_slope_both_eps',cl['slopes']==[1.,1.] and cl['stable_nonflat'])
        check('equal_floor_is_below_noise',a.classify_pair([0,0],[1e-5,1e-5],1e-5)['label']=='FLAT_OR_BELOW_NOISE')
        check('sign_flip_is_unstable',a.classify_pair([0,0],[.004,-.002],1e-5)['label']=='NONFLAT_UNSTABLE')
        check('epsilon_ratio_failure_is_unstable',a.classify_pair([0,0],[.004,.0001],1e-5)['label']=='NONFLAT_UNSTABLE')
        huge=a.classify_pair(np.array([-3e38,-1.5e38],np.float32),np.array([3e38,1.5e38],np.float32),1e-5)
        check('promote_FP32_before_difference_no_overflow',np.isfinite(huge['slopes']).all())
        rejects('NaN_score_rejected',lambda:a.classify_pair([0,0],[1,np.nan],1e-5))
        pr=a.compare_projection(0,0,1e-5,.002)
        check('flat_denominator_positive_ratio_null',pr['denominator']==1e-5/.004 and pr['AD_over_physical'] is None and pr['sign_agreement'] is None)
        check('zero_AD_not_sign_agreement_with_nonzero_physical',a.compare_projection(1,0,1e-5,.002)['sign_agreement'] is False)
        # All160 synthetic score values, no actual file reads or geometry.
        scores={k:np.zeros(160,np.float32) for k in a.CHANNELS};grads={k:np.ones((8,20),np.float32) for k in a.CHANNELS}
        directions=np.zeros((8,4,2,20),np.float64);offset=np.zeros_like(directions);clip=np.zeros((8,4,2),bool)
        for s in range(8):
            for j,joint in enumerate(a.JOINTS):
                for ei,eps in enumerate(a.EPS):
                    m,p=a.pair_slots(s,j,ei);directions[s,j,ei,joint]=1
                    for channel in a.CHANNELS:scores[channel][m]=-eps;scores[channel][p]=eps
        clip[0,0,0]=True
        rr,cc,noise=a.comparisons(scores,grads,directions,offset,clip)
        check('all64_pairs_per_each5_channels_retained',len(rr)==320 and len(cc)==160 and all(sum(r['channel']==k for r in rr)==64 for k in a.CHANNELS))
        check('clipped_is_orthogonal_not_filtered',sum(r['clipped'] for r in rr)==5 and all(r['stable_nonflat'] for r in rr))
        check('nominal_vs_effective_preserved',all(r['WM_AD_effective']==r['WM_AD_nominal_secondary']==1 for r in rr))
        summary=a.summaries(rr)
        check('all32_vs_unclipped_secondary_counts',all(r['rows']==32 for r in summary if r['subset']=='all32') and any(r['rows']==31 for r in summary if r['subset']=='stable_unclipped_secondary'))
        # Actual default subprocess: no config/data/model accesses.
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1')
        cmd=[sys.executable,'-B',str(a.OWN['CLI'])]
        process=subprocess.run(cmd,capture_output=True,text=True,timeout=20,env=env)
        (out/'default_stdout.json').write_text(process.stdout,encoding='utf-8');(out/'default_stderr.txt').write_text(process.stderr,encoding='utf-8')
        a.require(process.returncode==0,'Default metadata CLI failed: '+process.stderr)
        actual=json.loads(process.stdout)
        check('actual_defaultCLI_noTorch_Warp_NPZ_models',actual['Torch_imported'] is False and actual['Warp_imported'] is False and actual['NPZ_opened']==actual['checkpoint_loads']==0)
        # Now import Torch ONLY for explicit algebra/forward-hook fixtures.
        import torch
        torch.set_num_threads(1)
        check('Torch_fixture_CPU_uninitialized',not torch.cuda.is_initialized())
        def rng():return hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state()),protocol=5)+torch.get_rng_state().numpy().tobytes()).hexdigest()
        before_rng=rng()
        class AlgebraProxy(torch.nn.Module):
            def __init__(self):super().__init__();self.calls=0
            def forward(self,x):
                self.calls+=1;b=len(x);p=3
                distance=(-.003-.001*x[:,0:1]).expand(b,p)
                weights=torch.exp(-.5*(distance.clamp_min(0)/.001).square())
                return {'support':x[:,1:2].expand(b,64),'presence':weights[:,:1].expand(b,6),
                    'minimum_signed_distance_m':distance.amin(-1),'sampled_signed_distance_m':distance,
                    'soft_contact_weight':weights,'projected_contact_com_o_m':x[:,None,:3].expand(b,p,3),
                    'inward_normal_o':torch.ones((b,p,3),dtype=x.dtype)}
        class AlgebraA2(torch.nn.Module):
            def __init__(self):super().__init__();self.geometry_proxy=AlgebraProxy()
            def forward(self,x,m):
                g=self.geometry_proxy(x);P=((-g['sampled_signed_distance_m']-.003).clamp_min(0).amax(-1)/.001).square()
                S=g['support'].mean(-1);w=torch.ones_like(P)
                return w*S-.1*P,{'geometry_intent_penetration_prior':P,'geometry_intent_support_match':S,'geometry_intent_phase_weight':w}
        class AlgebraPenalty(torch.nn.Module):
            def __init__(self):super().__init__();self.base=AlgebraA2()
            def forward(self,x,m):
                _,t=self.base(x,m);return -.1*t['geometry_intent_penetration_prior'],t
        # No real WorldModel, weights or MJB; x1[:20]=x[:20]+2u exactly.
        x=torch.ones((2,58),dtype=torch.float32);u=torch.full((2,20),.25,dtype=torch.float32,requires_grad=True)
        pred=torch.cat((x[:,:20]+2*u,x[:,20:]),-1)
        po,ao=AlgebraPenalty(),AlgebraA2()
        hooks_before=len(ao.geometry_proxy._forward_hooks)
        ps,pt,pg=a.evaluate_objective(po,pred,np.array([0,149],np.int64))
        az,at,ag=a.evaluate_objective(ao,pred,np.array([0,149],np.int64))
        ch=a.channels(ps,pt,az,at)
        check('hook_one_existing_forward_per_objective',po.base.geometry_proxy.calls==ao.geometry_proxy.calls==1)
        check('hook_removed_after_success',len(ao.geometry_proxy._forward_hooks)==hooks_before)
        check('hook_saved_output_detached_copy',set(ag)==a.CAPTURE_KEYS and ag['support'].shape==(2,64) and np.array_equal(ag['sampled_signed_distance_m'],pg['sampled_signed_distance_m']))
        gs={}
        for i,k in enumerate(('BP','A2','CS','raw_P')):gs[k],=torch.autograd.grad(ch[k].sum(),u,retain_graph=i<3)
        expectedP=torch.zeros((2,20));expectedP[:,0]=4*pred[:,0].detach()
        expectedS=torch.zeros((2,20));expectedS[:,1]=2
        check('closedform_rawP_u_AD',torch.allclose(gs['raw_P'],expectedP,atol=2e-6,rtol=1e-6))
        check('closedform_support_u_AD',torch.equal(gs['CS'],expectedS))
        check('closedform_actualP_AND_A2_AD_components',torch.allclose(gs['BP'],-.1*expectedP,atol=1e-6) and torch.allclose(gs['A2'],expectedS-.1*expectedP,atol=1e-6))
        check('four_VJP_no_extra_forward',po.base.geometry_proxy.calls==ao.geometry_proxy.calls==1)
        check('phase149_memory_is149_not_terminal150',a.memory_for_phases(np.array([0,149]))['time'].flatten().tolist()==[0,149])
        rejects('phase150_score_outside_preregistered_active_task',lambda:a.memory_for_phases(np.array([150])))
        rejects('phase_fractional_rejected',lambda:a.memory_for_phases(np.array([.5])))
        rejects('malformed_proxy_output',lambda:run_bad_hook(a,torch,AlgebraProxy,x))
        check('hook_removed_after_exception',len(ao.geometry_proxy._forward_hooks)==hooks_before)
        try:
            with a.ProxyCapture(ao.geometry_proxy,2):
                ao.geometry_proxy(x);raise RuntimeError('explicit fixture error')
        except RuntimeError:pass
        check('hook_cleanup_preserves_original_exception',len(ao.geometry_proxy._forward_hooks)==hooks_before)
        rejects('double_geometry_forward_rejected',lambda:run_double(a,ao.geometry_proxy,x))
        check('hook_removed_after_duplicate_rejection',len(ao.geometry_proxy._forward_hooks)==hooks_before)
        diag=a.proxy_diagnostics(ag)
        check('proxy_nearsurface_diagnostics_use_capture_only',diag['rows']==2 and ao.geometry_proxy.calls==4 and diag['basis']==64)
        check('fixtures_preserve_Python_Numpy_TorchRNG',rng()==before_rng)
        check('actual_modules_never_imported',all(k not in sys.modules for k in ('full151_intent.binding_v3','full151_native.world_model_checkpoint','mujoco','warp')))
        check('fixture_CUDA_never_initialized',not torch.cuda.is_initialized())
        for k,v in source_pins.items():check('source_preserved_'+k,a.pin(v['path'])==v)
        source_text=Path(a.__file__).read_text(encoding='utf-8')
        check('actual_path_calls_both_real_factories_not_manual_math','call(\'objective_factory\',build_penalty_only_objective' in source_text and 'call(\'objective_factory\',build_contact_objective' in source_text)
        check('actual_preflight_authorization_before_NPZ',source_text.index('authorization_gate(c)')<source_text.index('with np.load(checked(inputs'))
        report={'schema':'full151105_trained_bonus_CPU_contract_proof:1','status':'PASS_PURE_CPU_CONTRACTS_NOT_REAL_SCORE','passed':sum(checks.values()),'total':len(checks),'checks':checks,
            'source_pins':source_pins,'protocol':a.pin(a.PREREG),'scope':{'actual_objective_constructs':0,'actual_WM_loads':0,'actual_full05_NPZ_reads':0,
            'actual_objective_geometry_or_WM_forwards':0,'Torch_used_for_closedform_hook_fixture':True,'fixture_module_forward_not_native_or_learned_model':True,
            'Env':0,'physics_steps':0,'GPU_used':False,'actual_dataset_score_or_gradient_qualification':False}}
        a._write(out,'report.json',report)
        ready={'schema':'full151105_trained_bonus_CPU_contract_READY:1','status':'PASS_PURE_CPU_CONTRACTS_NOT_REAL_SCORE','source_pins':source_pins,'cpu_report':a.pin(out/'report.json'),
            'cpu_checks':{'passed':report['passed'],'total':report['total']},'protocol':a.pin(a.PREREG),
            'api':{'entry':str(a.OWN['CLI']),'module':str(a.OWN['consumer']),'default':'no args declaration; --config JSON hash/metadata only','authorized':'CUDA_VISIBLE_DEVICES=-1; --config separately authorized JSON --run','config_schema':a.SCHEMA,'config_role':a.ROLE,'config_keys':sorted(a.KEYS)},
            'CPU_scientific_run_authorized':False,'GPU_run_authorized':False,'actual_score_ready':False}
        a._write(out,'READY.json',ready)
        print(json.dumps({'passed':report['passed'],'total':report['total'],'report':a.pin(out/'report.json'),'READY':a.pin(out/'READY.json'),'source_pins':source_pins},indent=2))
    except BaseException as error:
        a._write(out,'failure.json',{'schema':'full151105_trained_bonus_CPU_contract_failure:1','error':repr(error),'checks_completed':checks,'source_pins':source_pins,'actual_science_run':False})
        raise


def run_bad_hook(a,torch,Proxy,x):
    class Bad(Proxy):
        def forward(self,x):
            v=super().forward(x);v['support']=v['support'][:,:63];return v
    obj=Bad()
    try:
        with a.ProxyCapture(obj,len(x)):obj(x)
    finally:a.require(len(obj._forward_hooks)==0,'Bad hook leaked')


def run_double(a,obj,x):
    with a.ProxyCapture(obj,len(x)):obj(x);obj(x)


if __name__=='__main__':main()
