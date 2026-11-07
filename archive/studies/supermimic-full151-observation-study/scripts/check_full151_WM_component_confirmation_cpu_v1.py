"""Three SYNTHETIC confirmation cases; no sealed values/real weights are read."""
from pathlib import Path
import ast,copy,hashlib,json,os,sys,time
ROOT=Path('D:/workspace/supermimic-full151-observation-study')
FILES={'consumer':ROOT/'src/full151_gradient/component_confirmation105_v1.py',
       'CLI':ROOT/'scripts/confirm_full151_WM_component_cpu_v1.py','checker':Path(__file__)}


def pin(p):return {'path':str(Path(p).resolve()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}


def main():
    out=ROOT/'outputs/full151105_WM_component_confirmation_preparation_CPU_v1';out.mkdir()
    before={k:pin(v) for k,v in FILES.items()}
    for k,v in FILES.items():(out/(k+'_source_before.py')).write_bytes(v.read_bytes())
    os.environ['CUDA_VISIBLE_DEVICES']='-1';sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,'D:/workspace/dex-supermimic-research/src')
    import numpy as np
    import torch
    from full151_gradient import component_confirmation105_v1 as e
    from dex_supermimic.rl.dex_supermimic.networks import WorldModel
    torch.set_num_threads(4);start=time.perf_counter();checks={};details={}
    def reject(fn):
        try:fn()
        except ValueError:return True
        return False
    # 1. New EPS-major rows versus j-major pair map, plus actual numeric clip.
    x=np.zeros((8,58),np.float32);x[:,43:52]=np.eye(3,dtype=np.float32).reshape(9)
    center=np.full((8,20),.1,np.float32);center[0,0]=.2;center[1,0]=.001
    a={'snapshot_x':x,'snapshot_u':center,'branch_x_next':np.repeat(x,19,axis=0),'branch_u':np.repeat(center,19,axis=0),
       'direction__centers':center.copy(),'direction__minus':np.empty((8,4,2,20),np.float32),'direction__plus':np.empty((8,4,2,20),np.float32),
       'direction__d_eff':np.empty((8,4,2,20),np.float64),'direction__center_offset':np.empty((8,4,2,20),np.float64),'direction__clipped':np.empty((8,4,2),bool),
       'completed':np.ones(152,bool),'snapshot_completed':np.ones(8,bool),'restore_qualified':np.ones(152,bool),'restore_inventory_bitexact':np.ones(152,bool)}
    rows=[];snaps=[]
    for s in range(8):
        path=('A0','A2')[s//4];k=e.PHASES[s%4];snaps.append({'snapshot_id':s,'path':path,'k':k,'baseline3_qualified':True})
        for j,q in enumerate((0,4,8,12)):
            for ei,eps in enumerate((.002,.001)):
                unit=np.eye(20,dtype=np.float32)[q]
                minus=np.clip(center[s]-np.float32(eps)*unit,-.2,.2);plus=np.clip(center[s]+np.float32(eps)*unit,-.2,.2)
                if s==1 and j==0 and ei==1:minus[q]=np.float32(-0.)
                a['direction__minus'][s,j,ei]=minus;a['direction__plus'][s,j,ei]=plus
                a['direction__d_eff'][s,j,ei]=(plus.astype('f8')-minus.astype('f8'))/(2*eps)
                a['direction__center_offset'][s,j,ei]=(plus.astype('f8')+minus.astype('f8'))*.5-center[s].astype('f8')
                a['direction__clipped'][s,j,ei]=not(np.array_equal(plus,center[s]+np.float32(eps)*unit) and np.array_equal(minus,center[s]-np.float32(eps)*unit))
        for off in range(19):
            base=off<3;ei=None if base else (off-3)//8;j=None if base else (off-3)%8//2;sign=0 if base else (-1,1)[(off-3)%2]
            bid=s*19+off
            if not base:a['branch_u'][bid]=a['direction__plus' if sign==1 else 'direction__minus'][s,j,ei]
            rows.append({'row_id':bid,'snapshot_id':s,'path':path,'k':k,'type':'baseline' if base else 'direction','baseline_repeat':off if base else None,
                         'j':j,'epsilon_index':ei,'epsilon_rad':None if base else (.002,.001)[ei],'joint_index':None if base else (0,4,8,12)[j],
                         'sign':sign,'status':'COMPLETED','restorequalified':True,'actualu':a['branch_u'][bid].tolist()})
    data,identity,pairs=e.prepare_test(a,rows,snaps,np)
    bad=copy.deepcopy(a);bad['completed']=np.empty(0,bool)
    wrong=copy.deepcopy(rows);wrong[3]['joint_index']=4
    checks['all152_EPS_to_jmajor64_clip_signedzero_and_nonvacuous_masks']=bool(len(identity)==152 and len(pairs)==64
        and all(p['pair_id']==s*8+j*2+ei and p['minus_branch_id']==s*19+3+8*ei+2*j for s in range(8) for j in range(4) for ei in range(2) for p in [pairs[s*8+j*2+ei]])
        and pairs[0]['clipped'] is True and pairs[9]['clipped'] is False and np.signbit(a['direction__minus'][1,0,1,0])
        and np.array_equal(data['x152_FP32'],np.repeat(x,19,axis=0)) and reject(lambda:e.prepare_test(bad,rows,snaps,np)) and reject(lambda:e.prepare_test(a,wrong,snaps,np)))
    # 2. The frozen strict decision has no accuracy epsilon or outcome filter.
    metrics={e.ARMS[0]:{'pointwise_loss':1.,'common_pre_pair_loss':2.},e.ARMS[1]:{'pointwise_loss':1.,'common_pre_pair_loss':1.}}
    yes=e.confirmation_decision(metrics)['local_component_confirmation']
    tie=copy.deepcopy(metrics);tie[e.ARMS[1]]['common_pre_pair_loss']=2.
    trade=copy.deepcopy(metrics);trade[e.ARMS[1]]['pointwise_loss']=1.00000001
    nan=copy.deepcopy(metrics);nan[e.ARMS[1]]['common_pre_pair_loss']=float('nan')
    checks['strict_equal_boundary_tradeoff_nonfinite_decision']=bool(yes and not e.confirmation_decision(tie)['local_component_confirmation']
        and not e.confirmation_decision(trade)['local_component_confirmation'] and reject(lambda:e.confirmation_decision(nan)))
    # 3. Real native architecture but ONLY newly generated synthetic weights;
    # no original105/component files, test values, optimizer or AD.
    rng_before=torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):base=WorldModel(278,15,hidden=(512,512,512),amp=False)
    state=copy.deepcopy(base.state_dict());norm=e.c._tensor_digest(e.c._norm_state(base),torch)
    initial={'schema':e.c.ARTIFACT_SCHEMA,'WM':state,'frozen_s52':torch.maximum(base.delta_norm.std(),base.error_floor).detach(),
             'physical_xi46':torch.zeros(46),'normalizer_host_configuration':e.c._norm_config(base),'parent_checkpoint105':pin(__file__)}
    models=[]
    for arm in e.ARMS:
        payload={'schema':e.c.ARTIFACT_SCHEMA,'arm':arm,'iteration':1000,'WM_state':copy.deepcopy(state),'parent_checkpoint105':pin(__file__),
                 'physical_xi46':initial['physical_xi46'].clone(),'frozen_s52':initial['frozen_s52'].clone(),'normalizer_host_configuration':initial['normalizer_host_configuration'],
                 'frozen_norm_sha256':norm,'initial_WM_sha256':e.c._tensor_digest(state,torch),'recipe':copy.deepcopy(e.c.RECIPE),
                 'counts':{'WM_forwards':1000,'backward_calls':1000,'optimizer_steps':1000}}
        models.append(e.make_eval_component(payload,initial,arm,torch))
    tx=torch.from_numpy(x[:3]);tu=torch.zeros((3,20));dynamics=torch.cat((tx,torch.zeros((3,46+174))),-1)
    fingerprints=[e.c._tensor_digest(m.state_dict(),torch) for m in models]
    with torch.inference_mode():ys=[m(tx,tu,dynamics) for m in models]
    wrong_payload=copy.deepcopy(payload);wrong_payload['WM_state']['net.0.weight']=wrong_payload['WM_state']['net.0.weight'].double()
    dtype_rejected=reject(lambda:e.make_eval_component(wrong_payload,initial,e.ARMS[1],torch))
    checks['synthetic_native_component_strict_eval_noAD_weights_norm_RNG']=bool(dtype_rejected
        and all(not p.requires_grad for m in models for p in m.parameters())
        and all(not m.training and m.amp is False and e.c._tensor_digest(m.state_dict(),torch)==fingerprints[j] for j,m in enumerate(models))
        and all(torch.isfinite(t).all() and not t.requires_grad for y in ys for t in y)
        and all(torch.equal(a,b) for a,b in zip(ys[0],ys[1])) and torch.equal(torch.random.get_rng_state(),rng_before))
    require_sources=all(pin(v)==before[k] for k,v in FILES.items())
    if not require_sources or torch.cuda.is_initialized() or 'mujoco' in sys.modules or 'warp' in sys.modules:raise RuntimeError('Source/backend drift')
    report={'schema':'full151105_WM_component_confirmation_synthetic_CPU_contract:1','status':'PASS_SYNTHETIC_MAPPING_CRITERION_COMPONENT_ONLY' if all(checks.values()) else 'FAIL_SYNTHETIC_CONTRACT',
            'passed':sum(checks.values()),'total':len(checks),'checks':checks,'source_pins':before,
            'scope':{'synthetic_native_WM_constructors':4,'synthetic_native_WM_forwards_batch3':2,'actual_component_checkpoint_reads':0,
                     'actual_sealed_test_value_reads':0,'actual_confirmation_model_forwards':0,'AD_optimizer_Env_MJB_physics_GPU':0},'source_unchanged':True,'elapsed_seconds':time.perf_counter()-start}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    ready={'schema':'full151105_WM_component_confirmation_synthetic_READY:1','status':report['status'],'source_pins':before,
           'cpu_report':pin(out/'report.json'),'cpu_checks':{'passed':report['passed'],'total':report['total']},'actual_evaluation_authorized':False}
    (out/'READY.json').write_text(json.dumps(ready,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'READY':pin(out/'READY.json')}))
    return 0 if all(checks.values()) else 1


if __name__=='__main__':raise SystemExit(main())
