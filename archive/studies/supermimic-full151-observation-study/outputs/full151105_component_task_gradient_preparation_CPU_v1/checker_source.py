"""Four synthetic task-score/projection/guard cases; no real model/test data."""
import copy,json,os,random,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_gradient import component_task_gradient105_v1 as core
from full151_gradient import component_task_gradient_independent_math_v1 as independent

OUT=ROOT/'outputs/full151105_component_task_gradient_preparation_CPU_v1'

def fixture():
    x=np.zeros((8,58),np.float32);x[:,43:52]=np.eye(3,dtype=np.float32).reshape(1,9)
    a={'snapshot_x':x,'snapshot_u':np.zeros((8,20),np.float32),'donor_next_x':x.copy(),
      'goal_p':np.zeros((8,3),np.float32),'goal_R':np.broadcast_to(np.eye(3,dtype=np.float32),(8,3,3)).copy(),
      'branch_x_next':np.repeat(x,19,axis=0),'branch_u':np.zeros((152,20),np.float32),
      'branch_object_terms_k':np.tile(np.asarray([2,8],np.float32),(152,1)),
      'branch_object_terms_k_CPU_secondary':np.tile(np.asarray([2,8],np.float32),(152,1)),
      'branch_object_score_k':np.full(152,10,np.float32),'donor_object_score_k':np.full(8,10,np.float32),
      'direction__d_eff':np.zeros((8,4,2,20),np.float64),'direction__center_offset':np.zeros((8,4,2,20),np.float64),'direction__clipped':np.zeros((8,4,2),bool)}
    pairs=[]
    for s in range(8):
        for j,q in enumerate(core.JOINTS):
            for ei,eps in enumerate(core.EPS):
                m=s*19+3+8*ei+2*j;p=m+1;a['branch_u'][m,q]=-np.float32(eps);a['branch_u'][p,q]=np.float32(eps)
                d=(a['branch_u'][p].astype(np.float64)-a['branch_u'][m].astype(np.float64))/(2*eps);a['direction__d_eff'][s,j,ei]=d
                pairs.append({'pair_id':len(pairs),'snapshot_id':s,'path':('A0','A2')[s//4],'k':core.PHASES[s%4],'j':j,'joint_index':q,'epsilon_index':ei,'requested_epsilon_rad':eps,'minus_branch_id':m,'plus_branch_id':p})
    return a,pairs

def main():
    if OUT.exists():raise ValueError('Fresh preparation only')
    OUT.mkdir();sources={**{k:core.pin(v) for k,v in core.OWN.items()},**{k:core.pin(v) for k,v in core.DEPS.items()},'checker':core.pin(__file__),'independent_math':core.pin(independent.__file__)}
    core.c._write_json(OUT/'source_freeze_before_tests.json',sources)
    for key,v in sources.items():(OUT/(key+'_source.py')).write_bytes(Path(v['path']).read_bytes())
    checks={}
    def check(name,fn):fn();checks[name]=True
    def require(v):
        if not v:raise AssertionError('Synthetic closed-form contract')
    def reject(fn):
        try:fn()
        except ValueError:return
        raise AssertionError('Invalid input must block, not repair/subset')
    def projection():
        g=np.zeros(20,np.float32);g[0]=2;center=np.zeros(20,np.float32);minus=np.zeros((2,20),np.float32);plus=minus.copy()
        for ei,e in enumerate(core.EPS):minus[ei,0]=-np.float32(e);plus[ei,0]=np.float32(e*.5)
        rm=np.asarray([.9,.95],np.float32);rp=np.asarray([1.1,1.05],np.float32)
        expected=independent.projection_conventions(g,center,minus,plus,rm,rp,0)
        rows=[]
        for ei in range(2):
            rows.append({'snapshot_id':0,'joint_index':0,'actual_delta_u20_rad_F64':expected['actual_delta_u_rad_F64'][ei].tolist(),'actual_coordinate_denominator_rad':float(expected['actual_du_coordinate_rad_F64'][ei]),'actual_coordinate_defined':True,
              'physical_actual_coordinate_secant':float(expected['physical_actual_coordinate_secant'][ei]),'physical_legacy_nominal_epsilon_slope':float(expected['physical_legacy_nominal_slope'][ei]),'legacy_d_eff_dimensionless':expected['legacy_d_eff_dimensionless'][ei].tolist()})
        result=core.projected_rows(np.tile(g,(8,1)),rows,np)
        require(all(r['WM_AD_actual_delta_projected_per_coordinate']==2 for r in result))
        np.testing.assert_array_equal([r['WM_AD_actual_delta_projected_per_coordinate'] for r in result],expected['model_actual_coordinate_projection'])
        np.testing.assert_array_equal([r['WM_AD_legacy_effective_direction'] for r in result],expected['model_legacy_effective_projection'])
        require(all(abs(r['WM_AD_legacy_effective_direction']-1.5)<1e-6 for r in result))
    check('actual_coordinate_and_legacy_clipped_chord_denominators_distinct',projection)
    def coverage_and_blockers():
        a,pairs=fixture();b,g,rows,_=core.physical_descriptors(a,[],pairs,np)
        require(len(b)==8 and all(x['passed'] is True for x in b) and len(g)==32 and len(rows)==64 and all(not x['stable_nonflat'] and x['signal_floor']==1e-5 for x in g))
        zero=copy.deepcopy(a);m=pairs[0]['minus_branch_id'];p=pairs[0]['plus_branch_id'];zero['branch_u'][p,0]=zero['branch_u'][m,0]
        reject(lambda:core.physical_descriptors(zero,[],pairs,np))
        bad=copy.deepcopy(a);bad['branch_x_next'][0,0]=.1
        reject(lambda:core.physical_descriptors(bad,[],pairs,np))
    check('all32groups_all64rows_shared_floor_zero_den_or_failed_baseline_blocks',coverage_and_blockers)
    def mapping():
        a,pairs=fixture();reject(lambda:core.physical_descriptors(a,[],pairs[:-1],np))
        wrong=copy.deepcopy(pairs);wrong[0]['plus_branch_id']+=2
        reject(lambda:core.physical_descriptors(a,[],wrong,np))
    check('partial_or_wrong_endpoint_pair_mapping_blocks_before_model',mapping)
    def score_AD():
        import torch
        from full151_gradient.serial_probe105_calibration_v1 import controlled_object_score_FP32
        torch.set_num_threads(4);require(not torch.cuda.is_initialized());rng=torch.get_rng_state().clone();py=random.getstate();nr=np.random.get_state()
        u=torch.zeros((2,20),dtype=torch.float32);u[0,0]=.01;u.requires_grad_(True)
        state=torch.cat((torch.zeros((2,40)),u[:,:3],torch.eye(3).reshape(1,9).expand(2,-1),torch.zeros((2,6))),-1)
        terms,score=controlled_object_score_FP32(state,torch.zeros((2,3)),torch.eye(3).expand(2,-1,-1));gradient,=torch.autograd.grad(score.sum(),u)
        d=u.detach().numpy()[:,0].astype(np.float64);expected=-4*d/(.05**2)*np.exp(-(d/.05)**2)-16*d/(.04**2)*np.exp(-(d/.04)**2)
        bound=8*np.finfo(np.float32).eps*np.maximum(1.,np.abs(expected));require(np.all(np.abs(gradient.detach().numpy()[:,0]-expected)<=bound))
        require(torch.isfinite(gradient).all() and torch.equal(gradient[:,3:],torch.zeros((2,17))) and score.dtype==terms.dtype==torch.float32 and torch.equal(score,terms.sum(-1,dtype=torch.float32)))
        require(torch.equal(torch.get_rng_state(),rng) and random.getstate()==py and np.array_equal(np.random.get_state()[1],nr[1]) and not torch.cuda.is_initialized())
    check('native_F32_score_input_AD_matches_closed_form_including_zero_distance',score_AD)
    for value in sources.values():core.checked(value)
    report={'schema':'full151105_component_task_gradient_CPU_contract:1','status':'PASS_SYNTHETIC_TASK_SCORE_AND_FINITE_CHORD_CONTRACT_NO_REAL_MODEL','checks':checks,'passed':len(checks),'total':len(checks),'source_pins':sources,
      'synthetic_native_score_calls':1,'synthetic_input_VJP_calls':1,'real_test_values_decoded':0,'real_component_loads_or_WM_forwards':0,'real_case_uVJP_calls':0,'MJB_FK_Env_physics_GPU_calls':0}
    core.c._write_json(OUT/'report.json',report);ready={'schema':'full151105_component_task_gradient_CPU_READY:1','source_pins':sources,'cpu_report':core.pin(OUT/'report.json'),'cpu_checks':{'passed':len(checks),'total':len(checks)},'api':{'entry':str(core.OWN['CLI']),'consumer':str(core.OWN['consumer'])},'actual_exploratory_execution_authorized':False}
    core.c._write_json(OUT/'READY.json',ready);print(json.dumps({'status':report['status'],'passed':len(checks),'total':len(checks),'READY':core.pin(OUT/'READY.json')}))

if __name__=='__main__':main()
