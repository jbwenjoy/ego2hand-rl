"""One bounded CPU mathematical receipt; no candidate, optimization or physics."""
from pathlib import Path
from dataclasses import replace
import hashlib,importlib.util,json,os,sys,time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import prefix_curves as c
import prefix_control as u

PROTOCOL=Path('D:/workspace/supermimic-observation-dynamic-wrench-study/proposals/short_dynamic_tracking_v1/protocol.json')
PROTOCOL_SHA='94ea7cbe03f3e2aaa7bb4f7c8860bae8e599cd4e8e823cbe47220f73ffe101a3'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,x:Path(p).write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda x:x.item())+'\n',encoding='utf-8')

def close(a,b,tol=1e-9):
    if np.max(np.abs(np.asarray(a)-np.asarray(b)))>tol:raise AssertionError('Numeric mismatch '+str(np.max(np.abs(np.asarray(a)-np.asarray(b)))))

def rejects(call):
    try:call()
    except (ValueError,TypeError):return
    raise AssertionError('Expected rejection')

def synthetic(q=None):
    t=np.arange(16)*.02;q=np.tile((.01+.02*t)[:,None],(1,20)) if q is None else q
    pose=np.c_[t**2,t**3,.4+.01*t,np.ones(16),np.zeros((16,3))]
    v=np.c_[2*t,3*t*t,np.full(16,.01)];a=np.c_[np.full(16,2.),6*t,np.zeros(16)]
    omega=np.zeros((16,3));alpha=np.zeros((16,3))
    return c.build_prefix_curves(t,q,pose,v,a,omega,alpha)

def main():
    start=time.perf_counter();out=ROOT/'outputs/curves_control_cpu_v1';out.mkdir(parents=True,exist_ok=False)
    assert sha(PROTOCOL)==PROTOCOL_SHA
    protocol=read(PROTOCOL);sources={str(p):sha(p) for p in [Path(c.__file__),Path(u.__file__),Path(__file__),PROTOCOL]}
    for p in [Path(c.__file__),Path(u.__file__),Path(__file__)]: (out/p.name).write_bytes(p.read_bytes())
    write(out/'contract_before_tests.json',{'source_pins':sources,'q_stencils':c.Q_STENCILS,'control_grouping':u.GROUPING,'candidate_read':False,'optimization_or_physics':False})
    checks=[]
    def check(name,f):
        try:f();checks.append({'name':name,'pass':True})
        except Exception as exc:checks.append({'name':name,'pass':False,'error':repr(exc)})
    def truth(x):
        if not bool(x):raise AssertionError('Boolean check failed')
    t=np.arange(16)*.02;curve=synthetic()
    def polynomial():
        co=np.array([1.,2.,-3.,.4,.2,-.1]);h=.7
        def f(x,k):return np.polynomial.polynomial.polyval(x,np.polynomial.polynomial.polyder(co,k))
        herm=c.coefficients(np.array([f(0,0)]),np.array([f(h,0)]),np.array([f(0,1)]),np.array([f(h,1)]),np.array([f(0,2)]),np.array([f(h,2)]),h)
        for s in np.linspace(0,1,21):
            for k in range(3):close(c.evaluate(herm,s,h,k),f(s*h,k),1e-12)
    check('quintic analytic degree5 values_velocity_acceleration',polynomial)
    def stencils():
        q=np.tile((.2+.1*t+.03*t*t)[:,None],(1,20));d,a,n,n32=c.q_derivatives(q)
        close(d,np.tile((.1+.06*t)[:,None],(1,20)),1e-11);close(a,np.full_like(q,.06),5e-10)
        truth(np.array_equal(n[:-1],np.diff(q,axis=0)/.02));truth(np.array_equal(n[-1],n[-2]));truth(not np.array_equal(n,d))
        constant=np.repeat(q[:1],16,axis=0);truth(not np.any(c.q_derivatives(constant)[0]) and not np.any(c.q_derivatives(constant)[1]))
    check('declared_second_order_endpoints_and_nativeFD_separate',stencils)
    def continuity():
        for co in [curve.q_coefficients,curve.p_coefficients]:
            for i in range(14):
                for order in range(3):close(c.evaluate(co[i],1,.02,order),c.evaluate(co[i+1],0,.02,order),1e-8)
    check('quintic_q_and_p_C2_across14_boundaries',continuity)
    check('valid_whole_polynomial_range',lambda:truth(curve.q_range_check(np.full(20,-1.),np.full(20,1.))['passed']))
    def overshoot():
        qc=curve.q_coefficients.copy();qc[0,:,0]=c.coefficients(np.array([.01]),np.array([.01]),np.array([10.]),np.array([-10.]),np.array([0.]),np.array([0.]),.02)[:,0]
        bad=replace(curve,q_coefficients=c.immutable(qc));r=bad.q_range_check(np.full(20,-.01),np.full(20,.04));truth(not r['passed'] and r['max_hard_excess_rad']>0)
        truth(any(0<s<1 for s in r['all300_segments_joints'][0]['candidate_s']))
    check('root_range_catches_between_knot_overshoot',overshoot)
    def outward():
        q=np.full((16,20),.1);q[:,0]=np.linspace(0,.05,16);q[1,0]=.001;q[2,0]=.005
        r=synthetic(q).q_range_check(np.zeros(20),np.ones(20));truth(not r['passed'] and 0 in r['initial_outward_boundary_joint_indices'])
    check('near_lower_q0_outward_derivative_fail_not_clip',outward)
    check('curves_no_extrapolation',lambda:rejects(lambda:curve.sample([.3001])))
    check('wrong_qshape_reject',lambda:rejects(lambda:c.q_derivatives(np.zeros((15,20)))))
    check('nonfinite_q_reject',lambda:rejects(lambda:c.q_derivatives(np.full((16,20),np.nan))))
    def jacobian():
        for phi in [np.zeros(3),np.array([1e-7,-2e-7,3e-7]),np.array([.2,-.1,.3])]:
            v=np.array([.3,.2,-.1]);eps=1e-6
            close(c.right_jacobian_directional(phi,v),(c.right_jacobian(phi+eps*v)-c.right_jacobian(phi-eps*v))/(2*eps),2e-8)
            close(c.so3_exp(c.so3_log(c.so3_exp(phi))),c.so3_exp(phi),1e-12)
    check('SO3_exp_log_and_DJright_independent_FD',jacobian)
    check('near_pi_branch_rejected',lambda:rejects(lambda:c.so3_log(c.so3_exp(np.array([np.pi,0,0])))))
    def noncommuting():
        rotations=np.array([c.so3_exp([.03*np.sin(x),.2*x,.1*x*x]) for x in t]);pose=np.c_[t,t*t,.4+.01*t,np.ones(16),np.zeros((16,3))]
        # Direct rotation knots avoid a quaternion conversion oracle.
        rc=[];w=np.tile([.11,.2,-.03],(16,1));a=np.tile([.02,-.01,.03],(16,1))
        for i,h in enumerate(np.diff(t)):
            phi=c.so3_log(rotations[i].T@rotations[i+1]);d0=rotations[i].T@w[i];d1=np.linalg.solve(c.right_jacobian(phi),rotations[i+1].T@w[i+1]);a1=np.linalg.solve(c.right_jacobian(phi),rotations[i+1].T@a[i+1]-c.right_jacobian_directional(phi,d1)@d1)
            rc.append(c.coefficients(np.zeros(3),phi,d0,d1,rotations[i].T@a[i],a1,h))
        cr=replace(curve,rotation_knots=c.immutable(rotations),phi_coefficients=c.immutable(np.array(rc)));k=cr.sample(t)
        close(k['object_rotation_w'],rotations,1e-12);close(k['object_omega_world_rad_s'],w,1e-10);close(k['object_alpha_world_rad_s2'],a,2e-8)
        for x in [.011,.057,.123,.249]:
            eps=1e-6;z=cr.sample([x-eps,x,x+eps]);rdot=(z['object_rotation_w'][2]-z['object_rotation_w'][0])/(2*eps);wh=rdot@z['object_rotation_w'][1].T
            omega=np.array([wh[2,1]-wh[1,2],wh[0,2]-wh[2,0],wh[1,0]-wh[0,1]])/2
            close(omega,z['object_omega_world_rad_s'][1],1e-6);close((z['object_omega_world_rad_s'][2]-z['object_omega_world_rad_s'][0])/(2*eps),z['object_alpha_world_rad_s2'][1],1e-4)
    check('SO3_world_omega_alpha_all_endpoints_and_independent_FD',noncommuting)
    rawpin=protocol['inputs']['rawFD_design_arrays'];truth(sha(rawpin['path'])==rawpin['sha256'])
    with np.load(rawpin['path']) as z:raw={k:z[k][:16].copy() for k in ['time_s','object_pose_mapped_w','raw_v_world_m_s','raw_a_world_m_s2','raw_omega_world_rad_s','raw_alpha_world_rad_s2','body_rotation_w']}
    def raw_object():
        cr=c.build_prefix_curves(raw['time_s'],np.zeros((16,20)),raw['object_pose_mapped_w'],raw['raw_v_world_m_s'],raw['raw_a_world_m_s2'],raw['raw_omega_world_rad_s'],raw['raw_alpha_world_rad_s2']);s=cr.sample(t)
        for k,expected in [('object_rotation_w',raw['body_rotation_w']),('object_v_world_m_s',raw['raw_v_world_m_s']),('object_a_world_m_s2',raw['raw_a_world_m_s2']),('object_omega_world_rad_s',raw['raw_omega_world_rad_s']),('object_alpha_world_rad_s2',raw['raw_alpha_world_rad_s2'])]:close(s[k],expected,2e-7)
        truth(len(cr.sample(np.arange(61)*.005)['q_goal_rad'])==61)
    check('actual60b1_object_first16_full151_derivatives_preserved',raw_object)
    p=u.PDParameters(np.ones(20),np.tile([0.,-1.,-.1],(20,1)),np.full(20,-2.),np.full(20,2.),np.tile([-10.,10.],(20,1)),np.tile([-10.,10.],(20,1)))
    check('control15commands14updates_and_nonconstant',lambda:truth(all(u.export_control(curve,p,curve.q_goal_rad[0],np.full(20,.1))['gates'].values())))
    check('unsupported_gain_bias_rejected',lambda:rejects(lambda:u.PDParameters(np.ones(20),np.zeros((20,3)),np.full(20,-2.),np.full(20,2.),np.tile([-1.,1.],(20,1)),np.tile([-1.,1.],(20,1)))))
    check('nonfinite_control_reject',lambda:rejects(lambda:u.proposal_numpy(np.full((16,20),np.nan),np.zeros((16,20)),np.zeros(20),p)))
    check('wrong_phase_shape_reject',lambda:rejects(lambda:u.proposal_numpy(np.zeros((16,20)),np.zeros((15,20)),np.zeros(20),p)))
    check('constant_tracking_not_intervention',lambda:truth(not u.export_control(synthetic(np.zeros((16,20))),p,np.zeros(20),np.full(20,.1))['gates']['tracking_FP32_intervention_nonconstant']))
    def rate():
        pp=replace(p,bias=np.tile([0.,-1.,0.],(20,1)));q=np.repeat((np.arange(16)%2)[:,None],20,axis=1).astype(float)-.5
        truth(not u.export_control(synthetic(q),pp,q[0],np.full(20,.1))['gates']['actual14_updates_rate_le15'])
    check('executed_actual_rate_failure_retained',rate)
    check('reference_force_cap_failure_retained',lambda:truth(not u.export_control(curve,replace(p,actuator_force_bounds=np.tile([-.0001,.0001],(20,1))),curve.q_goal_rad[0],np.full(20,.1))['gates']['actual60_ref_PD_predictions_within_both_caps']))
    mp,cp=protocol['inputs']['model'],protocol['inputs']['actual_context'];actual,modelproof=u.load_actual_pd(mp['path'],cp['path'],model_sha256=mp['sha256'],context_sha256=cp['sha256'])
    check('actual_model_gear_gain_bias_caps_checked_without_MjData',lambda:truth(all(modelproof['model_checks'].values())))
    import torch
    rng=torch.get_rng_state().clone();tp=u.torch_parameters(actual);q=np.tile(np.linspace(-.1,.1,20,dtype=np.float32),(16,1));qd=np.tile(np.linspace(-.2,.2,20,dtype=np.float32),(16,1));q0=q[0];pu=np.linspace(-.03,.03,20,dtype=np.float32)
    def torch_parity():
        ff=u.static_prior_numpy(q0,pu,actual);tf=u.static_prior_torch(torch.from_numpy(q0),torch.from_numpy(pu),tp);truth(np.array_equal(ff.view(np.uint32),tf.numpy().view(np.uint32)))
        a=u.proposal_numpy(q,qd,ff,actual);b=u.proposal_torch(torch.from_numpy(q),torch.from_numpy(qd),tf,tp)
        truth(np.array_equal(a.view(np.uint32),b.numpy().view(np.uint32)));truth(np.array_equal(u.c2_numpy(a,actual).view(np.uint32),u.c2_torch(b,tp).numpy().view(np.uint32)))
        with torch.autocast('cpu',dtype=torch.bfloat16):out=u.c2_torch(u.proposal_torch(torch.from_numpy(q),torch.from_numpy(qd),tf,tp),tp)
        truth(out.dtype==torch.float32 and np.array_equal(out.numpy().view(np.uint32),u.c2_numpy(a,actual).view(np.uint32)))
        qq=torch.from_numpy(q.copy()).requires_grad_();u.proposal_torch(qq,torch.from_numpy(qd),tf,tp).sum().backward();truth(torch.isfinite(qq.grad).all() and qq.grad.abs().max()>0)
    check('NumPy_Torch_actualFP32_grouping_C2_outerAMP_gradient',torch_parity)
    purepin=protocol['inputs']['pure_C2'];spec=importlib.util.spec_from_file_location('original_pinned_C2',purepin['path']);pure=importlib.util.module_from_spec(spec);spec.loader.exec_module(pure)
    check('original_pinned_C2_Numpy_exact',lambda:truth(np.array_equal(u.c2_numpy(q,actual),pure.target_inset(torch.from_numpy(q),torch.tensor(actual.target_lower.copy()),torch.tensor(actual.target_upper.copy()),.020).numpy())))
    check('Torch_nonfinite_rejected',lambda:rejects(lambda:u.c2_torch(torch.full((20,),float('nan')),tp)))
    check('RNG_and_CPU_only',lambda:truth(torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized() and not torch.cuda.is_available() and torch.cuda.device_count()==0 and 'warp' not in sys.modules))
    check('sources_unchanged',lambda:truth(all(sha(p)==h for p,h in sources.items())))
    report={'status':'PASS' if all(x['pass'] for x in checks) else 'FAIL_RETAINED_NO_SOURCE_OVERWRITE','checks':checks,'count':len(checks),'source_pins':sources,'protocol':{'path':str(PROTOCOL),'sha256':PROTOCOL_SHA},'rawFD':rawpin,'q_stencils':c.Q_STENCILS,'control_grouping':u.GROUPING,'actual_model_contract':modelproof,'candidate_read_or_optimized':False,'physics_steps':0,'GPU_used':False,'Torch_CUDA_initialized':torch.cuda.is_initialized(),'visible_CUDA_count':torch.cuda.device_count(),'WM_gradient_or_Warp_parity_qualified':False,'wall_s_excluding_imports':time.perf_counter()-start}
    write(out/'report.json',report);print(json.dumps({'status':report['status'],'count':len(checks),'failures':[x for x in checks if not x['pass']],'report_SHA256':sha(out/'report.json')},indent=2));raise SystemExit(0 if report['status']=='PASS' else 1)

if __name__=='__main__':main()
