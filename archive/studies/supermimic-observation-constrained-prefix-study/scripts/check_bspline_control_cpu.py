"""Small basis/linear/Bernstein checks; NO feasibility LP or nonlinear solve."""
from pathlib import Path
import hashlib,json,os,sys,time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
from scipy import optimize
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import bspline_control as b
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,x:Path(p).write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda x:x.item())+'\n',encoding='utf-8')

def close(x,y,tol=1e-9):
    error=float(np.max(np.abs(np.asarray(x)-np.asarray(y))))
    if error>tol:raise AssertionError('numeric error '+str(error))
def truth(x):
    if not bool(x):raise AssertionError('Expected true')
def rejects(fn):
    try:fn()
    except (ValueError,TypeError):return
    raise AssertionError('Expected rejection')

def main():
    start=time.perf_counter();out=ROOT/'outputs/bspline_control_math_cpu_v1';out.mkdir(exist_ok=False)
    protocol=ROOT/'protocols/constrained-prefix-v1.json';assert sha(protocol)==b.PROTOCOL_SHA256
    sources={str(p):sha(p) for p in [Path(__file__),Path(b.__file__),protocol,b.FROZEN_CONTROL]}
    for p in [Path(__file__),Path(b.__file__)]: (out/p.name).write_bytes(p.read_bytes())
    write(out/'contract_before_checks.json',{'sources':sources,'phase0_solver_executed':False,'protocol':b.PROTOCOL_SHA256,'margins':{'control_rad':2e-6,'force_Nm':2e-6,'rate_rad_s':1e-4},'q_cert_atol_rad':1e-12})
    calls={k:0 for k in ('linprog','minimize','least_squares')};saved={k:getattr(optimize,k) for k in calls}
    def denied(k):
        def fail(*a,**kw):calls[k]+=1;raise AssertionError('Solver execution forbidden in mathematical check')
        return fail
    for k in calls:setattr(optimize,k,denied(k))
    checks=[]
    def check(name,fn):
        try:fn();checks.append({'name':name,'pass':True})
        except Exception as exc:checks.append({'name':name,'pass':False,'error':repr(exc)})
    try:
        old=b.load_frozen_fp32_control();params=old.PDParameters(np.full(20,2.,np.float32),np.tile([.01,-2.,-.02],(20,1)),np.full(20,-1.2),np.full(20,1.2),np.tile([-3.,3.],(20,1)),np.tile([-4.,4.],(20,1)))
        q0=np.linspace(-.1,.1,20);x0=np.tile(q0,17);delta=(np.arange(1,18)[:,None]**2)*np.linspace(1e-5,2e-5,20)[None,:];x=(q0+delta).ravel()
        tau=old.static_prior_numpy(q0,np.full(20,.1),params);contract=b.build_linear_contract(q0,params,tau,np.full(20,-1.),np.full(20,1.));e=b.bezier_basis();times=np.arange(61)*.005
        check('clamped_cubic22knots18controls14simpleinteriors',lambda:truth(len(b.make_basis().knots)==22 and b.NCONTROL==18 and np.array_equal(b.make_basis().knots[:4],np.zeros(4)) and np.array_equal(b.make_basis().knots[-4:],np.repeat(.3,4)) and np.array_equal(b.make_basis().knots[4:-4],b.BREAKS[1:-1])))
        def partition():
            bb=b.basis_at(np.linspace(0,.3,301));truth(bb.min()>=-1e-14);close(bb.sum(axis=1),1.,1e-14)
            close(b.basis_at(times,1).sum(axis=1),0.,1e-11);close(b.basis_at(times,2).sum(axis=1),0.,1e-9)
        check('partition_nonnegative_derivative_sums',partition)
        check('repeatq0_exact_values_zero_analytic_derivatives',lambda:truth(np.array_equal(b.evaluate_controls(x0,q0,times),np.tile(q0,(61,1))) and not np.any(b.evaluate_controls(x0,q0,times,1)) and not np.any(b.evaluate_controls(x0,q0,times,2))))
        check('hardP0_exact_and340layout',lambda:truth(np.array_equal(b.unpack_controls(x,q0)[0],q0) and np.array_equal(b.unpack_controls(x,q0)[1:].ravel(),x)))
        def fd():
            for t in [.013,.075,.124,.286]:
                eps=1e-6;close((b.evaluate_controls(x,q0,[t+eps])-b.evaluate_controls(x,q0,[t-eps]))/(2*eps),b.evaluate_controls(x,q0,[t],1),2e-8)
                close((b.evaluate_controls(x,q0,[t+eps],1)-b.evaluate_controls(x,q0,[t-eps],1))/(2*eps),b.evaluate_controls(x,q0,[t],2),2e-7)
        check('physical_seconds_derivative_independentFD',fd)
        def extraction():
            pdelta=x.reshape(17,20)-q0;qbez=q0+np.einsum('irc,cj->irj',e['q_cubic'][:,:,1:],pdelta);vbez=np.einsum('irc,cj->irj',e['qd_elevated_cubic'][:,:,1:],pdelta)
            for i in range(15):
                for s in [0.,.23,.79,1.]:
                    t=b.BREAKS[i]+s*(b.BREAKS[i+1]-b.BREAKS[i]);close(b.bernstein3(s)@qbez[i],b.evaluate_controls(x,q0,[t])[0],1e-12);close(b.bernstein3(s)@vbez[i],b.evaluate_controls(x,q0,[t],1)[0],1e-10)
        check('Bezier_q_and_degree_elevated_qd_all15intervals',extraction)
        def continuity():
            z=x.reshape(17,20)-q0;q=q0+np.einsum('irc,cj->irj',e['q_cubic'][:,:,1:],z);h=np.diff(b.BREAKS)
            for i in range(14):
                close(q[i,3],q[i+1,0],1e-12);close(3*(q[i,3]-q[i,2])/h[i],3*(q[i+1,1]-q[i+1,0])/h[i+1],1e-10)
                close(6*(q[i,3]-2*q[i,2]+q[i,1])/h[i]**2,6*(q[i+1,2]-2*q[i+1,1]+q[i+1,0])/h[i+1]**2,1e-8)
        check('C2_all14_simple_breaks',continuity)
        check('controlpoint_convexhull_and_velocity_report_only',lambda:truth(b.control_point_certificate(x,q0,np.full(20,-1.),np.full(20,1.))['passed'] and b.control_point_certificate(x,q0,np.full(20,-1.),np.full(20,1.))['velocity_hardware_limit_assumed'] is None))
        check('controlpoint_box_violation_not_repaired',lambda:truth(not b.control_point_certificate(x+2,q0,np.full(20,-1.),np.full(20,1.))['passed']))
        def commands():
            values=b.planned_values(contract,x);q=b.evaluate_controls(x,q0,b.BREAKS[:15]);v=b.evaluate_controls(x,q0,b.BREAKS[:15],1);direct=(tau.astype(float)-params.bias[:,0]-params.bias[:,1]*q-params.bias[:,2]*v)/params.kp
            close(values['commands15'],direct,1e-12)
            close(np.diff(values['commands15'],axis=0).ravel(),(contract['command_matrix'][20:]-contract['command_matrix'][:-20])@(x-contract['x0']),1e-12)
        check('linear_reference_damping_commands_and_rate_rows',commands)
        def forces():
            values=b.planned_values(contract,x)
            for i in range(15):
                for s in [0.,.37,.91,1.]:
                    t=b.BREAKS[i]+s*(b.BREAKS[i+1]-b.BREAKS[i]);q=b.evaluate_controls(x,q0,[t])[0];v=b.evaluate_controls(x,q0,[t],1)[0]
                    for kind,k in [('tracking',i),('hold',0)]:
                        direct=params.kp*values['commands15'][k]+params.bias[:,0]+params.bias[:,1]*q+params.bias[:,2]*v
                        close(b.bernstein3(s)@values[kind+'_rawforce_Bernstein'][i],direct,1e-10)
        check('ZOH_track_hold_rawPD_Bernstein_reconstruction_all15',forces)
        def matrix():
            truth(contract['A_ub'].shape==(5960,340));truth(len(contract['row_groups'])==8)
            v=np.cos(np.arange(340))/20;eps=1e-6;fd=(b.planned_values(contract,x+eps*v)['centered_constraint_residual']-b.planned_values(contract,x-eps*v)['centered_constraint_residual'])/(2*eps)
            close(fd,contract['A_ub']@v,3e-7);close(contract['A_ub']@x-contract['b_ub'],contract['A_ub']@(x-contract['x0'])-contract['b_centered'],1e-11)
        check('5960rows_centered_absolute_and_directionalFD',matrix)
        def margins():
            truth(contract['margins']['planning_rate_rad_s']==14.9999 and contract['margins']['target_domain_rad']==2e-6 and contract['margins']['raw_force_Nm']==2e-6)
            close(contract['planning_target_lower'],(params.target_lower+np.float32(.020)).astype(float)+2e-6,0.)
        check('fixed_tightening_no_originalgate_relaxation',margins)
        def epigraph():
            lp=b.phase0_l1_problem(contract);truth(lp['A_ub'].shape==(6640,680) and len(lp['bounds'])==680 and not lp['c'][:340].any() and np.all(lp['c'][340:]==1))
            dx=np.linspace(-.02,.02,340);aug=np.r_[x0+dx,np.abs(dx)];close((lp['A_ub']@aug-lp['b_ub'])[-680:].max(),0.,1e-12)
        check('phase0_L1_340epigraph_constructed_NOT_solved',epigraph)
        check('q0_outside_hardbound_reject_no_clip',lambda:rejects(lambda:b.build_linear_contract(np.full(20,-2.),params,tau,np.full(20,-1.),np.full(20,1.))))
        check('nonfinite_or_shape_reject',lambda:rejects(lambda:b.unpack_controls(np.full(340,np.nan),q0)))
        check('out_of_window_reject',lambda:rejects(lambda:b.basis_at([.3001])))
        def final():
            result=b.audit_post_C2_controls(x,q0,params,np.full(20,.1),np.full(20,-1.),np.full(20,1.))
            truth(result['proposal_postC2_identity_actual15']);truth(all(v['passed'] for v in result['postC2_real_affine_Bernstein_certificate'].values()));truth(all(result['FP32_reference61_caps_pass'].values()))
            truth(result['FP32_reference61_samples']['tracking_rawPD_affine61_FP32_Nm'].shape==(61,20));truth(not np.array_equal(result['native_forward_FD64_rad_s'],result['analytic_qd16_rad_s']))
            # Independent finalU32 real-affine check, not the old LP commands.
            z=x.reshape(17,20)-q0;q=q0+np.einsum('irc,cj->irj',e['q_cubic'][:,:,1:],z);v=np.einsum('irc,cj->irj',e['qd_elevated_cubic'][:,:,1:],z)
            for label in ['tracking','hold']:
                u=result['fp32_old_export'][label+'_u_exec_15_FP32_rad'].astype(float)
                direct=params.kp.astype(float)*u[:,None,:]+params.bias[:,0].astype(float)+params.bias[:,1].astype(float)*q+params.bias[:,2].astype(float)*v
                close(result['postC2_real_affine_Bernstein_coefficients'][label],direct,1e-12)
        check('final_postC2_U32_recert_and_separate61FP32_samples',final)
        check('old_FP32_source_pin_unchanged',lambda:truth(sha(b.FROZEN_CONTROL)==b.FROZEN_CONTROL_SHA))
        import torch
        check('no_solvers_physics_Warp_CUDA',lambda:truth(not any(calls.values()) and 'mujoco' not in sys.modules and 'warp' not in sys.modules and not torch.cuda.is_initialized() and not torch.cuda.is_available() and torch.cuda.device_count()==0))
        check('frozen_source_inputs_unchanged',lambda:truth(all(sha(p)==h for p,h in sources.items())))
    finally:
        for k,v in saved.items():setattr(optimize,k,v)
    report={'status':'PASS' if all(x['pass'] for x in checks) else 'FAIL_RETAINED_NO_SOURCE_OVERWRITE','checks':checks,'count':len(checks),'source_pins':sources,'protocol_sha256':b.PROTOCOL_SHA256,'forbidden_solver_calls':calls,'LP_or_nonlinear_solves':0,'physical_steps':0,'GPU_used':False,'scope':'Synthetic small mathematical tests only. No phase0 feasibility or candidate solve. Sufficient boxes/Bernstein restrictions, counterfactual moving-reference hold forces and analytic/nativeFD separation preserved.','wall_s_excluding_imports':time.perf_counter()-start}
    write(out/'report.json',report);print(json.dumps({'status':report['status'],'count':len(checks),'failures':[x for x in checks if not x['pass']],'report_SHA256':sha(out/'report.json')},indent=2));raise SystemExit(report['status']!='PASS')

if __name__=='__main__':main()
