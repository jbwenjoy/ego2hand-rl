"""Small synthetic math/negative proof; no LP or saved-reference fitting."""
from pathlib import Path
import ast,hashlib,importlib.util,json,sys,traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'src/full151/continuous_control.py'
OLD=Path('D:/workspace/supermimic-observation-prefix-study/src/prefix_control.py')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(path,name):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m)
    return m


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    out=Path(args.output)
    if out.exists():raise FileExistsError('Proof output must be fresh')
    out.mkdir(parents=True)
    (out/'continuous_control_source_before.py').write_bytes(SOURCE.read_bytes())
    (out/'checker_source_before.py').write_bytes(Path(__file__).read_bytes())
    pins={k:{'path':str(p),'sha256':sha(p)} for k,p in [('module',SOURCE),('checker',Path(__file__)),('frozen_FP32_control',OLD)]}
    checks={};details={}
    def ok(name,value):
        checks[name]=bool(value)
        if not checks[name]:raise AssertionError(name)
    def rejects(name,fn):
        try:fn()
        except ValueError:ok(name,True)
        else:ok(name,False)
    try:
        ok('frozen_old760_source_exact',sha(OLD)=='76026dbdcdfdecf4680820b24da088e1d123e86ceb8b42a2bbd60a5412103b6d')
        b=load(SOURCE,'full151_synthetic_control')
        old=load(OLD,'full151_synthetic_frozen760')
        B=b.basis();knots=B.knots;times=np.arange(151)*.02
        ok('basis153_degree3_clamped157knots',B.ncontrol==153 and B.degree==3 and len(knots)==157 and np.count_nonzero(knots==0)==4 and np.count_nonzero(knots==3)==4)
        ok('basis_partition_unity',np.max(np.abs(B.evaluate(times).sum(axis=1)-1))<1e-14)
        ok('basis_derivative_partition_zero',np.max(np.abs(B.evaluate(times,1).sum(axis=1)))<1e-12)
        ok('basis_endpoint_exact',np.array_equal(B.evaluate([0])[0],np.eye(153)[0]) and np.array_equal(B.evaluate([3])[0],np.eye(153)[-1]))
        constant=np.repeat((np.arange(20)/64)[None,:],153,axis=0)
        ok('centered_constant_bitexact',np.array_equal(b.eval_controls(constant,times),np.repeat(constant[:1],151,axis=0)))
        ok('constant_derivatives_exact_zero',not np.any(b.eval_controls(constant,times,1)) and not np.any(b.eval_controls(constant,times,2)))
        greville=np.array([knots[i+1:i+4].mean() for i in range(153)])
        slope=(np.arange(20)+1)/2560
        line=greville[:,None]*slope
        ok('linear_reproduction_q',np.max(np.abs(b.eval_controls(line,times)-times[:,None]*slope))<1e-15)
        ok('linear_reproduction_physical_qd',np.max(np.abs(b.eval_controls(line,times,1)-slope))<1e-14)
        ok('linear_qdd_zero',np.max(np.abs(b.eval_controls(line,times,2)))<1e-11)
        P=.04*np.sin(np.arange(153)[:,None]/152*np.pi)*(1+np.arange(20)[None,:]/40)
        t=np.array([.113,.997,1.834,2.777]);h=1e-6
        fd=(b.eval_controls(P,t+h)-b.eval_controls(P,t-h))/(2*h)
        ok('basis_derivative_local_FD',np.max(np.abs(fd-b.eval_controls(P,t,1)))<1e-8)
        e=b.bezier_matrices();s=.37
        bern=np.array([(1-s)**3,3*s*(1-s)**2,3*s*s*(1-s),s**3])
        qbez=np.einsum('irc,cj->irj',e['q_cubic'],P)
        dbez=np.einsum('irc,cj->irj',e['qd_elevated_cubic'],P)
        ok('Bezier_q_exact_curve',np.max(np.abs(np.einsum('r,irj->ij',bern,qbez)-b.eval_controls(P,times[:-1]+s*.02)))<1e-14)
        ok('Bezier_qd_physical_seconds_degree_elevation',np.max(np.abs(np.einsum('r,irj->ij',bern,dbez)-b.eval_controls(P,times[:-1]+s*.02,1)))<1e-12)
        kp=(1+np.arange(20)/32).astype(np.float32)
        bias=np.c_[np.zeros(20,np.float32),-kp,-np.float32(.02)*kp]
        lo=-np.ones(20,np.float32);hi=np.ones(20,np.float32)
        caps=np.tile(np.array([-4,4],np.float32),(20,1))
        params=old.PDParameters(kp,bias,lo,hi,caps,caps)
        tau=np.zeros(20,np.float32)
        P0,P1=P[0].copy(),P[1].copy()
        v0=b.eval_controls(P,[0],1)[0]
        u0=old.c2_numpy(old.proposal_numpy(P0,v0,tau,params),params)
        target=b.eval_controls(P,times)+.002*np.cos(times[:,None])
        j=7
        built=b.build_joint_lp(j,target,P0,P1,u0,tau,params,lo,hi)
        lp,c=built['linprog_kwargs'],built['contract'];x=P[2:,j]
        ok('LP453variables2402rows',lp['c'].shape==(453,) and lp['A_ub'].shape==(2402,453))
        ok('LP_single_HiGHS_fixed_budget',lp['method']=='highs' and lp['options']=={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'time_limit':5.,'maxiter':100000})
        ok('q_affine_absolute_free_controls',np.max(np.abs(c['q_matrix']@x+c['q_base']-b.eval_controls(P,times)[:,j]))<1e-14)
        ok('qd_affine_absolute_free_controls',np.max(np.abs(c['qd_matrix']@x+c['qd_base']-b.eval_controls(P,times,1)[:,j]))<1e-12)
        u=c['command_matrix']@x+c['command_base']
        expected=(float(tau[j])-float(bias[j,0])-float(bias[j,1])*b.eval_controls(P,times)[:150,j]-float(bias[j,2])*b.eval_controls(P,times,1)[:150,j])/float(kp[j])
        expected[0]=float(u0[j])
        ok('all150_commands_including_fixed_first',np.max(np.abs(u-expected))<1e-13 and not np.any(c['command_matrix'][0]) and c['command_base'][0]==float(u0[j]))
        f=np.einsum('irc,c->ir',c['force_Bernstein_matrix'],x)+c['force_Bernstein_base']
        direct=float(kp[j])*u[:,None]+float(bias[j,0])+float(bias[j,1])*qbez[:,:,j]+float(bias[j,2])*dbez[:,:,j]
        ok('all600_Bernstein_rawforce_coefficients',np.max(np.abs(f-direct))<1e-12)
        groups={g['name']:g for g in c['row_groups']}
        rg=groups['target_rate_upper'];upper=(lp['A_ub'][rg['start']:rg['stop'],:151]@x-lp['b_ub'][rg['start']:rg['stop']])
        ok('first_rate_uses_u1_minus_fixedu0',np.max(np.abs(upper-(np.diff(u)-.02*14.9999)))<1e-14)
        fit=np.abs(b.eval_controls(P,times)[:,j]-target[:,j]);smooth=np.abs(np.diff(P[:,j],n=2))
        fullx=np.r_[x,fit,smooth]
        expectedcost=fit.mean()+4*smooth.mean()
        ok('objective_meanL1_plus4meanCoefficientDiff',abs(lp['c']@fullx-expectedcost)<1e-16)
        ok('synthetic_all_linear_rows_feasible',np.max(lp['A_ub']@fullx-lp['b_ub'])<1e-12)
        ok('no_hold_counterfactual_rows',all('hold' not in g['name'] for g in c['row_groups']))
        samples=b.sample_controls(P)
        ok('full151_601_and_nativeFD_separate',samples['q_goal_rad'].shape==(151,20) and samples['planned_q601_rad'].shape==(601,20) and samples['native_forward_FD32_rad_s'].shape==(151,20) and not np.array_equal(samples['native_forward_FD64_rad_s'],samples['analytic_qd151_rad_s']))
        Pbefore=P.copy();paramsbefore={k:getattr(params,k).tobytes() for k in params.__dataclass_fields__}
        exported=b.audit_post_C2(P,params,tau,u0,lo,hi,old)
        ok('final_FP32_schedule150_and149_rates',exported['u_postC2_150_FP32_rad'].shape==(150,20) and exported['planned_rate149_rad_s'].shape==(149,20))
        ok('FP32_oracle_initial_identity',exported['gates']['same_fixed_initial_u0'] and not np.any(exported['initial_ctrl_to_first_target_jump_rad']))
        ok('postC2_force_Bernstein_and_FP32samples_distinct',exported['postC2_real_affine_rawforce_Bernstein_Nm'].shape==(150,4,20) and exported['reference_affine601_FP32_Nm'].shape==(601,20))
        ok('inputs_unchanged',np.array_equal(P,Pbefore) and all(getattr(params,k).tobytes()==v for k,v in paramsbefore.items()))
        badu=u0.astype(float).copy();badu[0]+=1e-10
        rejects('reject_nonFP32_fixedu',lambda:b.build_joint_lp(j,target,P0,P1,badu,tau,params,lo,hi))
        rejects('reject_invalid_joint_bool',lambda:b.build_joint_lp(True,target,P0,P1,u0,tau,params,lo,hi))
        rejects('reject_missing_targetframe',lambda:b.build_joint_lp(j,target[:150],P0,P1,u0,tau,params,lo,hi))
        rejects('reject_changed_regularizer',lambda:b.build_joint_lp(j,target,P0,P1,u0,tau,params,lo,hi,3.))
        badp=P1.copy();badp[0]=2
        rejects('reject_fixedP1_outside_hardbox',lambda:b.build_joint_lp(j,target,P0,badp,u0,tau,params,lo,hi))
        rejects('reject_extrapolation',lambda:B.evaluate([3.00001]))
        badparams={k:getattr(params,k).copy() for k in params.__dataclass_fields__};badparams['bias'][j,1]=0
        rejects('reject_unsupported_PD_bias',lambda:b.build_joint_lp(j,target,P0,P1,u0,tau,badparams,lo,hi))
        badinit=u0.copy();badinit[j]=np.float32(.99)
        rejects('reject_fixedu0_outside_insetplanning',lambda:b.build_joint_lp(j,target,P0,P1,badinit,tau,params,lo,hi))
        tree=ast.parse(SOURCE.read_text('utf-8'))
        calls={getattr(n.func,'attr',getattr(n.func,'id','')) for n in ast.walk(tree) if isinstance(n,ast.Call)}
        ok('source_has_no_solvercall',not calls.intersection({'linprog','minimize','step','forward','mj_forward','mj_step'}))
        ok('no_Torch_Warp_MuJoCo_imports',not any(k in sys.modules for k in ('torch','warp','mujoco')))
        ok('source_and_oracle_unchanged',all(sha(p['path'])==p['sha256'] for p in pins.values()))
        details={'max_synthetic_linear_residual':float(np.max(lp['A_ub']@fullx-lp['b_ub'])),
          'max_qd_FD_error':float(np.max(np.abs(fd-b.eval_controls(P,t,1)))),
          'LP_calls':0,'real_reference_reads':0,'FK_calls':0,'collision_calls':0,'Env':0,'physics_steps':0,'WM':0,'GPU':0,
          'scope':'Synthetic mathematics/source contract only. No real-data fit/LP was solved; no dynamics qualification.'}
        status='PASS'
    except Exception:
        status='FAILED_HARNESS_OR_CHECK';details['traceback']=traceback.format_exc()
    report={'schema':'full151_continuous_control_synthetic_CPU_proof:1','status':status,
      'source_pins':pins,'passed':sum(checks.values()),'total':len(checks),'checks':checks,'details':details}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n','utf-8')
    print(json.dumps({'status':status,'passed':report['passed'],'total':report['total'],'report':str(out/'report.json'),'report_sha256':sha(out/'report.json')}))
    return 0 if status=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
