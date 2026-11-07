"""Restore geometric feasibility using local phone planes and fresh full checks.

Only q0..11 change; phone pose and other8 joints stay at each saved candidate.
Plane sensitivities propose steps, not certified closest-point/physical gradients.
All accepted and final states are queried against all22/189 original constraints.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time
import numpy as np
from scipy.optimize import minimize

H=Path(__file__).resolve().parent;P=H.parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def checked(r):assert pin(r['path'])==r,'Changed pinned input '+r['path'];return Path(r['path'])
def load(p,name):
    sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def violation(f):return float(np.linalg.norm(np.minimum(f['values'][:214]/.001,0)))

def plain_failure(entry,parent,qpos,error):
    return dict(schema='functional_tripod_geometric_reference_candidate:1',candidate_id=entry['name']+'__fixed_phone_q_projection',
        source_entry=entry,parent_geometric_candidate=parent['candidate'],qpos27_F64=np.asarray(qpos,dtype=np.float64),qvel26_F64=np.zeros(26),
        q20_F64=np.asarray(qpos,dtype=np.float64)[:20],designated_geom_ids=[15,28,41],pad_bindings=[],control_qualified=False,
        geometry=dict(passed=False,phone_pair_count=0,self_pair_count=0,table_complete=False,unknown=[dict(projection_error=error)],fresh_final_evaluation=False),
        static_or_dynamic_holding_qualified=False,source_functional_pad_truth='UNKNOWN')

def matrix_from_phone_planes(E,f,helper):
    desc=helper.phone_gap_jacobian(E,f)
    J=np.zeros((214,12));used=[];unknown=[];row=0
    for g in f['state']['phone_geometry_rows']:
        key=int(g['geom_id']);d=desc[key]
        gradient=d.get('grad_q12')
        if gradient is not None:
            gradient=np.asarray(gradient);assert gradient.shape==(12,) and np.isfinite(gradient).all()
            J[row]=gradient
            if key in [15,28,41]:J[row+1]=-gradient
            used.append(key)
        else:unknown.append(key)
        row+=2 if key in [15,28,41] else 1
    assert row==25 and len(f['state']['self_geometry_rows'])==189
    # Self and unavailable/far-plane sensitivities are unmodelled, NOT a zero
    # physical derivative claim. Every proposal is checked with full geometry.
    return J,desc,dict(modelled_phone_geoms=used,unmodelled_phone_geoms=unknown,
        unmodelled_self_pairs=189,unmodelled_entries_are_zero_in_proposal_matrix_only=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);a=ap.parse_args();pr=read(a.protocol)
    assert pr['schema']=='functional_tripod_saved_geometry_plane_projection_protocol:1'
    assert pr['CPU_static_run_authorized'] is True and pr['neural_training_or_integration_authorized'] is False
    for r in [*pr['sources'].values(),*pr['inputs'].values()]:checked(r)
    assert pr['sources']['runner']==pin(__file__)
    out=Path(pr['output']);out.mkdir(exist_ok=False);write(out/'protocol.json',pr)
    base=load(checked(pr['sources']['frozen_geometry_solver']),'frozen_tripod_solver_for_projection')
    helper=load(checked(pr['sources']['plane_helper']),'phone_plane_helper_for_projection')
    original=read(checked(pr['inputs']['original_geometry_protocol']));previous=read(checked(pr['inputs']['original_geometry_report']))
    assert len(previous['candidates'])==len(original['entries'])==4
    start=time.perf_counter();E=base.Tripod(original,out);E.global_deadline=start+pr['budget']['global_wall_s']
    results=[];qp_count=0;accepted=0;line_search_calls=0;entry_summaries=[]
    for entry,parent in zip(original['entries'],previous['candidates']):
        old=read(checked(parent['candidate']));assert old['candidate_id']==entry['name']
        E.counts['entries_attempted']+=1
        try:
            E.configure(entry);x=np.asarray(old['x18'],dtype=np.float64)
            assert x.shape==(18,) and E.rest.tobytes()==np.asarray(old['qpos27_F64'],dtype=np.float64)[12:20].tobytes()
            fixed=x[12:].tobytes();logs=[];phase_error=None
            E.phase='GEOMETRY_PROJECTION';E.phase_deadline=min(E.global_deadline,time.perf_counter()+pr['budget']['entry_wall_s'])
            E.cheap_max=20000;E.full_limit=pr['budget']['full_queries_per_entry']
            E.E.set_budget(deadline_monotonic=E.phase_deadline,max_additional_heavy_queries=pr['budget']['heavy_queries_per_entry'])
            f=E.full(x);initial=violation(f)
        except Exception as err:
            candidate=plain_failure(entry,parent,old['qpos27_F64'],'INITIAL: '+repr(err))
            path=out/(entry['name']+'_candidate.json');base.write(path,candidate);state=out/(entry['name']+'_state.npz')
            np.savez_compressed(state,qpos27_F64=candidate['qpos27_F64'],qvel26_F64=candidate['qvel26_F64'])
            results.append(dict(candidate=pin(path),state_npz=pin(state),geometry_pass=False,candidate_id=candidate['candidate_id']))
            entry_summaries.append(dict(candidate_id=candidate['candidate_id'],geometry_pass=False,phase_error='INITIAL: '+repr(err)))
            if getattr(E,'trace',None) is not None:E.trace.close()
            E.counts['entries_returned']+=1;continue
        for iteration in range(pr['budget']['iterations_per_entry']):
            if f['passed']:break
            try:
                # A cache hit after a rejected trial need not leave E.d at f.
                # Synchronize the current native FK data before point Jacobians.
                E.full_cache.pop(x.tobytes(),None);f=E.full(x)
                J,desc,scope=matrix_from_phone_planes(E,f,helper)
                if any(desc[g].get('grad_q12') is None for g in [15,28,41]):
                    logs.append(dict(iteration=iteration,status='MISSING_PAD_PLANE_NO_STEP',scope=scope));break
                scale=np.full(12,pr['proposal']['joint_step_trust_rad'])
                # Scaled convex quadratic local proposal with explicit slack.
                A=J*scale[None,:]/.001;b=(f['values'][:214]-pr['proposal']['interior_target_margin_m'])/.001
                qpJ=np.c_[A,np.ones(214)]
                lower=np.maximum(-1,(E.m.jnt_range[:12,0]-x[:12])/scale)
                upper=np.minimum(1,(E.m.jnt_range[:12,1]-x[:12])/scale)
                initialz=np.r_[np.zeros(12),max(0.,-float(b.min()))+1e-10]
                fun=lambda z:.5*float(z[:12]@z[:12])+1000.*float(z[12]**2)
                jac=lambda z:np.r_[z[:12],2000.*z[12]]
                z=minimize(fun,initialz,jac=jac,method='SLSQP',bounds=list(zip(lower,upper))+[(0.,None)],
                    constraints=[dict(type='ineq',fun=lambda z:b+qpJ@z,jac=lambda z:qpJ)],
                    options=dict(maxiter=100,ftol=1e-12,disp=False));qp_count+=1
                note=dict(iteration=iteration,QP_status=int(z.status),QP_success=bool(z.success),QP_message=str(z.message),
                    proposed_dq12=(scale*z.x[:12]).tolist(),proposal_slack=float(z.x[12]),initial_violation=violation(f),scope=scope,line_search=[])
                got=False
                for fraction in pr['proposal']['line_search_fractions']:
                    trial=x.copy();trial[:12]+=fraction*scale*z.x[:12]
                    assert trial[12:].tobytes()==fixed and np.all(trial[:12]>=E.m.jnt_range[:12,0]) and np.all(trial[:12]<=E.m.jnt_range[:12,1])
                    actual=E.full(trial);line_search_calls+=1
                    note['line_search'].append(dict(fraction=fraction,full_violation=violation(actual),full_geometry_pass=bool(actual['passed'])))
                    if actual['passed'] or violation(actual)<violation(f)-1e-9:
                        x=trial;f=actual;got=True;accepted+=1;break
                note['accepted']=got;logs.append(note)
                if not got:break
            except Exception as err:
                phase_error=repr(err);logs.append(dict(iteration=iteration,error=phase_error));break
        E.phase='FINAL';E.phase_deadline=min(E.global_deadline,time.perf_counter()+pr['budget']['final_wall_s'])
        E.cheap_max+=100;E.full_limit+=2;E.E.set_budget(deadline_monotonic=E.phase_deadline,max_additional_heavy_queries=500)
        try:candidate=E.final_candidate(x,entry['name']+'__fixed_phone_q_projection')
        except Exception as err:
            phase_error=repr(err)
            try:candidate=E.final_candidate(x,entry['name']+'__fixed_phone_q_projection',saved_result=f,final_failure=repr(err))
            except Exception as fallback_error:
                phase_error=phase_error+'; FALLBACK: '+repr(fallback_error)
                candidate=plain_failure(entry,parent,f['k']['qpos27'],phase_error)
        candidate['parent_geometric_candidate']=parent['candidate'];candidate['projection_iteration_log']=logs
        candidate['projection_scope']='Fixed phone and q12..19; approximate phone-only support-plane sensitivities. All unmodelled self/AABB derivatives excluded from physical claims. Every proposal/final uses full original geometry; no static force LP or relaxed gate.'
        path=out/(entry['name']+'_candidate.json');base.write(path,candidate);state=out/(entry['name']+'_state.npz')
        np.savez_compressed(state,qpos27_F64=candidate['qpos27_F64'],qvel26_F64=candidate['qvel26_F64'])
        E.trace.close();E.counts['entries_returned']+=1
        results.append(dict(candidate=pin(path),state_npz=pin(state),geometry_pass=bool(candidate['geometry']['passed']),candidate_id=candidate['candidate_id']))
        entry_summaries.append(dict(candidate_id=candidate['candidate_id'],initial_violation=initial,final_accepted_violation=violation(f),
            geometry_pass=bool(candidate['geometry']['passed']),accepted_iterations=sum(v.get('accepted',False) for v in logs),
            phone_pose_and_other8_x_parameters_unchanged=x[12:].tobytes()==fixed,phase_error=phase_error))
        print(json.dumps(entry_summaries[-1]),flush=True)
    assert E.E._parameter_digest()==E.model_initial
    report=dict(schema='functional_tripod_geometric_reference_result:1',status='COMPLETED_FOUR_SAVED_STATE_PHONE_PLANE_PROJECTIONS',
        protocol=pin(a.protocol),model_mjb=previous['model_mjb'],parent_report=pr['inputs']['original_geometry_report'],candidates=results,
        summaries=entry_summaries,counts=E.counts,static_geometry_counts=E.E.counts,
        geometric_QP_calls=qp_count,accepted_projection_steps=accepted,line_search_full_geometry_calls=line_search_calls,
        qualified_geometry_candidates=sum(v['geometry_pass'] for v in results),elapsed_s=time.perf_counter()-start,
        model_partial_parameter_digest_unchanged=True,holding_or_ego_retarget_qualified=False,new_neural_training=False,
        scope='Adaptive feasibility restoration after bounded finite-difference search, not unseen validation. No material/solver/gate change. Support-plane proposal is not exact smooth gap gradient, native contact force or holding proof.')
    write(out/'report.json',report);print(json.dumps(dict(report=pin(out/'report.json'),qualified=report['qualified_geometry_candidates'])))

if __name__=='__main__':main()
