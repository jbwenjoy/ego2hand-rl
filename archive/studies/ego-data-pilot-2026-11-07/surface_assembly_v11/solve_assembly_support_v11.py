"""Assembly first, then ideal support under hard geometry; no learning.

A0 cheap FK fits ten independent assembly equations. A1 keeps those equations
and full collision-window inequalities hard. B optimizes support inside the
same hard constraints. New native contacts always provide final evidence.
"""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
from scipy.optimize import least_squares,minimize

ROOT=Path(__file__).resolve().parent;P=ROOT.parent
sys.path.insert(0,str(P/'native_contact_design_v7'))
from evaluator_contact_design_v7 import NativeContactDesignEvaluator
from scan_body_axis_reference_v7 import pin,read,write,checked,clean,lp_from_probe
from strict_support_lp_guarded_v7 import solve_strict_support
sys.path.insert(0,str(P/'surface_assembly_v11'))
from assembly_kinematics_v11 import AssemblyKinematics
from anchor_validation_v11 import validate_anchor_bundle
sys.path.insert(0,str(P/'wrench_residual_v6'))
from scaled_wrench_residual_lp_v6 import scaled_wrench_residual_lp

CFG={'seed_names':['original','tilt5'],'q_trust_rad':None,'q_planner_margin_rad':1e-9,
 'palm_translation_trust_m':.025,'rotvec_coordinate_trust_rad':.5,
 'compression_target_m':25e-6,'pad_gap_window_m':[-.0001,.0002],
 'closure_gate_m':.0002,'normal_gate_deg':1.,'eq_position_scale_m':.001,
 'max_abs_hard_eq10_dimensionless':1e-5,
 'A0_max_FK_queries':6000,'A0_wall_s':20.,'A0_max_nfev':200,
 'A1_full_queries':450,'A1_heavy_queries':6000,'A1_wall_s':60.,'A1_maxiter':40,
 'B_full_queries':600,'B_heavy_queries':6000,'B_wall_s':90.,'B_maxiter':60,
 'final_heavy_queries_per_seed':3000,'final_wall_s_per_seed':45.,
 'FD14':[1e-5]*8+[1e-6]*3+[1e-5]*3,'full_cache_size':128,'cheap_cache_size':1024,
 'shortlist_per_seed':2,'native_forward_max':12,'final_native_LP_max':8,
 'normal_cap_per_finger_N':3.,'common_force_point':'Actual hand-triangle A, conditionally inside preloaded phone; not native con.pos',
 'actual_initial_3N_budget_is_descriptive':True}


class Budget(RuntimeError):pass


def bits(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',action='store_true')
    a=ap.parse_args();protocol=read(a.protocol)
    if protocol['schema']!='assembly_then_hard_geometry_support_protocol:11' or protocol['config']!=CFG:raise ValueError('Frozen rules')
    for rec in [*protocol['source_pins'].values(),*protocol['inputs'].values()]:checked(rec)
    if protocol['source_pins']['runner']!=pin(__file__):raise ValueError('Exact runner')
    anchor_bundle,anchor_json_checks=validate_anchor_bundle(protocol['inputs']['surface_anchors']['path'],protocol['inputs']['surface_anchors'])
    if anchor_bundle['branch']!=protocol['anchor_branch']:raise ValueError('Explicit source mapping branch')
    for key,total in (('cheap_FK_contract',14),('native_J_contract',87),('native_LP_contract',4)):
        proof=read(checked(protocol['inputs'][key]))
        if proof['passed']!=total or proof['total']!=total or any(v is not True for v in proof['checks'].values()):raise ValueError('Prior numeric contract '+key)
    out=Path(protocol['output_path'])
    if out.exists():raise ValueError('No overwrite/retry')
    if not a.run:print(json.dumps({'status':'METADATA_ONLY_NO_MODEL','config':CFG}));return
    if protocol['CPU_run_authorized'] is not True:raise ValueError('Explicit CPU binding')
    out.mkdir();write(out/'protocol.json',protocol);ev=None;kin=None;LPcalls=0;seeds=[];phase='load';start=time.perf_counter()
    attemptfile=(out/'attempts.jsonl').open('x',encoding='utf8')
    try:
        old=read(checked(protocol['inputs']['old_native_report']))
        oldc=next(c for s in old['seed_results'] if s['seed']==0 for c in s['shortlist'] if c['rank']==0)
        x46=np.asarray(oldc['x46']);qbase=x46[:20].copy();xb=np.r_[qbase[:8],x46[20:26]]
        design=read(checked(protocol['inputs']['reference_design']))
        tilt=next(v for v in design['four_proposed_seeds'] if v['name']=='tilt_to_force_cone')
        initial=[xb.copy(),np.r_[qbase[:8],tilt['phone_root_translation_palm_m'],tilt['phone_rotation_vector_palm_rad']]]
        ev=NativeContactDesignEvaluator.from_existing_pilot(max_heavy_queries=40000,digest_check_period=128);e=ev.geometry
        kin=AssemblyKinematics(e,anchors_path=protocol['inputs']['surface_anchors']['path'],expected_anchor_pin=protocol['inputs']['surface_anchors'],qrest12=qbase[8:]);e.anchors=kin.hand_points.copy();e.anchor_targets=kin.phone_points.copy();e.material=kin.phone_points.copy()
        if kin.anchor_pin!=protocol['inputs']['surface_anchors'] or not bits(qbase,np.asarray(kin.anchors['source_pose']['qpos27'])[:20]):raise ValueError('Same declared original material/source pose')
        write(out/'algorithm_surface_configuration.json',{'source_anchor_pin':protocol['inputs']['surface_anchors'],
          'source_mapping_branch':protocol['anchor_branch'],'custom_anchor_JSON_checks':anchor_json_checks,'custom_anchor_compiled_checks':kin.compiled_anchor_checks,
          'hand_geom_local_points':kin.hand_points,'phone_body_material_points':kin.phone_points,
          'hand_geom_local_outward_normals':kin.hand_normals,'phone_body_outward_normals':kin.phone_normals,
          'phone_body_tangent_rows':kin.phone_tangents_body_rows,
          'original_ego_material_targets_body_m':[v['original_ego_material_target_phone_body_m'] for v in kin.anchors['anchors']],
          'common_force_point_rule':CFG['common_force_point'],'algorithm_only_physical_model_unchanged':True})
        lo=np.r_[e.qlo[:8]+1e-9,xb[8:11]-.025,xb[11:]-.5]
        hi=np.r_[e.qhi[:8]-1e-9,xb[8:11]+.025,xb[11:]+.5]
        xs=np.r_[np.full(8,.3),np.full(3,.02),np.full(3,.3)];zl=(lo-xb)/xs;zh=(hi-xb)/xs
        cosine=float(np.cos(np.deg2rad(1.)));fd=np.asarray(CFG['FD14'])/xs
        for seedno,x0 in enumerate(initial):
            cheapcache={};fullcache={};pool=[];best_bad=None;attempt=0;successful=0;stage='A0';stage_count=0
            last_cheap_requested=None;best_cheap=None
            stage_limit=6000;stage_deadline=time.perf_counter()+20.;cheap_start_count=kin.counts['query_attempted'];stage_reports={}
            def X(z):return xb+xs*np.asarray(z)
            def cheap(z):
                nonlocal last_cheap_requested
                x=X(z);key=x.tobytes()
                last_cheap_requested=x.copy()
                if key in cheapcache:return cheapcache[key]
                if stage=='A0' and (kin.counts['query_attempted']-cheap_start_count>=6000 or time.perf_counter()>=stage_deadline):raise Budget('A0 cheap FK budget')
                value=kin.query(x);cheapcache[key]=value
                if len(cheapcache)>1024:cheapcache.pop(next(iter(cheapcache)))
                return value
            def eq(z):
                k=cheap(z);return np.r_[k['closure2x3'].ravel()/.001,k['normal_eq2x2'].ravel()]
            def branch(z):return cheap(z)['antidot2']-cosine
            def gate(k):
                return bool(np.max(np.linalg.norm(k['closure2x3'],axis=1))<=.0002
                  and np.max(np.degrees(np.arccos(np.clip(k['antidot2'],-1,1))))<=1.
                  and np.max(abs(np.r_[k['closure2x3'].ravel()/.001,k['normal_eq2x2'].ravel()]))<=1e-5)
            def motion(z,center):return float(np.sum((z-center)**2)+1e-7*np.sum(z*z))
            def score(v):return v['LP']['scaled_loss'] if v['LP'] is not None else 1000.+v['motion']
            def register(v):
                nonlocal pool,best_bad
                violation=float(np.sum(np.minimum(v['constraints']['values_m']/.001,0)**2)+np.sum(eq(v['z14'])**2)+np.sum(np.minimum(branch(v['z14']),0)**2))
                if best_bad is None or violation<best_bad['violation']:
                    best_bad={'value':v,'violation':violation}
                if not(v['geometry_gate'] and v['assembly_gate']):return
                close=[i for i,w in enumerate(pool) if np.max(abs(v['z14']-w['z14']))<1e-4]
                if close:
                    i=close[0]
                    if score(v)<score(pool[i]):pool[i]=v
                else:pool.append(v)
                pool.sort(key=score);pool=pool[:2]
            def full(z,purpose='base'):
                nonlocal attempt,successful,stage_count
                x=X(z);key=x.tobytes()
                if key in fullcache:
                    v=fullcache[key]
                    if purpose=='base':register(v)
                    return v
                if stage_count>=stage_limit or time.perf_counter()>=stage_deadline:raise Budget(stage+' full evaluation budget')
                attempt+=1;stage_count+=1
                attemptfile.write(json.dumps(clean({'event':'attempt','seed':seedno,'stage':stage,'attempt':attempt,'purpose':purpose,'x14':x}))+ '\n');attemptfile.flush()
                k=cheap(z);s=e.evaluate(k['q20'],x[8:11],x[11:],exact_all=False)
                if s['unknown']:
                    write(out/f'unknown_seed{seedno}_attempt{attempt}.json',s);raise ValueError('Unknown collision evidence')
                s['fixed_material_line_scope']='Caller v11 pinned source-region material points; exact source and algorithmic1mm branches remain distinct. Not actual native contact line.'
                if (not bits(s['q20_rad'],k['q20'])
                    or np.max(abs(s['anchors_world_m']-k['A_hand_world_m']))>5e-12
                    or np.max(abs(s['anchor_targets_world_m']-k['B_phone_world_m']))>5e-12):
                    raise ValueError('Same-x cheap/full FK surface operands required')
                c=e.constraint_vector(s,pad_gap_window_m=[-.0001,.0002],nonpad_buffer_m=0.,self_buffer_m=0.)
                v={'x14':x,'z14':np.asarray(z).copy(),'kinematics':k,'static':s,'constraints':c,'LP':None,
                  'assembly_gate':gate(k),'geometry_gate':c['all_satisfied'],'motion':motion(z,zcenter),'attempt':attempt,
                  'eq10_scaled':eq(z),'max_abs_eq10_scaled':float(np.max(abs(eq(z))))}
                fullcache[key]=v
                if len(fullcache)>128:fullcache.pop(next(iter(fullcache)))
                successful+=1
                attemptfile.write(json.dumps(clean({'event':'returned','seed':seedno,'stage':stage,'attempt':attempt,
                  'assembly_gate':v['assembly_gate'],'geometry_gate':v['geometry_gate'],'min_geometry_m':min(c['values_m']),
                  'eq10':eq(z),'antidot2':k['antidot2']}))+ '\n');attemptfile.flush()
                if purpose=='base':register(v)
                return v
            def inequalities(z,purpose='base'):
                v=full(z,purpose);return np.r_[v['constraints']['values_m']/.001,branch(z)]
            def support(z,purpose='base'):
                nonlocal LPcalls
                v=full(z,purpose)
                if v['LP'] is None:
                    s=v['static'];k=v['kinematics']
                    # Common hand-surface A is used in both object lever and hand Jacobian.
                    lp=scaled_wrench_residual_lp(k['A_hand_world_m'],k['n_hand_world'],s['support_operands']['object_COM'],
                      float(e.m.body_mass[e.oid]),e.mu_actual,s['anchor_J2x3x26'][:,:,:20],
                      s['qfrc_bias26_Nm'][:20]-s['qfrc_passive26_Nm'][:20],s['reachable_torque_lower20_Nm'],s['reachable_torque_upper20_Nm'],
                      finger_group_ids=['finger1','finger2'],normal_cap_per_finger_N=3.,gravity_world=e.m.opt.gravity)
                    LPcalls+=1;v['LP']=lp
                    if not lp['hard_tau_group_cone_feasible']:write(out/f'unknown_LP_seed{seedno}_attempt{v["attempt"]}.json',lp);raise ValueError('Unknown conditional support LP')
                if purpose=='base':register(v)
                return float(v['LP']['scaled_loss']**2+1e-7*np.sum(np.asarray(z)**2))
            def J(fun,z):
                base=np.atleast_1d(fun(z));jac=np.empty((base.size,14))
                for j,h in enumerate(fd):
                    step=min(h,zh[j]-z[j])
                    if step<1e-10:step=-min(h,z[j]-zl[j])
                    if abs(step)<1e-14:jac[:,j]=0.;continue
                    zz=z.copy();zz[j]+=step;actual=zz[j]-z[j]
                    jac[:,j]=(np.atleast_1d(fun(zz))-base)/actual
                return jac
            def budgeted_error(err):return isinstance(err,Budget) or type(err).__name__=='StaticBudgetExceeded'
            z0=(x0-xb)/xs;z=z0.copy();phase=f'seed{seedno}_A0'
            def fit(z):
                nonlocal best_cheap
                r=np.r_[eq(z),10*np.minimum(branch(z),0),1e-7*(z-z0)]
                cost=float(r@r)
                if best_cheap is None or cost<best_cheap['cost']:best_cheap={'z14':z.copy(),'cost':cost}
                return r
            try:
                sol=least_squares(fit,z,jac=lambda z:J(fit,z),bounds=(zl,zh),max_nfev=200,ftol=1e-12,xtol=1e-12,gtol=1e-10)
                z=sol.x.copy();singular=np.linalg.svd(sol.jac[:10],compute_uv=False)
                stage_reports['A0']={'status':int(sol.status),'message':sol.message,'nfev':sol.nfev,'njev':sol.njev,'optimality':float(sol.optimality),
                  'eq10_jacobian_singular_values':singular,'eq10_local_numeric_rank_relative1e_minus8':int(np.sum(singular>singular[0]*1e-8))}
            except RuntimeError as err:
                if not budgeted_error(err):raise
                if best_cheap is not None:z=best_cheap['z14'].copy()
                stage_reports['A0']={'budget_stop':repr(err),'selected_best_finite_cheap_visit_including_FD':best_cheap,
                  'last_requested_cheap_x14':last_cheap_requested}
            fitted=cheapcache.get(X(z).tobytes())
            if fitted is None:
                stage='A1';fitted=cheap(z)
            write(out/f'seed{seedno}_A0.json',{'z14':z,'kinematics':fitted,'report':stage_reports['A0'],'geometry_or_support_qualified':False})
            stage='A1';phase=f'seed{seedno}_A1';stage_count=0;stage_limit=450;stage_deadline=time.perf_counter()+60.
            e.set_budget(deadline_monotonic=stage_deadline,max_additional_heavy_queries=6000)
            zcenter=z.copy()
            cons=[{'type':'eq','fun':eq,'jac':lambda z:J(eq,z)},
                  {'type':'ineq','fun':inequalities,'jac':lambda z:J(lambda zz:inequalities(zz,'FD'),z)}]
            try:
                v=full(z)
                if v['geometry_gate'] and v['assembly_gate']:
                    stage_reports['A1']={'status':'ALREADY_PASSED_SAVED_GEOMETRY_AND_ASSEMBLY_GATES'}
                else:
                    sol=minimize(lambda zz:motion(zz,zcenter),z,jac=lambda zz:2*(zz-zcenter)+2e-7*zz,
                      method='SLSQP',bounds=list(zip(zl,zh)),constraints=cons,options={'maxiter':40,'ftol':1e-10,'disp':False})
                    z=sol.x.copy();full(z);stage_reports['A1']={'status':int(sol.status),'message':sol.message,'nit':sol.nit,'success':bool(sol.success)}
            except RuntimeError as err:
                if not budgeted_error(err):raise
                stage_reports['A1']={'budget_stop':repr(err)}
            if pool:
                stage='B';phase=f'seed{seedno}_B';stage_count=0;stage_limit=600;stage_deadline=time.perf_counter()+90.
                e.set_budget(deadline_monotonic=stage_deadline,max_additional_heavy_queries=6000);z=pool[0]['z14'].copy()
                try:
                    sol=minimize(support,z,jac=lambda zz:J(lambda zzz:support(zzz,'FD'),zz).ravel(),method='SLSQP',
                      bounds=list(zip(zl,zh)),constraints=cons,options={'maxiter':60,'ftol':1e-10,'disp':False})
                    z=sol.x.copy();support(z);stage_reports['B']={'status':int(sol.status),'message':sol.message,'nit':sol.nit,'success':bool(sol.success)}
                except RuntimeError as err:
                    if not budgeted_error(err):raise
                    stage_reports['B']={'budget_stop':repr(err)}
            else:stage_reports['B']={'status':'NOT_RUN_NO_ASSEMBLED_COLLISION_FEASIBLE_BASE'}
            stage='final';phase=f'seed{seedno}_final';e.set_budget(deadline_monotonic=time.perf_counter()+45.,max_additional_heavy_queries=3000)
            chosen=list(pool) if pool else ([best_bad['value']] if best_bad else []);finals=[]
            for rank,v in enumerate(chosen[:2]):
                x=v['x14'];q=v['kinematics']['q20'];geo=ev.final_geometry(q,x[8:11],x[11:],pad_gap_window_m=[-.0001,.0002])
                u=x46[26:].copy();raw=None;control_origin='original_seed_u0_no_conditional_LP'
                if v['LP'] is not None:
                    raw=(np.asarray(v['LP']['tau_act_required_Nm'])-v['static']['PD_offset20'])/v['static']['PD_gain20']
                    if np.all(raw>=e.m.actuator_ctrlrange[:,0]) and np.all(raw<=e.m.actuator_ctrlrange[:,1]):u=raw;control_origin='legal_conditional_common_A_inverse_PD'
                    else:control_origin='original_seed_u0_because_conditional_inverse_outside_ctrl_range'
                native=ev.evaluate(q,x[8:11],x[11:],u);nativelp,reason=lp_from_probe(native);strict=None
                material_diagnostics=[]
                for contact in native['contact_design'].get('phone_hand_contacts') or []:
                    finger=contact['finger_index']
                    if finger not in (1,2):continue
                    body=(contact['position_world_m']-native['raw']['phone_root_world_m'])@native['raw']['phone_body_R_world']
                    source_target=np.asarray(kin.anchors['anchors'][finger-1]['original_ego_material_target_phone_body_m'])
                    angle=np.arccos(np.clip(float(np.dot(contact['normal_force_on_object_world'],v['kinematics']['n_hand_world'][finger-1])),-1,1))
                    material_diagnostics.append({'native_contact_index':contact['native_contact_index'],'finger_index':finger,
                      'native_point_phone_body_m':body,'fixed_phone_material_point_body_m':kin.phone_points[finger-1],
                      'distance_to_fixed_phone_material_point_m':float(np.linalg.norm(body-kin.phone_points[finger-1])),
                      'original_ego_material_target_body_m':source_target,'distance_to_original_ego_target_m':float(np.linalg.norm(body-source_target)),
                      'native_vs_conditional_force_normal_angle_rad':float(angle)})
                if nativelp is not None:
                    LPcalls+=1
                    if geo['geometry_gate_pass'] and v['assembly_gate'] and nativelp['hard_bounds_pass']:
                        pd=native['contact_design']['PD_raw'];strict=solve_strict_support(nativelp,pd_gain20=pd['PD_gain20'],pd_offset20=pd['PD_offset20'],ctrl_range20x2=pd['actuator_ctrlrange20x2']);LPcalls+=1
                ideal=bool(geo['geometry_gate_pass'] and v['assembly_gate'] and strict is not None and strict['strict_force_moment_pass'])
                repeats=[];entry={};budget_descr=None;state=None
                if ideal:
                    for k in range(2):repeats.append(ev.evaluate(q,x[8:11],x[11:],strict['proposed_ctrl20']))
                    checks={key:bits(repeats[0]['raw'][key],repeats[1]['raw'][key]) for key in ('qpos27','ctrl20','qacc26','qfrc_actuator26','qfrc_constraint26','a_COM_world_m_s2','alpha_world_rad_s2')}
                    checks['physical_contacts']=bits(repeats[0]['raw']['contact_rows24'][:,1:],repeats[1]['raw']['contact_rows24'][:,1:])
                    ag=repeats[0]['native_contact_aggregates'];entry={'known':all(not rr['unknown'] for rr in repeats),'repeat_bits':all(checks.values()),
                      'ctrl_unclamped':all(bits(rr['PD']['requested_ctrl20'],rr['PD']['effective_ctrl20']) for rr in repeats),
                      'two_pads_loaded':all(ag[k]['normal_force_signed_N']>.1 for k in ('thumb_pad','index_pad')),
                      'nonpad_table_zero':all(ag[k]['normal_force_signed_N']<=1e-8 for k in ('phone_nonpad','phone_table'))}
                    budget_descr={'initial_pad_normal_N':[ag[k]['normal_force_signed_N'] for k in ('thumb_pad','index_pad')],
                      'initial_each_pad3N_design_budget_pass':all(ag[k]['normal_force_signed_N']<=3 for k in ('thumb_pad','index_pad')),'hard_dynamic_gate':False}
                    state=out/f'seed{seedno}_rank{rank}_state.npz';rr=repeats[0]['raw']
                    np.savez_compressed(state,qpos27=rr['qpos27'],qvel26=rr['qvel26'],ctrl20=rr['ctrl20'],x46=np.r_[q,x[8:],rr['ctrl20']])
                record={'seed':seedno,'rank':rank,'candidate':v,'final_geometry':geo,'native_probe':native,'current_native_LP':nativelp,
                  'initial_native_probe_ctrl_source':control_origin,'conditional_inverse_ctrl20':raw,
                  'actual_material_contact_diagnostics':material_diagnostics,
                  'LP_unavailable_reason':reason,'strict_native_support_LP':strict,'ideal_native_support_candidate':ideal,
                  'fresh_command_probes':repeats,'native_entry':entry,'actual_initial_design_budget':budget_descr,
                  'holding_diagnostic_eligible':bool(ideal and entry and all(entry.values())),'candidate_state':None if state is None else pin(state),
                  'holding_or_faithful_ego_qualified':False}
                path=out/f'seed{seedno}_rank{rank}.json';write(path,record)
                finals.append({'rank':rank,'result':pin(path),'geometry_pass':geo['geometry_gate_pass'],'assembly_gate':v['assembly_gate'],
                  'ideal_native_support':ideal,'holding_diagnostic_eligible':record['holding_diagnostic_eligible'],'candidate_state':record['candidate_state']})
            seeds.append({'seed':seedno,'name':CFG['seed_names'][seedno],'stages':stage_reports,'full_attempts':attempt,'full_returned':successful,'finals':finals})
            print(json.dumps(clean(seeds[-1])),flush=True)
        attemptfile.close();audit=ev.audit_model();audit['passed']=True
        write(out/'report.json',{'schema':'assembly_then_hard_geometry_support_result:11','status':'COMPLETED_BOUNDED_TWO_SEED_STAGED_SEARCH',
          'protocol':pin(a.protocol),'anchor_branch':protocol['anchor_branch'],'seed_results':seeds,'holding_diagnostic_candidates':sum(v['holding_diagnostic_eligible'] for s in seeds for v in s['finals']),
          'counts':ev.counts,'static_counts':e.counts,'cheap_FK_counts':kin.counts,'LP_calls':LPcalls,
          'model_audit':audit,'cheap_model_audit':kin.audit_model(),'wall_seconds':time.perf_counter()-start,
          'new_steps_training_GPU':0,'scope':'Fixed material surface assembly and hard geometry precede support. Common A is conditional, not actual native con.pos. Original fixed model/wrist; no faithful ego, calibrated pad, policy or holding qualification.'})
    except BaseException as err:
        attemptfile.close();write(out/'failure.json',{'phase':phase,'error':repr(err),'seed_results':seeds,'LP_calls':LPcalls,
          'counts':None if ev is None else ev.counts,'cheap_counts':None if kin is None else kin.counts,'automatic_retry':False});raise


if __name__=='__main__':main()
