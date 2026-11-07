"""Bounded q8/phone6 surface-pair and wrench candidate search, no learning.

Fixed actual filled-triangle material points/normals, not arbitrary centroids.
The optimizer uses static FK and conditional ideal forces. Final probes always
recompute native multi-contact and complete independent geometry separately.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parent;P=ROOT.parent
sys.path.insert(0,str(P/'native_contact_design_v7'))
from evaluator_contact_design_v7 import NativeContactDesignEvaluator
from scan_body_axis_reference_v7 import pin,read,write,checked,lp_from_probe
from strict_support_lp_guarded_v7 import solve_strict_support
sys.path.insert(0,str(P/'wrench_residual_v6'))
from scaled_wrench_residual_lp_v6 import scaled_wrench_residual_lp

CONFIG={'seed_names':['original','tilt5'],'variables':'q_first8,t_palm3,rotvec_palm3; remaining12q fixed',
 'q_trust_rad':.45,'translation_trust_m':.025,'rotvec_coordinate_trust_rad':.5,
 'compression_target_m':.000025,'pad_gap_window_m':[-.0001,.0002],
 'normal_alignment_gate_deg':1.,'anchor_closure_gate_m':.0002,
 'max_search_evaluations_per_seed':1000,'max_search_heavy_queries_per_seed':6000,
 'max_search_wall_seconds_per_seed':120.,'max_final_heavy_queries_per_seed':3000,
 'max_final_wall_seconds_per_seed':60.,'max_nfev':100,
 'FD_steps14':[.0001]*8+[.00001]*3+[.0001]*3,
 'shortlist_per_seed':3,'cache_max_entries':128,'final_geometries_max':6,
 'native_forward_max':18,'final_native_LP_max':12,'normal_cap_per_finger_N':3.,
 'weights':{'closure_per_0p1mm':5.,'normal_difference':10.,'scaled_wrench6':100.,
   'negative_geometry_per_0p1mm':50.,'q_style_per_0p3rad':.02,
   'translation_style_per_0p02m':.005,'rotvec_style_per_0p3rad':.005}}


class SearchBudget(RuntimeError):pass


def same(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',action='store_true')
    a=ap.parse_args();protocol=read(a.protocol)
    if protocol['schema']!='fixed_surface_material_q8_phone6_search_protocol:8' or protocol['config']!=CONFIG:raise ValueError('Frozen design')
    for rec in [*protocol['source_pins'].values(),*protocol['inputs'].values()]:checked(rec)
    if protocol['source_pins']['runner']!=pin(__file__):raise ValueError('Current source')
    anchors=read(checked(protocol['inputs']['surface_anchors']))
    if anchors['schema']!='saved_filled_triangle_paired_surface_anchors:7' or any(v is not True for v in anchors['checks'].values()):raise ValueError('Actual triangle anchors')
    out=Path(protocol['output_path'])
    if out.exists():raise ValueError('No overwrite/retry')
    if not a.run:print(json.dumps({'status':'METADATA_ONLY_NO_MODEL','config':CONFIG}));return
    if protocol['CPU_run_authorized'] is not True:raise ValueError('Explicit CPU boundary')
    out.mkdir();write(out/'protocol.json',protocol);start=time.perf_counter();ev=None;logs=[];seed_results=[];LP_calls=0
    phase='load'
    try:
        old=read(checked(protocol['inputs']['native_search_report']))
        source=next(c for s in old['seed_results'] if s['seed']==0 for c in s['shortlist'] if c['rank']==0)
        x46=np.asarray(source['x46']);qbase=x46[:20].copy();xbase=np.r_[qbase[:8],x46[20:26]]
        if not same(anchors['source_pose']['qpos27'],source['native']['raw']['qpos27']):raise ValueError('Same anchor/source pose')
        reference_design=read(checked(protocol['inputs']['reference_design']))
        tilt=next(s for s in reference_design['four_proposed_seeds'] if s['name']=='tilt_to_force_cone')
        seeds=[xbase.copy(),np.r_[qbase[:8],tilt['phone_root_translation_palm_m'],tilt['phone_rotation_vector_palm_rad']]]
        ev=NativeContactDesignEvaluator.from_existing_pilot(max_heavy_queries=20000,digest_check_period=128)
        e=ev.geometry
        # Algorithm inputs only; physical model and all original collider arrays remain unchanged.
        e.anchors=np.array([s['surfaces']['hand']['projected_point_geom_local_m'] for s in anchors['anchors']])
        e.anchor_targets=np.array([s['surfaces']['phone']['projected_point_phone_body_m'] for s in anchors['anchors']])
        e.material=e.anchor_targets.copy()
        nHlocal=np.array([s['surfaces']['hand']['outward_normal_geom_local'] for s in anchors['anchors']])
        nObody=np.array([s['surfaces']['phone']['outward_normal_phone_body'] for s in anchors['anchors']])
        if [s['hand_geom_id'] for s in anchors['anchors']]!=list(e.pads):raise ValueError('Exact thumb/index geom order')
        for normal in (nHlocal,nObody):
            if np.max(abs(np.linalg.norm(normal,axis=1)-1))>1e-8:raise ValueError('Unit actual normals')
        write(out/'algorithm_surface_configuration.json',{'hand_geom_local_points':e.anchors,
          'phone_body_material_points':e.anchor_targets,'hand_geom_local_outward_normals':nHlocal,
          'phone_body_outward_normals':nObody,'algorithm_arrays_only_physical_model_unchanged':True,
          'source_anchors':protocol['inputs']['surface_anchors'],
          'original_ego_material_points_unchanged':[v['original_ego_material_target_phone_body_m'] for v in anchors['anchors']]})
        lo=np.r_[np.maximum(e.qlo[:8],qbase[:8]-.45),xbase[8:11]-.025,xbase[11:14]-.5]
        hi=np.r_[np.minimum(e.qhi[:8],qbase[:8]+.45),xbase[8:11]+.025,xbase[11:14]+.5]
        scale_x=np.r_[np.full(8,.3),np.full(3,.02),np.full(3,.3)]
        for seed_index,x0 in enumerate(seeds):
            phase='search_'+str(seed_index);deadline=time.perf_counter()+120.
            e.set_budget(deadline_monotonic=deadline,max_additional_heavy_queries=6000)
            cache={};shortlist=[];search_count=0;solver=None;stop=None
            def evaluate(x,purpose):
                nonlocal search_count,LP_calls,shortlist
                x=np.asarray(x,dtype=np.float64);key=x.tobytes()
                if key in cache:
                    v=cache[key]
                    if purpose=='base' and not v['registered_base']:
                        v['registered_base']=True;register(v)
                    return v
                if search_count>=1000 or time.perf_counter()>=deadline:raise SearchBudget('Registered native-FK/query/wall budget')
                search_count+=1;q=qbase.copy();q[:8]=x[:8]
                s=e.evaluate(q,x[8:11],x[11:14],exact_all=False)
                if s['unknown']:
                    write(out/('unknown_seed'+str(seed_index)+'_eval'+str(search_count)+'.json'),s)
                    raise ValueError('Unknown static geometry retained')
                constraints=e.constraint_vector(s,pad_gap_window_m=[-.0001,.0002],nonpad_buffer_m=0.,self_buffer_m=0.)
                R=s['phone_R_world'];GR=np.array([e.d.geom_xmat[g].reshape(3,3) for g in e.pads])
                nH=np.einsum('nij,nj->ni',GR,nHlocal);nO=nObody@R.T
                A=s['anchors_world_m'];B=s['anchor_targets_world_m']
                if seed_index==0 and search_count==1:
                    for key,actual in (('hand',A),('phone',B)):
                        expected=np.array([v['surfaces'][key]['initial_projected_point_world_m'] for v in anchors['anchors']])
                        if np.max(abs(actual-expected))>5e-12:raise ValueError('Initial actual-surface coordinate contract '+key)
                lp=scaled_wrench_residual_lp(B,nH,s['support_operands']['object_COM'],float(e.m.body_mass[e.oid]),e.mu_actual,
                  s['anchor_J2x3x26'][:,:,:20],s['qfrc_bias26_Nm'][:20]-s['qfrc_passive26_Nm'][:20],
                  s['reachable_torque_lower20_Nm'],s['reachable_torque_upper20_Nm'],finger_group_ids=['finger1','finger2'],
                  normal_cap_per_finger_N=3.,gravity_world=e.m.opt.gravity)
                LP_calls+=1
                if not lp['hard_tau_group_cone_feasible'] or lp['scaled_residual6'] is None:
                    write(out/('unknown_LP_seed'+str(seed_index)+'_eval'+str(search_count)+'.json'),lp)
                    raise ValueError('Unknown conditional point-force LP')
                closure=B-A+25e-6*nH
                residual=np.r_[5*closure.ravel()/1e-4,10*(nH+nO).ravel(),100*np.asarray(lp['scaled_residual6']),
                  50*np.minimum(constraints['values_m'],0)/1e-4,
                  .02*(x[:8]-qbase[:8])/.3,.005*(x[8:11]-xbase[8:11])/.02,.005*(x[11:14]-xbase[11:14])/.3]
                if not np.isfinite(residual).all():raise ValueError('Nonfinite proposal residual')
                angles=np.arccos(np.clip(np.sum(nH*(-nO),axis=1),-1,1))
                v={'x14':x.copy(),'q20':q,'static':s,'LP':lp,'constraints':constraints,'residual':residual,
                  'cost':float(residual@residual),'closure_world_m':closure,'normal_angles_rad':angles,
                  'registered_base':purpose=='base','evaluation':search_count,'purpose':purpose}
                cache[key]=v
                if len(cache)>128:cache.pop(next(iter(cache)))
                logs.append({'seed':seed_index,'evaluation':search_count,'purpose':purpose,'x14':x.copy(),
                  'cost':v['cost'],'LP_scaled_loss':lp['scaled_loss'],'max_closure_m':float(np.linalg.norm(closure,axis=1).max()),
                  'normal_angles_deg':np.degrees(angles),'minimum_geometry_constraint_m':float(min(constraints['values_m']))})
                if purpose=='base':register(v)
                return v
            def register(v):
                nonlocal shortlist
                if not any(np.max(abs((v['x14']-w['x14'])/scale_x))<1e-4 for w in shortlist):
                    shortlist.append(v);shortlist.sort(key=lambda w:w['cost']);shortlist=shortlist[:3]
            def fun(x):return evaluate(x,'base')['residual']
            def jac(x):
                v=evaluate(x,'base');J=np.empty((len(v['residual']),14))
                for j,h in enumerate(CONFIG['FD_steps14']):
                    step=min(h,hi[j]-x[j])
                    if step<1e-9:step=-min(h,x[j]-lo[j])
                    if abs(step)<1e-12:J[:,j]=0;continue
                    z=x.copy();z[j]+=step;actual=z[j]-x[j]
                    J[:,j]=(evaluate(z,'FD')['residual']-v['residual'])/actual
                return J
            try:
                sol=least_squares(fun,x0,jac=jac,bounds=(lo,hi),x_scale=scale_x,
                  max_nfev=100,ftol=1e-10,xtol=1e-10,gtol=1e-8)
                solver={'status':int(sol.status),'message':sol.message,'nfev':sol.nfev,'njev':sol.njev,
                  'optimality':float(sol.optimality),'cost':float(sol.cost),'success':bool(sol.success)}
            except (SearchBudget,RuntimeError) as err:
                if not isinstance(err,SearchBudget) and type(err).__name__!='StaticBudgetExceeded':raise
                # Retain finite shortlist on registered budget stops only, no retry.
                stop=repr(err)
            e.set_budget(deadline_monotonic=time.perf_counter()+60.,max_additional_heavy_queries=3000)
            finals=[]
            for rank,v in enumerate(shortlist):
                phase='final_'+str(seed_index)+'_'+str(rank);x=v['x14'];q=v['q20']
                geo=ev.final_geometry(q,x[8:11],x[11:14],pad_gap_window_m=[-.0001,.0002])
                gain=np.asarray(v['static']['PD_gain20']);offset=np.asarray(v['static']['PD_offset20'])
                u=(np.asarray(v['LP']['tau_act_required_Nm'])-offset)/gain
                legal=bool(np.all(u>=e.m.actuator_ctrlrange[:,0]) and np.all(u<=e.m.actuator_ctrlrange[:,1]))
                native=ev.evaluate(q,x[8:11],x[11:14],u) if legal else None
                current,reason=lp_from_probe(native) if legal else (None,'Raw inverse control outside original bounds; candidate rejected without clipping')
                strict=None
                if current is not None:
                    LP_calls+=1
                    if current['hard_bounds_pass'] and geo['geometry_gate_pass']:
                        pd=native['contact_design']['PD_raw']
                        strict=solve_strict_support(current,pd_gain20=pd['PD_gain20'],pd_offset20=pd['PD_offset20'],
                          ctrl_range20x2=pd['actuator_ctrlrange20x2'])
                        LP_calls+=1
                anchor_gate=bool(np.linalg.norm(v['closure_world_m'],axis=1).max()<=.0002
                  and np.max(np.degrees(v['normal_angles_rad']))<=1.)
                ideal=bool(geo['geometry_gate_pass'] and anchor_gate and strict is not None and strict['strict_force_moment_pass'])
                repeats=[];entry={};state=None
                if ideal:
                    for k in range(2):repeats.append(ev.evaluate(q,x[8:11],x[11:14],strict['proposed_ctrl20']))
                    equal={k:same(repeats[0]['raw'][k],repeats[1]['raw'][k]) for k in ('qpos27','ctrl20','qacc26','qfrc_actuator26','qfrc_constraint26','a_COM_world_m_s2','alpha_world_rad_s2')}
                    equal['physical_contacts']=same(repeats[0]['raw']['contact_rows24'][:,1:],repeats[1]['raw']['contact_rows24'][:,1:])
                    ag=repeats[0]['native_contact_aggregates']
                    entry={'known':all(not z['unknown'] for z in repeats),'repeat_bits':all(equal.values()),
                      'ctrl_unclamped':all(same(z['PD']['requested_ctrl20'],z['PD']['effective_ctrl20']) for z in repeats),
                      'both_pads_loaded':all(ag[k]['normal_force_signed_N']>.1 for k in ('thumb_pad','index_pad')),
                      'initial_pad_budget3N':all(ag[k]['normal_force_signed_N']<=3 for k in ('thumb_pad','index_pad')),
                      'nonpad_table_force_zero':all(ag[k]['normal_force_signed_N']<=1e-8 for k in ('phone_nonpad','phone_table'))}
                    state=out/f'seed{seed_index}_rank{rank}_state.npz'
                    z=repeats[0]['raw'];np.savez_compressed(state,qpos27=z['qpos27'],qvel26=z['qvel26'],ctrl20=z['ctrl20'],x46=np.r_[q,x[8:14],z['ctrl20']])
                record={'seed':seed_index,'rank':rank,'x14':x,'q20':q,'optimizer_candidate':v,
                  'final_geometry':geo,'native_probe':native,'current_native_LP':current,'LP_unavailable_reason':reason,
                  'strict_native_support_LP':strict,'surface_anchor_gate_pass':anchor_gate,
                  'ideal_surface_native_support_candidate':ideal,'fresh_command_probes':repeats,'native_entry':entry,
                  'holding_diagnostic_eligible':bool(ideal and entry and all(entry.values())),
                  'candidate_state':None if state is None else pin(state),'faithful_ego_or_holding_qualified':False}
                path=out/f'seed{seed_index}_rank{rank}.json';write(path,record)
                finals.append({'rank':rank,'result':pin(path),'cost':v['cost'],'geometry_pass':geo['geometry_gate_pass'],
                  'anchor_gate':anchor_gate,'ideal_native_support':ideal,'holding_diagnostic_eligible':record['holding_diagnostic_eligible']})
            seed_results.append({'seed':seed_index,'name':CONFIG['seed_names'][seed_index],'solver':solver,'stop':stop,
              'search_evaluations':search_count,'shortlist':finals})
            print(json.dumps(seed_results[-1]),flush=True)
        audit=ev.audit_model();audit['passed']=True
        write(out/'evaluation_log.json',logs)
        write(out/'report.json',{'schema':'fixed_surface_material_q8_phone6_search_result:8','protocol':pin(a.protocol),
          'status':'COMPLETED_BOUNDED_TWO_SEED_SURFACE_WRENCH_SEARCH','seed_results':seed_results,
          'holding_diagnostic_candidates':sum(c['holding_diagnostic_eligible'] for s in seed_results for c in s['shortlist']),
          'counts':ev.counts,'static_counts':e.counts,'LP_calls':LP_calls,'model_audit':audit,
          'wall_seconds':time.perf_counter()-start,'new_steps_training_GPU':0,
          'scope':'Fixed true triangle material points/normals, q8+phone6 bounded static ideal search; all final contacts/native support independently recomputed. No virtual patch/torsion, calibrated pad, faithful ego trajectory, policy or holding qualification.'})
    except BaseException as err:
        write(out/'evaluation_log_failure.json',logs)
        write(out/'failure.json',{'phase':phase,'error':repr(err),'seed_results':seed_results,
          'counts':None if ev is None else ev.counts,'LP_calls':LP_calls,'automatic_retry':False});raise


if __name__=='__main__':main()
