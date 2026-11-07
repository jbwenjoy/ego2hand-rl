"""Saved projection arithmetic only. Reviewer authored plane helper, not runner.

Does not execute FK, distance queries, Jacobians, optimizers, forces or NN.
Saved query bounds are not re-certified by a new query; they are source-bound.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,sys
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent.parent
MATH=P/'phone_episode_acquisition_v1/independent_dataset_review_v1/audit_saved_dataset_v1.py'
assert hashlib.sha256(MATH.read_bytes()).hexdigest()=='3af410ac90fba95b8eb0774706747daa301e3ce9938999016b04549d786be1cf'
sp=importlib.util.spec_from_file_location('independent_saved_math',MATH);A=importlib.util.module_from_spec(sp);sp.loader.exec_module(A)
read=A.read;pin=A.pin;checked=A.checked;bits=A.bits;close=A.close;require=A.require;write=A.write
def loadpath(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def margins(c):
    g=c['geometry'];s=g['source_full_state'];v=[];names=[]
    require(len(s['phone_geometry_rows'])==22 and len(s['self_geometry_rows'])==189,'All22phone/189self saved rows')
    require(len({x['geom_id'] for x in s['phone_geometry_rows']})==22 and any(x['geom_id']==0 for x in s['phone_geometry_rows']),'Unique phone pairs/table')
    for row in s['phone_geometry_rows']:
        if row['geom_id'] in [15,28,41]:
            require(row['status']=='EXACT_NUMERICAL_IDEAL_CONVEX','All3 designated tip gaps exact source query')
            v.extend([row['lower_bound_m']+.0001,.0002-row['upper_bound_m']]);names.extend([row['name']+'_pad_lower',row['name']+'_pad_upper'])
        else:v.append(row['lower_bound_m']);names.append(row['name']+'_nonpad')
    for row in s['self_geometry_rows']:v.append(row['lower_bound_m']);names.append('/'.join(row['names'])+'_self')
    require(len(v)==214 and np.isfinite(v).all(),'Exact214 saved geometric inequalities')
    return np.asarray(v),names
def surface(surf,V,label):
    vid=np.asarray(surf['triangle_vertex_ids']);tri=np.asarray(surf['triangle_vertices_local_m']);w=np.asarray(surf['barycentric']);point=np.asarray(surf['point_local_m']);n=np.asarray(surf['outward_unit_normal_local'])
    require(vid.shape==(3,) and np.issubdtype(vid.dtype,np.integer) and np.all(vid>=0) and np.all(vid<len(V)),label+' real vertex indices')
    bits(tri,V[vid],label+' actual compiled triangle vertices')
    close(w@tri,point,label+' filled-triangle bary reconstruction');require(min(w)>=-1e-10 and abs(w.sum()-1)<1e-10,label+' bary domain')
    close(np.linalg.norm(n),1.,label+' unit normal')
    require(float(np.max((V-point)@n))<1e-10,label+' convex outward supporting normal')
    close(.5*np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0])),surf['triangle_area_m2'],label+' triangle area')
    return point,n
def audit(report_path):
    r=read(report_path);require(r['schema']=='functional_tripod_geometric_reference_result:1' and r['status']=='COMPLETED_FOUR_SAVED_STATE_PHONE_PLANE_PROJECTIONS','Completed projection result')
    p=read(checked(r['protocol']));require(p['schema']=='functional_tripod_saved_geometry_plane_projection_protocol:1','Frozen projection protocol')
    for item in [*p['sources'].values(),*p['inputs'].values()]:checked(item)
    parent=read(checked(r['parent_report']));require(r['parent_report']==p['inputs']['original_geometry_report'],'Parent four failures bound')
    require(len(r['candidates'])==len(parent['candidates'])==len(r['summaries'])==4,'All4 pass/failure records retained')
    data=loadpath(report_path.parent/'compiled_geometry_and_limits.npz')
    original_data=loadpath(Path(r['parent_report']['path']).parent/'compiled_geometry_and_limits.npz')
    require(set(data)==set(original_data),'Same compiled geometry/range export')
    for key in data:bits(data[key],original_data[key],'Unchanged saved compiled model operand '+key)
    acquired=read(P/'phone_episode_acquisition_v1/acquired_v1/report.json');ipos=np.array(acquired['model_audits']['0.0003125']['body_ipos_body_m'])
    qrange=data['jnt_range'][:20];all_results=[];full_logs=0;qps=accepted=lines=jac_calls=0
    for index,(rec,oldrec,summary) in enumerate(zip(r['candidates'],parent['candidates'],r['summaries'])):
        c=read(checked(rec['candidate']));old=read(checked(oldrec['candidate']));require(c['parent_geometric_candidate']==oldrec['candidate'],'Explicit parent candidate')
        require(c['candidate_id']==old['candidate_id']+'__fixed_phone_q_projection'==rec['candidate_id']==summary['candidate_id'],'Candidate identity')
        q=np.asarray(c['qpos27_F64']);v=np.asarray(c['qvel26_F64']);x=np.asarray(c['x18']);oldx=np.asarray(old['x18']);oldq=np.asarray(old['qpos27_F64'])
        z=A.load(rec['state_npz']);require(set(z)=={'qpos27_F64','qvel26_F64'},'Exact saved state arrays')
        bits(z['qpos27_F64'],q,'Candidate NPZ qpos');bits(z['qvel26_F64'],v,'Candidate NPZ qvel');bits(v,np.zeros(26),'Zero initial speed')
        bits(q[12:20],oldq[12:20],'Other8 joint positions fixed');bits(q[20:27],oldq[20:27],'Phone free pose bytes fixed')
        bits(x[12:],oldx[12:],'Phone18-vector parameter bytes fixed');bits(q[:12],x[:12],'Actual q0..11 variable ordering')
        require(np.all(q[:20]>=qrange[:,0]) and np.all(q[:20]<=qrange[:,1]),'True joint bounds without clamp')
        m,names=margins(c);mold,_=margins(old);g=c['geometry'];s=g['source_full_state']
        require(g['fresh_final_evaluation'] is True and g['unknown']==[] and g['table_complete'] is True,'Fresh final full/known/table')
        require(g['phone_pair_count']==22 and g['self_pair_count']==189,'Complete explicit geometry counts')
        close(g['constraint_values'][:214],m,'Independent214 gap margins');require(g['constraint_labels'][:214]==names,'Margin labels')
        source=read(checked(c['source_record']));sq=np.asarray(source['native_probe']['raw']['qpos27']);R=A.rotation(q[23:27]);R0=A.rotation(sq[23:27]);pos=q[20:23]
        dr=float(np.linalg.norm(pos-sq[20:23]));angle=float(np.linalg.norm(A.log3(R@R0.T)))
        close(g['constraint_values'][214:],[.035-dr,math.pi/6-angle],'Named functional phone trusts')
        close(c['functional_phone_pose_deviation']['translation_m'],dr,'Phone translation deviation');close(c['functional_phone_pose_deviation']['angle_rad'],angle,'Phone angle deviation')
        passed=bool(np.all(np.asarray(g['constraint_values'])>=0))
        require(passed==g['passed']==rec['geometry_pass']==summary['geometry_pass'],'All4 geometry result flags recomputed')
        require(c['designated_geom_ids']==[15,28,41] and len(c['pad_bindings'])==3,'Actual three designated tips')
        Ahand=[];nhand=[];Bphone=[];nphone=[];anchor_to_phone=[];qgap=[]
        bygeom={row['geom_id']:row for row in s['phone_geometry_rows']}
        for j,b in enumerate(c['pad_bindings']):
            gid=[15,28,41][j];require(b['hand_geom_id']==gid,'Pad binding identity')
            hp,hn=surface(b['hand_surface'],data[f'geom{gid}_vertices'],'hand'+str(gid))
            pp,pn=surface(b['phone_surface'],data['phone_body_vertices'],'phone'+str(gid))
            nearest=b['phone_nearest_to_hand_anchor_surface'];npnt,nn=surface(nearest,data['phone_body_vertices'],'nearestPhone'+str(gid))
            bits(b['geom_local_hand_anchor_m'],hp,'Geom-local hand point ABI');bits(b['phone_body_material_point_m'],pp,'Body material point ABI');bits(b['phone_body_material_outward_normal'],pn,'Body material normal ABI')
            aw=np.array(b['hand_anchor_world_m']);require(np.isfinite(aw).all(),'Finite world material point')
            close(aw,s['anchors_world_m'][j],'Bindings/world anchor export consistency')
            Ahand.append(aw);nhand.append(np.array(b['hand_outward_unit_normal_world']));Bphone.append(R@pp+pos);nphone.append(R@pn)
            body_anchor=R.T@(aw-pos);close(nearest['projection_distance_m'],np.linalg.norm(body_anchor-npnt),'Point-to-filled-phone projection distance')
            anchor_to_phone.append(float(np.linalg.norm(body_anchor-npnt)))
            row=bygeom[gid];query=row['query_phone_frame'];nbody=np.array(query['normal_world']);nw=R@nbody
            close(b['force_on_object_unit_normal_world'],-nw,'Force-on-object sign relative hand-exit query')
            close(np.linalg.norm(b['geom_local_force_on_object_normal']),1.,'Local force direction unit')
            Pset=np.array(query['object_support_points_world_m']);Hset=np.array(query['hand_support_points_world_m'])
            supportgap=float(nbody@(Hset.mean(0)-Pset.mean(0)));close(supportgap,row['exact_gap_m'],'Saved support-plane gap sign arithmetic')
            close(b['query_gap_m'],row['exact_gap_m'],'Binding actual query gap')
            qgap.append(float(row['exact_gap_m']))
        Ahand=np.array(Ahand);nhand=np.array(nhand);Bphone=np.array(Bphone);nphone=np.array(nphone)
        closure=Bphone-Ahand+.000025*nhand;normal_angle=np.degrees(np.arccos(np.clip(-np.sum(nhand*nphone,axis=1),-1,1)))
        close(closure,c['surface_alignment']['closure3x3_m'],'Three material closure descriptors');close(normal_angle,c['surface_alignment']['normal_angles_deg'],'Three normal alignment descriptors')
        COM=pos+R@ipos;lever=Ahand-COM
        close(lever,c['lever_descriptors']['hand_anchor_to_COM_world3x3_m'],'Three actual anchor COM lever arms')
        area=.5*np.linalg.norm(np.cross(Ahand[1]-Ahand[0],Ahand[2]-Ahand[0]));close(area,c['lever_descriptors']['three_anchor_triangle_area_m2'],'Triangle area descriptor')
        for j,mat in enumerate(c['source_material_deviation']):
            ego=np.array(mat['original_ego_material_target_body_m']);actual=R.T@(Ahand[j]-pos)
            close(actual,mat['actual_hand_anchor_phone_body_m'],'Ego material body coordinate');close(np.linalg.norm(actual-ego),mat['anchor_to_original_ego_target_distance_m'],'Ego material point deviation')
            close(np.linalg.norm((R-R0)@ego+pos-sq[20:23]),mat['fixed_material_world_shift_from_source_m'],'Functional source material world shift')
        current=oldx[:12].copy();current_violation=float(np.linalg.norm(np.minimum(mold/.001,0)));log=c['projection_iteration_log']
        accepted_entry=0;line_entry=0;qp_entry=0
        for step in log:
            if 'QP_status' not in step:require(False,'Unexpected projection phase error '+str(step))
            qp_entry+=1;dq=np.array(step['proposed_dq12']);require(dq.shape==(12,) and np.max(abs(dq))<=p['proposal']['joint_step_trust_rad']+1e-12,'Every proposed true q step within0.02rad')
            close(step['initial_violation'],current_violation,'Periteration current merit')
            require(step['scope']['unmodelled_self_pairs']==189 and step['scope']['unmodelled_entries_are_zero_in_proposal_matrix_only'] is True,'No zero-self physical derivative claim')
            jac_calls+=2*len(step['scope']['modelled_phone_geoms'])
            ls=step['line_search'];line_entry+=len(ls)
            require(len(ls)<=len(p['proposal']['line_search_fractions']),'Finite line search budget')
            for i,trial in enumerate(ls):require(trial['fraction']==p['proposal']['line_search_fractions'][i],'Fixed ordered line-search fractions')
            if step['accepted']:
                require(bool(ls),'Accepted step has full check');chosen=ls[-1]
                require(chosen['full_geometry_pass'] or chosen['full_violation']<current_violation-1e-9,'Accepted only from fresh full merit/pass')
                current=current+chosen['fraction']*dq;current_violation=chosen['full_violation'];accepted_entry+=1
            else:require(not any(t['full_geometry_pass'] or t['full_violation']<current_violation-1e-9 for t in ls),'No skipped favorable trial/replacement')
        close(current,q[:12],'Reconstructed true accepted q0..11 trajectory')
        close(current_violation,float(np.linalg.norm(np.minimum(m/.001,0))),'Final accepted geometric merit')
        close(summary['initial_violation'],float(np.linalg.norm(np.minimum(mold/.001,0))),'Parent initial merit');close(summary['final_accepted_violation'],current_violation,'Reported final merit')
        require(summary['accepted_iterations']==accepted_entry and summary['phase_error'] is None,'Accepted count/no programming error concealed')
        qps+=qp_entry;lines+=line_entry;accepted+=accepted_entry
        evpath=report_path.parent/(old['candidate_id']+'_evaluations.jsonl');events=[json.loads(x) for x in evpath.read_text().splitlines()]
        require(events and events[-1]['stage']=='FINAL','Last actual query is final, not a search cache')
        close(events[-1]['x18'],x,'Fresh final query actual pose')
        require(events[-1]['geometry_passed']==passed,'Fresh final status')
        full_logs+=len(events)
        all_results.append(dict(candidate_id=c['candidate_id'],candidate=rec['candidate'],geometry_pass=passed,unknown=g['unknown'],
            minimum_214_margin_m=float(m.min()),violations=[dict(label=n,margin_m=float(z)) for n,z in zip(names,m) if z<0],
            full_queries=len(events),QP_calls=qp_entry,accepted_iterations=accepted_entry,line_search_calls=line_entry,
            max_total_q12_change_rad=float(np.max(abs(q[:12]-oldq[:12]))),middle_q_change4_rad=(q[8:12]-oldq[8:12]).tolist(),
            fixed_phone_qrest8_exact=True,pad_exact_gaps3_m=qgap,closure_norms3_m=np.linalg.norm(closure,axis=1).tolist(),normal_angles3_deg=normal_angle.tolist(),
            material_hand_anchor_phone_distance3_m=anchor_to_phone,phone_translation_from_v11_m=dr,phone_angle_from_v11_rad=angle,
            ego_material_deviation2_m=[x['anchor_to_original_ego_target_distance_m'] for x in c['source_material_deviation']],anchor_triangle_area_m2=float(area),
            holding_qualified=False))
    require(full_logs==r['counts']['full_geometry_evaluations']==r['static_geometry_counts']['evaluations']==27,'All27 real full query logs')
    require(qps==r['geometric_QP_calls']==8 and lines==r['line_search_full_geometry_calls']==16 and accepted==r['accepted_projection_steps']==6,'All QP/line/accepted counters')
    require(jac_calls==r['counts']['separating_plane_point_jac_calls']==112,'All112 read-only helper point-J calls')
    require(sum(x['geometry_pass'] for x in all_results)==r['qualified_geometry_candidates']==2,'Two passes /two failures preserved')
    require(r['counts']['MJB_reads']==1 and r['counts']['cheap_FK']==r['counts']['cheap_comPos']==31 and r['counts']['entries_returned']==r['counts']['entries_attempted']==4,'Static workload metadata')
    for k in ['mj_forward','integration_steps','NN_training_GPU','LP_calls']:require(r['counts'][k]==0,'No new dynamic/NN/forceLP '+k)
    termpath=report_path.parent.parent/'process_run1/terminal.json';terminal=read(termpath)
    require(terminal['returncode']==0 and terminal['protocol']==r['protocol'],'Actual once terminal/protocol')
    for key in ['stdout','stderr']:checked(terminal[key])
    require(terminal['stderr']['bytes']==0 and not any(x in sys.modules for x in ['mujoco','torch','warp']),'Saved-only review no native/NN import')
    checks={k:True for k in ['all4_source_parent_states_and_pins','full214_geometry_and2_trust_margins_recomputed','two_passes_fresh_final_with_failures_kept',
        'all_proposed_q12_steps_and_actual_accepted_path','phone_pose_qrest8_and_real_limits_unchanged','compiled_filled_triangle_bary_support_binding',
        'closure_normals_querygap_and_force_sign_arithmetic','COM_levers_and_original_ego_material_deviations','all_static_query_Jac_QP_counts_and_no_dynamics','terminal_source_and_helper_scope_disclosed']}
    result=dict(schema='functional_tripod_projection_saved_independent:1',status='PASS_SAVED_PROJECTION_ARITHMETIC_TWO_GEOMETRY_PASSES_NOT_HOLDING',
        engineering_qualified=True,producer_report=pin(report_path),protocol=r['protocol'],parent_report=r['parent_report'],checks=checks,passed=len(checks),total=len(checks),candidates=all_results,
        source_pins=dict(auditor=pin(Path(__file__)),frozen_independent_math=pin(MATH),source_freeze=pin(H/'SOURCE_FREEZE_v1.json')),
        actual_terminal=pin(termpath),max_F64_arithmetic_errors=dict(A.ERRORS),
        independence_scope='Reviewer did not author or execute Root projection runner; reviewer authored its plane helper. Independent saved arithmetic here, no independent geometry/FK/Jac/reintegration. World hand FK and exact-query bounds remain source-bound; no full-model hash claim from partial digest.',
        qualification=dict(dynamic_holding=False,real_ego=False,functional_pad_truth=False,point_LP_feasibility_required=False),
        actual_new_model_FK_geometry_queries_Jac_LP_physics_NN_training_calls=0)
    write(H/'report.json',result);return result
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
    require(not (H/'report.json').exists() and not (H/'failure.json').exists(),'Fresh once saved assessment')
    try:
        r=audit(a.report);print(json.dumps(dict(report=pin(H/'report.json'),passed=r['passed'],total=r['total'],geometry_passes=sum(x['geometry_pass'] for x in r['candidates']))))
    except BaseException as err:
        write(H/'failure.json',dict(error=repr(err),source=pin(Path(__file__)),producer_report=pin(a.report),automatic_retry=False,new_native_calls=0));raise
if __name__=='__main__':main()
