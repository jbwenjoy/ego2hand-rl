"""Saved operands only; reviewer authored the solver/helper, Root executed geometry.

No FK, geometry query, Jacobian, optimizer, native/model load, physics or NN.
The frozen v1 saved-arithmetic helpers are reused without invoking their audit.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, math, sys
import numpy as np

H=Path(__file__).resolve().parent
P=H.parent.parent
OLD=H.parent/'saved_arithmetic_review_v1/audit_saved_compression_v1.py'
assert hashlib.sha256(OLD.read_bytes()).hexdigest()=='0a5809ea9f4499bf4aeb8cd159e87b0293c768d3dda2836e1f6859f9ab78d5c6'
sp=importlib.util.spec_from_file_location('frozen_compression_saved_arithmetic',OLD)
B=importlib.util.module_from_spec(sp);sp.loader.exec_module(B)
A=B.A
read,pin,checked,bits,close,require,write=B.read,B.pin,B.checked,B.bits,B.close,B.require,B.write

def audit(path):
    require(pin(path)['sha256']=='b58c29f433901efaf2c2e671cfa3902dd78621cad16275b927eb6d37905e8cdf','Root terminal report pin')
    r=read(path);p=read(checked(r['protocol']))
    require(r['schema']=='functional_tripod_geometric_reference_result:1' and r['status']=='COMPLETED_EXPLICIT_SUBSET_TRIPOD_COMPRESSION_TARGET_SEARCH','Completed explicit subset result')
    for item in [*p['sources'].values(),*p['inputs'].values(),r['model_mjb']]:checked(item)
    oldr=read(checked(r['retained_v1_report']));oldp=read(checked(oldr['protocol']))
    failed=read(checked(r['retained_v1_numeric_boundary_candidate']))
    require(r['retained_v1_report']==p['inputs']['retained_compression_v1_report'] and r['retained_v1_numeric_boundary_candidate']==p['inputs']['retained_numeric_boundary_candidate'],'Retained false result exact pins')
    require(p['hard_geometry']==oldp['hard_geometry'] and p['budget']==oldp['budget'],'No hard gate or budget changes')
    require(p['compression_target_window_m']==oldp['compression_target_window_m']==[-.00009,-.00006] and p['compression_target_center_m']==oldp['compression_target_center_m']==-.000075,'Same separate compression target')
    expected_proposal=dict(oldp['proposal']);expected_proposal['interior_target_margin_m']=2e-6
    require(p['proposal']==expected_proposal,'Only proposal interior becomes2um')
    parent=read(checked(r['parent_report']));eligible=[x for x in parent['candidates'] if x['geometry_pass']]
    require(len(eligible)==2 and len(r['candidates'])==len(r['summaries'])==len(p['selected_parent_candidates'])==1,'One explicit subset of original two eligible')
    selected=p['selected_parent_candidates'][0]
    require(selected in [x['candidate'] for x in eligible] and r['explicit_selected_original_eligible_subset']==[selected],'Original eligible selected pin')
    rec=r['candidates'][0];c=read(checked(rec['candidate']));old=read(checked(selected));summary=r['summaries'][0]
    require(c['parent_geometric_candidate']==selected and c['source_entry']==old['source_entry']==failed['source_entry'],'Source entry and original parent binding')
    require(c['candidate_id']==rec['candidate_id']==summary['candidate_id']==old['source_entry']['name']+'__compression_target_projection','String candidate identity')
    require(c['parent_geometric_candidate']!=r['retained_v1_numeric_boundary_candidate'],'Failed state is provenance only, not warmstart')
    data=B.loadpath(path.parent/'compiled_geometry_and_limits.npz')
    parentdata=B.loadpath(Path(r['parent_report']['path']).parent/'compiled_geometry_and_limits.npz')
    require(set(data)==set(parentdata),'Identical compiled export fields')
    for k in data:bits(data[k],parentdata[k],'Compiled geometry/limits preserved '+k)
    q=np.array(c['qpos27_F64']);v=np.array(c['qvel26_F64']);x=np.array(c['x18']);oq=np.array(old['qpos27_F64']);ox=np.array(old['x18'])
    z=A.load(rec['state_npz']);require(set(z)=={'qpos27_F64','qvel26_F64'},'Exact two-key saved state')
    bits(z['qpos27_F64'],q,'NPZ qpos');bits(z['qvel26_F64'],v,'NPZ qvel');bits(v,np.zeros(26),'Zero speed')
    bits(q[12:],oq[12:],'Phone qpos and other8 fixed bytes');bits(x[12:],ox[12:],'Phone parameter bytes fixed');bits(q[:12],x[:12],'q12 ordering')
    qr=data['jnt_range'][:20];require(np.all(q[:20]>=qr[:,0]) and np.all(q[:20]<=qr[:,1]),'Actual joint hard ranges')
    m,names=B.margins(c);fm,fnames=B.margins(failed);g=c['geometry'];s=g['source_full_state']
    close(g['constraint_values'][:214],m,'Recomputed original214 margins');require(g['constraint_labels'][:214]==names,'Original214 names')
    require(g['fresh_final_evaluation'] is True and g['unknown']==[] and g['table_complete'] is True and g['phone_pair_count']==22 and g['self_pair_count']==189,'Fresh full22/189/table and no unknown')
    source=read(checked(c['source_record']));sq=np.array(source['native_probe']['raw']['qpos27']);R=A.rotation(q[23:27]);R0=A.rotation(sq[23:27]);pos=q[20:23]
    dr=np.linalg.norm(pos-sq[20:23]);angle=np.linalg.norm(A.log3(R@R0.T))
    close(g['constraint_values'][214:],[.035-dr,math.pi/6-angle],'Unchanged named phone trusts')
    hard=bool(np.all(np.array(g['constraint_values'])>=0));tm=B.target_margins(c);target=bool(np.all(tm>=0))
    require(hard==g['passed']==rec['geometry_pass']==summary['geometry_pass']==True,'Original strict>=0 geometry pass')
    require(target==c['compression_target']['passed']==c['compression_target_pass']==rec['compression_target_pass']==summary['compression_target_pass']==True,'Separate target pass')
    close(c['compression_target']['guarded_margins6_m'],tm,'Recomputed compression target6')
    require(c['geometry_and_compression_target_pass'] is True and c['native_loaded_or_holding_qualified'] is False and r['holding_or_ego_retarget_qualified'] is False,'No load/holding conflation')
    require(float(fm.min())<0 and failed['geometry']['passed'] is False and failed['compression_target_pass'] is True and sum(fm<0)==1,'v1 tiny negative remains false')
    nonpad=[(row['lower_bound_m'],row['name']) for row in s['phone_geometry_rows'] if row['geom_id'] not in [15,28,41]]
    nmin,nname=min(nonpad);require(nmin>0 and nmin<2e-6,'Positive actual hard margin,2um remains search target not gate')
    rows={row['geom_id']:row for row in s['phone_geometry_rows']};gaps=[]
    for b in c['pad_bindings']:
        gid=b['hand_geom_id'];hp,hn=B.surface(b['hand_surface'],data[f'geom{gid}_vertices'],'Hand'+str(gid))
        pp,pn=B.surface(b['phone_surface'],data['phone_body_vertices'],'Phone'+str(gid))
        bits(b['geom_local_hand_anchor_m'],hp,'Filled hand material binding');bits(b['phone_body_material_point_m'],pp,'Filled phone material binding')
        query=rows[gid]['query_phone_frame'];nb=np.array(query['normal_world']);hs=np.array(query['hand_support_points_world_m']);ps=np.array(query['object_support_points_world_m'])
        close(nb@(hs.mean(0)-ps.mean(0)),rows[gid]['exact_gap_m'],'Saved support-plane gap sign')
        close(b['force_on_object_unit_normal_world'],-R@nb,'Object force direction sign')
        close(b['query_gap_m'],rows[gid]['exact_gap_m'],'Actual query gap in binding');gaps.append(rows[gid]['exact_gap_m'])
    close(gaps,c['compression_target']['exact_gaps3_m'],'Actual three query gaps, not target constants')
    log=c['projection_iteration_log'];require(len(log)==1,'One QP iteration retained')
    step=log[0];dq=np.array(step['proposed_dq12']);require(np.max(abs(dq))<=.02 and step['accepted'] is True,'Actual proposal step limit and acceptance')
    require(len(step['line_search'])==1 and step['line_search'][0]['fraction']==1.,'First full line-search accepted')
    close(ox[:12]+dq,q[:12],'Accepted q path reconstructed');close(step['initial_violation'],B.merit(old),'Original initial merit')
    close(step['initial_target_margins6_m'],B.target_margins(old),'Original initial target operands')
    close(summary['initial_violation'],B.merit(old),'Summary initial merit');close(summary['final_accepted_violation'],B.merit(c),'Summary final merit')
    require(step['scope']['unmodelled_self_pairs']==189 and step['scope']['unmodelled_entries_are_zero_in_proposal_matrix_only'] is True,'Self Jacobian heuristic scope')
    events=[json.loads(line) for line in (path.parent/(old['source_entry']['name']+'_evaluations.jsonl')).read_text().splitlines()]
    require(len(events)==4 and events[-1]['stage']=='FINAL','Four full query records with fresh final')
    for i,e in enumerate(events):
        bits(e['x18'],ox if i<2 else x,'Recorded initial/final actual x18')
        require(e['counts']['full_geometry_evaluations']==i+1 and e['unknown']==[],'Monotone complete query records')
    require(not np.array_equal(events[0]['x18'],failed['x18']),'Actual first evaluation not v1 failed warmstart')
    require(r['counts']['full_geometry_evaluations']==r['static_geometry_counts']['evaluations']==4 and r['counts']['separating_plane_point_jac_calls']==2*len(step['scope']['modelled_phone_geoms'])==14,'Full and helper counters')
    require(r['geometric_QP_calls']==r['accepted_projection_steps']==r['line_search_full_geometry_calls']==1 and r['qualified_geometry_candidates']==r['compression_target_candidates']==r['geometry_and_compression_candidates']==1,'One full accepted proposal and eligible candidate')
    require(r['counts']['MJB_reads']==r['counts']['entries_returned']==r['counts']['entries_attempted']==1 and r['counts']['cheap_FK']==r['counts']['cheap_comPos']==5,'Source execution workload')
    for k in ['mj_forward','integration_steps','LP_calls','NN_training_GPU']:require(r['counts'][k]==0,'Zero executed dynamic/NN '+k)
    termpath=path.parent.parent/'process_v2_run1/terminal.json';term=read(termpath)
    require(term['returncode']==0 and term['protocol']==r['protocol'] and term['source']==p['sources']['runner'],'Actual terminal source/protocol join')
    checked(term['stdout']);checked(term['stderr']);require(term['stderr']['bytes']==0,'No stderr')
    require(not any(k in sys.modules for k in ['mujoco','warp','torch']),'Saved review no simulator or NN import')
    checks={k:True for k in ['source_protocol_and_retained_v1_pins','explicit_original_eligible_subset_no_failed_warmstart','original214_and_phone_trust_recomputed','compression6_separate_from_hard_gate','v1_negative_failure_retained_v2_positive','fixed_phone_rest8_true_joint_bounds','filled_triangle_binding_gap_and_normal_sign','single_q12_proposal_and_full_final_records','four_full_fourteen_Jac_one_QP_counters','terminal_and_zero_new_native_calls']}
    out=dict(schema='functional_tripod_compression_saved_arithmetic:2',status='PASS_SAVED_EXPLICIT_SUBSET_NUMERIC_FEASIBILITY_ARITHMETIC',engineering_qualified=True,producer_report=pin(path),protocol=r['protocol'],checks=checks,passed=len(checks),total=len(checks),
        candidate=rec,candidate_id=c['candidate_id'],original_eligible_parent=selected,retained_v1_report=r['retained_v1_report'],retained_v1_candidate=r['retained_v1_numeric_boundary_candidate'],
        original_hard_geometry_pass=hard,compression_target_pass=target,pad_exact_gaps3_m=gaps,minimum_original214_margin_m=float(m.min()),minimum_nonpad_margin_m=float(nmin),minimum_nonpad_name=nname,
        interior_search_target_m=2e-6,interior_is_not_new_hard_gate=True,retained_v1_minimum_margin_m=float(fm.min()),retained_v1_failing_label=fnames[int(np.argmin(fm))],retained_v1_geometry_pass=False,
        fixed_phone_and_rest8_rawbytes=True,max_single_q12_step_rad=float(np.max(abs(dq))),middle_q_change4_rad=(q[8:12]-oq[8:12]).tolist(),counts=r['counts'],
        source_pins=dict(auditor=pin(Path(__file__)),frozen_saved_arithmetic=pin(OLD),source_freeze=pin(H/'SOURCE_FREEZE_v1.json')),terminal=pin(termpath),max_F64_arithmetic_errors=dict(A.ERRORS),
        independence_scope='Reviewer authored compression source and plane helper; Root executed the geometry search. Saved arithmetic only: no independent FK/query/Jac replay. Saved exact bounds and world hand FK remain source-bound; partial model parameter digest is not a full model hash.',
        actual_new_model_FK_query_Jac_LP_physics_NN_calls=0,native_loaded_or_holding_qualified=False)
    write(H/'report.json',out);return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
    require(not (H/'report.json').exists() and not (H/'failure.json').exists(),'Fresh once saved audit')
    try:
        r=audit(a.report);print(json.dumps(dict(report=pin(H/'report.json'),passed=r['passed'],total=r['total'],minimum_nonpad_margin_m=r['minimum_nonpad_margin_m'],pad_exact_gaps3_m=r['pad_exact_gaps3_m'])))
    except BaseException as e:
        write(H/'failure.json',dict(error=repr(e),source=pin(Path(__file__)),producer_report=pin(a.report),automatic_retry=False,new_native_calls=0));raise

if __name__=='__main__':main()
