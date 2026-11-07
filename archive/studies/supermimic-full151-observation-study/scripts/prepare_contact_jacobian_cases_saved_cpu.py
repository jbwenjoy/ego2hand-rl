"""Select source-bound local Jacobian diagnostic cases from saved contacts only."""
from pathlib import Path
import argparse,collections,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/full151_failed_candidate_geometry_v1'
LEGACY=Path('D:/workspace/supermimic-observation-retarget-study/outputs/native_filtered_contact_local_checks_v2/report.json')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=lambda p:{'path':str(p),'sha256':sha(p)}
J=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
def load(p,keys=None):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}

def run(output):
    if output.exists():raise FileExistsError(output)
    receipt=J(RUN/'receipt.json');protocol=J(receipt['protocol']['path']);catalog=load(RUN/'catalog.npz');a=load(RUN/'sampled_native_geometry.npz',['selected_distance_m','generated_mask','tie_count','contact_count'])
    raw=[json.loads(s) for s in (RUN/'native_contacts.jsonl').read_text().splitlines()]
    if sha(RUN/'receipt.json')!='6d217c95479087896bfd7f70c27f8bb4edb5f58facac49aee55d7057481d6607' or len(raw)!=601:raise ValueError('Wrong actual saved geometry')
    ref=load(protocol['inputs']['reference']['path'],['planned_time601_s','planned_q601_rad','object_COM_world_m','object_rotation_w'])
    parent=Path(protocol['inputs']['failed_receipt']['path']).parent;pd=load(parent/'actual_PD_parameters.npz',['q_lower_rad','q_upper_rad'])
    pairs=catalog['pairs'];kind=catalog['pair_kind'];names=catalog['geom_names'];tied=[]
    for row in raw:
        groups=collections.defaultdict(list)
        for v in row['records']:groups[v[0]].append(v)
        for pair,entries in groups.items():
            minimum=min(x[4] for x in entries);tie=[x for x in entries if abs(x[4]-minimum)<=1e-9]
            if len(tie)<2:continue
            p=np.asarray([x[5] for x in tie]);n=np.asarray([np.asarray(x[6])*(1 if (x[2],x[3])==tuple(pairs[pair]) else -1) for x in tie])
            tied.append({'sample':row['sample'],'time_s':row['time_s'],'pair_index':pair,'pair_kind':str(kind[pair]),'geom_names':[str(names[k]) for k in pairs[pair]],'contact_count':len(entries),'diagnostic_tie_count':len(tie),'distance_m':minimum,
                         'canonical_normal_spread_norm':float(np.linalg.norm(n[:,None]-n[None,:],axis=-1).max()),'position_spread_m':float(np.linalg.norm(p[:,None]-p[None,:],axis=-1).max()),'native_contact_indices':[x[1] for x in tie]})
    ranking={k:sorted([v for v in tied if v['pair_kind']==k and v['sample'] not in [17,335]],key=lambda v:(-v['canonical_normal_spread_norm'],-v['position_spread_m'],v['sample'],v['pair_index'])) for k in ['hand_object','hand_self']}
    if (ranking['hand_object'][0]['sample'],ranking['hand_object'][0]['pair_index'])!=(489,243) or (ranking['hand_self'][0]['sample'],ranking['hand_self'][0]['pair_index'])!=(330,193):raise ValueError('Declared saved-case ranking changed')
    declared=[('worst_generated_hand_object',17,228),('worst_generated_hand_self',335,77),('max_tied_normal_spread_hand_object',489,243),('max_tied_normal_spread_hand_self',330,193)]
    cases=[]
    for label,s,p in declared:
        entries=[x for x in raw[s]['records'] if x[0]==p];minimum=min(x[4] for x in entries);ties=[x for x in entries if abs(x[4]-minimum)<=1e-9];selected=min(entries,key=lambda x:x[4]);q=ref['planned_q601_rad'][s]
        cases.append({'case_id':label,'sample':s,'time_s':float(ref['planned_time601_s'][s]),'pair_index':p,'pair_kind':str(kind[p]),'geom_names':[str(names[k]) for k in pairs[p]],'selected_distance_m':minimum,'base_contact_count':len(entries),'base_diagnostic_tie_count':len(ties),'selected_native_contact_index':selected[1],
                      'base_pair_records':entries,'all_sample_native_records':raw[s]['records'],'required_unique_branch_gate':label.startswith('worst_'),'tie_only_directional_diagnostic':not label.startswith('worst_'),
                      'source64_q20':q.tolist(),'qhard_plus_1e_minus5_crossing_joint_indices':np.flatnonzero(q+1e-5>pd['q_upper_rad']).tolist(),'qhard_minus_1e_minus5_crossing_joint_indices':np.flatnonzero(q-1e-5<pd['q_lower_rad']).tolist(),
                      'smallest_hard_margin_rad':float(np.minimum(q-pd['q_lower_rad'],pd['q_upper_rad']-q).min()),'saved_Jacobians_available':False})
    old=J(LEGACY)
    if len(old['checks'])!=56 or not all(old['checks'].values()):raise ValueError('Wrong prior56 source qualification')
    report={'schema':'full151_contact_row_oracle_source_proposal:1','status':'READY_PURE_HELPER_AND_SAVED_CASES_NATIVE_PROBE_PENDING','selection_rule':'Fixed worst HO17/pair228 and worst self335/pair77. For each category separately, excluding those states, select maximum canonical tied-normal spread, then point spread, earliest sample/pair. No FD/Jacobian values were available or used.',
            'source_pins':{'actual601_receipt':pin(RUN/'receipt.json'),'raw_records':pin(RUN/'native_contacts.jsonl'),'catalog':pin(RUN/'catalog.npz'),'selected_geometry':pin(RUN/'sampled_native_geometry.npz'),'reference':protocol['inputs']['reference'],'legacy56_local_report':pin(LEGACY),'native_geometry':protocol['sources']['native_geometry'],'new_row_helper':pin(ROOT/'src/full151/contact_row_oracle_v2.py'),'case_producer':pin(__file__)},
            'saved_geometry_counts':{'samples':601,'catalog_pairs':273,'raw_witnesses':sum(len(x['records']) for x in raw),'diagnostic_tied_pair_sample_rows':len(tied),'tie_samples':int((a['tie_count']>1).any(1).sum())},
            'selected_cases':cases,
            'row_interface':{'function':'full151.contact_row_oracle_v2.geometry_rows','inputs':['B601[:,2:]','sample_index[M]','contactdist_m[M]','actual_Jdist20[M,20]'],'output':'A_step delta_free <= rhs_m with rhs_m=d+.002999, raw metres,3020 joint-major columns j*151+c','all_generated_witnesses':True,'selected_min_or_tie_averaging':False,'basis_fixed_P0_P1_excluded':True},
            'new_query_adapter_requirements':['Perform exactly one native geometry evaluate at each proposed q/object pose, with frozen original MJB options and actual geom_size/pos context. Save all raw native contacts and selected-pair diagnostics.',
                'For each generated witness obtain two real world point Jacobians at the same native contact.pos and its raw geom/body order. ddist/dq=n_AtoB@(J_B-J_A) restricted to robot20 DOFs. Count mj_jac separately. Do not attach the selected-min Jacobian to other witnesses.',
                'Native indices identify only the current query records, not persistent mesh features across perturbations. Keep raw order/normal, canonical-order normal and matching diagnostics separately.',
                'Finite rows or zero columns are not a smoothness or global native-distance certificate. Ties keep all witnesses; they are diagnostic branch evidence, never averaged into a qualified gradient.',
                'Absent pairs stay NaN/UNKNOWN with no contact row; empty rowsets or zero gradients do not certify clearance. Actual trial geometry must be freshly re-queried over the complete grid before accepting a coupled step.'],
            'finite_probe_contract_from_root':{'states':[17,335,489,330],'base_queries':4,'coordinate_joints':20,'epsilon_rad':1e-5,'signs':[-1,1],'perturbation_queries':160,'total_geometry_queries':164,'distance_FD_gate_m_per_rad':.001,
                'unique_cases':'The two worst unique rows require all20 coordinate tests to remain source-branch/normal stable and pass the preregistered local distance gate. Any changed/absent/tied branch is reported unqualified, never dropped to pass.',
                'tie_cases':'Keep central and one-sided changes plus each frozen-witness directional prediction; no global/smooth qualification is assigned to ties.',
                'bound_crossings':'Do not clamp or omit. Record that a coordinate finite-difference outside hard q bounds is static geometry derivative evidence only, not a feasible control/action direction.'},
            'prior56_scope':{'status':old['status'],'check_count':56,'largest_unique_normalstable_distance_FD_error_m_per_rad':max(x['unique_normalstable_distance_max_abs_error_m_per_rad'] for c in old['case_rows'] for x in c['FD']),
                             'retained_allcommon_failure_example_m_per_rad':old['case_rows'][0]['FD'][0]['all_common_distance_max_abs_error_m_per_rad'],'warning':'Those four old geometric states were local unique-branch tests. They did not qualify the present601-row set, allcontact witnesses, ties, geometry switches or a physical gradient.'},
            'native_probe_executed_by_this_producer':False,'real_FK_collision_solver_Env_WM_physics_GPU_calls':0,'physical_reference_qualified':False,'contact_target_ready':False}
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'case_arrays.npz',sample_index=np.array([x[1] for x in declared]),pair_index=np.array([x[2] for x in declared]),time_s=ref['planned_time601_s'][[x[1] for x in declared]],source64_q20=ref['planned_q601_rad'][[x[1] for x in declared]],object_COM_world_m=ref['object_COM_world_m'][[x[1] for x in declared]],object_rotation_w=ref['object_rotation_w'][[x[1] for x in declared]])
    (output/'all_tied_pair_diagnostics.json').write_text(json.dumps({'rows':tied,'rankings':ranking},indent=2),encoding='utf-8')
    (output/'proposal.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    ready={'schema':'full151_contact_row_oracle_source_READY:1','status':report['status'],'proposal':pin(output/'proposal.json'),'case_arrays':pin(output/'case_arrays.npz'),'all_ties':pin(output/'all_tied_pair_diagnostics.json'),'helper':pin(ROOT/'src/full151/contact_row_oracle_v2.py'),'pure23_proof':pin(ROOT/'outputs/contact_row_oracle_CPU_v1/report.json'),'units6_delta_proof':pin(ROOT/'outputs/contact_row_oracle_units_CPU_v2/report.json'),'real_native_probe_authorized_by_this_receipt':False}
    (output/'READY.json').write_text(json.dumps(ready,indent=2),encoding='utf-8');return ready
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();print(json.dumps(run(args.output),indent=2))
