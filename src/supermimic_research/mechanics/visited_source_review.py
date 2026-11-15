"""Independent saved-source, EFC and repeated original-candidate replay review."""
import contextlib
import json
import subprocess
import tomllib
from pathlib import Path

import mujoco
import numpy as np

from ..artifacts import file_pin,verify_file_pin
from ..frozen_sources import freeze_sources,verify_source_snapshot
from ..runtime.controlled_evaluation import write
from .warp_saved_review import review_saved_warp_contacts
from .pen_replay_fidelity import compare_response

ROOT=Path(__file__).resolve().parents[3]


def graph_trace(packet,index):
    return {key:packet[val][index].astype(float) for key,val in [('post_qpos','Q'),('post_qvel','V'),('post_ctrl','U'),('post_GPU_time','T')]}


def patterns(z,tick,m,body):
    rows=z['raw_contact_tick_plus23'];selected=np.flatnonzero(rows[:,0].astype(int)==tick);pairs={}
    for i in selected:
        row=rows[i];g,h=map(int,row[1:3])
        if (m.geom_bodyid[g]==body)==(m.geom_bodyid[h]==body):continue
        pair=tuple(sorted((g,h)));dim=int(row[3]);n=1 if dim==1 else 2*(dim-1)
        ids=z['GPU_contact_efc_addresses'][i,:n];edges=0
        if ids[0]>=0:
            lo,hi=z['GPU_efc_ranges'][tick:tick+2];force=z['GPU_efc_force_flat'][lo:hi]
            if np.any(ids>=len(force)) or np.any(ids<0):raise ValueError('Original active constraint addresses required')
            edges=int(np.count_nonzero(force[ids]>0))
        counts=pairs.setdefault(pair,[0,0,0]);counts[0]+=1;counts[1]+=int(row[18]>0);counts[2]+=edges
    return [[*pair,*counts] for pair,counts in sorted(pairs.items())]


def main(a):
    r=json.loads(a.report.read_text())
    if r['schema']=='pen_matched_amplitude_sources:1':
        from .matched_source_review import main as matched_main
        return matched_main(a,r)
    cfg=tomllib.loads(verify_file_pin(r['config']).read_text());cp=verify_file_pin(r['config'])
    gates=tomllib.loads((cp.parent/cfg['gates']).read_text());original=json.loads(verify_file_pin(r['original_run']).read_text())
    old_review=json.loads(verify_file_pin(r['original_review']).read_text());old_efc=json.loads(verify_file_pin(old_review['EFC']).read_text())
    if a.output.exists():raise ValueError('Fresh independent visited-source review')
    verify_source_snapshot(json.loads(verify_file_pin(r['source_receipt']).read_text()))
    for pin in r['sources']:verify_file_pin(pin)
    if 'installed_source_receipt' in r:verify_source_snapshot(json.loads(verify_file_pin(r['installed_source_receipt']).read_text()))
    profile=r.get('ccd_profile','ORIGINAL_RECONSTRUCTION')
    profile_options=r.get('collision_numerics')
    old_reference_differs=profile=='TIGHT'
    a.output.mkdir();receipt=freeze_sources(sorted((ROOT/'src/supermimic_research').rglob('*.py'))+[ROOT/'tools/review_pen_visited_sources.py'],
        a.output/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write(a.output/'source_receipt.json',receipt)
    expected={(c['name'],c['dt']) for c in r['cases']}
    try:
        with (a.output/'EFC_console.log').open('w',encoding='utf8') as log,contextlib.redirect_stdout(log):efc=review_saved_warp_contacts(a.report,a.output/'EFC',expected)
    except ValueError:
        if not (a.output/'EFC/report.json').exists():raise
        efc=json.loads((a.output/'EFC/report.json').read_text())
    comparisons=[];anchors=[];all_spreads=[];contracts=[];contact_patterns=[]
    for anchor in r['anchors']:
        mode,dt,k=anchor['mode'],anchor['dt'],anchor['k'];n=round(.02/dt);lo,hi=k*n,(k+1)*n
        c=next(c for c in original['cases'] if c['name']==mode and c['dt']==dt);d=c['planning'][k]
        z=dict(np.load(verify_file_pin(c['trace'])));old=dict(np.load(verify_file_pin(d['prediction'])));source=dict(np.load(verify_file_pin(d['source'])))
        saved=dict(np.load(verify_file_pin(anchor['reconstructed_source'])));m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])))
        jids=np.array([m.joint(f'robot/right_finger{i}_joint{j}').id for i in range(1,6) for j in range(1,5)])
        qids,dofs=m.jnt_qposadr[jids],m.jnt_dofadr[jids];aids=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in jids])
        body,qa,da=(c['model_layout'][key] for key in ['object_body','object_qposadr','object_dofadr'])
        oe=next(c for c in old_efc['cases'] if c['name']==mode and c['dt']==dt);old_w=np.load(verify_file_pin(oe['wrenches']))['world_COM_wrench'][lo:hi]
        actual={key:z[key][lo:hi] for key in ['post_qpos','post_qvel','post_ctrl','post_GPU_time']};old_pred=graph_trace(old,d['selected']);old_pred_w=old['W'][d['selected']].astype(float)
        candidates=[c for c in r['candidate_replays'] if c['mode']==mode and c['dt']==dt and c['k']==k]
        instrumented=[c for c in r['cases'] if c['mode']==mode and c['dt']==dt and c['k']==k]
        cand_packets={c['repeat']:dict(np.load(verify_file_pin(c['trace']))) for c in candidates}
        inst_packets={c['repeat']:dict(np.load(verify_file_pin(c['trace']))) for c in instrumented}
        inst_w={c['repeat']:np.load(verify_file_pin(next(e for e in efc['cases'] if e['name']==c['name'] and e['dt']==dt)['wrenches']))['world_COM_wrench'] for c in instrumented}
        reconstruction=json.loads(verify_file_pin(anchor['reconstruction']).read_text())
        model_checks={}
        if 'profile_model_arrays' in reconstruction:
            profile_model=dict(np.load(verify_file_pin(reconstruction['profile_model_arrays'])))
            old_model=dict(np.load(verify_file_pin(reconstruction['original_model_arrays'])))
            profile_scalars=json.loads(verify_file_pin(reconstruction['profile_model_scalars']).read_text())
            old_scalars=json.loads(verify_file_pin(reconstruction['original_model_scalars']).read_text())
            changed_arrays=sorted(key for key in old_model if profile_model[key].tobytes()!=old_model[key].tobytes())
            target_tolerance=profile_options['ccd_tolerance'] if profile_options else float(old_model['opt.ccd_tolerance'][0])
            model_checks=dict(model_keys_exact=set(profile_model)==set(old_model),only_CCD_tolerance_changed=set(changed_arrays)<=set(['opt.ccd_tolerance']),
                all_model_scalars_exact=profile_scalars==old_scalars,declared_CCD_tolerance=np.array_equal(profile_model['opt.ccd_tolerance'],np.full_like(old_model['opt.ccd_tolerance'],target_tolerance)),
                declared_field_diff=changed_arrays==reconstruction['changed_arrays'] and not reconstruction['changed_scalars'])
        base_checks=dict(**model_checks,source_keys_exact=set(source)==set(saved),all_source_bytes_exact=all(source[key].tobytes()==saved[key].tobytes() for key in source),
            static_role_explicit=anchor['Data_scalar_role']=='RECONSTRUCTED_NOT_ORIGINAL',both_paths_three_repeats=len(candidates)==3 and len(instrumented)==3)
        def comparison(label,p,pw,q,qw,repeat,candidate=None):
            metrics,checks=compare_response(p,q,pw,qw,qids,dofs,qa,da,gates)
            within=label in ['CANDIDATE_VS_INSTRUMENTED','CANDIDATE_VS_REPEAT0','INSTRUMENTED_VS_REPEAT0','ALL57_VS_REPEAT0']
            role='WITHIN_PROFILE_FIDELITY' if within else ('OLD_NUMERICAL_MODEL_RESPONSE_DIFFERENCE' if old_reference_differs else 'HISTORICAL_SAME_NUMERICS_DIAGNOSTIC')
            entry=dict(mode=mode,dt=dt,k=k,repeat=repeat,label=label,role=role,used_for_profile_qualification=within or profile=='ORIGINAL_RECONSTRUCTION',
                metrics=metrics,checks=checks,passed=all(checks.values()))
            if candidate is not None:entry['candidate']=candidate
            return entry
        for repeat in range(3):
            packet=cand_packets[repeat];p=graph_trace(packet,d['selected']);pw=packet['W'][d['selected']].astype(float)
            inst=inst_packets[repeat];iw=inst_w[repeat];scalars=json.loads(verify_file_pin(anchor['Data_scalars']).read_text())
            checks=dict(**base_checks,all57_targets_exact=np.array_equal(packet['U'][:,:,aids],np.broadcast_to(old['PD_targets'][:,None,:],packet['U'][:,:,aids].shape)),
                observed_source_read_only=all(q['source_untouched'] for q in candidates),
                instrumented_initial_qv_exact=np.array_equal(inst['pre_qpos'][0],source['qpos'][0].astype(float)) and np.array_equal(inst['pre_qvel'][0],source['qvel'][0].astype(float)),
                instrumented_initial_warm_exact=np.array_equal(inst['pre_qacc_warmstart'][0],source['qacc_warmstart'][0]),
                instrumented_PD_exact=np.array_equal(inst['post_ctrl'][:,aids],np.broadcast_to(old['PD_targets'][d['selected']],(n,20)).astype(float)),
                original_clock=np.array_equal(inst['post_GPU_time'],actual['post_GPU_time']),
                no_aid=not np.any(inst['post_qfrc_applied']) and not np.any(inst['post_xfrc_applied']),
                mocap_fixed=np.array_equal(inst['post_mocap_pos'],np.broadcast_to(source['mocap_pos'][0],inst['post_mocap_pos'].shape).astype(float)) and np.array_equal(inst['post_mocap_quat'],np.broadcast_to(source['mocap_quat'][0],inst['post_mocap_quat'].shape).astype(float)),
                capacity=not np.any(inst['overflow']) and int(inst['post_nefc'].max())<=scalars['njmax'],
                finite_active=all(np.isfinite(inst[key]).all() for key in ['pre_qpos','pre_qvel','post_qpos','post_qvel','post_qfrc_constraint','post_qacc','post_qacc_warmstart']))
            contracts.append(dict(mode=mode,dt=dt,k=k,repeat=repeat,checks=checks))
            for label,q,qw in [('CANDIDATE_VS_ORIGINAL_ACTUAL',actual,old_w),('CANDIDATE_VS_ORIGINAL_FORECAST',old_pred,old_pred_w)]:comparisons.append(comparison(label,p,pw,q,qw,repeat))
            for label,q,qw in [('INSTRUMENTED_VS_ORIGINAL_ACTUAL',actual,old_w),('INSTRUMENTED_VS_ORIGINAL_FORECAST',old_pred,old_pred_w)]:comparisons.append(comparison(label,inst,iw,q,qw,repeat))
            comparisons.append(comparison('CANDIDATE_VS_INSTRUMENTED',p,pw,inst,iw,repeat))
            if repeat:
                comparisons.append(comparison('CANDIDATE_VS_REPEAT0',p,pw,graph_trace(cand_packets[0],d['selected']),cand_packets[0]['W'][d['selected']].astype(float),repeat))
                comparisons.append(comparison('INSTRUMENTED_VS_REPEAT0',inst,iw,inst_packets[0],inst_w[0],repeat))
                all_spreads.extend(comparison('ALL57_VS_REPEAT0',graph_trace(packet,j),packet['W'][j].astype(float),graph_trace(cand_packets[0],j),cand_packets[0]['W'][j].astype(float),repeat,j) for j in range(57))
            contact_patterns.append(dict(mode=mode,dt=dt,k=k,repeat=repeat,
                instrumented=[patterns(inst,t,m,body) for t in range(n)],original_actual=[patterns(z,lo+t,m,body) for t in range(n)],
                solver_niter=inst['post_solver_niter'].astype(int).tolist(),original_actual_solver_niter=z['post_solver_niter'][lo:hi].astype(int).tolist()))
        selected_rows=[q for q in comparisons if q['mode']==mode and q['dt']==dt and q['k']==k]
        anchors.append(dict(mode=mode,dt=dt,k=k,original_selected=d['selected'],repeat_selected=[q['recomputed_selected'] for q in candidates],
            results={label:[sum(q['passed'] for q in selected_rows if q['label']==label),sum(q['label']==label for q in selected_rows)] for label in sorted({q['label'] for q in selected_rows})}))
    counts={label:[sum(q['passed'] for q in comparisons if q['label']==label),sum(q['label']==label for q in comparisons)] for label in sorted({q['label'] for q in comparisons})}
    verify_source_snapshot(receipt,live=True)
    eligible=[e for e in comparisons if e['used_for_profile_qualification']]+all_spreads
    contracts_ok=all(all(e['checks'].values()) for e in contracts)
    profile_qualified=contracts_ok and efc['passed']==efc['total'] and all(e['passed'] for e in eligible)
    result=dict(schema='pen_visited_source_saved_review:2',run=file_pin(a.report),source_receipt=file_pin(a.output/'source_receipt.json'),
        ccd_profile=profile,collision_numerics=profile_options,old_reference_model_differs=old_reference_differs,within_profile_qualified=profile_qualified,
        EFC=file_pin(a.output/'EFC/report.json'),EFC_counts=[efc['passed'],efc['total']],anchors=anchors,comparisons=comparisons,
        contract_passed=sum(sum(e['checks'].values()) for e in contracts),contract_total=sum(len(e['checks']) for e in contracts),contracts=contracts,
        comparison_counts=counts,all57_repeat_passed=sum(e['passed'] for e in all_spreads),all57_repeat_total=len(all_spreads),all57_repeats=all_spreads,
        contact_patterns=contact_patterns,original_Data_scalar_observations_available=False,neural_updates=0,native_integrations=0,
        scope='Within-profile qualification excludes old numerical-model response differences. Every original selected candidate through two paths/three repeats, all57 repeat spread, independent EFC, active pair/edge patterns and solver counters. Static Data metadata remains reconstructed, original lost import list remains unavailable. No confidence intervals, contact-cause proof, expert promotion, policy/ego success or global fidelity qualification.')
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),EFC=result['EFC_counts'],contracts=[result['contract_passed'],result['contract_total']],comparisons=counts,all57=[result['all57_repeat_passed'],len(all_spreads)],anchors=anchors)))
    if not profile_qualified:raise ValueError('Preserve failed within-profile visited-source replay gates')
