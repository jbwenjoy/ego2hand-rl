"""Independent matched-table, repeated native/EFC and sparse-inertia review."""
import contextlib
import json
import subprocess
import tomllib
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from ..artifacts import file_pin,verify_file_pin
from ..frozen_sources import freeze_sources,verify_source_snapshot
from ..runtime.controlled_evaluation import write
from .warp_saved_review import review_saved_warp_contacts
from .pen_replay_fidelity import compare_response
from .visited_source_review import graph_trace
from .native_inertia import expand_sparse_inertia
ROOT=Path(__file__).resolve().parents[3]


def load(pin):return dict(np.load(verify_file_pin(pin)))
def doc(pin):return json.loads(verify_file_pin(pin).read_text())


def main(a,r):
    cfg=tomllib.loads(verify_file_pin(r['config']).read_text());cp=verify_file_pin(r['config'])
    gates=tomllib.loads((cp.parent/cfg['gates']).read_text());original=doc(r['original_run'])
    if a.output.exists():raise ValueError('Fresh matched-source review')
    for key in ['source_receipt','installed_source_receipt']:verify_source_snapshot(doc(r[key]))
    for pin in r['sources']:verify_file_pin(pin)
    a.output.mkdir();receipt=freeze_sources(sorted((ROOT/'src/supermimic_research').rglob('*.py'))+[ROOT/'tools/review_pen_visited_sources.py'],
        a.output/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(a.output/'source_receipt.json',receipt)
    try:
        with (a.output/'EFC_console.log').open('w',encoding='utf8') as log,contextlib.redirect_stdout(log):
            efc=review_saved_warp_contacts(a.report,a.output/'EFC',{(c['name'],c['dt']) for c in r['cases']})
    except ValueError:
        if not (a.output/'EFC/report.json').exists():raise
        efc=json.loads((a.output/'EFC/report.json').read_text())
    contracts=[];comparisons=[];closures=[];effects=[];not_comparable=[]
    original_offsets=load(original['candidates'])['raw_offsets'];proposal=doc(original['proposal']);center=(2*load(proposal['artifacts']['selected'])['delta20']).astype(np.float32)
    for anchor in r['anchors']:
        mode,dt,k=(anchor[key] for key in ['mode','dt','k']);n=round(.02/dt)
        c=next(c for c in original['cases'] if c['name']==mode and c['dt']==dt);old=load(c['planning'][k]['prediction']);source=load(c['planning'][k]['source'])
        saved=load(anchor['restored_source']);obs=doc(anchor['observed_Data_layout']);scalars=doc(anchor['Data_scalars']);reconstruction=doc(anchor['reconstruction'])
        model=load(anchor['restored_model_arrays']);old_model=load(c['fixed_model_arrays']);ms=doc(anchor['restored_model_scalars'])
        m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])));body,qa,da=(c['model_layout'][key] for key in ['object_body','object_qposadr','object_dofadr'])
        jids=np.array([m.joint(f'robot/right_finger{i}_joint{j}').id for i in range(1,6) for j in range(1,5)]);qids=m.jnt_qposadr[jids];dofs=m.jnt_dofadr[jids]
        aids=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in jids]);ref=load(c['initialization']['nominal_mean_wrench']);packets={}
        common=dict(source_keys=set(saved)==set(source),all_source_bytes=all(saved[key].tobytes()==source[key].tobytes() for key in source),
            recorded_scalars=scalars==obs['scalars']==reconstruction['Data_static_scalars'],recorded_scalar_role=reconstruction['Data_static_role']=='MATCHED_RECORDED_OBSERVATIONS',
            model_keys=set(model)==set(old_model),all_model_bytes=all(model[key].tobytes()==old_model[key].tobytes() for key in model),
            model_scalars=ms==doc(c['fixed_model_scalars']),model_unchanged=anchor['model_hashes_before']==anchor['model_hashes_after'])
        def compare(label,p,pw,q,qw,amplitude,repeat,candidate=None):
            metrics,checks=compare_response(p,q,pw,qw,qids,dofs,qa,da,gates)
            row=dict(mode=mode,dt=dt,k=k,amplitude=amplitude,repeat=repeat,label=label,metrics=metrics,checks=checks,passed=all(checks.values()))
            if candidate is not None:row['candidate']=candidate
            comparisons.append(row)
        for amp in [1,4]:
            entries=[e for e in r['candidate_replays'] if e['mode']==mode and e['k']==k and e['amplitude']==amp]
            instruments=[e for e in r['cases'] if e['mode']==mode and e['k']==k and e['amplitude']==amp]
            if len(entries)!=3 or len(instruments)!=3:raise ValueError('Three repeats per matched table/path')
            cand={e['repeat']:load(e['trace']) for e in entries};inst={e['repeat']:load(e['trace']) for e in instruments}
            w={e['repeat']:load(next(x for x in efc['cases'] if x['name']==e['name'])['wrenches'])['world_COM_wrench'] for e in instruments}
            raw=center+original_offsets[k]*np.float32(amp/4)
            ema=np.float32(.5)*np.clip(np.float32(.5)*raw,-.6,.6)+np.float32(.5)*old['controller__processed_actions']
            intended=ref['reference_hand'][k+1]+ema-old['encoder_bias20'];pd=np.clip(intended,old['PD_lower20'],old['PD_upper20'])
            for repeat in range(3):
                entry=next(e for e in entries if e['repeat']==repeat);instrument=next(e for e in instruments if e['repeat']==repeat);d=cand[repeat];z=inst[repeat];j=entry['selected']
                quat=d['Q'][:,-1,qa+3:qa+7].astype(float);angle=(Rotation.from_quat(quat[:,[1,2,3,0]])*Rotation.from_quat(ref['quaternion_wxyz'][k+1][[1,2,3,0]]).inv()).magnitude()
                state=(((d['Q'][:,-1,qa:qa+3]-ref['COM_world_m'][k+1])/.005)**2).sum(1)+(angle/np.deg2rad(5))**2
                state+=.01*(((d['Q'][:,-1,qids]-ref['reference_hand'][k+1])/.02)**2).mean(1)+.01*((ema/.02)**2).mean(1)
                mean=d['W'].astype(float).mean(1);target=old['support_target_world'];support=(((mean[:,:3]-target[:3])/.1)**2).mean(1)+(((mean[:,3:]-target[3:])/.005)**2).mean(1)
                score=state+float(old['support_weight'])*bool(old['endpoint_velocity_valid'] and ref['valid_interval'][k])*support
                feasible=d['score_feasible'].astype(bool);violation=d['score_violation'];ids=np.flatnonzero(feasible) if feasible.any() else np.flatnonzero(violation==violation.min())
                best=int(ids[np.argmin(d['score_total'][ids])])
                checks=dict(**common,raw_table=np.array_equal(raw,d['raw_candidates']),EMA=np.max(abs(ema-d['EMA_candidates']))<1e-7,
                    intended_PD=np.max(abs(intended-d['intended_PD']))<2e-7,clamped_PD=np.max(abs(pd-d['PD_targets']))<2e-7,
                    all57_executed_PD=np.array_equal(d['U'][:,:,aids],np.broadcast_to(d['PD_targets'][:,None,:],(57,n,20))),
                    full_orientation=np.allclose(angle,d['score_orientation_error_rad'],atol=2e-6,rtol=1e-6),
                    score_FP32=np.all(abs(score-d['score_total'])<=1e-3+1e-5*abs(score)),argmin=j==best==instrument['selected'],
                    selected_PD=np.array_equal(z['post_ctrl'][:,aids],np.broadcast_to(d['PD_targets'][j],(n,20)).astype(float)),
                    source_qv=np.array_equal(z['pre_qpos'][0],source['qpos'][0].astype(float)) and np.array_equal(z['pre_qvel'][0],source['qvel'][0].astype(float)),
                    source_warm=np.array_equal(z['pre_qacc_warmstart'][0],source['qacc_warmstart'][0]),source_isolated=entry['source_untouched'] and instrument['restored_source_exact'],
                    clock=np.array_equal(d['T'][j].astype(float),z['post_GPU_time']),no_aid=not np.any(z['post_qfrc_applied']) and not np.any(z['post_xfrc_applied']),
                    capacity=not np.any(z['overflow']) and int(z['post_nefc'].max())<=scalars['njmax'])
                contracts.append(dict(mode=mode,k=k,amplitude=amp,repeat=repeat,checks={key:bool(val) for key,val in checks.items()}))
                compare('CANDIDATE_VS_INSTRUMENTED',graph_trace(d,j),d['W'][j].astype(float),z,w[repeat],amp,repeat)
                if repeat:
                    fixed=entries[0]['selected']
                    compare('CANDIDATE_REPEAT_R0_ACTION',graph_trace(d,fixed),d['W'][fixed].astype(float),graph_trace(cand[0],fixed),cand[0]['W'][fixed].astype(float),amp,repeat)
                    for index in range(57):compare('ALL57_REPEAT',graph_trace(d,index),d['W'][index].astype(float),graph_trace(cand[0],index),cand[0]['W'][index].astype(float),amp,repeat,index)
                    if np.array_equal(z['post_ctrl'],inst[0]['post_ctrl']):compare('INSTRUMENTED_REPEAT_MATCHED_PD',z,w[repeat],inst[0],w[0],amp,repeat)
                    else:not_comparable.append(dict(mode=mode,k=k,amplitude=amp,repeat=repeat,reason='Selected PD changed; different actions cannot be scored as replay error'))
                mass=float(model['body_mass'][0,body]);force=z['post_qfrc_smooth']+z['post_qfrc_constraint'];residual=[];block_error=[];ma_error=[];smooth_error=[]
                for tick in range(n):
                    M=expand_sparse_inertia(z['post_M'][tick],model['M_rownnz'],model['M_rowadr'],model['M_colind'],m.nv)
                    residual.append((M@z['post_qacc'][tick]-force[tick])[da:da+3]);block_error.append(np.max(abs(M[da:da+3,da:da+3]-mass*np.eye(3))))
                    ma_error.append((M@z['post_qacc'][tick]-z['post_efc_Ma'][tick])[da:da+3]);smooth_error.append((M@z['post_qacc_smooth'][tick]-z['post_qfrc_smooth'][tick])[da:da+3])
                residual=np.array(residual);peak=int(np.linalg.norm(residual,axis=1).argmax())
                closure_checks=dict(free_mass_block=max(block_error)<1e-7,finite_terms=bool(np.isfinite(residual).all()),
                    explicit_smooth_force=np.max(abs(z['post_qfrc_smooth']-(z['post_qfrc_actuator']+z['post_qfrc_passive']-z['post_qfrc_bias'])))<1e-5)
                closures.append(dict(mode=mode,k=k,amplitude=amp,repeat=repeat,checks={key:bool(val) for key,val in closure_checks.items()},
                    max_linear_Ma_residual_N=float(np.linalg.norm(residual,axis=1).max()),max_dt_linear_Ma_residual_N_s=float(dt*np.linalg.norm(residual,axis=1).max()),
                    max_stored_Ma_difference_N=float(np.linalg.norm(ma_error,axis=1).max()),max_smooth_solve_residual_N=float(np.linalg.norm(smooth_error,axis=1).max()),
                    peak_tick=peak,solver_niter=z['post_solver_niter'].astype(int).tolist()))
            packets[amp]=cand[0]
        effects.append(dict(mode=mode,k=k,source=c['planning'][k]['source'],
            zero_candidate_PD_exact=np.array_equal(packets[1]['PD_targets'][0],packets[4]['PD_targets'][0]),
            zero_candidate_state_max_abs=float(abs(packets[1]['Q'][0]-packets[4]['Q'][0]).max()),
            matched_direction_max_endpoint_COM_difference_m=float(np.linalg.norm(packets[1]['Q'][:,-1,qa:qa+3]-packets[4]['Q'][:,-1,qa:qa+3],axis=1).max()),
            matched_direction_max_mean_force_difference_N=float(np.linalg.norm(packets[1]['W'].mean(1)[:,:3]-packets[4]['W'].mean(1)[:,:3],axis=1).max())))
    contract_passed=sum(sum(e['checks'].values()) for e in contracts);contract_total=sum(len(e['checks']) for e in contracts)
    counts={label:[sum(e['passed'] for e in comparisons if e['label']==label),sum(e['label']==label for e in comparisons)] for label in sorted({e['label'] for e in comparisons})}
    qualified=contract_passed==contract_total and efc['passed']==efc['total'] and all(e['passed'] for e in comparisons) and all(all(c['checks'].values()) for c in closures) and not not_comparable
    verify_source_snapshot(receipt,live=True)
    result=dict(schema='pen_matched_amplitude_review:1',run=file_pin(a.report),source_receipt=file_pin(a.output/'source_receipt.json'),EFC=file_pin(a.output/'EFC/report.json'),EFC_counts=[efc['passed'],efc['total']],
        contract_passed=contract_passed,contract_total=contract_total,contracts=contracts,comparison_counts=counts,comparisons=comparisons,
        closures=closures,source_matched_effects=effects,not_comparable=not_comparable,within_cohort_qualified=qualified,
        native_integrations=0,neural_updates=0,scope='Same exact observed source/controller/model under two action tables. Chosen graph/instrumented and fixed-PD repeats only, all57 repeat gates and independent EFC. Sparse mass/acceleration decomposition localizes recorded residual, not native stopping reason or task/expert/gradient/ego qualification.')
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),contracts=[contract_passed,contract_total],EFC=result['EFC_counts'],counts=counts,qualified=qualified)))
    if not qualified:raise ValueError('Retain failed matched-source fidelity/closure evidence')
