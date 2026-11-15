"""Independent finite-response,retained anomaly,context and timing arithmetic."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import mujoco
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.mechanics.warp_packet_decode import decode_saved_case


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh immutable outcome analysis')
    paths={k:ROOT/'runs'/v/'report.json' for k,v in dict(benchmark='source_epoch_benchmark_v1',review='source_epoch_benchmark_review_v1',amortized='source_epoch_amortized_v1',replay='source_epoch_impulse_replay_v1').items()}
    data={k:json.loads(p.read_text()) for k,p in paths.items()};run=data['benchmark'];review=data['review'];replay=data['replay'];out.mkdir()
    receipt=freeze_sources([Path(__file__).resolve(),ROOT/'src/supermimic_research/mechanics/warp_packet_decode.py'],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    checks={};original=run['cases'][69];model=mujoco.MjModel.from_binary_path(str(verify_file_pin(original['model'])));physical=dict(np.load(verify_file_pin(original['physical_context'])));z=dict(np.load(verify_file_pin(original['trace'])))
    metrics,decoded=decode_saved_case(original,z,model,physical);impulse=np.linalg.norm(decoded['linear_impulse_residual_N_s'],axis=1);tick=int(impulse.argmax())
    checks['original_failure_retained']=not metrics['checks']['linear_impulse'] and not review['checks']['row69_EFC_linear_impulse']
    integration=z['post_qvel'][:,:3]-z['pre_qvel'][:,:3]-.0025*z['post_qacc'][:,:3]
    anomaly=dict(case=original['name'],tick=tick,physical_time_s=float(z['post_GPU_time'][tick]),metrics=metrics,
        maximum_velocity_vs_qacc_increment_error_m_s=float(np.linalg.norm(integration,axis=1).max()),
        pre_velocity_continuity_exact=bool(np.array_equal(z['pre_qvel'][1:],z['post_qvel'][:-1])),
        impulse_residual_N_s=decoded['linear_impulse_residual_N_s'][tick].tolist(),root_cause='UNRESOLVED;matching labels/generalized force and consistent qacc integration do not explain the isolated force-to-motion mismatch.')
    for i,c in enumerate(replay['cases']):
        packet=dict(np.load(verify_file_pin(c['trace'])));value,_=decode_saved_case(original,packet,model,physical)
        checks[f'replay{i}_physical']=all(value['checks'].values());checks[f'replay{i}_metric']=bool(np.isclose(value['max_linear_impulse_residual_N_s'],c['metrics']['max_linear_impulse_residual_N_s'],atol=1e-15,rtol=1e-12))
    # Compare the finite central responses themselves,retaining the invalid pair.
    y=np.load(verify_file_pin(review['responses']))['y'];hp=json.loads(verify_file_pin(review['scale_health']).read_text());wm=torch.load(verify_file_pin(hp['checkpoint']),map_location='cpu',weights_only=False);scale=wm['arms']['DIRECT']['scales']['y_scale']
    pairs=[];blocks=dict(object12=slice(40,52),wrench6=slice(78,84),Fn6=slice(84,90))
    for source in run['sources']:
        for direction in range(2):
            for epsilon in [.002,.005]:
                groups={}
                for mode in ['STRICT','READ_LOCKED_EPOCH']:
                    for sign in [1,-1]:
                        ids=[i for i,c in enumerate(run['cases']) if c['source_name']==source['name'] and c['mode']==mode and c['profile']['direction']==direction and c['profile']['epsilon']==epsilon and c['profile']['sign']==sign]
                        if len(ids)!=3:raise ValueError('Complete finite-response repeats')
                        groups[(mode,sign)]=ids
                qualified=all(all(review['metadata'][i]['checks'].values()) for ids in groups.values() for i in ids);row=dict(source=source['name'],direction=direction,epsilon=epsilon,qualified=qualified,blocks={})
                for name,sl in blocks.items():
                    normalized={key:y[ids,sl]/scale[sl] for key,ids in groups.items()}
                    fd={mode:(normalized[(mode,1)].mean(0)-normalized[(mode,-1)].mean(0))/(2*epsilon) for mode in ['STRICT','READ_LOCKED_EPOCH']}
                    difference=float(np.linalg.norm(fd['STRICT']-fd['READ_LOCKED_EPOCH']));radius=sum(float(np.max(np.linalg.norm(values-values.mean(0),axis=1))) for values in normalized.values())/(2*epsilon)
                    bound=radius+1e-5/epsilon;within=difference<=bound;row['blocks'][name]=dict(normalized_FD_difference_L2=difference,observed_radius_bound=radius,registered_endpoint_floor_contribution=1e-5/epsilon,within=within)
                    checks[f'{source["name"]}_D{direction}_E{epsilon}_{name}_FD_envelope']=within
                pairs.append(row)
    # Quantify,never silently equate W1 and W32 derived Model constants.
    training=json.loads(verify_file_pin(run['training']).read_text());contexts=[]
    for arm in training['arms']:
        parent=torch.load(verify_file_pin(arm['checkpoint']),map_location='cpu',weights_only=False)['native']['native']['model'];entry=next(s for s in run['sources'] if s['actor']==arm['arm']);current=torch.load(verify_file_pin(entry['source']),map_location='cpu',weights_only=False)['native']['native']['model'];diff=[]
        for key,value in current['arrays'].items():
            expanded=np.broadcast_to(value,parent['arrays'][key].shape)
            if not np.array_equal(expanded,parent['arrays'][key]):diff.append(dict(array=key,W1_shape=list(value.shape),W32_shape=list(parent['arrays'][key].shape),max_abs_difference=float(abs(expanded.astype(float)-parent['arrays'][key].astype(float)).max())))
        checks[arm['arm']+'_declared_W1_W32_differences']={v['array'] for v in diff}=={'body_invweight0','dof_invweight0','actuator_acc0'}
        contexts.append(dict(actor=arm['arm'],actor_checkpoint=arm['checkpoint'],W1_source=entry['source'],scalars_exact=current['scalars']==parent['scalars'],different_arrays=diff))
    totals=data['amortized']['total_s'];ratio=totals['STRICT']/totals['READ_LOCKED_EPOCH'];checks['whole_batch_ratio']=bool(np.isclose(ratio,data['amortized']['pooled_ratio'],atol=1e-12,rtol=1e-12));checks['qualified_FD_pairs']=sum(p['qualified'] for p in pairs)==23
    verify_source_snapshot(receipt,live=True)
    result=dict(schema='source_epoch_outcome_analysis:1',inputs={k:file_pin(p) for k,p in paths.items()},source_receipt=file_pin(out/'source_receipt.json'),original_anomaly=anomaly,replay_summaries=replay['summaries'],
        finite_response_pairs=pairs,qualified_FD_pairs=sum(p['qualified'] for p in pairs),total_FD_pairs=len(pairs),W1_W32_contexts=contexts,whole_batch_ratio=ratio,
        physical_qualification='INCOMPLETE_ORIGINAL_IMPULSE_FAILURE_RETAINED',checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,
        scope='Arithmetic and raw replay checks preserve the failed initial physical gate. Fresh30-repeat controls did not reproduce it;no causal explanation or automatic TRAIN enablement. Restore timing applies only to measured batch.')
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],qualified_FD_pairs=result['qualified_FD_pairs'],original_anomaly=anomaly,W1_W32_contexts=contexts)),flush=True)
    if not all(checks.values()):raise ValueError('Retain outcome arithmetic discrepancy')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
