"""Saved-only final300/resume sampler audit: no NN, native or optimization."""
from pathlib import Path
import argparse,json,os,sys,time
import numpy as np
from evaluate_online import preflight

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.learning import online_contract as C,online_data as D,world_model as W
from supermimic_research.learning.checkpoint import _log_prefix_valid
from supermimic_research.artifacts import file_pin


def finite(value,torch):
    if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite(v,torch) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite(v,torch) for v in value)
    return True


def run(args):
    _,c,registry,root,result,health,*_,out=preflight(args)
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch
    torch.set_num_threads(4);tick=time.perf_counter()
    prior_path=registry.resolve(c['training']['resume_artifact'])
    prior=torch.load(prior_path,map_location='cpu',weights_only=False)
    final=torch.load(registry.resolve_pin(result['checkpoint']),map_location='cpu',weights_only=False)
    C.need(prior['schema']==C.CHECKPOINT and prior['update']==10 and final['update']==300,'Fixed own10 to own300')
    C.validate_actor_payload(final,dict(kind='own_online',training_protocol=health['protocol']),health,root,c,C.source_pins())
    checks={};details={}
    checks['all_selected_sources_inputs_and_final_header']=True
    checks['raw_root_norm_scales_context_frozen']=all(W.raw_tree_equal(prior[k],final[k]) for k in ('normalizers','loss_scales','physical_context'))
    checks['raw_all_policy_target_norms_frozen']=all(W.raw_tree_equal(prior['arms'][m]['norm'],final['arms'][m]['norm']) for m in prior['arms'])
    checks['all_networks_targets_and_saved_gradients_finite']=finite(final['shared_WM'],torch) and finite(final['arms'],torch) and finite(final['gradient_snapshots'],torch)
    names=['WM',*[(mode+'_'+kind) for mode in C.R.MODES for kind in ('Actor','Critic')]]
    checks['exact_five_Adam_and_gradient_sets']=set(final['optimizers'])==set(final['gradient_snapshots'])==set(names)
    optimizer_details={}
    for name,opt in final['optimizers'].items():
        states=opt['state'];snap=final['gradient_snapshots'][name]
        if name=='WM':network=final['shared_WM']
        else:
            mode,kind=name.rsplit('_',1);network=final['arms'][mode]['stateactor' if kind=='Actor' else 'statecritic']
        ids=[index for group in opt['param_groups'] for index in group['params']]
        gradient_names=list(snap['post_clip'])
        ok=bool(states) and len(states)==len(snap['post_clip'])==len(snap['pre_clip'])
        ok=ok and ids==list(states) and len(set(ids))==len(ids) and set(snap['pre_clip'])==set(gradient_names)
        ok=ok and all(float(v['step'])==300 and finite(v,torch) and bool((v['exp_avg_sq']>=0).all()) for v in states.values())
        ok=ok and snap['tag'].startswith('PRE') and finite(snap,torch)
        for index,gradient_name in zip(ids,gradient_names):
            value=network[gradient_name]
            for tensor in (states[index]['exp_avg'],states[index]['exp_avg_sq'],snap['pre_clip'][gradient_name],snap['post_clip'][gradient_name]):
                ok=ok and tensor.shape==value.shape and tensor.dtype==value.dtype
        optimizer_details[name]=dict(parameters=len(states),all_step300=ok)
    details['optimizers']=optimizer_details;checks['all_five_Adam_moments_step300_and_named_gradient_shapes']=all(v['all_step300'] for v in optimizer_details.values())
    old_states={'WM':prior['shared_WM']}|{m+'_'+key:prior['arms'][m][key] for m in prior['arms'] for key in ('stateactor','statecritic','statecritic_target')}
    new_states={'WM':final['shared_WM']}|{m+'_'+key:final['arms'][m][key] for m in final['arms'] for key in ('stateactor','statecritic','statecritic_target')}
    details['network_changed_tensors']={key:sum(not W.raw_tree_equal(old_states[key][name],v) for name,v in state.items()) for key,state in new_states.items()}
    checks['actual_network_and_target_updates_after10']=all(value>0 for value in details['network_changed_tensors'].values())
    checks['all_parent_and_current_log_prefixes']=all(_log_prefix_valid(v) for v in final['parent_log_prefixes']+[final['log_prefix']])
    log_path=Path(final['log_prefix']['path']);rows=[json.loads(v) for v in log_path.read_bytes().splitlines()]
    checks['exact_updates11_to300_without_extra']=len(rows)==290 and [row['update'] for row in rows]==list(range(11,301))
    generator=torch.Generator(device='cpu');generator.set_state(prior['sample_generator']);pools={};maps={}
    for cycle in range(1,4):
        pool=D.merge_pools([r['dataset_report'] for r in result['collections'][:cycle]],registry);pools[cycle]=pool
        maps[cycle]=pool['mapping']
    checks['all336_cumulative_row_identities']=maps[3]==C.read(registry.resolve_pin(final['pool_identity_map'])) and maps[1]==C.read(registry.resolve_pin(prior['pool_identity_map'])) and maps[3][:len(maps[1])]==maps[1]
    sample_ok=True
    for row in rows:
        cycle=C.cursor(row['update'])['cycle'];pool=pools[cycle];canonical=pool['canonical'];n=int(canonical['WM_eligible'].sum())
        expected={name:torch.randint(size,(count,),generator=generator,device='cpu').tolist() for name,size,count in [('online',n,4),('old_original',684,2),('old_tripod',153,2)]}
        eligible=D.actor_indices(canonical,2);pick=torch.randint(len(eligible),(8,),generator=generator,device='cpu').numpy()
        expected_actor=canonical['row_id'][eligible[pick]].tolist()
        sample_ok=sample_ok and row['cursor']==C.cursor(row['update']) and all(row['WM_sampling'][name]['local_indices']==value for name,value in expected.items()) and row['Actor_global_row_ids']==expected_actor
    checks['replayed_all290_sampler_draws_exact']=sample_ok
    checks['saved_final_sampler_equals_replayed_state']=W.raw_tree_equal(generator.get_state(),final['sample_generator'])
    checks['raw_global_RNG_restored_from10_and_preserved']=W.raw_tree_equal(prior['RNG'],final['RNG'])
    checks['learning_call_budget_online300']=final['learning_calls']==C.expected_learning_calls(300,c['training']['microbatch']) and final['counts']==dict(WM_updates=300,paired_updates=300)
    checks['three_actual_collections_previous_complete_models']=len(result['collections'])==3 and [r['actor_input']['stage_id'] for r in result['collections']]==[0,1,2]
    for receipt in result['collections']:
        C.actor_metadata(receipt['actor_input'],receipt['cycle'],c,registry,C.source_pins(),root)
    reports=[C.read(registry.resolve_pin(r['report'])) for r in result['collections']]
    details['collection_rows']=[r['rows'] for r in reports];details['native_steps']=[r['counts']['integration_steps'] for r in reports]
    checks['all_collection_model_audits']=all(all(a['passed'] is True for a in report['model_audits'].values()) for report in reports)
    evidence=Path(result['checkpoint']['path']).parent/'health_0300.npz'
    with np.load(evidence,allow_pickle=False) as z:
        probe_keys=[key for key in z.files if key.startswith('expected_')]
        checks['all_saved_readback_predictions_raw_equal']=len(probe_keys)==5 and all(z[key].dtype==z['restored_'+key[9:]].dtype and z[key].shape==z['restored_'+key[9:]].shape and z[key].tobytes()==z['restored_'+key[9:]].tobytes() for key in probe_keys)
    checks['CPU_saved_only_no_CUDA']=not torch.cuda.is_initialized()
    report=dict(schema='supermimic_research.online300_saved_audit:1',checks=checks,passed=sum(checks.values()),total=len(checks),details=details,
        prior_checkpoint=file_pin(prior_path),checkpoint=result['checkpoint'],health=result['health'],training_terminal=file_pin(args.terminal),auditor=file_pin(__file__),
        own_CPU_checkpoint_reads=2,NN_constructors=0,NN_FW=0,backward=0,optimizer_steps=0,native_steps=0,GPU_calls=0,
        elapsed_s=time.perf_counter()-tick,scope='Direct saved tensors and full sampler replay only; no physical or policy qualification. Named optimizer shape binding uses saved order and pinned source, without network construction.')
    C.need(all(checks.values()),'Saved online300 audit failure: '+','.join(k for k,v in checks.items() if not v))
    out.mkdir(parents=True);C.write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),passed=report['passed'],total=report['total'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--result',type=Path,required=True);p.add_argument('--terminal',type=Path,required=True);p.add_argument('--zero-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a)
