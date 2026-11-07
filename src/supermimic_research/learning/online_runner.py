"""Thin collection/update orchestration over the shared numerical core.

Native collection is an isolated CPU subprocess. Only declared complete TRAIN
rows enter the pooled caches. Frozen offline artifacts supply a new stage's
weights/norms/scales, never its Adam history. No automatic retry or refill.
"""
from pathlib import Path
import copy,datetime,hashlib,json,os,subprocess,sys,time,traceback
import numpy as np
from supermimic_research.artifacts import file_pin
from . import runner as R,world_model as WM,update_step as U,online_data as D,online_contract as C
from .checkpoint import checkpoint_health,_log_prefix_valid


def _collect(config_path,c,registry,root,sources,out,cycle,actor_input,profile_pin):
    C.need(C.fingerprint(C.source_pins()) == C.fingerprint(sources), 'Sources unchanged before collection')
    target=out/f'collection_cycle{cycle:02d}'
    protocol=out/f'collection_cycle{cycle:02d}.protocol.json'
    p=dict(schema=C.COLLECTION,configuration_file=file_pin(config_path),configuration=c,
        artifact_manifest=file_pin(registry.manifest),sources=sources,actor_input=actor_input,
        collection_cycle=cycle,profile_plan=profile_pin,native_authorized=True,output=str(target))
    C.write(protocol,p)
    from .collection_worker import inspect
    inspect(protocol)  # metadata/lineage first, no additional backend calls
    command=[sys.executable,'-X','utf8',str(Path(__file__).resolve().parents[3]/'scripts/research.py'),
             'collect-online','--protocol',str(protocol),'--run']
    stdout=out/f'collection_cycle{cycle:02d}.stdout.log';stderr=out/f'collection_cycle{cycle:02d}.stderr.log'
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='-1';tick=time.perf_counter()
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with stdout.open('xb') as so,stderr.open('xb') as se:
        process=subprocess.Popen(command,stdout=so,stderr=se,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        C.write(out/f'collection_cycle{cycle:02d}.process.json',dict(pid=process.pid,command=command,protocol=file_pin(protocol)))
        code=process.wait()
    terminal=out/f'collection_cycle{cycle:02d}.terminal.json'
    C.write(terminal,dict(schema='supermimic_research.collection_terminal:1',pid=process.pid,returncode=code,
        command=command,protocol=file_pin(protocol),source=sources['learning/collection_worker.py'],sources=sources,
        stdout=file_pin(stdout),stderr=file_pin(stderr),started_UTC=started,
        ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-tick,automatic_retry=False))
    C.need(code==0,'Native collection child failed; preserve output, no automatic retry')
    report=C.read(target/'report.json')
    C.need(C.fingerprint(C.source_pins())==C.fingerprint(sources),'Sources unchanged after collection')
    return dict(cycle=cycle,protocol=file_pin(protocol),terminal=file_pin(terminal),report=file_pin(target/'report.json'),
        dataset_report=report['dataset_report'],worker_receipt=file_pin(target/'worker_receipt.json'),actor_input=actor_input)


def _optimizers(s):
    optimizers={'WM':U.make_optimizer(s['model'],learning_rate=s['c']['optimizer']['WM_lr'])};arms={}
    for mode,a in s['arms'].items():
        arms[mode]={key:U.make_optimizer(a[key],learning_rate=s['c']['optimizer'][key+'_lr']) for key in ('actor','critic')}
        optimizers[mode+'_Actor']=arms[mode]['actor'];optimizers[mode+'_Critic']=arms[mode]['critic']
    C.need(all(not opt.state for opt in optimizers.values()),'New stage starts with five fresh Adam states')
    return optimizers,arms


def train(config_path):
    path,c,registry=C.configuration(config_path)
    C.need(c['training']['authorized'] is True and c['online']['native_authorized'] is True,'Actual learning/native configuration required')
    root=C.root_metadata(c,registry);sources=C.source_pins();inputs=C.selected_inputs(c,registry,root)
    out=(path.parent/c['training']['output']).resolve();runs=Path(__file__).resolve().parents[3]/'runs'
    C.need(out.is_relative_to(runs) and not out.exists(),'Fresh experiment output under unified runs')
    out.mkdir(parents=True);tick=time.perf_counter();phase='setup';update=0;collections=[];counts=dict(WM_updates=0,paired_updates=0)
    learning_calls=dict(WM_supervised_FW=0,WM_backward=0,WM_optimizer_steps=0,Actor_FW=0,WM_imagination_FW=0,
        Critic_real_FW=0,Critic_imagination_FW=0,Actor_backward_microbatches=0,Actor_optimizer_steps=0,Critic_backward=0,Critic_optimizer_steps=0)
    protocol=out/'protocol.json'
    C.write(protocol,dict(schema=C.PROTOCOL,configuration_file=file_pin(path),configuration=c,sources=sources,
        inputs=inputs,root_checkpoint=root['checkpoint'],numerical_scope='new source and microbatch recipe; no old-run resume',
        stop_after_first_healthy=True,automatic_retry=False))
    try:
        s=R._setup(path,smoke_only=False,warm_validator=lambda ck:C.validate_root_payload(ck,root))
        torch=s['torch'];s['scales']=s['warm_loss_scales'];C.need(s['policy_warmstart'] and s['scales'] is not None,'Root policy and loss scales required')
        C.need((len(s['old']['x']),len(s['new']['x']))==(684,153),'Frozen offline TRAIN caches only')
        C.write(out/'offline_replay_selection.json',dict(original=s['old_selection'],tripod=s['dataset'].selection_receipt()))
        provenance=dict(config=file_pin(path),artifact_manifest=file_pin(registry.manifest),sources=sources,inputs=inputs,
            starting_checkpoint=s['starting_checkpoint'],normalizer_origin_checkpoint=root['checkpoint'])
        s['provenance']=provenance
        frozen=dict(normalizers=WM.move_tensors(copy.deepcopy(s['norm']),'cpu'),
            scales=WM.move_tensors(copy.deepcopy(s['scales']),'cpu'),physical=WM.move_tensors(copy.deepcopy(s['physical']),'cpu'),
            policy_norms=WM.move_tensors({m:copy.deepcopy(U.arm_states({m:a})[m]['norm']) for m,a in s['arms'].items()},'cpu'))
        optimizers,arm_opts=_optimizers(s);parent_cp=root['checkpoint'];parent_health=root['root_health'];parent_logs=[];resume=None
        if c['training']['resume_artifact']:
            rp=registry.resolve(c['training']['resume_artifact']);hp=registry.pin(c['training']['resume_health']);h=C.read(registry.resolve_pin(hp))
            C.validate_own_header(h,file_pin(rp),c,sources,root,registry)
            C.need(h['update']<300,'Complete online stage cannot resume extra updates')
            resume=torch.load(rp,map_location='cpu',weights_only=False)
            ai=dict(kind='own_online',stage_id=h['cursor']['cycle'],checkpoint=file_pin(rp),health=hp,training_protocol=h['protocol'])
            C.validate_actor_payload(resume,ai,h,root,c,sources)
            C.need(WM.raw_tree_equal(resume['normalizers'],frozen['normalizers']) and WM.raw_tree_equal(resume['loss_scales'],frozen['scales']) and WM.raw_tree_equal(resume['physical_context'],frozen['physical']),'Resume retains root scales/norm/context')
            s['model'].load_state_dict(resume['shared_WM']);U.restore_arm_states(s['arms'],resume['arms'])
            for name,opt in optimizers.items():opt.load_state_dict(resume['optimizers'][name])
            update=resume['update'];counts=resume['counts'];learning_calls=resume['learning_calls'];collections=resume['collections'];C.verify_collections(collections,registry)
            registry.resolve_pin(resume['pool_identity_map'])
            parent_logs=resume['parent_log_prefixes']+[resume['log_prefix']]
            C.need(all(_log_prefix_valid(v) for v in parent_logs),'Immutable parent log prefixes')
            parent_cp=file_pin(rp);parent_health=hp
        from supermimic_research.runtime.native_collection import profile_plan
        from supermimic_research.runtime import phone_task,structured
        C.write(out/'profile_plan.json',profile_plan());profile_pin=file_pin(out/'profile_plan.json')
        setup_s=time.perf_counter()-tick;times=[];gradients={};start_update=update+1
        update_config=U.UpdateConfig(imagination_horizon=2,microbatch=c['training']['microbatch'])
        logpath=out/'updates.jsonl';resume_RNG_pending=resume is not None
        with logpath.open('x',encoding='utf8') as log:
            first_cycle=update//100+1
            for cycle in range(first_cycle,4):
                if len(collections)<cycle:
                    C.need(len(collections)==cycle-1 and update==(cycle-1)*100,'No repeated/skipped collection')
                    actor_input=dict(kind='offline1000' if cycle==1 else 'own_online',stage_id=cycle-1,
                        checkpoint=parent_cp,health=parent_health,training_protocol=root['root_protocol'] if cycle==1 else C.read(registry.resolve_pin(parent_health))['protocol'])
                    phase=f'collection_cycle{cycle}'
                    collections.append(_collect(path,c,registry,root,sources,out,cycle,actor_input,profile_pin))
                    C.verify_collections(collections,registry)
                    C.write(out/f'collections_cycle{cycle}.json',collections)
                else:C.need(len(collections)==cycle,'Resume reuses existing cycle pool without recollection')
                phase=f'cache_cycle{cycle}';cache_tick=time.perf_counter()
                merged=D.merge_pools([r['dataset_report'] for r in collections],registry)
                pool=D.prepare_pool(merged,s['geometry'],s['physical'],device=s['device'],chunk_rows=4)
                data=pool['canonical'];eligible=pool['Actor_indices'];C.need(len(eligible)>0,'Known complete prealive Actor starts')
                map_path=out/f'pool_identity_map_cycle{cycle}.json';C.write(map_path,pool['mapping']);map_pin=file_pin(map_path)
                if resume_RNG_pending:
                    prior_map=C.read(registry.resolve_pin(resume['pool_identity_map']))
                    C.need(pool['mapping'][:len(prior_map)]==prior_map and (update%100==0 or len(pool['mapping'])==len(prior_map)), 'Exact reconstructed prior cumulative row/episode identities')
                C.write(out/f'pool_selection_cycle{cycle}.json',dict(selection=pool['selection'],WM_selection=pool['WM_selection'],geometry_cache=pool['geometry_cache']))
                bank=R._real_bank(data,device=s['device'],torch=torch)
                rewards={m:torch.tensor(v,dtype=torch.float64,device=s['device']) for m,v in pool['rewards'].items()}
                if resume_RNG_pending:
                    s['generator'].set_state(resume['sample_generator']);WM.restore_rng(resume['RNG'],device=s['device']);resume_RNG_pending=False
                cache_s=time.perf_counter()-cache_tick
                while update<cycle*100:
                    update+=1;phase=f'cycle{cycle}_update{update}';begin=time.perf_counter();wall_begin=time.time()
                    batch,sampled=D.mixed_WM_batch(pool['WM_cache'],s['old'],s['new'],s['generator'],torch)
                    wm=U.supervised_WM_step(s['model'],optimizers['WM'],batch,s['norm'],s['scales'],s['physical'],codec=structured,phase=f'PRE_ONLINE_WM_{update}',materialize_metrics=False)
                    learning_calls['WM_supervised_FW']+=wm['counts']['WM_FW'];learning_calls['WM_backward']+=wm['counts']['WM_backward'];learning_calls['WM_optimizer_steps']+=wm['counts']['WM_optimizer_steps']
                    real,indices,device_indices=R._real_batch(bank,eligible,s['generator'],8,device=s['device'],torch=torch)
                    ac=U.matched_actor_critic_step(world_model=s['model'],arms=s['arms'],optimizers=arm_opts,real_batch=real,
                        mode_rewards={m:v[device_indices] for m,v in rewards.items()},tasks=s['tasks'],geometry=s['geometry'],codec=structured,
                        norms=s['norm'],physical=s['physical'],wm_step=phone_task.wm_step_current_only,config=update_config,
                        phase=f'PRE_ONLINE_{update}',materialize_metrics=False)
                    for a in ac.values():
                        for name,value in a['counts'].items():learning_calls[name]+=value
                    gradients={'WM':wm['gradient_snapshot']}|{m+'_'+kind.title():a['gradient_snapshots'][kind] for m,a in ac.items() for kind in ('actor','critic')}
                    metrics=U.materialize_metrics_tree(dict(WM_loss=wm['loss'],modes={m:{k:a[k] for k in ('real_TD_loss','imagined_Actor_loss')} for m,a in ac.items()}))
                    if s['device'].startswith('cuda:'):torch.cuda.synchronize(s['device'])
                    elapsed=time.perf_counter()-begin;times.append(elapsed);counts=dict(WM_updates=update,paired_updates=update)
                    row=dict(update=update,cursor=C.cursor(update),**metrics,WM_sampling=sampled,Actor_global_row_ids=data['row_id'][indices].tolist(),update_seconds=elapsed,
                        begin_unix_s=wall_begin,end_unix_s=time.time())
                    log.write(json.dumps(row)+'\n');log.flush()
                    if update==10 or update%100==0:
                        phase=f'health_update{update}';health_tick=time.perf_counter();os.fsync(log.fileno());prefix=logpath.read_bytes()
                        checkpoint=out/f'checkpoint_{update:04d}.pt'
                        payload=dict(schema=C.CHECKPOINT,update=update,cursor=C.cursor(update),counts=counts.copy(),learning_calls=learning_calls.copy(),protocol=file_pin(protocol),
                            configuration=c,provenance=provenance,root_checkpoint=root['checkpoint'],normalizer_origin_checkpoint=root['checkpoint'],
                            parent_checkpoint=parent_cp,parent_health=parent_health,collections=collections,pool_identity_map=map_pin,
                            shared_WM=s['model'].state_dict(),arms=U.arm_states(s['arms']),normalizers=s['norm'],loss_scales=s['scales'],physical_context=s['physical'],
                            optimizers={k:o.state_dict() for k,o in optimizers.items()},gradient_snapshots=gradients,sample_generator=s['generator'].get_state(),
                            RNG=WM.capture_rng(device=s['device']),parent_log_prefixes=parent_logs,
                            log_prefix=dict(path=str(logpath),bytes=len(prefix),sha256=hashlib.sha256(prefix).hexdigest(),records=update-start_update+1,range=[start_update,update]))
                        WM.atomic_checkpoint(checkpoint,payload)
                        def extra(ck):
                            C.verify_collections(ck['collections'],registry);registry.resolve_pin(ck['pool_identity_map'])
                            return dict(online_cursor=ck['cursor']==C.cursor(update) and ck['counts']==dict(WM_updates=update,paired_updates=update),
                                learning_call_budget=ck['learning_calls']==C.expected_learning_calls(update,c['training']['microbatch']),
                                actual_sources_unchanged=C.fingerprint(C.source_pins())==C.fingerprint(sources),
                                fixed_root_normalizers=WM.raw_tree_equal(s['norm'],frozen['normalizers']) and WM.raw_tree_equal(s['scales'],frozen['scales']) and WM.raw_tree_equal(s['physical'],frozen['physical']),
                                fixed_policy_normalizers=WM.raw_tree_equal({m:U.arm_states({m:a})[m]['norm'] for m,a in s['arms'].items()},frozen['policy_norms']),
                                root_and_parent=ck['root_checkpoint']==root['checkpoint'] and ck['parent_checkpoint']==parent_cp and ck['parent_health']==parent_health,
                                collections_and_row_map=ck['collections']==collections and ck['pool_identity_map']==map_pin)
                        F=WM.normalize_features(pool['WM_cache']['feature_F64'][:8],s['norm'])
                        health,evidence=checkpoint_health(checkpoint,s,optimizers,gradients,counts,F,real['obs'],schema=C.CHECKPOINT,extra_checks=extra)
                        with (out/f'health_{update:04d}.npz').open('xb') as stream:np.savez_compressed(stream,**evidence);stream.flush();os.fsync(stream.fileno())
                        mean=float(np.mean(times[-min(25,len(times)):]))
                        health.update(schema=C.HEALTH,checkpoint=file_pin(checkpoint),update=update,cursor=C.cursor(update),protocol=file_pin(protocol),configuration=c,provenance=provenance,
                            root_checkpoint=root['checkpoint'],normalizer_origin_checkpoint=root['checkpoint'],parent_checkpoint=parent_cp,parent_health=parent_health,
                            collections=collections,pool_identity_map=map_pin,learning_calls=learning_calls.copy(),setup_s=setup_s,current_cycle_cache_s=cache_s,elapsed_s=time.perf_counter()-tick,
                            measured_update_seconds=mean,ETA_measured_updates=min(25,len(times)),remaining_updates=300-update,ETA_learning_only_s=(300-update)*mean,
                            remaining_collection_cycles=3-len(collections),observed_collection_s=[C.read(registry.resolve_pin(r['terminal']))['elapsed_s'] for r in collections],
                            ETA_scope='Learning-only estimate. Remaining policy-dependent CPU collection/cache/health/IO and resume setup unmeasured.',
                            checkpoint_health_IO_s=time.perf_counter()-health_tick,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated() if s['device'].startswith('cuda:') else 0,
                            first_healthy_pausepoint=update==10,stops_here=update==10,device=s['device'])
                        hp=out/f'health_{update:04d}.json';C.write(hp,health);parent_cp=file_pin(checkpoint);parent_health=file_pin(hp)
                        print(json.dumps(dict(update=update,health=health['passed'],total=health['total'],ETA_learning_only_s=health['ETA_learning_only_s'])),flush=True)
                        if update==10:
                            result=dict(status='STOPPED_FIRST_HEALTHY_ONLINE_CHECKPOINT_WITH_ETA',checkpoint=parent_cp,health=parent_health,completed_updates=update,target_updates=300,collections=collections,policy_benefit_qualified=False)
                            C.write(out/'result.json',result);return result
        result=dict(status='COMPLETED_THREE_COLLECTION_UPDATE_CYCLES_NOT_POLICY_QUALIFICATION',checkpoint=parent_cp,health=parent_health,completed_updates=update,collections=collections,policy_benefit_qualified=False)
        C.write(out/'result.json',result);return result
    except BaseException as error:
        C.write(out/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),update=update,counts=counts,learning_calls=learning_calls,
            partial_current_update_possible=True,last_complete_paired_update=counts['paired_updates'],collections=collections,automatic_retry=False))
        raise
