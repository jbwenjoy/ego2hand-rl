"""Equal sample and Adam budget control for minibatch composition and Actor-step size."""
from pathlib import Path
import sys,time,json,subprocess
import torch
from train_pen_multiclip_ppo import (ROOT,OBJECTIVE,COLLISION,PRELOAD,CLIPS,clip_config,multiclip_inputs,
    parent_state,noise_for,paired_samples,norm_state,check_bytes,init_learners,evaluate_counts,
    multiclip_w32,backend_scope,factory,snapshot,save_checked,restore,cpu_tree,file_pin,verify_file_pin,freeze_sources,write_json)
from supermimic_research.learning.pen_balanced_ppo import mixed_update
from supermimic_research.learning.owned_pen_ppo import validate_recorded_rollout


def main(out):
    out.mkdir(exist_ok=False);cfgdir=out/'config';cfgdir.mkdir()
    torch.set_num_threads(4);pin,actorpin,parent=parent_state();obj,_=clip_config(cfgdir,'clip_003',PRELOAD);manifest=factory()
    protocol=dict(parent=pin,parent_actor=actorpin,clips=multiclip_inputs(OBJECTIVE,obj),pairs=16,controls_per_clip_pair=64,
        worlds_per_clip=32,epochs=4,minibatches_per_pair_epoch=16,transitions_per_minibatch_per_clip=128,
        profiles=['balanced','balanced_kl'],limit=.03,seed=7411,factory=manifest['generated'],
        scope='Both rollouts collected before any update. GAE normalized per clip. Same 4096 transitions and 64 joint-clipped Adam updates/pair for both. balanced_kl backtracks Actor parameters only if either task reset or sampled-state mean KL exceeds .03 against the pre-pair policy. Critic and Adam moments take full update. Diagnostic constraint on sampled states.')
    write_json(out/'intent.json',protocol)
    files=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'tools/train_pen_multiclip_ppo.py',ROOT/'tools/evaluate_pen_dev_clip.py',ROOT/'tools/benchmark_pen_scaled.py',OBJECTIVE,COLLISION]
    receipt=freeze_sources(files,out/'snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt)
    actors={};reports={}
    try:
        for idx,profile in enumerate(protocol['profiles']):
            folder=out/profile;folder.mkdir();rows=[];ck=None;profile_protocol=dict(protocol,active_profile=profile)
            with torch.jit.optimized_execution(False),backend_scope(COLLISION) as collision:
                for first,last in [(1,1),(2,16)]:
                    with multiclip_w32(OBJECTIVE,obj,manifest) as tasks:
                        ls=init_learners(tasks,parent,'shared');initial={k:l.observations['policy'].clone() for k,l in ls.items()};norm=norm_state(ls[CLIPS[0]])
                        if ck:
                            payload=torch.load(verify_file_pin(ck),weights_only=False,map_location='cpu')
                            audit=restore(ls,payload,'shared',profile_protocol);del payload
                            write_json(folder/'first_checkpoint.json',dict(checkpoint=ck,restored=audit))
                        for pair in range(first,last+1):
                            start=time.perf_counter();logs={}
                            for n,clip in enumerate(CLIPS):
                                l=ls[clip];l.normalization_batch=None
                                with paired_samples(l.actor,noise_for(2*pair-1+n).to(l.device)):m=l.collect(64)
                                h=l.consistency()
                                if not all(h['checks'].values()):raise ValueError('Collector health: '+str(h))
                                validate_recorded_rollout(m,l.cfg.ppo_gamma)
                                valid=(m['contact_geometry_max']<=l.adapter.objective_constants['geometry_limits']).all(-1)&~m['engine_overflow']&m['no_applied_force']&(m['contact_errors']==0)&(m['contact_nonhand_max_sum'][...,0]<=.1)
                                phase=m['phase_used'];logs[clip]=dict(valid=int(valid.sum()),terminal=int(m['terminated'].sum()),phase1=int((phase==1).sum()),max_phase=int(phase.max()),late=int((valid&(phase>=((protocol['clips'][clip]['frames']-1)//2+1))).sum()))
                            stats=mixed_update(ls,CLIPS,initial,pair,.03 if profile=='balanced_kl' else None)
                            evaluate_counts(ls,'shared',2*pair);check_bytes(norm,norm_state(ls[CLIPS[0]]),'frozen normalizer')
                            rows.append(dict(pair=pair,clips=logs,update=stats,seconds=time.perf_counter()-start))
                            with (folder/'rows.jsonl').open('a') as f:f.write(json.dumps(rows[-1],allow_nan=False)+'\n')
                        dest=folder/('first_full.pt' if last==1 else 'final_full.pt')
                        state=snapshot(ls,'shared',receipt,2*last,profile_protocol);loaded=save_checked(dest,state);ck=file_pin(dest);del state,loaded
                        if last==16:
                            l=ls[CLIPS[0]];p=out/(profile+'.pt')
                            torch.save(dict(actor=cpu_tree(l.actor.state_dict()),config=parent['config'],iteration=2781+32,batch_backend=dict(factory=manifest['generated'],collision_backend=collision.contract['backend_id']),added_native_samples=65536,added_optimizer_calls=1024),p);actors[profile]=file_pin(p)
            rp=folder/'report.json';write_json(rp,dict(profile=profile,rows=rows,final=ck,first=file_pin(folder/'first_checkpoint.json')));reports[profile]=file_pin(rp)
        write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),actors=actors,profiles=reports))
    except BaseException as e:
        write_json(out/'failure.json',dict(error=repr(e)));raise
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
