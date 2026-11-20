"""Same-source first-control action-distribution interventions on both references."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
import numpy as np
from qualify_two_clip_runtime import ROOT,config,OBJECTIVE,COLLISION,backend_scope,factory,multiclip_w32,file_pin,write_json
from supermimic_research.artifacts import verify_file_pin
from supermimic_research.runtime import owned_pen_state as native
from supermimic_research.runtime.source_validation_epoch import SourceValidationEpoch
from supermimic_research.learning.native_pen_ppo import model_pair


def main(run,out):
    out.mkdir(exist_ok=False);r=ROOT/'runs';full=file_pin(run/'report.json');report=json.loads(verify_file_pin(full).read_bytes())
    actors={'parent':file_pin(r/'pen_scaled_train_v1/actor_02781.pt'),**report['actors']}
    obj,_=config(out,'clip_003',r/'pen_refit_dev_preload_v1/report.json');seeds=[7411,7412,7413,7414]
    write_json(out/'intent.json',dict(run=full,seeds=seeds,design='From each of two fixed starting poses compare Gaussian mean and exploration std swaps with common normal noise. Exact same W32 full snapshot before every probe. 4 noise draws of 32 worlds, one 20 ms control. Counterfactual parts mixed only at inference.'))
    src=json.loads((run/'source_receipt.json').read_bytes());rows=[];torch.set_num_threads(4)
    with torch.jit.optimized_execution(False),torch.no_grad(),backend_scope(COLLISION),multiclip_w32(OBJECTIVE,obj,factory()) as tasks:
        policies={}
        for k,p in actors.items():
            st=torch.load(verify_file_pin(p),map_location='cpu',weights_only=False)
            a,_,_=model_pair('cuda:0',SimpleNamespace(**st['config']));a.load_state_dict(st['actor']);policies[k]=a.eval()
        for clip,rt in tasks.items():
            rt.support_weight=0.
            snap=native.capture(rt,src);obs=rt.observation()['policy'];dists={k:a.distribution(obs) for k,a in policies.items()}
            spec='specialist_'+clip[-3:]
            with SourceValidationEpoch(snap['sources'],snap['engine_sources']) as epoch:
                for seed in seeds:
                    n=torch.randn(32,20,generator=torch.Generator().manual_seed(seed)).to(rt.device)
                    for mean,std in [('parent','parent'),('shared','shared'),('parent','shared'),('shared','parent'),(spec,spec),('parent',spec),(spec,'parent')]:
                        audit=native.restore(rt,snap,source_epoch=epoch)
                        assert audit['all_array_bytes_equal'] and audit['all_scalar_values_equal']
                        raw=dists[mean].mean+dists[std].scale*n
                        _,_,term,trunc,info=rt.step(raw)
                        geom=rt.contact_views['geometry_max']
                        ok=(geom<=rt.objective_constants['geometry_limits']).all(-1)&(rt.contact_views['nonhand_max_sum'][:,0]<=.1)
                        assert not rt.contact_views['errors'].any() and not rt.contact_views['global_error'].any()
                        assert not rt.env.sim.wp_data.qfrc_applied.numpy().any() and not rt.env.sim.wp_data.xfrc_applied.numpy().any()
                        rows.append(dict(clip=clip,seed=seed,mean=mean,std=std,valid=int(ok.sum()),survive=int((~(term|trunc)).sum()),max_mm=float(geom[:,0].max()*1000),
                            median_mm=float(geom[:,0].median()*1000),clipped_components=int((raw.abs()>1).sum()),applied_delta_l2=float((raw.clamp(-1,1)-(dists['parent'].mean+dists['parent'].scale*n).clamp(-1,1)).norm(dim=-1).mean())))
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=rows,native=sum(32*8 for row in rows)))
    for clip in tasks:
        cs=[r for r in rows if r['clip']==clip]
        print(clip,{(m,s):sum(r['valid'] for r in cs if (r['mean'],r['std'])==(m,s)) for m,s in dict.fromkeys((r['mean'],r['std']) for r in cs)})
if __name__=='__main__':main(*map(lambda s:Path(s).resolve(),sys.argv[1:]))
