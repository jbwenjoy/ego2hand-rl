"""Offline full-precision-accounted first mixed update on the actual frozen collection."""
from pathlib import Path
import json,sys,copy,time
from types import SimpleNamespace
import torch
from train_pen_multiclip_ppo import ROOT,CLIPS,parent_state,model_pair,file_pin,verify_file_pin,write_json
from supermimic_research.learning.pen_balanced_ppo import mixed_update

def main(out):
    out.mkdir(exist_ok=False);torch.set_num_threads(2)
    pin,_,parent=parent_state();folder=ROOT/'runs/pen_twoclip_parent_batches_v1'
    report=json.loads((folder/'report.json').read_bytes());batches={k:torch.load(verify_file_pin(report['clips'][k]['batch']),map_location='cpu',weights_only=False) for k in CLIPS}
    initial={k:b['start_obs'] for k,b in batches.items()};results={}
    for name,limit in [('balanced',None),('balanced_kl',.03)]:
        cfg=SimpleNamespace(**parent['config']);a,c,opt=model_pair('cpu',cfg)
        a.load_state_dict(parent['modules']['actor']);c.load_state_dict(parent['modules']['critic']);opt.load_state_dict(copy.deepcopy(parent['optimizer']))
        with torch.no_grad():base={k:a.distribution(v) for k,v in initial.items()}
        learners={k:SimpleNamespace(actor=a,critic=c,optimizer=opt,cfg=cfg,device='cpu',memory=b['memory'],observations={'critic':b['bootstrap_obs']},iteration=0,optimizer_steps=0) for k,b in batches.items()}
        stats=mixed_update(learners,CLIPS,initial,1,limit)
        with torch.no_grad():
            for k,obs in initial.items():
                new=a.distribution(obs);prev=base[k]
                stats[k+'_reset_kl']=float(torch.distributions.kl_divergence(prev,new).sum(-1).mean())
                stats[k+'_applied_mean_shift']=float((new.mean.clamp(-1,1)-prev.mean.clamp(-1,1)).norm(dim=-1).mean())
        assert stats['actual_adam']==64 and {int(v['step']) for v in opt.state.values()}=={88992+64}
        results[name]=stats
    write_json(out/'report.json',dict(source=file_pin(folder/'report.json'),parent=pin,results=results));print(results)
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
