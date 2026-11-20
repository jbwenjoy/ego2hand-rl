"""Audit two-clip PPO budgets, frozen statistics and geometry resets on saved packets."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.learning.native_pen_ppo import model_pair
from supermimic_research.learning.pen_matched_recovery import saved_bytes_equal
from supermimic_research.strict_json import write_json


def read(pin):
    return json.loads(verify_file_pin(pin).read_bytes())


def audit_memory(m,log,frames):
    phase=m['phase_used'];geometry=m['contact_geometry_max']
    geometric=(geometry>torch.tensor([.002,.0002,.001],dtype=geometry.dtype)).any(-1)
    nonhand=m['contact_nonhand_max_sum'][...,0]>.1
    invalid=geometric|nonhand|(m['contact_errors']!=0)|m['contact_global_error'].bool().any(-1).unsqueeze(-1)|~m['contact_objective_phase_valid']
    torch.testing.assert_close(invalid,m['contact_objective_geometry_failure'])
    term=invalid|m['contact_objective_native_terminated']
    torch.testing.assert_close(term,m['terminated'])
    torch.testing.assert_close(m['phase_after_reset'],torch.where(term|m['truncated'],1,phase+1))
    torch.testing.assert_close(m['phase_after_reset'][:-1],phase[1:])
    valid=~geometric & ~nonhand & ~m['engine_overflow'] & m['no_applied_force'] & (m['contact_errors']==0)
    hist=torch.bincount(phase[valid],minlength=frames)
    assert hist.tolist()==log['valid_phase_histogram'] and int(term.sum())==log['terminal']
    assert int(phase.eq(1).sum())==log['first_phase_count'] and int(phase.max())==log['max_phase']
    def count(mask):return int(mask.sum())
    return dict(phase1=count(phase==1),phase1_terminated=count((phase==1)&term),terminal=count(term),
                geometry_failure=count(invalid),native_task_termination=count(m['contact_objective_native_terminated']),
                object_penetration_gt_2mm=count(geometry[...,0]>.002),self_penetration_gt_0p2mm=count(geometry[...,1]>.0002),
                joint_gt_1mrad=count(geometry[...,2]>.001),overflow=count(m['engine_overflow']),
                nonhand_gt_0p1N=count(nonhand),noaid=count(m['no_applied_force']),valid_count=count(valid),
                object_penetration_quantiles_mm=(1000*torch.quantile(geometry[...,0].double(),torch.tensor([0.,.5,.95,1.],dtype=torch.float64))).tolist(),
                phase_min=int(phase.min()),phase_max=int(phase.max()),phase_transitions_checked=int(phase[1:].numel()))


def main(run,out):
    torch.set_num_threads(2)
    report=json.loads((run/'report.json').read_bytes());intent=read(report['intent'])
    parent=torch.load(verify_file_pin(intent['parent']),map_location='cpu',weights_only=False)
    assert parent['iteration']==2781 and parent['optimizer_steps']==88992
    a,c,_=model_pair('cpu',SimpleNamespace(**parent['config']))
    a.load_state_dict(parent['modules']['actor']);c.load_state_dict(parent['modules']['critic'])
    clips=('clip_001','clip_003')
    summary={}
    for profile,ppin in report['profiles'].items():
        profile_report=read(ppin)
        logs=[json.loads(line) for line in verify_file_pin(profile_report['iterations']).read_bytes().splitlines()]
        assert [r['clip'] for r in logs]==[clips[k%2] for k in range(32)]
        assert sum(r['native'] for r in logs)==524288 and sum(r['transitions'] for r in logs)==65536
        assert sum(r['optimizer_calls'] for r in logs)==1024
        assert all(all(r['health']['checks'].values()) for r in logs)
        disk=read(profile_report['first_restore']);flags=disk['restored']
        assert flags['rng_equal'] and flags['all_models_and_adam_byte_equal']
        assert all(v['native']['all_scalar_values_equal'] and v['native']['all_array_bytes_equal'] and v['memory_equal'] and v['stored_observations_equal'] for v in flags['clips'].values())
        checkpoints={};overall={}
        for pin,completed in ((disk['checkpoint'],1),(profile_report['final_checkpoint'],32)):
            state=torch.load(verify_file_pin(pin),weights_only=False,map_location='cpu')
            assert state['completed_batches']==completed and state['owners']==flags['ownership']
            if profile=='shared':assert len(state['models'])==1 and state['owners'][clips[0]]==state['owners'][clips[1]]
            else:assert len(state['models'])==2 and state['owners'][clips[0]]!=state['owners'][clips[1]]
            for owner,model in state['models'].items():
                for key in ['actor','critic']:
                    expected={k:v for k,v in parent['modules'][key].items() if k.startswith('norm.')}
                    assert expected and saved_bytes_equal(expected,{k:v for k,v in model[key].items() if k.startswith('norm.')})
                updates=32*sum(state['owners'][log['clip']]==owner for log in logs[:completed])
                assert set(int(v['step']) for v in model['optimizer']['state'].values())=={88992+updates}
                if completed==32:
                    actor=torch.load(verify_file_pin(report['actors'][owner]),map_location='cpu',weights_only=False)
                    assert saved_bytes_equal(actor['actor'],model['actor'])
            mem={}
            for clip in clips:
                active=[r for r in logs[:completed] if r['clip']==clip]
                n=len(active);v=state['collectors'][clip];fields=v['fields']
                assert [fields[k] for k in ['iteration','optimizer_steps','training_transitions','calibration_transitions']]==[n,32*n,2048*n,0]
                if not active:
                    assert v['memory'] is None
                    continue
                m=v['memory'];frames=int(intent['clips'][clip]['frames'])
                mem[clip]=audit_memory(m,active[-1],frames)
                if completed==1:
                    with torch.no_grad():
                        eps=torch.randn(64,32,20,generator=torch.Generator().manual_seed(7411+1009))
                        obs=m['obs'].reshape(-1,161);dist=a.distribution(obs);sim=(dist.loc+dist.scale*eps.reshape(-1,20)).reshape_as(m['actions'])
                        error=float(abs(sim-m['actions']).max())
                        assert error<1e-4
                        mem[clip]['parent_sample_cpu_error']=error
            checkpoints[completed]=dict(memory=mem,checkpoint=pin)
            del state
        for clip in clips:
            rows=[row for row in logs if row['clip']==clip]
            first,last=rows[0],rows[-1]
            overall[clip]=dict(transitions=sum(r['transitions'] for r in rows),valid_count=sum(r['valid_count'] for r in rows),
                late_valid=sum(r['late_valid'] for r in rows),phase1_count=sum(r['first_phase_count'] for r in rows),
                terminations=sum(r['terminal'] for r in rows),max_phase=max(r['max_phase'] for r in rows),
                first_valid=first['valid_count'],last_valid=last['valid_count'],first_phase1=first['first_phase_count'],last_phase1=last['first_phase_count'],
                max_penetration_mm=max(r['max_object_penetration_mm'] for r in rows))
        summary[profile]=dict(checkpoints=checkpoints,by_clip=overall,matched_stream=[(r['sample_noise_sha256'],r['permutation_seed']) for r in logs])
    assert summary['shared']['matched_stream']==summary['specialists']['matched_stream']
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'report.json',dict(source=file_pin(run/'report.json'),reviewer=file_pin(__file__),summary=summary,
        scope='Saved-state CPU audit of actual retained first/last rollout memory, both clip reset chains, all network/Adam and normalization ownership, exported weights, and complete training logs. Intermediate batches only have aggregate health, no full packet re-derived.'))
    print({k:v['by_clip'] for k,v in summary.items()})
    print({k:v['checkpoints'][32]['memory']['clip_003'] for k,v in summary.items()})
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
