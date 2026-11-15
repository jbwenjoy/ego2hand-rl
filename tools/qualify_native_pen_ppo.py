"""Live collector/checkpoint qualification before native PPO optimizer launch."""
import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from supermimic_research.artifacts import file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_native_policy import NativePenPolicyAdapter
from supermimic_research.learning.native_pen_ppo import NativePenPPO,cpu_tree


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


@torch.no_grad()
def reference_end_source(adapter):
    """Declared PRE-START final-frame source, not a teleport inside an episode."""
    d=adapter.env.sim.data;f=adapter.features;a=adapter.action
    adapter.motion.time_steps.fill_(368)
    pose=f.reference_pose_world()
    d.qpos[:,f.qa:f.qa+7]=pose
    d.qpos[:,f.qids]=adapter.motion.joint_pos
    d.qvel.zero_();d.qacc_warmstart.zero_();d.time.zero_()
    a._raw_actions.zero_();a._processed_actions.zero_();a._targets.copy_(a.limit_target(adapter.motion.joint_pos));a._targets_valid.fill_(True)
    f.robot.set_joint_position_target(a._targets,joint_ids=a._target_ids)
    d.ctrl[:,a._position_ctrl_ids]=a._targets
    adapter.env.sim.forward();adapter.env.sim.sense()


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or cfg['qualification_controls']!=8:raise ValueError('Fresh declared8-control qualification')
    out.mkdir();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})
    paths += [Path(__file__).resolve(),cp,(cp.parent/cfg['adapter_config']).resolve()]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    from mjlab.sim.sim import Simulation
    original=Simulation.step;observed=0
    def step(sim):
        nonlocal observed
        original(sim);observed+=sim.num_envs
    Simulation.step=step;checks={};result={}
    try:
        with torch.jit.optimized_execution(False):
            adapter=NativePenPolicyAdapter((cp.parent/cfg['adapter_config']).resolve(),32)
            try:
                trainer=NativePenPPO(adapter,cfg['ppo']);trainer.collect(8);consistency=trainer.consistency()
                checks.update({'collector_'+k:v for k,v in consistency['checks'].items()})
                state=trainer.checkpoint(receipt);torch.save(state,out/'collector_checkpoint.pt')
                torch.save(cpu_tree(trainer.memory),out/'collector_memory.pt')
                result['collector']=consistency;result['checkpoint']=file_pin(out/'collector_checkpoint.pt')
                result['memory']=file_pin(out/'collector_memory.pt')
                checks['no_optimizer_state']=not trainer.optimizer.state_dict()['state'] and trainer.optimizer_steps==0
            finally:adapter.close()
            fresh=NativePenPolicyAdapter((cp.parent/cfg['adapter_config']).resolve(),32)
            try:
                loaded=NativePenPPO(fresh,cfg['ppo']);audit=loaded.load(torch.load(out/'collector_checkpoint.pt',map_location='cpu',weights_only=False))
                checks.update({'load_'+key:bool(v) for key,v in audit.items() if isinstance(v,bool)})
                checks['load_native_arrays']=audit['native']['all_array_bytes_equal'];checks['load_native_scalars']=audit['native']['all_scalar_values_equal']
                result['load_audit']=audit
            finally:fresh.close()
            final=NativePenPolicyAdapter((cp.parent/cfg['adapter_config']).resolve(),1)
            try:
                reference_end_source(final);last=NativePenPPO(final,cfg['ppo'])
                last.collect(1,forced_actions=torch.zeros(1,1,20,device=last.device));m=last.memory
                last_consistency=last.consistency();checks.update({'reference_end_'+key:v for key,v in last_consistency['checks'].items()})
                checks.update(reference_end_truncated=bool(m['truncated'][0,0]) and not bool(m['terminated'][0,0]),
                    reference_end_source_phase=int(m['phase_source'][0,0])==368,
                    reference_end_terminal_phase=int(m['phase_next'][0,0])==369,
                    reference_end_reset_phase=int(m['phase_after_reset'][0,0])==1,
                    reference_end_correct_target=torch.equal(m['reward'],m['raw_reward']+last.cfg.ppo_gamma*m['terminal_next_value']))
                torch.save(cpu_tree(m),out/'reference_end_memory.pt');result['reference_end_memory']=file_pin(out/'reference_end_memory.pt')
                result['reference_end']=last_consistency
            finally:final.close()
    finally:Simulation.step=original
    checks['actual_native_budget']=observed==2056
    verify_source_snapshot(receipt,live=True)
    result.update(schema='pen_native_PPO_collector_qualification:1',config=file_pin(cp),source_receipt=file_pin(out/'source_receipt.json'),
        checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=observed,actual_optimizer_steps=0,
        role='DIAGNOSTIC_NEVER_FIT',scope='32world8controls plus1world1control final-frame pre-start timeout fixture, fresh state load. No optimizer/full task/force-repeat/ego qualification.')
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],integrations=observed)))
    if not all(checks.values()):raise ValueError('Keep collector/serialization discrepancy before training')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
