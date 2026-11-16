"""Fresh clipped-action and natural terminal pending/rollback checks on GPU."""
import argparse
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools'),str(ROOT)]
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.explicit_collision_backend import backend_scope,load_contract
from supermimic_research.runtime.backend_pen_runtime import BackendSupportPenRuntime
from supermimic_research.runtime.full_rehash_scope import FullRehashVerifierScope,implementation_pins
from supermimic_research.runtime.pen_live_packets import LivePenPackets
from supermimic_research.runtime import owned_pen_state as native
from supermimic_research.learning.pen_matched_recovery import WorkCounter
from supermimic_research.learning.world_model import raw_tree_equal
from supermimic_research.strict_json import write_json
from run_pen_backend_control_study import actor_initializer
from evaluate_pen_long_control import context_pins


def read(p):return json.loads(Path(p).read_text())


def main(args):
    started=time.perf_counter();cp=args.config.resolve();cfg=tomllib.loads(cp.read_text())
    if (cfg['schema'],cfg['mode'],cfg['arms'],cfg['reference_controls_max'],cfg['native_budget_max'],cfg['clipped_RAW_joint0'],cfg['verifier'])!=(1,'FRESH_NATIVE4_CLIPPED_AND_TERMINAL_LIFECYCLE_ONLY',['RESIDUAL','SUPPORT'],368,5920,1.4,'SINGLE_RESOLVE_FULL_REHASH'):raise ValueError('Fixed natural terminal lifecycle required')
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['study_config','node_review','protocol','registration','output']};study=tomllib.loads(paths['study_config'].read_text());sp={k:(paths['study_config'].parent/study[k]).resolve() for k in ['backend_manifest','objective_config','training_report','plant_context_report']}
    inputs={k:file_pin(p) for k,p in {**sp,**{k:paths[k] for k in ['study_config','node_review','protocol']}}.items()};contract=load_contract(sp['backend_manifest']);training=read(sp['training_report']);review=read(paths['node_review'])
    if any(r['passed']!=r['total'] or not all(r['checks'].values()) for r in [training,review]):raise ValueError('Completed training/node review required')
    contexts=context_pins(read(sp['plant_context_report']));actors={a['arm']:a['checkpoint'] for a in training['arms']}
    source_paths=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,paths['protocol'],ROOT/'tools/run_pen_backend_control_study.py',ROOT/'tools/evaluate_pen_long_control.py']
    def freeze(out):
        receipt=freeze_sources(source_paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt);return receipt
    if args.register:
        out=paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh lifecycle registration required')
        out.mkdir();freeze(out);write_json(paths['registration'],dict(schema='pen_terminal_lifecycle_registration:1',config=file_pin(cp),inputs=inputs,actors=actors,contexts=contexts,implementation_pins=implementation_pins(),source_receipt=file_pin(out/'source_receipt.json'),native_budget_max=5920,optimizer_steps=0));print(json.dumps(dict(registration=file_pin(paths['registration']))),flush=True);return
    reg=read(paths['registration'])
    if reg['config']!=file_pin(cp) or reg['inputs']!=inputs or reg['actors']!=actors or reg['contexts']!=contexts or reg['implementation_pins']!=implementation_pins():raise ValueError('Registered lifecycle inputs changed')
    verify_source_snapshot(read(verify_file_pin(reg['source_receipt'])),live=True);out=paths['output']
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh lifecycle output required')
    out.mkdir();receipt=freeze(out);counter=WorkCounter();runtime=packets=None;arms=[];rows=[];stage='construction'
    from mjlab.sim.sim import Simulation
    original=Simulation.step
    def step(sim):
        if packets is not None and packets.recorder.enabled:packets.recorder.before_step()
        result=counter.native_call(original,sim)
        if packets is not None and packets.recorder.enabled:packets.recorder.after_step()
        return result
    Simulation.step=step
    try:
        torch.set_num_threads(4)
        with backend_scope(sp['backend_manifest']) as backend,torch.no_grad(),torch.jit.optimized_execution(False):
            try:
                for arm in cfg['arms']:
                    stage=arm+'_fresh';folder=out/arm;folder.mkdir();state=torch.load(verify_file_pin(actors[arm]),map_location='cpu',weights_only=False);runtime=BackendSupportPenRuntime(sp['objective_config'],1);actor,admission=actor_initializer(runtime,state,actors[arm],contract=contract,plant_pin=contexts[arm],expected_iteration=2000)
                    before_actor={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()};packets=LivePenPackets(runtime,receipt,counter,folder,'QUALIFY_CLIPPED_TERMINAL_LIFECYCLE_NEVER_FIT');scope=FullRehashVerifierScope(cfg['verifier'],reg['implementation_pins']);arm_rows=[]
                    with scope:
                        raw=actor.distribution(runtime.observation()['policy'],validate_args=False).mean[0].cpu().numpy().copy();s0,s0pin=packets.capture(folder,'initial');clipped=raw.copy();clipped[0]=1.4;stage=arm+'_CLIPPED'
                        clip,endclip=packets.query(clipped,folder/'CLIPPED',stage,s0,s0pin);rows.append(clip);packets.restore(s0pin,arm+'_restore_initial');clipped_contract=clip['RAW_clip_count']>=1 and clip['sampled_RAW20'][0]==float(np.float32(1.4)) and clip['applied_RAW20'][0]==1.
                        terminal=preterminal=terminal_pin=preterminal_pin=last_raw=None;terminal_case=None
                        for index in range(368):
                            stage=arm+'_P'+str(index+1).zfill(3);raw=actor.distribution(runtime.observation()['policy'],validate_args=False).mean[0].cpu().numpy().copy();sf=folder/('P'+str(index+1).zfill(3));sf.mkdir();source,sourcepin=packets.capture(sf,'source')
                            case,end=packets.query(raw,sf/'BASE',stage,source,sourcepin);rows.append(case);arm_rows.append(case);write_json(out/'work.json',dict(stage=stage,**counter.metadata()))
                            if case['terminated'] or case['truncated']:
                                terminal,terminal_pin=end,case['end'];preterminal,preterminal_pin=source,sourcepin;last_raw=raw;terminal_case=case;break
                        if terminal is None:raise ValueError('No natural terminal/completion within original368 controls')
                        stage=arm+'_terminal_guard';before=counter.metadata();guard=False;message=None
                        try:runtime.step(torch.as_tensor(last_raw,device=runtime.device)[None])
                        except ValueError as e:message=str(e);guard=message=='Explicitly reset terminal worlds before next step'
                        unchanged=native.capture(runtime,receipt);guard_equal=all(raw_tree_equal(terminal[k],unchanged[k]) for k in ['native','python','rng']);guard_counter=before==counter.metadata();guard_pending=bool(runtime.pending[0])
                        packets.restore(preterminal_pin,arm+'_preterminal');pending_cleared=not bool(runtime.pending[0]);stage=arm+'_REPLAY_PRETERMINAL';replay,replay_end=packets.query(last_raw,folder/'REPLAY_PRETERMINAL',stage,preterminal,preterminal_pin);rows.append(replay)
                        packets.restore(terminal_pin,arm+'_terminal_end');pending_restored=bool(runtime.pending[0]);phase_after=int(runtime.motion.time_steps[0]);expected_phase=terminal_case['phase']+1
                    arms.append(dict(arm=arm,admission=admission,clipped_contract=clipped_contract,terminal_guard=dict(denied_before_native=guard,message=message,state_unchanged=guard_equal,external_counter_unchanged=guard_counter,pending=guard_pending),preterminal_pending_cleared=pending_cleared,terminal_pending_restored=pending_restored,
                        phase_after_terminal=phase_after,expected_phase_after_terminal=expected_phase,selected_controls=len(arm_rows),missing_controls=list(range(len(arm_rows)+1,369)),terminal_case=terminal_case['name'],terminal_kind='FAILURE' if terminal_case['terminated'] else 'REFERENCE_COMPLETION',
                        replay_phase_equal=replay['phase']==terminal_case['phase'],replay_terminal_same=replay['terminated']==terminal_case['terminated'] and replay['truncated']==terminal_case['truncated'],replay_end_qpos_max_abs=float(abs(terminal['native']['data']['arrays']['qpos']-replay_end['native']['data']['arrays']['qpos']).max()),
                        physical_invalid_rows=[c['name'] for c in arm_rows if not c['physical_audit']['valid']],restores=packets.restores,captures=packets.captures,actor_unchanged=raw_tree_equal(before_actor,actor.state_dict()),scope=scope.audit()))
                    runtime.close();runtime=packets=None
                checks=dict(both2_arms=len(arms)==2,all_clipped_contracts=all(a['clipped_contract'] for a in arms),all_input_reward=all(c['packed_input_bytes'] and all(c['reward_checks'].values()) and c['source_reward_equal'] for c in rows),all8_tick_packets=all(c['native_integrations']==8 for c in rows),
                    both_natural_pending_guards=all(all(v for k,v in a['terminal_guard'].items() if k!='message') for a in arms),all_pending_rollback=all(a['preterminal_pending_cleared'] and a['terminal_pending_restored'] for a in arms),all_phase_no_autoreset=all(a['phase_after_terminal']==a['expected_phase_after_terminal'] and a['replay_phase_equal'] for a in arms),
                    all6_full_restores=sum(len(a['restores']) for a in arms)==6 and all(all(all(v['complete_state_equal'].values()) and v['external_counter_unchanged'] for v in a['restores']) for a in arms),all_actor_bytes=all(a['actor_unchanged'] for a in arms),all_scopes=all(a['scope']['validated'] and a['scope']['bindings_restored'] for a in arms),
                    actual_attempted_accounting=counter.native==counter.native_attempted==len(rows)*8<=5920,no_optimizer=counter.optimizer==counter.optimizer_attempted==0)
                backend.verify(cold=True);verify_source_snapshot(receipt,live=True)
                result=dict(schema='pen_clipped_terminal_lifecycle:1',config=file_pin(cp),registration=file_pin(paths['registration']),inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),checks=checks,passed=sum(checks.values()),total=len(checks),arms=arms,rows=rows,**counter.metadata(),normalizer_fits=0,wall_s=time.perf_counter()-started,
                    scope='Lifecycle only;physical-invalid rows retained,not qualified policy/trajectory evidence. No WM/acceptance controller or full54-case launch;natural completion claimed only if observed. Same-source future differences remain explicit.')
                write_json(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],**counter.metadata())),flush=True)
                if not all(checks.values()):raise ValueError('Lifecycle discrepancy retained')
            except Exception as e:
                write_json(out/'failure.json',dict(type=type(e).__name__,message=str(e),stage=stage,rows=rows,arms=arms,**counter.metadata(),resumable=False));raise
            finally:
                for rt in list(backend.runtime_set):rt.close()
                runtime=packets=None
    finally:Simulation.step=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--register',action='store_true');main(p.parse_args())
