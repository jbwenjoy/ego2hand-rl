"""Fresh-process strict replay with paired read-only solver observation scopes."""
import argparse,json,subprocess,sys,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import mujoco
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_support_runtime import SupportPenRuntime
from supermimic_research.runtime.pen_owned_probe_recorder import OwnedProbeRecorder,restore_source
from supermimic_research.runtime.gpu_model_snapshot import model_leaves
from supermimic_research.mechanics.warp_packet_decode import decode_saved_case
from supermimic_research.mechanics.solver_balance import linear_solver_balance

FIELDS=['solver_niter','nefc','qacc','qacc_smooth','qvel','qfrc_smooth','qfrc_constraint','qfrc_bias','qfrc_passive','qfrc_actuator','efc.Ma','M']


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


class SolverObserver(OwnedProbeRecorder):
    def after_step(self):
        super().after_step()
        if self.capture_solver:
            for key in FIELDS:self.solver_rows[key].append(self.solver_arrays[key].numpy()[0].copy())


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());rp=(cp.parent/cfg['probe_report']).resolve();run=json.loads(rp.read_text());pp=(cp.parent/cfg['preparation_report']).resolve();prep=json.loads(pp.read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or a.replica not in cfg['replicas'] or cfg['repeats']!=3 or cfg['restore_mode']!='STRICT' or run['restore_mode']!='STRICT' or prep['probe']!=file_pin(rp) or prep['qualified_rows']!=808:raise ValueError('Fresh exact registered strict replay')
    chosen=[]
    for name in cfg['failure_cases']:
        old=next(c for c in run['cases'] if c['name']==name);row=next(r for r in prep['metadata'] if r['name']==name)
        if row['valid'] or row['checks']['EFC_linear_impulse']:raise ValueError('Retained failed original condition required')
        opposite=next(c for c in run['cases'] if c['source_name']==old['source_name'] and c['profile']['direction']==old['profile']['direction'] and c['profile']['epsilon']==old['profile']['epsilon'] and c['profile']['repeat']==0 and c['profile']['sign']==-old['profile']['sign'])
        if not next(r for r in prep['metadata'] if r['name']==opposite['name'])['valid']:raise ValueError('Original matched control must pass')
        chosen.extend([dict(original=old,selected_role='ORIGINAL_IMPULSE_FAILURE'),dict(original=opposite,selected_role='MATCHED_ORIGINAL_PASS')])
    if len(chosen)!=8 or cfg['scopes']!=['PACKET_BASELINE','PACKET_PLUS_SOLVER_FIELDS']:raise ValueError('Eight conditions and two observer scopes')
    out.mkdir();protocol=(cp.parent/cfg['protocol']).resolve();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),cp,protocol]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    write(out/'intent.json',dict(config=file_pin(cp),probe=file_pin(rp),preparation=file_pin(pp),replica=a.replica,selection=chosen,cases=48,native_integrations=384,optimizer_steps=0))
    from mjlab.sim.sim import Simulation
    original=Simulation.step;observed=0;recorder=None;runtime=None;cases=[];checks={};sources={};contexts={};models={};options={};ordinal=cfg['replicas'].index(a.replica)
    def step(sim):
        nonlocal observed
        if recorder is not None and recorder.enabled:recorder.before_step()
        original(sim);observed+=sim.num_envs
        if recorder is not None and recorder.enabled:recorder.after_step()
    Simulation.step=step
    try:
        with torch.no_grad(),torch.jit.optimized_execution(False):
            for actor in ['RESIDUAL','SUPPORT']:
                runtime=SupportPenRuntime((cp.parent/cfg['objective_config']).resolve(),1);runtime.support_weight=float(runtime.objective_cfg['support_weights'][actor]);recorder=SolverObserver(runtime)
                arrays,_=model_leaves(runtime.env.sim.wp_data);recorder.solver_arrays={k:arrays[k] for k in FIELDS}
                selected=[c for c in chosen if c['original']['actor']==actor]
                for repeat in range(cfg['repeats']):
                    offset=(repeat+ordinal)%len(selected);order=selected[offset:]+selected[:offset]
                    scopes=cfg['scopes'][::-1] if (repeat+ordinal)%2 else cfg['scopes']
                    for entry in order:
                        c=entry['original'];key=c['source_name']
                        if key not in sources:
                            sources[key]=torch.load(verify_file_pin(c['source']),map_location='cpu',weights_only=False);models[key]=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['model'])));contexts[key]=dict(np.load(verify_file_pin(c['physical_context'])))
                            saved=sources[key]['native']['native']['model'];optkeys=['opt.solver','opt.iterations','opt.ls_iterations','opt.graph_conditional','opt.integrator','opt.disableflags','opt.enableflags','opt.cone']
                            options[key]=dict(scalars={k:saved['scalars'][k] for k in optkeys},arrays={k:saved['arrays'][k].tolist() for k in ['opt.timestep','opt.tolerance','opt.ls_tolerance','opt.gravity']})
                        source=sources[key];old=dict(np.load(verify_file_pin(c['trace'])));raw=old['actions20'][0]
                        for scope in scopes:
                            audit=restore_source(runtime,source);name=f'{c["name"]}_{scope}_R{repeat}';folder=out/name;folder.mkdir();recorder.capture_solver=scope=='PACKET_PLUS_SOLVER_FIELDS';recorder.solver_rows={k:[] for k in FIELDS}
                            exact_observation=all(torch.equal(v.cpu(),source['observation'][k]) for k,v in runtime.observation().items());recorder.begin(raw);before=observed
                            runtime.step(torch.tensor(raw,device=runtime.device)[None]);packet=recorder.finish();np.savez_compressed(folder/'trace.npz',**packet)
                            metrics,_=decode_saved_case(c,packet,models[key],contexts[key]);local=dict(exact_restore=audit['all_array_bytes_equal'] and audit['all_scalar_values_equal'],observation=exact_observation,native_count=observed-before==8,
                                actual_PD=np.array_equal(packet['post_ctrl'][:,c['aids']],old['post_ctrl'][:,c['aids']]))
                            row=dict(name=name,original_name=c['name'],source_name=key,selected_role=entry['selected_role'],actor=actor,scope=scope,repeat=repeat,trace=file_pin(folder/'trace.npz'),metrics=metrics,
                                original_trace=c['trace'],source=c['source'],actual_native_integrations=observed-before,restore_audit=audit,
                                original_endpoint_max_abs=dict(qpos=float(np.max(abs(packet['post_qpos']-old['post_qpos']))),qvel=float(np.max(abs(packet['post_qvel']-old['post_qvel'])))))
                            if recorder.capture_solver:
                                fields={k:np.array(v) for k,v in recorder.solver_rows.items()};np.savez_compressed(folder/'solver_fields.npz',**fields)
                                for key2,packetkey in [('qacc','post_qacc'),('qvel','post_qvel'),('qfrc_constraint','post_qfrc_constraint')]:local['epoch_'+key2]=np.array_equal(fields[key2].astype(np.float64),packet[packetkey])
                                balance=linear_solver_balance(packet,fields,c['physics']['mass'],c['physics']['gravity'],c['model_layout']['object_dofadr'],float(source['native']['native']['model']['arrays']['opt.timestep'][0]));np.savez_compressed(folder/'linear_balance.npz',**balance)
                                row.update(solver_fields=file_pin(folder/'solver_fields.npz'),linear_balance=file_pin(folder/'linear_balance.npz'),solver_niter=fields['solver_niter'].tolist(),balance_max_norm={k:float(np.linalg.norm(v,axis=1).max()) for k,v in balance.items()})
                            checks.update({name+'_'+k:bool(v) for k,v in local.items()});cases.append(row);write(folder/'report.json',row)
                    print(json.dumps(dict(replica=a.replica,actor=actor,repeat=repeat,cases=len(cases),impulse_failures=sum(not c['metrics']['checks']['linear_impulse'] for c in cases),native=observed)),flush=True)
                runtime.close();runtime=None;recorder=None
            checks.update(declared_cases=len(cases)==48,native_count=observed==384)
            verify_source_snapshot(receipt,live=True)
            result=dict(schema='strict_pen_solver_observation:1',config=file_pin(cp),protocol=file_pin(protocol),probe=file_pin(rp),preparation=file_pin(pp),source_receipt=file_pin(out/'source_receipt.json'),intent=file_pin(out/'intent.json'),replica=a.replica,
                cases=cases,actual_GPU_options=options,checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=observed,optimizer_steps=0,role=cfg['role'],private_solver_context_captured=False,
                scope='Two read-only observer scopes in one fresh process;targeted selected cases,not population incidence,solver termination cause,WM training or replacement of original impulse failures.')
            write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],impulse_failures=sum(not c['metrics']['checks']['linear_impulse'] for c in cases),native=observed)),flush=True)
            if not all(checks.values()):raise ValueError('Retain observation contract discrepancy')
    finally:
        if runtime:runtime.close()
        Simulation.step=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--replica',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
