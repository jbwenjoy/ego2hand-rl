"""Independent file review of targeted solver observations and scope variation."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import mujoco
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.mechanics.warp_packet_decode import decode_saved_case
from supermimic_research.mechanics.controlled_pen_records import pack_case
from supermimic_research.mechanics.solver_balance import linear_solver_balance


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    out=a.output.resolve();runs=[json.loads(p.read_text()) for p in a.reports]
    if out.exists() or not out.is_relative_to(ROOT/'runs') or [r['replica'] for r in runs]!=['A','B'] or any(r['passed']!=r['total'] or r['private_solver_context_captured'] for r in runs) or runs[0]['probe']!=runs[1]['probe']:raise ValueError('Fresh two-process public-field review')
    out.mkdir();protocol=verify_file_pin(runs[0]['protocol']);paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),protocol]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    probe=json.loads(verify_file_pin(runs[0]['probe']).read_text());originals={c['name']:c for c in probe['cases']};checks={};rows=[];packets={};models={};contexts={};sources={};failed_ticks=[]
    for run in runs:
        verify_source_snapshot(json.loads(verify_file_pin(run['source_receipt']).read_text()),live=True)
        for ci,c in enumerate(run['cases']):
            original=originals[c['original_name']];key=original['source_name'];tag=run['replica']+'_'+c['name'];packet=dict(np.load(verify_file_pin(c['trace'])));packets[(run['replica'],c['original_name'],c['scope'],c['repeat'])]=packet
            if key not in models:
                models[key]=mujoco.MjModel.from_binary_path(str(verify_file_pin(original['model'])));contexts[key]=dict(np.load(verify_file_pin(original['physical_context'])));sources[key]=torch.load(verify_file_pin(original['source']),map_location='cpu',weights_only=False)
                verify_source_snapshot(sources[key]['native']['sources'],live=True)
                for pin in sources[key]['native']['engine_sources']:verify_file_pin(pin)
            metrics,decoded=decode_saved_case(original,packet,models[key],contexts[key]);record=pack_case(models[key],original,packet,decoded,np.array(original['qids']),np.array(original['dofs']),np.array(original['aids']),packet['initial_warmstart'],original['physics'])
            geometry=bool(np.all(record['geometry_flags'][0]<=[.002,.0002,.001]));valid=bool(record['valid'][0] and all(metrics['checks'].values()) and geometry)
            checks[tag+'_metrics_exact']=metrics==c['metrics'];checks[tag+'_initial_source']=np.array_equal(packet['pre_qpos'][0],sources[key]['native']['native']['data']['arrays']['qpos'][0]) and np.array_equal(packet['pre_qvel'][0],sources[key]['native']['native']['data']['arrays']['qvel'][0])
            checks[tag+'_eight_native']=len(packet['post_qpos'])==8
            row=dict(replica=run['replica'],name=c['name'],original_name=c['original_name'],scope=c['scope'],repeat=c['repeat'],selected_role=c['selected_role'],valid=valid,geometry=geometry,metrics=metrics)
            if 'solver_fields' in c:
                fields=dict(np.load(verify_file_pin(c['solver_fields'])));saved=dict(np.load(verify_file_pin(c['linear_balance'])));source=sources[key];dt=float(source['native']['native']['model']['arrays']['opt.timestep'][0]);d=original['model_layout']['object_dofadr'];mass=original['physics']['mass']
                balance=linear_solver_balance(packet,fields,mass,original['physics']['gravity'],d,dt)
                checks[tag+'_balance_exact']=set(balance)==set(saved) and all(np.array_equal(v,saved[k]) for k,v in balance.items())
                for fk,pk in [('qacc','post_qacc'),('qvel','post_qvel'),('qfrc_constraint','post_qfrc_constraint')]:checks[tag+'_epoch_'+fk]=np.array_equal(fields[fk].astype(np.float64),packet[pk])
                ticks=np.flatnonzero(np.linalg.norm(decoded['linear_impulse_residual_N_s'],axis=1)>=5e-7)
                for tick in ticks:
                    failed_ticks.append(dict(replica=run['replica'],original_name=c['original_name'],repeat=c['repeat'],tick=int(tick),solver_niter=int(fields['solver_niter'][tick]),iteration_cap=int(source['native']['native']['model']['scalars']['opt.iterations']),
                        impulse_Ns=float(np.linalg.norm(decoded['linear_impulse_residual_N_s'][tick])),balance_N={k:float(np.linalg.norm(v[tick])) for k,v in balance.items() if k.endswith('_N')},
                        physics=dict(passive=fields['qfrc_passive'][tick,d:d+3].tolist(),actuator=fields['qfrc_actuator'][tick,d:d+3].tolist())))
                row['balance_max_norm']={k:float(np.linalg.norm(v,axis=1).max()) for k,v in balance.items()}
            rows.append(row)
    scope_pairs=[]
    for run in runs:
        names=sorted({c['original_name'] for c in run['cases']})
        for name in names:
            for repeat in range(3):
                left=packets[(run['replica'],name,'PACKET_BASELINE',repeat)];right=packets[(run['replica'],name,'PACKET_PLUS_SOLVER_FIELDS',repeat)];differences={}
                for key in ['post_qpos','post_qvel','post_qacc','post_qfrc_constraint','GPU_efc_force_flat']:
                    x,y=left[key],right[key];same_shape=x.shape==y.shape
                    differences[key]=dict(exact=bool(same_shape and np.array_equal(x,y)),max_abs=float(np.max(abs(x-y))) if same_shape and x.size else None,shapes=[list(x.shape),list(y.shape)])
                scope_pairs.append(dict(replica=run['replica'],original_name=name,repeat=repeat,fields=differences))
    summaries=[]
    for replica in ['A','B']:
        for scope in ['PACKET_BASELINE','PACKET_PLUS_SOLVER_FIELDS']:
            for role in ['ORIGINAL_IMPULSE_FAILURE','MATCHED_ORIGINAL_PASS']:
                cs=[r for r in rows if r['replica']==replica and r['scope']==scope and r['selected_role']==role]
                summaries.append(dict(replica=replica,scope=scope,selected_role=role,cases=len(cs),physical_invalid=sum(not c['valid'] for c in cs),impulse_failures=sum(not c['metrics']['checks']['linear_impulse'] for c in cs),max_impulse_Ns=max(c['metrics']['max_linear_impulse_residual_N_s'] for c in cs)))
    checks.update(declared_cases=len(rows)==96,declared_scope_pairs=len(scope_pairs)==48,native_accounted=sum(r['actual_native_integrations'] for r in runs)==768,zero_updates=not any(r['optimizer_steps'] for r in runs),same_actual_GPU_options=runs[0]['actual_GPU_options']==runs[1]['actual_GPU_options'])
    verify_source_snapshot(receipt,live=True)
    result=dict(schema='strict_pen_solver_independent_review:1',reports=[file_pin(p) for p in a.reports],probe=runs[0]['probe'],protocol=file_pin(protocol),source_receipt=file_pin(out/'source_receipt.json'),
        rows=rows,summaries=summaries,failed_extra_field_ticks=failed_ticks,scope_pairs=scope_pairs,exact_scope_pairs=sum(all(v['exact'] for v in p['fields'].values()) for p in scope_pairs),
        checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,
        physical_status='RETAINED_STRICT_PATH_IMPULSE_FAILURES' if any(not r['valid'] for r in rows) else 'SELECTED_REPLAYS_PASS_ORIGINAL_FAILURES_RETAINED',
        scope='Independent saved-data review of two fresh processes;public Ma residual is not private SolverContext gradient. Early solver iteration counts and observer/order numerical variation do not identify the termination criterion or establish causality. Original8 TRAIN rejects and epoch counterexample stay unchanged.')
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],summaries=summaries,failed_extra_ticks=len(failed_ticks),solver_niters=sorted({r['solver_niter'] for r in failed_ticks}),exact_scope_pairs=result['exact_scope_pairs'])),flush=True)
    if not all(checks.values()):raise ValueError('Retain independent observer-review discrepancy')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reports',type=Path,nargs=2,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
