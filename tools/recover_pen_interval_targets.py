"""Recover saved interval paths and their original row/pair coverage;no fitting."""
import argparse
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import mujoco
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.learning.backend_probe_review import require_response_probe
from supermimic_research.runtime.explicit_collision_backend import load_contract
from supermimic_research.mechanics.pen_interval_targets import CHANNELS,interval_targets


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(args):
    started=time.perf_counter();cp=args.config.resolve();cfg=tomllib.loads(cp.read_text())
    if cfg['schema']!=1 or cfg['mode']!='RECOVER_SAVED_INTERVAL_TARGETS_NO_FIT':raise ValueError('Read-only path recovery required')
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['common_data','reward_chain','backend_manifest','objective_config','protocol','registration','output']}
    inputs={k:file_pin(p) for k,p in paths.items() if k not in ['registration','output']};common=read(paths['common_data']);reward=read(paths['reward_chain']);contract=load_contract(paths['backend_manifest'])
    if (common['schema']!='backend_pen_response_common_data:1' or common['collision_backend']!=contract or common['passed']!=common['total'] or not all(common['checks'].values())
        or reward['schema']!='backend_native_reward_chain_audit:1' or reward['passed']!=reward['total'] or not all(reward['checks'].values())
        or reward['inputs']['common_data']!=inputs['common_data'] or reward['inputs']['objective_config']!=inputs['objective_config'] or reward['inputs']['backend_manifest']!=inputs['backend_manifest']):
        raise ValueError('Exact completed common data/reward/backend context required')
    for r in [common,reward]:verify_source_snapshot(read(verify_file_pin(r['source_receipt'])),live=True)
    source_paths=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,paths['protocol']]
    def freeze(out):
        receipt=freeze_sources(source_paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt);return receipt
    if args.register:
        out=paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh interval target registration required')
        out.mkdir();freeze(out);write(paths['registration'],dict(schema='pen_interval_targets_registration:1',config=file_pin(cp),inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),
            channels=CHANNELS,rows={'TRAIN':3240,'VALIDATE':1620},qualified_rows={'TRAIN':3202,'VALIDATE':1591},expected_groups={'TRAIN':960,'VALIDATE':480},
            actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0))
        print(json.dumps(dict(registration=file_pin(paths['registration']))),flush=True);return
    reg=read(paths['registration'])
    if reg['config']!=file_pin(cp) or reg['inputs']!=inputs or reg['channels']!=CHANNELS:raise ValueError('Registered recovery changed')
    verify_source_snapshot(read(verify_file_pin(reg['source_receipt'])),live=True)
    out=paths['output']
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh interval target output required')
    out.mkdir();receipt=freeze(out);objective=tomllib.loads(paths['objective_config'].read_text());contexts=torch.load(verify_file_pin(reward['source_contexts']),map_location='cpu',weights_only=False)
    checks={};cohorts={};models={};rows_processed=0;stage='admission'
    try:
        with patch.object(mujoco,'mj_step',side_effect=AssertionError('Saved path recovery')),patch.object(mujoco,'mj_forward',side_effect=AssertionError('No geometry regeneration')):
            for role in ['TRAIN','VALIDATE']:
                stage=role+'_admission';prep=read(verify_file_pin(common['preparations'][role]));raw=read(verify_file_pin(reward['inputs'][role]));spec_path=verify_file_pin(raw['config'])
                require_response_probe(raw,tomllib.loads(spec_path.read_text()),contract,reward['inputs']['response_registration'],file_pin(spec_path))
                if (reward['inputs'][role+'_preparation']!=common['preparations'][role] or prep['probe']!=reward['inputs'][role]
                    or prep['passed']!=prep['total'] or not all(prep['checks'].values()) or prep['total_rows']!=reg['rows'][role] or prep['qualified_rows']!=reg['qualified_rows'][role]):
                    raise ValueError('Exact frozen raw/prepared/reward row alignment required')
                for r in [prep,raw]:verify_source_snapshot(read(verify_file_pin(r['source_receipt'])),live=True)
                bank=dict(np.load(verify_file_pin(common['outputs'][role])));components=np.load(verify_file_pin(reward['cohorts'][role]['outputs']))
                mask=np.asarray([m['valid'] for m in prep['metadata']],bool)
                if not np.array_equal(mask,bank['valid']) or not np.array_equal(mask,components['physical_valid']):raise ValueError('Original physical row masks differ')
                rows=[];values=[];transformed=[];loss=[];penalty=[];fail=[];error_bits=[];global_bits=[];case_indices=[]
                for m in prep['metadata']:
                    c=raw['cases'][m['case_index']];stage=c['name'];mp=verify_file_pin(c['model'])
                    if str(mp) not in models:models[str(mp)]=mujoco.MjModel.from_binary_path(str(mp))
                    model=models[str(mp)];names=[model.geom(i).name or '' for i in range(model.ngeom)]
                    context=contexts[c['source_name']];phase_valid=1<=context['phase']<context['frames']
                    trace=dict(np.load(verify_file_pin(c['trace'])));window=dict(np.load(verify_file_pin(c['window'])))
                    v=interval_targets(trace,window,model.geom_bodyid,names,c['model_layout']['object_body'],objective,phase_valid=phase_valid)
                    ch=dict(v['checks']);ch['row_identity']=m['row_index']==len(rows) and m['source_name']==c['source_name'] and m['name']==c['name']
                    ch['native_reward_geometry']=abs(float(v['geometry_penalty'])-float(components['values'][len(rows),reward['channels'].index('geometry_penalty')]))<=2e-6
                    rows.append(dict(row_index=m['row_index'],case_index=m['case_index'],name=c['name'],source_name=c['source_name'],original_physical_valid=m['valid'],checks=ch,
                        maximum_trace_window_difference=abs(v['path4']-v['trace_path4']).tolist()))
                    values.append(v['path4']);transformed.append(v['log1p_squared_ratio4']);loss.append(v['geometry_loss']);penalty.append(v['geometry_penalty']);fail.append(v['geometry_failure']);error_bits.append(v['errors']);global_bits.append(v['global_error']);case_indices.append(m['case_index']);rows_processed+=1
                values=np.asarray(values);transformed=np.asarray(transformed);failure=np.asarray(fail,bool)
                packed=dict(path4=values,log1p_squared_ratio4=transformed,geometry_loss=np.asarray(loss),geometry_penalty=np.asarray(penalty),geometry_failure=failure,
                    errors=np.asarray(error_bits,np.int64),global_error=np.asarray(global_bits,np.int64),physical_valid=mask,case_indices=np.asarray(case_indices,np.int64))
                np.savez_compressed(out/(role+'_path_targets.npz'),**packed)
                groups=[];response=[]
                for gi,g in enumerate(prep['response_groups']):
                    p,m=g['rows']['1'],g['rows']['-1'];complete=len(p)==len(m)==2;qualified=bool(complete and mask[p+m].all())
                    if (g['complete'],g['qualified'])!=(complete,qualified):raise ValueError('Original pair masks changed')
                    fd=(values[p].mean(0)-values[m].mean(0))/(2*g['epsilon']) if complete else None
                    groups.append(dict(group_index=gi,source_name=g['source_name'],direction=g['direction'],epsilon=g['epsilon'],complete=complete,qualified=qualified,finite_response_per_PD=None if fd is None else fd.tolist()))
                    if qualified:response.append(fd)
                response=np.asarray(response).reshape(-1,4);coverage={}
                for j,name in enumerate(CHANNELS):
                    covered=values[mask,j];fd=response[:,j]
                    coverage[name]=dict(physical_valid_rows=len(covered),positive_rows=int((covered>0).sum()),zero_rows=int((covered==0).sum()),maximum=float(covered.max()),
                        eligible_pairs=len(fd),nonzero_finite_responses=int((fd!=0).sum()),FD_RMS_per_PD=float(np.sqrt(np.mean(fd*fd))),FD_abs_max_per_PD=float(abs(fd).max()))
                checks[role+'_all_rows']=len(rows)==reg['rows'][role] and int(mask.sum())==reg['qualified_rows'][role]
                checks[role+'_all_original_groups']=len(groups)==reg['expected_groups'][role]
                checks[role+'_all_trace_window_reward_checks']=all(all(r['checks'].values()) for r in rows)
                checks[role+'_all_finite_targets']=bool(all(np.isfinite(v).all() for v in packed.values()))
                cohorts[role]=dict(rows=len(rows),qualified_rows=int(mask.sum()),outputs=file_pin(out/(role+'_path_targets.npz')),row_audits=rows,groups=groups,coverage=coverage,
                    failure_rows=int(failure.sum()),error_rows=int(np.count_nonzero(error_bits)),global_error_rows=int(np.count_nonzero(global_bits)),
                    checks_passed=sum(sum(x['checks'].values()) for x in rows),checks_total=sum(len(x['checks']) for x in rows),
                    zero_channel_scope='No positive/nonzero response coverage cannot qualify a learned failure/contact channel. Physical-invalid rows retained but not admitted to fitting.')
        verify_source_snapshot(receipt,live=True)
        report=dict(schema='pen_saved_interval_targets:1',config=file_pin(cp),registration=file_pin(paths['registration']),inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),
            channels=CHANNELS,cohorts=cohorts,checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,
            wall_s=time.perf_counter()-started,scope='Recovered all original saved interval paths,not newly simulated data or endpoint geometry. Original invalid masks/all groups retained;no NN fit/gradient/Actor/ego claim.')
        write(out/'report.json',report)
        if not all(checks.values()):raise ValueError('Interval target correspondence failure retained')
        print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[report['passed'],report['total']],rows=rows_processed,coverage={k:v['coverage'] for k,v in cohorts.items()})),flush=True)
    except Exception as e:
        write(out/'failure.json',dict(type=type(e).__name__,message=str(e),stage=stage,rows_processed=rows_processed,actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--register',action='store_true');main(p.parse_args())
