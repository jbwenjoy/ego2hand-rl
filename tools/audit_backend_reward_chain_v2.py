"""Frozen read-only reconstruction of all native reward/control branches."""
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
from supermimic_research.runtime.explicit_collision_backend import load_contract
from supermimic_research.runtime.backend_pen_probe import require_backend_probe
from supermimic_research.learning.backend_probe_review import require_response_probe
from supermimic_research.mechanics.pen_reward_chain import CHANNELS,source_context,native_components,controller
from supermimic_research.learning.pen_native_response import repeated_contrast_spread


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def summarize(values,metadata,groups,sources):
    complete={};records=[]
    for gi,g in enumerate(groups):
        p,m=g['rows']['1'],g['rows']['-1'];done=len(p)==len(m)==2
        qualified=bool(done and all(metadata[i]['valid'] for i in p+m))
        if (g['complete'],g['qualified'])!=(done,qualified):raise ValueError('Original masks/completeness changed')
        row=dict(group_index=gi,source_name=g['source_name'],direction=g['direction'],epsilon=g['epsilon'],
                 complete=done,qualified=qualified,raw_amplitude=4*g['epsilon'],finite_response_per_RAW=None)
        if done:
            plus,minus=values[p],values[m];fd=(plus.mean(0)-minus.mean(0))/(8*g['epsilon'])
            spreads={channel:repeated_contrast_spread(plus[:,j:j+1],minus[:,j:j+1],np.ones(1)) for j,channel in enumerate(CHANNELS)}
            row['finite_response_per_RAW']=fd.tolist();row['repeat_spreads']=spreads
            complete[g['source_name'],g['direction'],g['epsilon']]=dict(group=gi,qualified=qualified,fd=fd,spreads=spreads)
        records.append(row)
    screens=[];passed={channel:0 for channel in CHANNELS}
    for source in sources:
        for direction in range(20):
            a,b=complete.get((source,direction,.002)),complete.get((source,direction,.005));terms={}
            for j,channel in enumerate(CHANNELS):
                relative=None;cosine=None;ok=False
                if a is not None and b is not None:
                    x,y=a['fd'][j],b['fd'][j]
                    relative=float(abs(x-y)/abs(y)) if y!=0 else None
                    cosine=float(np.sign(x*y)) if x*y!=0 else None
                    ok=bool(a['qualified'] and b['qualified'] and relative is not None and relative<=.25
                        and cosine is not None and cosine>=.9 and a['spreads'][channel]['contrast_exceeds_observed_spread']
                        and b['spreads'][channel]['contrast_exceeds_observed_spread'])
                terms[channel]=dict(physical_screen=ok,relative_amplitude_error=relative,cosine=cosine);passed[channel]+=ok
            screens.append(dict(source_name=source,direction=direction,channels=terms))
    ids=[i for i,r in enumerate(records) if r['qualified']]
    summaries={}
    for j,channel in enumerate(CHANNELS):
        a=np.asarray([records[i]['finite_response_per_RAW'][j] for i in ids])
        summaries[channel]=dict(qualified_pairs=len(ids),FD_RMS_per_RAW=float(np.sqrt(np.mean(a*a))) if len(a) else None,
                               FD_abs_max_per_RAW=float(abs(a).max()) if len(a) else None)
    geometry=[]
    for i in ids:
        v=np.asarray(records[i]['finite_response_per_RAW']);den=abs(v[4])+abs(v[5])+.2*abs(v[6])
        if den>0:geometry.append(abs(v[5])/den)
    return dict(groups=records,scalar_amplitude_screens=screens,physical_screen_direction_counts=passed,
        component_summary=summaries,geometry_absolute_share_of_tracking_geometry_support=dict(
            pairs=len(geometry),median=float(np.median(geometry)) if geometry else None,
            p95=float(np.quantile(geometry,.95)) if geometry else None,max=float(max(geometry)) if geometry else None,
            share_above_10_percent=int(sum(v>.1 for v in geometry))),
        scope='Finite scalar contrasts per unit RAW;original two-repeat/two-amplitude .25/.9 screens,not infinitesimal/reward-gradient or Actor admission. Geometry share uses sum of absolute component changes,not a cancellation-prone signed ratio.')


def main(args):
    started=time.perf_counter();cp=args.config.resolve();cfg=tomllib.loads(cp.read_text())
    if cfg['schema']!=1 or cfg['mode']!='READ_ONLY_NATIVE_REWARD_CHAIN':raise ValueError('Frozen native reconstruction protocol required')
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['TRAIN','VALIDATE','TRAIN_preparation','VALIDATE_preparation','common_data',
        'response_registration','backend_manifest','objective_config','adapter_config','protocol','registration','output']}
    inputs={k:file_pin(v) for k,v in paths.items() if k not in ['registration','output']}
    contract=load_contract(paths['backend_manifest']);objective=tomllib.loads(paths['objective_config'].read_text());adapter=tomllib.loads(paths['adapter_config'].read_text())
    common=read(paths['common_data'])
    if common['schema']!='backend_pen_response_common_data:1' or common['collision_backend']!=contract or common['passed']!=common['total'] or not all(common['checks'].values()):
        raise ValueError('Exact reviewed new-backend common products required')
    sources=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,paths['protocol']]
    def freeze(out):
        receipt=freeze_sources(sources,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        write(out/'source_receipt.json',receipt);return receipt
    if args.register:
        out=paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh reward-chain registration required')
        out.mkdir();freeze(out);write(paths['registration'],dict(schema='backend_reward_chain_registration:1',config=file_pin(cp),inputs=inputs,
            source_receipt=file_pin(out/'source_receipt.json'),one_case_formula_preflight=file_pin(ROOT/'runs/pen_reward_chain_preflight_v1.log'),
            actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0))
        print(json.dumps(dict(registration=file_pin(paths['registration']))),flush=True);return
    registration=read(paths['registration'])
    if registration['config']!=file_pin(cp) or registration['inputs']!=inputs:raise ValueError('Registered reward analysis changed')
    verify_source_snapshot(read(verify_file_pin(registration['source_receipt'])),live=True)
    out=paths['output']
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh reward analysis output required')
    out.mkdir();receipt=freeze(out);checks={};cohorts={};all_contexts={};native_rows=0;phase='admission'
    try:
        for role in ['TRAIN','VALIDATE']:
            phase=role+'_admission';raw=read(paths[role]);prepared=read(paths[role+'_preparation']);spec_path=verify_file_pin(raw['config']);spec=tomllib.loads(spec_path.read_text())
            require_response_probe(raw,spec,contract,inputs['response_registration'],file_pin(spec_path))
            if prepared['probe']!=inputs[role] or common['preparations'][role]!=inputs[role+'_preparation'] or prepared['passed']!=prepared['total'] or not all(prepared['checks'].values()):
                raise ValueError('Exact completed raw/prepared products required')
            for entry in [raw,prepared]:verify_source_snapshot(read(verify_file_pin(entry['source_receipt'])),live=True)
            exemplar=next(c for c in raw['cases'] if c['available']);contexts={};source_diagnostics=[]
            for source in raw['sources']:
                state=torch.load(verify_file_pin(source['source']),map_location='cpu',weights_only=False);require_backend_probe(state,contract)
                verify_source_snapshot(state['native']['sources'],live=True)
                if state['native']['metadata']['objective_config']!=inputs['objective_config']:raise ValueError('Source reward config differs')
                context=contexts[source['name']]=source_context(state,exemplar);all_contexts[source['name']]=context
                base=torch.tensor(state['base_raw'][None],requires_grad=True);PD,_,applied=controller(base,context)
                diagonal=torch.autograd.grad(PD.sum(),base)[0][0].numpy()
                source_diagnostics.append(dict(name=source['name'],source_qualified=source['qualified'],
                    base_raw_clipped_components=int((abs(state['base_raw'])>1).sum()),analytic_PD_RAW_diagonal=diagonal.tolist(),
                    quarter_gain_components=int((diagonal==.25).sum()),zero_gain_components=int((diagonal==0).sum())))
            values=[];rows=[]
            with patch.object(mujoco,'mj_step',side_effect=AssertionError('Read-only reward review')),patch.object(mujoco,'mj_forward',side_effect=AssertionError('Saved packets only')):
                for m in prepared['metadata']:
                    c=raw['cases'][m['case_index']];phase=c['name'];trace=dict(np.load(verify_file_pin(c['trace'])));window=dict(np.load(verify_file_pin(c['window'])))
                    v,ch,error=native_components(c,trace,window,contexts[c['source_name']],adapter['reward'],objective)
                    values.append(v);rows.append(dict(row_index=m['row_index'],case_index=m['case_index'],name=c['name'],source_name=c['source_name'],
                        original_physical_valid=m['valid'],checks=ch,errors=error));native_rows+=1
            values=np.asarray(values);np.savez_compressed(out/(role+'_native_components.npz'),values=values,physical_valid=np.asarray([m['valid'] for m in prepared['metadata']],bool))
            checks[role+'_all_rows_retained']=len(rows)==prepared['total_rows'] and all(r['row_index']==i for i,r in enumerate(rows))
            checks[role+'_all_formula_and_control_checks']=all(all(r['checks'].values()) for r in rows)
            checks[role+'_finite_components']=bool(np.isfinite(values).all())
            cohorts[role]=dict(rows=len(rows),qualified_rows=prepared['qualified_rows'],row_audits=rows,source_controller=source_diagnostics,
                outputs=file_pin(out/(role+'_native_components.npz')),row_check_passed=sum(sum(r['checks'].values()) for r in rows),
                row_check_total=sum(len(r['checks']) for r in rows),max_errors={k:max(r['errors'][k] for r in rows) for k in rows[0]['errors']},
                native_reward_responses=summarize(values,prepared['metadata'],prepared['response_groups'],[s['name'] for s in raw['sources']]))
        verify_source_snapshot(receipt,live=True);torch.save(all_contexts,out/'source_contexts.pt')
        report=dict(schema='backend_native_reward_chain_audit:1',config=file_pin(cp),registration=file_pin(paths['registration']),inputs=inputs,
            source_receipt=file_pin(out/'source_receipt.json'),channels=CHANNELS,source_contexts=file_pin(out/'source_contexts.pt'),cohorts=cohorts,
            checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,
            scalar_reward_scope='Full reward is reconstructed from independently checked recorded tracking/geometry/support components;no separate raw reward tensor was saved. Both fixed0/.2 objectives are evaluated without changing native data.',
            missing_WM_outputs=['8-substep geometry_max3','8-substep nonhand_max','geometry/global-error failure flags'],
            scope='Native controller/reward decomposition and finite RAW responses only. Endpoint/mean-wrench WM cannot supply all path-dependent geometry terms. No model reward-gradient,Actor,task,ego or acceleration admission.',wall_s=time.perf_counter()-started)
        write(out/'report.json',report)
        if not all(checks.values()):raise ValueError('Retain native reward reconstruction discrepancy')
        print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[report['passed'],report['total']],rows=native_rows)),flush=True)
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),phase=phase,reviewed_rows=native_rows,
            actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--register',action='store_true');main(p.parse_args())
