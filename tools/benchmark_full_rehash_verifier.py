"""Fixed immutable source-pin microbenchmark;not native-loader integration."""
import argparse
import cProfile
import json
import pstats
import subprocess
import sys
import time
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.strict_file_verification import verify_file_pin_single_resolve


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(args):
    cp=args.config.resolve();cfg=tomllib.loads(cp.read_text())
    if (cfg['schema'],cfg['repeats'],cfg['source_names'])!=(1,5,['RESIDUAL_N8841_P024','RESIDUAL_N8861_P018']):raise ValueError('Two frozen source catalogs x5 matched timing rounds required')
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['TRAIN','VALIDATE','protocol','registration','output']}
    catalogs=[]
    for role,name in zip(['TRAIN','VALIDATE'],cfg['source_names']):
        r=read(paths[role]);entry=next(s for s in r['sources'] if s['name']==name);state=torch.load(verify_file_pin(entry['source']),map_location='cpu',weights_only=False)
        manifest=state['native']['sources'];verify_source_snapshot(manifest,live=True)
        # Original validate_state_sources order:snapshots/observed,then engine.
        pins=[p for record in manifest['records'] for p in [record['snapshot'],record['observed']]]+state['native']['engine_sources']
        catalogs.append(dict(name=name,source=entry['source'],pins=pins))
    inputs={role:file_pin(paths[role]) for role in ['TRAIN','VALIDATE']};inputs['protocol']=file_pin(paths['protocol'])
    source_paths=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,paths['protocol'],ROOT/'tests/test_strict_file_verification.py']
    def freeze(out):
        receipt=freeze_sources(source_paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt);return receipt
    if args.register:
        out=paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh verifier registration required')
        out.mkdir();freeze(out);write(paths['registration'],dict(schema='single_resolve_full_rehash_registration:1',config=file_pin(cp),inputs=inputs,
            catalogs=catalogs,source_receipt=file_pin(out/'source_receipt.json'),native_or_optimizer_budget=0))
        print(json.dumps(dict(registration=file_pin(paths['registration']))),flush=True);return
    reg=read(paths['registration'])
    if reg['config']!=file_pin(cp) or reg['inputs']!=inputs or reg['catalogs']!=catalogs:raise ValueError('Registered verifier inputs changed')
    verify_source_snapshot(read(verify_file_pin(reg['source_receipt'])),live=True)
    out=paths['output']
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh verifier microbenchmark output required')
    out.mkdir();receipt=freeze(out);methods=dict(ORIGINAL=verify_file_pin,SINGLE_RESOLVE=verify_file_pin_single_resolve);rows=[];profiles={};checks={};calls=0
    def verify_all(pins,fn):
        nonlocal calls
        result=[]
        for pin in pins:result.append(str(fn(pin)));calls+=1
        return result
    for catalog in catalogs:
        pins=catalog['pins'];gold=verify_all(pins,methods['ORIGINAL']);candidate=verify_all(pins,methods['SINGLE_RESOLVE'])
        checks[catalog['name']+'_canonical_paths_exact']=gold==candidate
        for repeat in range(5):
            order=['ORIGINAL','SINGLE_RESOLVE'] if repeat%2==0 else ['SINGLE_RESOLVE','ORIGINAL']
            for name in order:
                start=time.perf_counter();observed=verify_all(pins,methods[name]);wall=time.perf_counter()-start
                checks[f'{catalog["name"]}_{name}_{repeat}_exact']=observed==gold
                rows.append(dict(catalog=catalog['name'],mode=name,repeat=repeat,pins=len(pins),wall_s=wall))
        for name,fn in methods.items():
            prof=cProfile.Profile();prof.enable();verify_all(pins,fn);prof.disable();label=catalog['name']+'_'+name
            prof.dump_stats(str(out/(label+'.prof')));s=pstats.Stats(prof)
            functions=[dict(file=f,line=line,function=function,calls=v[1],self_s=v[2],cumulative_s=v[3]) for (f,line,function),v in s.stats.items()]
            profiles[label]=dict(profile=file_pin(out/(label+'.prof')),resolve_calls=sum(v['calls'] for v in functions if v['function']=='resolve'),
                final_path_calls=sum(v['calls'] for v in functions if '_getfinalpathname' in v['function']),
                full_file_reads=sum(v['calls'] for v in functions if v['function']=="<method 'read' of '_io.BufferedReader' objects>"),
                functions=functions)
    summary={}
    for catalog in catalogs:
        times={mode:float(np.median([r['wall_s'] for r in rows if r['catalog']==catalog['name'] and r['mode']==mode])) for mode in methods}
        summary[catalog['name']]=dict(median_wall_s=times,original_over_candidate_ratio=times['ORIGINAL']/times['SINGLE_RESOLVE'],
            time_reduction_fraction=1-times['SINGLE_RESOLVE']/times['ORIGINAL'])
        a,b=profiles[catalog['name']+'_ORIGINAL'],profiles[catalog['name']+'_SINGLE_RESOLVE']
        checks[catalog['name']+'_resolve_count_half']=a['resolve_calls']==2*b['resolve_calls']==2*len(catalog['pins'])
        checks[catalog['name']+'_full_read_calls_equal']=a['full_file_reads']==b['full_file_reads']
    verify_source_snapshot(receipt,live=True)
    r=dict(schema='single_resolve_full_rehash_benchmark:1',config=file_pin(cp),registration=file_pin(paths['registration']),inputs=inputs,
        source_receipt=file_pin(out/'source_receipt.json'),catalogs=catalogs,timings=rows,profiles=profiles,summary=summary,
        checks=checks,passed=sum(checks.values()),total=len(checks),complete_hash_verifications=calls,
        actual_native_integrations=0,optimizer_steps=0,default_verifier_changed=False,native_integration_tested=False,
        scope='Explicit candidate always rereads/full-hashes bytes and checks size,one rather than two Path.resolve calls. Same ordinary concurrency/TOCTOU limitations as original,no cache/epoch. Fixed-file microbenchmark only;not source-loader/collection/training speedup or Actor/ego admission.')
    write(out/'report.json',r)
    if not all(checks.values()):raise ValueError('Keep verifier equivalence/count discrepancy')
    print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[r['passed'],r['total']],summary=summary)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--register',action='store_true');main(p.parse_args())
