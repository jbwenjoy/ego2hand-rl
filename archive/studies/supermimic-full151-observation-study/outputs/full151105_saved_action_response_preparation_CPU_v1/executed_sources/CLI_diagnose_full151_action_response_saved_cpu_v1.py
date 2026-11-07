"""Default JSON/hash only. Root alone authorizes the saved-only decomposition."""
import argparse,json,os,sys
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_gradient.action_response452105_v1 import declaration,metadata_preflight,run_analysis


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config');p.add_argument('--run',action='store_true');args=p.parse_args();exit_code=0
    if args.run:
        if not args.config:p.error('--run requires an authorized --config')
        r=run_analysis(args.config);exit_code=0 if r['fixed_visible_state_interpretation_permitted'] else 1
        result={k:r[k] for k in ('schema','status','config','fixed_visible_state_interpretation_permitted','elapsed_seconds')}
        if 'counts' in r:result['counts']=r['counts']
        plan=json.loads(Path(args.config).read_text(encoding='utf-8-sig'));result['report_path']=str(Path(plan['output'])/'report.json')
    elif args.config:
        r=metadata_preflight(args.config);result={k:r[k] for k in ('status','config_pin','Torch_imported','NPZ_opened','decompositions')};result['unique_source_files']=len(r['sources'])
    else:result=declaration()
    print(json.dumps(result,indent=2,allow_nan=False))
    return exit_code


if __name__=='__main__':raise SystemExit(main())
