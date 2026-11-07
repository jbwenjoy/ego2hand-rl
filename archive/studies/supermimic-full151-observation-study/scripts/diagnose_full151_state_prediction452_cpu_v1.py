"""Default JSON/hash only; actual fixed-model run needs Root's CPU plan."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_gradient.state_prediction452105_v1 import declaration,metadata_preflight,run_analysis


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config');p.add_argument('--run',action='store_true');args=p.parse_args()
    if args.run:
        if not args.config:p.error('--run requires explicit --config')
        r=run_analysis(args.config)
        # Full pins and all per-feature/stratum results remain in report.json.
        result={k:r[k] for k in ('schema','status','config','counts','prediction_rows','WM_batch_size','elapsed_seconds')}
        plan=json.loads(Path(args.config).read_text(encoding='utf-8-sig'));result['report_path']=str(Path(plan['output'])/'report.json')
        result['all452_blocks']=r['statistics']['all452']['blocks']
    elif args.config:
        r=metadata_preflight(args.config);result={k:r[k] for k in ('status','config_pin','Torch_imported','NPZ_opened','model_loads','WM_forwards','actual_inputs_assembled')}
        result['unique_source_files']=len(r['unique_sources'])
    else:result=declaration()
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
