"""Root explicit exploratory CPU task-score gradients; default JSON/hash only."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from full151_gradient import component_task_gradient105_v1 as core

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path);p.add_argument('--output',type=Path);p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.protocol is None:
        if a.run:p.error('--run requires --protocol and --output')
        print(json.dumps(core.declaration()));return
    if not a.run:
        r=core.metadata_preflight(a.protocol,a.output);print(json.dumps({'status':r['status'],'protocol':r['protocol_pin'],'test_array_values_decoded':0,'trusted_model_loads_FW_AD':0}));return
    if a.output is None:p.error('--run requires a fresh --output')
    r=core.run_analysis(a.protocol,a.output);print(json.dumps({'status':r['status'],'report':str(a.output/'report.json'),'counts':r['counts'],'coverage':r['coverage'],'elapsed_seconds':r['elapsed_seconds']}))

if __name__=='__main__':main()
