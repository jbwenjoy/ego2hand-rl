"""Read-only world-model JVP versus never-fit native action secants."""
from pathlib import Path
import argparse,json,os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.learning.response_training import restore_response
from supermimic_research.learning.response_diagnostics import probe_metrics


def main(args):
    out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified diagnostic output')
    health=json.loads(args.health.read_text());probe=json.loads(args.probes.read_text())
    if health['passed']!=health['total'] or not all(x['passed'] for x in probe['barriers']) or probe['role']!='CHECK_ONLY_NEVER_FIT_OR_SELECT':raise ValueError('Reviewed healthy own checkpoint and CHECK probes')
    for pin in health['sources']+probe['sources']:verify_file_pin(pin)
    saved=torch.load(verify_file_pin(health['checkpoint']),map_location='cpu',weights_only=False);device=torch.device(saved['device'])
    torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,_,_=restore_response(saved,device);raw=np.load(verify_file_pin(probe['probes']));metrics,arrays=probe_metrics(model,raw,device)
    out.mkdir(parents=True);np.savez_compressed(out/'derivatives.npz',**arrays)
    result=dict(schema='mug_response_native_derivative_review:1',health=file_pin(args.health),probes=file_pin(args.probes),
        metrics=metrics,derivatives=file_pin(out/'derivatives.npz'),neural_updates=0,native_steps=0,
        sources=[file_pin(__file__),file_pin(ROOT/'src/supermimic_research/learning/response_diagnostics.py')])
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(report=file_pin(out/'report.json'),metrics=metrics)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--health',type=Path,required=True);p.add_argument('--probes',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
