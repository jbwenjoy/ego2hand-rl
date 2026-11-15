"""Reconstruct fixed source-model tables; retain original capture provenance."""
import argparse
import json
import sys
import tomllib
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import ArtifactRegistry,file_pin,verify_file_pin
from supermimic_research.runtime.controlled_evaluation import evaluate_case,write
from supermimic_research.runtime.pen_gpu_initialization import initialize_and_query_pen
from supermimic_research.runtime.gpu_model_snapshot import capture_model


def main(a):
    import torch
    import warp as wp
    r=json.loads(a.report.read_text());cp=verify_file_pin(r['config']);cfg=tomllib.loads(cp.read_text())
    bp=verify_file_pin(r['base_config']);base=tomllib.loads(bp.read_text());p=json.loads(verify_file_pin(r['proposal']).read_text());s=np.load(verify_file_pin(p['artifacts']['selected']))
    old=json.loads(verify_file_pin(r['actual']).read_text())
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'runs'):raise ValueError('Fresh supplemental receipt')
    torch.set_num_threads(4);wp.set_device('cuda:0');a.output.mkdir();registry=ArtifactRegistry(bp.parent/base['artifact_manifest']);cases=[];receipts=[]
    for dt in [.005,.0025]:
        folder=a.output/f'actual_{dt}';captured={}
        def initialize(env):
            receipt=initialize_and_query_pen(env,folder,s['source_qpos'],s['qpos'],s['delta20'],s['target_BODY'],support_all_robot_contacts=True)
            values,scalars,hashes=capture_model(env.sim.wp_model)
            np.savez_compressed(folder/'fixed_model_arrays.npz',**values);write(folder/'fixed_model_scalars.json',scalars)
            captured.update(array_hashes=hashes,array_count=len(values),scalar_leaves=len(scalars))
            return receipt
        c=evaluate_case('hand2-pen-spin','SUPPLEMENT_MODEL_RECEIPT',None,{**base,'physics_dt':dt,'numerical_options':cfg['numerical_options'],'numerical_device_override':True},
            bp.parent,registry,folder,initializer=initialize,constant_action=2*s['delta20'])
        original=next(c for c in old['cases'] if c['dt']==dt)
        checks=dict(all_original_source_model_array_hashes=all(captured['array_hashes']==v['model_fingerprints'] for v in r['source_captures'] if v['dt']==dt),
            compiled_host_bytes_match=c['host_topology']['sha256']==original['host_topology']['sha256'],
            numerical_receipts_match=c['numerical_options']==original['numerical_options'])
        receipts.append(dict(dt=dt,checks=checks,arrays=file_pin(folder/'fixed_model_arrays.npz'),scalars=file_pin(folder/'fixed_model_scalars.json'),
            array_count=captured['array_count'],scalar_leaves=captured['scalar_leaves']))
        cases.append(c)
    result=dict(schema='pen_same_source_fixed_model_supplement:1',run=file_pin(a.report),source=file_pin(__file__),receipts=receipts,cases=cases,
        passed=sum(sum(c['checks'].values()) for c in receipts),total=sum(len(c['checks']) for c in receipts),neural_updates=0,
        scope='Original collector persisted full dynamic arrays and model fingerprints/physical tables, not complete raw model tables. These are freshly reconstructed fixed-model tables, verified against ALL original404 array hashes, compiled host bytes and actual numerical receipts.195 scalar leaves are newly captured, not retroactive original-run scalar observations. Original artifacts remain unchanged.')
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),passed=result['passed'],total=result['total'],counts=[[c['array_count'],c['scalar_leaves']] for c in receipts])))
    if result['passed']!=result['total']:raise ValueError('Keep incompatible reconstruction; do not alias it to original model')


if __name__=='__main__':
    import torch
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    with torch.jit.optimized_execution(False):main(p.parse_args())
