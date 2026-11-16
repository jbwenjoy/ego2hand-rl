"""Read-only first18 selected-native data and recursive state/warm forecasts."""
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
from supermimic_research.mechanics.pen_selected_response_data import prepare
from supermimic_research.learning.pen_diagnostics import rollout_diagnostics,window_indices,decode_pen_torch
from supermimic_research.learning.pen_path_response import PathResponseWM,ARMS,WEIGHTS
from supermimic_research.learning.pen_response import INPUT_SCHEMA,OUTPUT_SCHEMA
from supermimic_research.learning.world_model import raw_tree_equal
from supermimic_research.strict_json import write_json


def read(p):return json.loads(Path(p).read_text())


def main(args):
    started=time.perf_counter();cp=args.config.resolve();cfg=tomllib.loads(cp.read_text())
    if (cfg['schema'],cfg['mode'],cfg['models'],cfg['horizons'],cfg['cases'],cfg['reference_controls'],cfg['device'])!=(1,'READ_ONLY_RECURSIVE_SELECTED_NATIVE_CONTROLS',ARMS,[1,2,4,8,16],18,368,'cpu'):raise ValueError('Fixed inspected recurrent diagnostic required')
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['review_index','model_fit','protocol','registration','preparation','output']};index=read(paths['review_index']);reg=read(verify_file_pin(index['registration']));fit=read(paths['model_fit'])
    if len(index['reviewed_cases'])!=18 or [x['name'] for x in index['reviewed_cases']]!=[x['name'] for x in reg['cases'][:18]] or fit['update']!=2000 or fit['training_running'] or reg['inputs']['model_fit']!=file_pin(paths['model_fit']) or reg['inputs']['model_checkpoint']!=fit['checkpoint']:raise ValueError('Exact reviewed first18/final model owner required')
    inputs={k:file_pin(paths[k]) for k in ['review_index','model_fit','protocol']};inputs['model_checkpoint']=fit['checkpoint'];sources=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,paths['protocol']]
    def freeze(out):
        receipt=freeze_sources(sources,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt);return receipt
    if args.register:
        out=paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh diagnostic registration required')
        out.mkdir();freeze(out);write_json(paths['registration'],dict(schema='pen_selected_model_rollout_registration:1',config=file_pin(cp),inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),horizons=cfg['horizons'],models=ARMS,planned_cases=18,declared_windows={str(h):18*(369-h) for h in cfg['horizons']},native_budget=0,optimizer_budget=0));print(json.dumps(dict(registration=file_pin(paths['registration']))),flush=True);return
    registration=read(paths['registration'])
    if registration['config']!=file_pin(cp) or registration['inputs']!=inputs:raise ValueError('Registered diagnostic changed')
    verify_source_snapshot(read(verify_file_pin(registration['source_receipt'])),live=True);torch.set_num_threads(4)
    out=paths['preparation'].parent if args.prepare else paths['output']
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh diagnostic stage output required')
    out.mkdir();receipt=freeze(out)
    with patch.object(mujoco,'mj_step',side_effect=AssertionError('No native integration')),patch.object(mujoco,'mj_forward',side_effect=AssertionError('No forward simulation')):
        if args.prepare:
            data,metadata,data_inputs,row_checks=prepare(index);np.savez_compressed(out/'data.npz',**data);write_json(out/'metadata.json',dict(rows=metadata));counts={}
            for h in cfg['horizons']:
                available=sum(max(sum(data['case_id']==i)-h+1,0) for i in range(18));ids=window_indices(data,h);counts[str(h)]=dict(declared_windows=18*(369-h),available_complete_windows=int(available),physical_qualified_windows=len(ids),physical_masked_windows=int(available)-len(ids),unavailable_after_stopping=18*(369-h)-int(available))
            checks=dict(all1765=len(data['x'])==1765,all18_cases=len(np.unique(data['case_id']))==18,original8_masks=int((~data['valid']).sum())==8,all_encoding_checks=all(all(c.values()) for c in row_checks),all_window_denominators=all(c['declared_windows']==c['physical_qualified_windows']+c['physical_masked_windows']+c['unavailable_after_stopping'] for c in counts.values()))
            result=dict(schema='pen_selected_model_rollout_data:1',config=file_pin(cp),registration=file_pin(paths['registration']),inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),data=file_pin(out/'data.npz'),metadata=file_pin(out/'metadata.json'),packet_inputs=data_inputs,checks=checks,passed=sum(checks.values()),total=len(checks),window_counts=counts,actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,wall_s=time.perf_counter()-started)
        else:
            prep=read(paths['preparation'])
            if prep['registration']!=file_pin(paths['registration']) or prep['passed']!=prep['total'] or not all(prep['checks'].values()):raise ValueError('Completed exact data preparation required')
            verify_source_snapshot(read(verify_file_pin(prep['source_receipt'])),live=True);data=dict(np.load(verify_file_pin(prep['data'])));checkpoint=torch.load(verify_file_pin(fit['checkpoint']),map_location='cpu',weights_only=False);arrays={};results={};checks={}
            for name in ARMS:
                saved=checkpoint['arms'][name];ctx=saved['context']
                if saved['update']!=2000 or saved['input_schema']!=INPUT_SCHEMA or saved['output_schema']!=OUTPUT_SCHEMA or saved['weights']!=WEIGHTS[name]:raise ValueError('Own final model/input/output/weight schema required')
                model=PathResponseWM(ctx['scales'],ctx['physics'],ctx['path_units'],width=512);model.load_state_dict(saved['model'],strict=True);model.eval();model.requires_grad_(False);before={k:v.clone() for k,v in model.state_dict().items()}
                rows,a=rollout_diagnostics(model,data,'cpu',horizons=cfg['horizons']);ids=a['H1_row_indices'][:,0]
                with torch.no_grad():direct=decode_pen_torch(torch.tensor(data['x'][ids,:55]),model.prediction(torch.tensor(data['x'][ids],dtype=torch.float32))).numpy()
                checks[name+'_H1_direct_parity']=np.array_equal(direct,a['H1_states'][:,1]);checks[name+'_unchanged_buffers_weights']=raw_tree_equal(before,model.state_dict())
                for h in cfg['horizons']:
                    checks[name+'_H'+str(h)+'_shared_window_indices']=np.array_equal(a[f'H{h}_row_indices'],window_indices(data,h));rows[str(h)].update(prep['window_counts'][str(h)])
                arrays.update({name+'_'+k:v for k,v in a.items()});results[name]=rows
            np.savez_compressed(out/'arrays.npz',**arrays);result=dict(schema='pen_selected_recursive_model_diagnostic:1',config=file_pin(cp),registration=file_pin(paths['registration']),preparation=file_pin(paths['preparation']),inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),arrays=file_pin(out/'arrays.npz'),models=results,checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,wall_s=time.perf_counter()-started,scope='CPU fixed recorded-PD open-loop prediction only;state/warm observed once per window and recursively predicted. No policy/reference/controller generation or future state reset. All18 inspected development trajectories/all3 models;finite-only errors conditional,nonfinite failures explicit. Not Actor/gradient/native/ego admission;full54 remains separate.')
    verify_source_snapshot(receipt,live=True);write_json(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],native=0,optimizer=0)),flush=True)
    if not all(checks.values()):raise ValueError('Recurrent diagnostic discrepancy retained')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);g=p.add_mutually_exclusive_group();g.add_argument('--register',action='store_true');g.add_argument('--prepare',action='store_true');main(p.parse_args())
