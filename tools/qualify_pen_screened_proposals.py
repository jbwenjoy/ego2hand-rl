"""Read-only CPU real-model gradient/proposal checks including actual clipped RAW."""
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.shared_snapshot_transport import load_shared
from supermimic_research.learning.pen_live_decision import from_owned_snapshot
from supermimic_research.learning.pen_path_response import PathResponseWM
from supermimic_research.learning.pen_path_decomposition import gradients
from supermimic_research.learning.backend_reward_gradient import batch_context
from supermimic_research.learning.pen_native_screening import proposal,choose
from supermimic_research.strict_json import write_json
from run_pen_native_screened_rollout import declarations
import tomllib


def read(p):return json.loads(Path(p).read_text())


def main():
    started=time.perf_counter();out=ROOT/'runs/pen_screened_proposal_qa_v1';fitp=ROOT/'runs/pen_path_fit_v1/report.json';fit=read(fitp);cp=ROOT/'configs/pen_native_screened_rollout_design_v1.toml';design=tomllib.loads(cp.read_text());node=read(ROOT/'runs/pen_live_node_qualification_v1/report.json');paths=[ROOT/'runs/pen_terminal_lifecycle_v1'/a/'CLIPPED/report.json' for a in ['RESIDUAL','SUPPORT']]
    if out.exists():raise ValueError('Fresh real-model proposal QA required')
    inputs=dict(model=fit['checkpoint'],fit=file_pin(fitp),design=file_pin(cp),clipped_cases=[file_pin(p) for p in paths],node=file_pin(ROOT/'runs/pen_live_node_qualification_v1/report.json'))
    out.mkdir();sources=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'tools/run_pen_native_screened_rollout.py',ROOT/'configs/pen_native_screened_rollout_v1.toml',ROOT/'docs/pen-native-screened-rollout-protocol-v1.md'];receipt=freeze_sources(sources,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt)
    ck=torch.load(verify_file_pin(fit['checkpoint']),map_location='cpu',weights_only=False);cases=declarations(design);rows=[];arrays={};torch.set_num_threads(4)
    objective=tomllib.loads(verify_file_pin(node['inputs']['objective_config']).read_text());adapter=(Path(node['inputs']['objective_config']['path']).parent/objective['adapter_config']).resolve();reward=tomllib.loads(adapter.read_text())['reward']
    normal=[c for c in node['branches'] if c['name'] in ['GZIP_SHARED_IMMUTABLE_RESIDUAL_BASE','GZIP_SHARED_IMMUTABLE_SUPPORT_BASE']];clipped=[read(p) for p in paths]
    for name,saved in ck['arms'].items():
        ctx=saved['context'];model=PathResponseWM(ctx['scales'],ctx['physics'],ctx['path_units'],width=512);model.load_state_dict(saved['model']);model.eval();model.requires_grad_(False)
        for ci,case in enumerate(normal+clipped):
            source=load_shared(case['source']);raw=np.load(verify_file_pin(case['trace']))['actions20'][0];decision=from_owned_snapshot(source,case['phase'],case,raw);g=gradients(model,torch.tensor(decision['input102'][None]),torch.tensor(raw[None]),batch_context({'live':decision['controller_context']},['live'],'cpu'),reward,objective);ga={k:v.numpy() for k,v in g.items()};prefix=name+'_'+str(ci);arrays.update({prefix+'_'+k:v for k,v in ga.items()})
            for mode,mi in [('OBSERVABLE',0),('FULL',2)]:
                for goal in range(2):
                    p=proposal(raw,ga['RAW'][0,mi,goal],prediction_finite=bool(np.isfinite(ga['values']).all()));checks=dict(actual_PD_bytes=np.array_equal(ga['computed_PD'][0],decision['actual_PD20']),finite_gradient=bool(np.isfinite(ga['RAW']).all()),clipped_chain_zero=ci<2 or ga['RAW'][0,mi,goal,0]==0,
                        original_RAW_preserved=p['raw'][0]==raw[0] if ci>=2 else True,changed_coordinates_unclipped=not bool(np.any((p['raw']!=raw)&(abs(p['raw'])>1))))
                    rows.append(dict(model=name,case=case['name'],mode=mode,goal=goal,checks=checks,duplicate_BASE=p['duplicate_BASE'],reason=p['reason']))
    checks=dict(all54_unique=len(cases)==len(set(c['name'] for c in cases))==54,all18_each_repeat=all(sum(c['repetition']==r for c in cases)==18 for r in range(3)),all48_real_gradient_rows=len(rows)==48,all_real_proposal_contracts=all(all(r['checks'].values()) for r in rows),zero_optimizer_native=True)
    np.savez_compressed(out/'arrays.npz',**arrays);verify_source_snapshot(receipt,live=True);r=dict(schema='pen_real_model_proposal_CPU_qa:1',inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),arrays=file_pin(out/'arrays.npz'),checks=checks,passed=sum(checks.values()),total=len(checks),rows=rows,actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,wall_s=time.perf_counter()-started,
        scope='CPU real frozen model/source proposal and saturated controller-chain QA. Not CUDA parity or native proposal success. All54 case declaration checked,full matrix not launched by this tool.')
    write_json(out/'report.json',r);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[r['passed'],r['total']],rows=len(rows))),flush=True)
    if not all(checks.values()):raise ValueError('Real proposal discrepancy retained')


if __name__=='__main__':main()
