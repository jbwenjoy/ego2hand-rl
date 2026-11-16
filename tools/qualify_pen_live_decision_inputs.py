"""Regress the live-input adapter on all36 frozen sources,not live execution."""
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.learning.pen_live_decision import from_owned_snapshot
from supermimic_research.learning.world_model import raw_tree_equal
from supermimic_research.runtime.backend_pen_probe import require_backend_probe
from supermimic_research.runtime.explicit_collision_backend import load_contract
from supermimic_research.strict_json import write_json


def read(p):return json.loads(Path(p).read_text())


def main():
    bp=ROOT/'runs/pen_native_suggestions_bank_v1/report.json';bank=read(bp);out=ROOT/'runs/pen_live_decision_inputs_v1'
    if out.exists() or bank['passed']!=bank['total'] or not all(bank['checks'].values()):raise ValueError('Fresh admitted input regression required')
    verify_source_snapshot(read(verify_file_pin(bank['source_receipt'])),live=True);contract=load_contract(verify_file_pin(bank['inputs']['backend_manifest']))
    evaluation=read(verify_file_pin(bank['inputs']['evaluation']));fit=read(verify_file_pin(bank['inputs']['fit']));common=read(verify_file_pin(fit['inputs']['common_data']));native=read(verify_file_pin(evaluation['inputs']['native_reward']))
    contexts=torch.load(verify_file_pin(native['source_contexts']),map_location='cpu',weights_only=False);rows=[];checks={}
    for role,pp in common['preparations'].items():
        p=read(verify_file_pin(pp));raw=read(verify_file_pin(p['probe']));exemplar=next(c for c in raw['cases'] if c['available']);data=np.load(verify_file_pin(common['outputs'][role]))
        for entry in [s for s in bank['sources'] if s['role']==role]:
            source=torch.load(verify_file_pin(entry['original_source']['source']),map_location='cpu',weights_only=False);require_backend_probe(source,contract)
            case=entry['template'] if entry['template'] is not None else exemplar;result=from_owned_snapshot(source['native'],source['phase'],case,source['base_raw']);original=entry['original_base_row']
            numeric=True if original is None else np.array_equal(result['input102'],data['x'][original].astype(np.float32))
            rows.append(dict(name=entry['name'],role=role,observed_original_BASE=original is not None,clipped_components=result['clipped_components'],checks=dict(
                finite_input=bool(np.isfinite(result['input102']).all()),old_actual_input_bytes=numeric,original_context_bytes=raw_tree_equal(result['controller_context'],contexts[entry['name']]),sampled_RAW_bytes=np.array_equal(result['sampled_RAW20'],source['base_raw']))))
    checks=dict(all36_sources=len(rows)==36,all30_original_inputs=sum(r['observed_original_BASE'] for r in rows)==30,all_row_checks=all(all(r['checks'].values()) for r in rows),six_original_excluded_sources_retained=sum(not r['observed_original_BASE'] for r in rows)==6)
    if not all(checks.values()):raise ValueError('Live decision input regression mismatch: '+repr([r['name'] for r in rows if not all(r['checks'].values())]))
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__)],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt)
    result=dict(schema='pen_live_decision_input_regression:1',bank=file_pin(bp),source_receipt=file_pin(out/'source_receipt.json'),rows=rows,checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,
        scope='All36 historical owned snapshots and30 existing network input byte comparisons only. Six clipped sources retain sampled/applied distinction. No new live capture/restore/next-step,Actor or trajectory qualification.')
    write_json(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],inputs=30,sources=36)),flush=True)


if __name__=='__main__':main()
