"""Explicit recovery of completed eight-case data after final sys-enumeration failure."""
import argparse
import json
import sys
import tomllib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import verify_source_snapshot
from supermimic_research.runtime.controlled_evaluation import write


def main(a):
    folder=a.collection.resolve();intent=json.loads((folder/'intent.json').read_text());progress=json.loads((folder/'progress.json').read_text())
    cfg=tomllib.loads(verify_file_pin(intent['config']).read_text())
    log=a.log.read_text(encoding='utf8');receipt=json.loads(verify_file_pin(intent['source_receipt']).read_text())
    if (a.output.exists() or (folder/'report.json').exists() or cfg.get('experiment')!='VELOCITY'
            or "NameError: name 'sys' is not defined" not in log
            or 'for name,module in list(sys.modules.items())' not in log):
        raise ValueError('Only the recorded final sys-enumeration failure, no existing final report')
    verify_source_snapshot(receipt,live=True)
    expected={(mode,dt) for mode in cfg['controllers'] for dt in cfg['dt']}
    cases=progress['cases']
    if len(cases)!=8 or {(c['name'],c['dt']) for c in cases}!=expected:raise ValueError('All predeclared eight completed cases required')
    for c in cases:
        casefile=folder/f"{c['name']}_{c['dt']}"/'report.json'
        if json.loads(casefile.read_text())!=c:raise ValueError('Progress differs from completed per-case report')
        for key in ['trace','host_topology','GPU_physical_context','fixed_model_arrays','fixed_model_scalars']:verify_file_pin(c[key])
        for d in c['planning']:
            verify_file_pin(d['source']);verify_file_pin(d['prediction'])
        if c['candidate_physics_ticks']!=57*len(c['planning'])*round(.02/c['dt']):raise ValueError('Incomplete candidate work')
    a.output.mkdir()
    result=dict(**intent,cases=cases,task_sources=[],source_import_enumeration_available=False,
        collection_status='EIGHT_CASES_COMPLETED_FINAL_METADATA_ASSEMBLY_FAILED',
        recovery=dict(source=file_pin(__file__),intent=file_pin(folder/'intent.json'),progress=file_pin(folder/'progress.json'),log=file_pin(a.log),
            completed_case_reports=[file_pin(folder/f"{c['name']}_{c['dt']}"/'report.json') for c in cases],
            native_integrations=0,neural_updates=0,
            scope='Read-only recovery of complete cases. Source superset equality rechecked live before repair. Original executed-module enumeration was never saved and is unavailable; not retroactively reconstructed from a new process. Original failed collection remains unchanged.'))
    write(a.output/'report.json',result)
    print(json.dumps(dict(report=file_pin(a.output/'report.json'),cases=[dict(mode=c['name'],dt=c['dt'],duration=c['summary']['metrics']['actual_duration_s'],qualified=c['summary']['qualified']) for c in cases])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True);p.add_argument('--log',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
