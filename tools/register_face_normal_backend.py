"""Freeze the exact statically-qualified source identity, not a promotion claim."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.runtime.explicit_collision_backend import digest,load_contract


def main():
    output=ROOT/'configs/collision_backend_face_normal_v1.json'
    if output.exists():raise ValueError('No backend identity replacement')
    ledger_path=ROOT/'provenance/pen_ccd_recovery_candidate_v1/ledger.json';ledger=json.loads(ledger_path.read_text())
    summary=json.loads(verify_file_pin(ledger['summary']).read_text())
    if summary['passed']!=summary['total'] or not summary['candidate_static_fixture_checks_passed'] or summary['production_backend_or_state_admitted']:
        raise ValueError('Exact static-only admission evidence required')
    report_path=verify_file_pin(ledger['roots']['face_normal']);report=json.loads(report_path.read_text())
    identity=dict(schema='pinned_EPA_face_normal_backend:1',sources=report['source_variants'],installed_sources=report['installed_sources'],
        adapter_source=file_pin(ROOT/'src/supermimic_research/runtime/explicit_collision_backend.py'),static_ledger=file_pin(ledger_path),static_report=file_pin(report_path),
        settings=dict(horizon=96,face_factor=20,max_axis_retries=3,ccd_iterations=50,ccd_tolerance=1e-6,quiet=False),
        scope='Exact static-qualified artifact. Full scene/dynamics/state/training admission is a separate study result.')
    for p in identity['sources']+identity['installed_sources']:verify_file_pin(p)
    output.write_text(json.dumps(dict(schema='explicit_collision_backend_manifest:1',backend_id=digest(identity),identity=identity),indent=2)+'\n',encoding='utf-8')
    contract=load_contract(output);print(json.dumps(dict(manifest=file_pin(output),backend_id=contract['backend_id'])),flush=True)


if __name__=='__main__':main()
