"""Pin actual completed collector metadata for one saved-only export.

No future hashes are invented. --describe works before native data exist.
The default generated protocol is CPUFalse; Root may explicitly authorize
saved arithmetic after real terminal completion. No native or NN route exists.
"""
from pathlib import Path
import argparse,ast,hashlib,json
H=Path(__file__).resolve().parent;P=H.parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--describe',action='store_true');ap.add_argument('--collector-report',type=Path);ap.add_argument('--collector-terminal',type=Path)
    ap.add_argument('--output',type=Path);ap.add_argument('--protocol-output',type=Path,default=H/'protocol_CPUFalse_v1.json');ap.add_argument('--authorize-saved-run',action='store_true');a=ap.parse_args()
    if a.describe:
        print(json.dumps(dict(schema='phone_tripod_replay_dataset_SOURCE_declaration:1',report_schema='phone_tripod_replay_training_dataset:1',
            status='SOURCE_ONLY_PENDING_ACTUAL_COLLECTOR_REPORT',inputs=['collector_report','collector_terminal','family_plan','task_bundle','physical_parameters','geometry_metadata','bridge_qualification'],
            partition_arrays=['train','monitor'],families=dict(train=list(range(12)),monitor=list(range(12,16))),
            expected_consumer_review=dict(schema='phone_tripod_replay_dataset_saved_review:1',dataset_report='pin to exported dataset report',engineering_qualified=True),
            max_source_rows=640,raw_and_partial_rows_retained=True,new_native_NN_FK_calls=0)));return
    if not all([a.collector_report,a.collector_terminal,a.output]):ap.error('Actual report+terminal and fresh output required; no future pins')
    r=read(a.collector_report);t=read(a.collector_terminal)
    assert r['schema']=='phone_tripod_task_acquisition_result:1' and t['returncode']==0 and t['protocol']==r['protocol']
    parent=read(r['protocol']['path'])
    p=dict(schema='phone_tripod_replay_dataset_CPU_protocol:1',saved_only_run_authorized=bool(a.authorize_saved_run),native_NN_FK_authorized=False,
        output=str(a.output.resolve()),cross_split_alias_policy='BLOCK_QUALIFICATION_PRESERVE_ALL_ROWS',
        inputs=dict(collector_report=pin(a.collector_report),collector_terminal=pin(a.collector_terminal),family_plan=r['family_plan'],task_bundle=r['task_bundle'],
            physical_parameters=pin(P/'phone_object_wrench_labels_v1/completed_v1/physical_parameters.json'),geometry_metadata=pin(P/'phone_geometry_features_v1/completed_v1/compiled_constants.json'),bridge_qualification=parent['inputs']['bridge_qualification']),
        sources=dict(exporter=pin(H/'export_dataset_v1.py'),adapter=pin(H/'dataset_adapter_v1.py'),protocol_maker=pin(Path(__file__)),
            frozen_decode_window=pin(P/'phone_causal_impulse_labels_v1/export_saved_labels_v2.py'),scalar_codec=pin(P/'phone_transition_dataset_v1/transition_codec_v1.py'),
            pure_structured_encoder_source=pin(P/'phone_structured_transition_v1/structured_codec.py')),
        contract=dict(full_labels_only='completeH finite no-warning, no partialH labels',prefix='finite observed prefix labels separate, never train as H',
            WM_eligible='all physically valid completeH rows, including failed task endpoints',Actor_start_eligible='completeH & pre-boundary-task alive; terminal target retained and absorbing',
            task_terminal='source control-boundary POST-H .05m/45deg; any-native-tick event is diagnostic only',
            truncation='collection0.4s may bootstrap; reference2s timeout separate',
            field_map_source='export_dataset_v1.FIELD_MAP',fresh_or_oldCHECK_TEST_arrays_opened=False,actual_new_model_FK_forward_steps_NN_calls=0))
    assert not Path(p['output']).exists()
    for f in [H/'dataset_adapter_v1.py',H/'export_dataset_v1.py',Path(__file__)]:ast.parse(f.read_text())
    with a.protocol_output.open('x',encoding='utf8') as f:json.dump(p,f,indent=2,allow_nan=False)
    print(json.dumps(dict(protocol=pin(a.protocol_output),run_authorized=p['saved_only_run_authorized'],output=p['output'])))
if __name__=='__main__':main()
