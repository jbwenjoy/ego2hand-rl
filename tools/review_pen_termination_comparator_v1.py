"""Independent saved-state/packet/arithmetic review; no native step or forward."""
import argparse
import copy
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time
import tomllib
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.strict_json import write_json


def read(p): return json.loads(Path(p).read_text())


def main(a):
    started=time.perf_counter(); cp=a.config.resolve(); cfg=tomllib.loads(cp.read_text())
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['registration','reader_registration','output','reader_output','protocol']}
    reg=read(paths['registration'])
    if reg['config']!=file_pin(cp) or reg['queries']!=240 or reg['native_cap']!=1920:
        raise ValueError('Exact registered comparator required')
    verify_source_snapshot(read(verify_file_pin(reg['source_receipt'])),live=True)
    legacy=runpy.run_path(str(verify_file_pin(reg['inputs']['independent_reader_source'])))
    analytic=legacy['analytic_checks']()
    def freeze(out):
        sources=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,paths['protocol']]
        receipt=freeze_sources(sources,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        write_json(out/'source_receipt.json',receipt); return receipt
    if a.register:
        out=paths['reader_registration'].parent
        if out.exists() or not all(analytic.values()): raise ValueError('Fresh successful analytical reader registration required')
        out.mkdir(); freeze(out)
        write_json(paths['reader_registration'],dict(schema='pen_termination_comparator_reader_registration:1',config=file_pin(cp),worker_registration=file_pin(paths['registration']),
            source_receipt=file_pin(out/'source_receipt.json'),checks=analytic,passed=sum(analytic.values()),total=len(analytic),actual_native_integrations=0,optimizer_steps=0,
            scope='Independent analytical checks only;actual saved comparator review pending.'))
        print(json.dumps(dict(registration=file_pin(paths['reader_registration']),checks=[sum(analytic.values()),len(analytic)],native=0)),flush=True); return
    rr=read(paths['reader_registration'])
    if rr['config']!=file_pin(cp) or rr['worker_registration']!=file_pin(paths['registration']) or rr['passed']!=rr['total'] or not all(rr['checks'].values()):
        raise ValueError('Reader registration changed')
    verify_source_snapshot(read(verify_file_pin(rr['source_receipt'])),live=True)
    if a.replica not in ['A','B']: raise ValueError('Explicit replica required')
    folder=paths['output']/a.replica; rp=folder/'report.json'; producer=read(rp)
    if (folder/'failure.json').exists() or producer['schema']!='pen_termination_comparator:1' or producer['registration']!=file_pin(paths['registration']) or producer['reader_registration']!=file_pin(paths['reader_registration']) or producer['passed']!=producer['total'] or producer['total']!=7 or not all(producer['checks'].values()) or producer['actual_native_integrations']!=producer['attempted_native_integrations'] or producer['actual_native_integrations']!=960 or producer['optimizer_steps']!=0 or producer['normalizer_fits']!=0 or producer['setup_extra_native']!=0 or producer['setup_native_unknown']:
        raise ValueError('Exact successful120-query/960-tick producer required')
    before=[file_pin(p) for p in sorted(folder.rglob('*')) if p.is_file()]
    out=paths['reader_output']/a.replica
    if out.exists(): raise ValueError('Fresh read-only result required')
    import mujoco
    import torch
    from supermimic_research.runtime.shared_snapshot_transport import load_shared,tree_digest
    from supermimic_research.learning.world_model import raw_tree_equal
    from supermimic_research.mechanics.pen_screened_review import ScreenedCaseReviewer
    from supermimic_research.mechanics.pen_solve_exit_epoch_v2 import physical_view
    from supermimic_research.mechanics.warp_packet_decode import decode_saved_case
    torch.set_num_threads(4)
    verify_source_snapshot(read(verify_file_pin(producer['source_receipt'])),live=True)
    objective=tomllib.loads(verify_file_pin(reg['inputs']['objective_config']).read_text()); adapter=(Path(reg['inputs']['objective_config']['path']).parent/objective['adapter_config']).resolve()
    math_reader=SimpleNamespace(reward=tomllib.loads(adapter.read_text())['reward'],objective=objective)
    prepared=read(verify_file_pin(reg['inputs']['preparation'])); originals={r['name']:r['case'] for r in prepared['rows']}
    plan=reg['schedules'][a.replica]; checks=dict(exact_schedule=[r['declaration'] for r in producer['rows']]==plan and len(plan)==120,
        prerequisites=producer['prerequisites']==reg['prerequisites'])
    models={}; physicals={}; sources={}; observations={}; traces={}; records=[]; elements=0
    near=legacy['near']; differences=legacy['differences']; projection_near=legacy['projection_near']
    with patch.object(mujoco,'mj_step',side_effect=AssertionError('No native steps in saved reader')),patch.object(mujoco,'mj_forward',side_effect=AssertionError('No forward in saved reader')):
        for build in producer['builds']:
            x,y=[load_shared(build[k]) for k in ['before','after']]
            checks[build['teacher']+'_'+build['algorithm']+'_graph_state']=all(raw_tree_equal(x[k],y[k]) for k in ['native','python','rng','metadata']) and build['clock_native_estimate']==build['copy_events']==0 and build['counter_before']==build['counter_after'] and x['metadata']['termination_backend']==reg['identities'][build['algorithm']]
        for pp in producer['profiles']:
            profile=read(verify_file_pin(pp)); alg=profile['contract']['identity']['algorithm']
            checks[pp['path']+'_profile']=profile['entered'] and profile['bindings_restored'] and profile['contract']==reg['identities'][alg] and (alg=='INSTALLED_ORIGINAL_OR' or profile['kernel']['bindings_restored'])
        for pin,decl in zip(producer['rows'],plan):
            row=read(verify_file_pin(pin['report'])); c=read(verify_file_pin(row['case'])); source=load_shared(c['source']); old=row['original_case']; contract=reg['identities'][decl['algorithm']]
            key=json.dumps(old['source'],sort_keys=True)
            if key not in sources: sources[key]=load_shared(old['source'])
            original=sources[key]; expected_metadata=copy.deepcopy(original['metadata']); expected_metadata['termination_backend']=contract
            unchanged={k:raw_tree_equal(original[k],source[k]) for k in original if k not in ['metadata','sources','engine_sources']}
            audit=row['source_restore']['import_audit']; imported=dict(original,metadata=expected_metadata)
            mk=c['model']['path']
            if mk not in models: models[mk]=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['model']))); physicals[mk]=dict(np.load(verify_file_pin(c['physical_context'])))
            packet,end,decision,raw=ScreenedCaseReviewer.packet(math_reader,c,source,models[mk],physicals[mk])
            z=dict(np.load(verify_file_pin(c['trace']))); oldz=dict(np.load(verify_file_pin(old['trace']))); obs=torch.load(verify_file_pin(row['regenerated_observation']),map_location='cpu',weights_only=False)
            qchecks=dict(original_condition=old==originals[decl['original_name']], declaration=row['declaration']==decl,
                original_metadata_no_rule='termination_backend' not in original['metadata'], new_metadata_exact=source['metadata']==expected_metadata,
                no_numeric_or_inventory_change=all(unchanged.values()), backend_identity=row['backend']==contract,
                original_source_pin=row['source_restore']['original_source']==old['source'], original_digest=tree_digest(original)==audit['original_tree_sha256']==old['source']['full_tree_sha256'],
                imported_digest=tree_digest(imported)==audit['imported_tree_sha256'], metadata_audit=audit['original_metadata']==original['metadata'] and audit['imported_metadata']==expected_metadata,
                import_all_original_fields=audit['unchanged_top_level_fields']=={k:True for k in original if k!='metadata'} and audit['rollback_exact'] and audit['native_python_rng_unchanged'],
                all_packet_checks=all(packet['checks'].values()), actual_PD=np.array_equal(z['post_ctrl'][:,c['aids']],oldz['post_ctrl'][:,old['aids']]),
                raw_original=np.array_equal(raw,oldz['actions20'][0]), restore=all(row['source_restore']['complete_state_equal'].values()) and row['source_restore']['external_counter_unchanged'] and row['source_restore']['original_source_unmutated'],
                observation=old['name'] not in observations or raw_tree_equal(observations[old['name']],obs),
                original_differences=differences(z,oldz)==row['differences_from_original'], original_mask=row['original_physical_valid']==old['physical_audit']['valid'],
                original_impulse_mask=row['original_impulse_valid']==old['physical_audit']['checks']['EFC_linear_impulse'],
                new_mask=packet['physical']['valid']==row['new_physical_valid'],new_impulse_mask=packet['physical']['checks']['EFC_linear_impulse']==row['new_impulse_valid'],
                recorded_checks=row['passed']==row['total'] and all(row['checks'].values()))
            observations.setdefault(old['name'],obs); pair=(decl['algorithm'],decl['repeat'],old['name'])
            if pair in traces: qchecks['scope_differences']=differences(z,traces[pair])==row['differences_between_scopes']
            else: traces[pair]=z
            if decl['scope']=='PACKET_PLUS_SOLVE_EXIT':
                o=dict(np.load(verify_file_pin(row['solve_exit']))); f={k[5:]:v[:,0] for k,v in o.items() if k.startswith('data.')}
                qchecks.update(copies8=all(len(v)==8 for v in o.values()),finite_copies=all(np.isfinite(v).all() for v in o.values()),
                    epochs=np.array_equal(f['qpos'],z['pre_qpos']) and np.array_equal(f['qvel'],z['pre_qvel']) and np.array_equal(f['qacc'],z['post_qacc']) and np.array_equal(f['qfrc_constraint'],z['post_qfrc_constraint']),
                    applied=np.array_equal(f['qfrc_applied'],z['post_qfrc_applied']) and np.array_equal(f['xfrc_applied'],z['post_xfrc_applied']),
                    clock=bool(np.all(abs(z['post_GPU_time']-f['time']-np.float32(c['dt']))<=4*np.finfo(np.float32).eps*(1+abs(f['time'])))),copy_events=row['copy_events']==8)
                _,decoded=decode_saved_case(c,z,models[mk],physicals[mk]); arrays=legacy['independent_arrays'](z,physical_view(o,26),source['native']['model']['arrays'],c,models[mk].opt.gravity,decoded,50)
                saved=dict(np.load(verify_file_pin(row['balance']))); epoch=read(verify_file_pin(row['epoch_review'])); elements+=sum(v.size for v in arrays.values())
                qchecks.update(array_inventory=arrays.keys()==saved.keys(),all_independent_arrays=all(near(v,saved[k]) for k,v in arrays.items()),
                    original_impulse=near(arrays['original_total'],decoded['linear_impulse_residual_N_s']), epoch_summary=epoch['passed']==epoch['total'] and all(epoch['checks'].values()))
                peak=int(np.linalg.norm(arrays['original_total'],axis=1).argmax()); norm=float(np.linalg.norm(arrays['original_total'][peak])); terms=['free_mass_chart','Ma_accumulation','actual_scaled_gradient','gradient_force_recovery','smooth_minus_gravity']
                qchecks['peak_summary']=epoch['peak_tick']==peak and near(norm,epoch['peak_original_impulse_Ns']) and all(near(np.linalg.norm(arrays[k][peak]),epoch['peak_component_norms_Ns'][k]) and projection_near(arrays[k][peak],arrays['original_total'][peak],epoch['peak_signed_projections'][k]) for k in terms)
                qchecks['terminal_summary']=all(np.array_equal(arrays['terminal_'+k],v) for k,v in epoch['terminal_predicates'].items())
            else: qchecks['no_copy_events']=row['copy_events']==0 and 'solve_exit' not in row
            checks.update({decl['name']+'_'+k:bool(v) for k,v in qchecks.items()})
            records.append(dict(declaration=decl,original_valid=row['original_impulse_valid'],new_valid=row['new_impulse_valid'],checks=qchecks))
    for pin in before: verify_file_pin(pin)
    checks['same_file_inventory']=[p['path'] for p in before]==[str(p.resolve()) for p in sorted(folder.rglob('*')) if p.is_file()]
    checks['all120_rows']=len(records)==120; checks['all60_observed']=sum(r['declaration']['scope']=='PACKET_PLUS_SOLVE_EXIT' for r in records)==60
    out.mkdir(parents=True); receipt=freeze(out)
    write_json(out/'report.json',dict(schema='pen_termination_comparator_saved_review:1',registration=file_pin(paths['reader_registration']),producer=file_pin(rp),
        source_receipt=file_pin(out/'source_receipt.json'),input_file_capsule=dict(records_before_read=before,all_bytes_verified_after=True),rows=records,
        checks=checks,passed=sum(checks.values()),total=len(checks),recomputed_array_elements=elements,actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,wall_s=time.perf_counter()-started,
        scope='Independent metadata/source/numerical comparison and separately implemented saved arithmetic;canonical raw/EFC decoder and physical-prefix view reused. All120 current process queries retained;no physical backend/Actor/ego admission.'))
    verify_source_snapshot(receipt,live=True)
    print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[sum(checks.values()),len(checks)],array_elements=elements,native=0)),flush=True)
    if not all(checks.values()): raise ValueError('Independent comparator review discrepancy retained')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',type=Path,required=True); parser.add_argument('--register',action='store_true'); parser.add_argument('--replica',choices=['A','B']); main(parser.parse_args())
