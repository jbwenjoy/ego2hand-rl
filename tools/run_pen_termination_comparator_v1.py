"""Explicit new backend/source import; all240 original/candidate queries retained."""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.runtime.pen_solve_exit_observer import require_closed_full54
from supermimic_research.runtime.pen_termination_design import schedule
from supermimic_research.runtime.pen_termination_profile import execution_identity, TerminationProfile
from supermimic_research.strict_json import write_json


def read(path):
    return json.loads(Path(path).read_text())


def settings(cp):
    cfg = tomllib.loads(cp.read_text())
    if (cfg['schema'], cfg['mode'], cfg['replicas'], cfg['queries_per_replica'], cfg['native_cap_per_replica'], cfg['optimizer_budget'], cfg['normalizer_fit_budget']) != (1, 'FIXED_TEN_CONDITION_TERMINATION_COMPARATOR', ['A', 'B'], 120, 960, 0, 0):
        raise ValueError('Exact comparator population/budget required')
    paths = {k: (cp.parent / v).resolve() for k, v in cfg.items() if isinstance(v, str) and k not in ['mode']}
    return cfg, paths


def main(a):
    started = time.perf_counter(); cp = a.config.resolve(); cfg, paths = settings(cp)
    proof = require_closed_full54(paths['observation_config'])
    oldreg = read(paths['prior_worker_registration']); prep = read(paths['preparation'])
    for pin in oldreg['inputs'].values(): verify_file_pin(pin)
    if (file_pin(paths['preparation']) != oldreg['inputs']['preparation']
            or file_pin(paths['observation_config']) != oldreg['inputs']['observation_config']
            or proof['registration'] != oldreg['inputs']['full54_registration']):
        raise ValueError('Exact historical source/observation/full54 ownership required')
    if prep['passed'] != prep['total'] or not all(prep['checks'].values()): raise ValueError('Complete ten-condition preparation required')
    cpu = read(paths['cpu_qualification']); tests = paths['cpu_tests'].read_text(encoding='utf-8-sig')
    if cpu['passed'] != cpu['total'] or cpu['total'] != 7 or not all(cpu['checks'].values()) or 'Ran 601 tests' not in tests or '\nOK\n' not in tests:
        raise ValueError('Completed candidate597 and comparator601 CPU tests required')
    verify_source_snapshot(read(verify_file_pin(cpu['source_receipt'])), live=True)
    backend_pin = oldreg['inputs']['backend_manifest']; backend = read(verify_file_pin(backend_pin))
    installed = oldreg['inputs']['installed_solver']
    identities = {alg: execution_identity(alg, backend['backend_id'], installed) for alg in ['INSTALLED_ORIGINAL_OR', 'DROP_IMPROVEMENT_ONLY']}
    inputs = {k: file_pin(paths[k]) for k in ['preparation', 'observation_config', 'prior_worker_registration', 'cpu_qualification', 'cpu_tests', 'prior_evidence', 'protocol']}
    inputs.update(backend_manifest=backend_pin, objective_config=oldreg['inputs']['objective_config'], installed_solver=installed,
                  independent_reader_source=file_pin(ROOT / 'tools/review_pen_solve_exit_saved.py'))
    sources = sorted((ROOT / 'src').rglob('*.py')) + [Path(__file__), cp, paths['protocol'], ROOT / 'tools/review_pen_termination_comparator_v1.py']
    def freeze(folder):
        receipt = freeze_sources(sources, folder / 'source_snapshot', ROOT, subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
        write_json(folder / 'source_receipt.json', receipt); return receipt
    if a.register:
        from supermimic_research.runtime.shared_snapshot_transport import load_shared, tree_digest
        from supermimic_research.runtime.pen_termination_source_import import import_historical_state
        preflight = []
        for item in prep['rows']:
            state = load_shared(item['case']['source'])
            for algorithm, contract in identities.items():
                imported, audit = import_historical_state(state, dict(state['metadata'], termination_backend=contract), contract)
                if tree_digest(state) != audit['original_tree_sha256']: raise ValueError('Historical source mutated')
                preflight.append(dict(name=item['name'], algorithm=algorithm, source=item['case']['source'], audit=audit))
        out = paths['registration'].parent
        if out.exists(): raise ValueError('Fresh comparator registration required')
        out.mkdir(); freeze(out)
        write_json(paths['registration'], dict(schema='pen_termination_comparator_registration:1', config=file_pin(cp), inputs=inputs,
            prerequisites=proof, identities=identities, source_receipt=file_pin(out / 'source_receipt.json'),
            schedules={rep: schedule(prep['rows'], rep) for rep in cfg['replicas']}, source_import_cpu_preflight=preflight,
            queries=240, native_cap=1920, actual_native_integrations=0, optimizer_steps=0, normalizer_fits=0))
        print(json.dumps(dict(registration=file_pin(paths['registration']), imports=len(preflight), native=0)), flush=True); return
    reg = read(paths['registration'])
    if reg['config'] != file_pin(cp) or reg['inputs'] != inputs or reg['identities'] != identities or reg['prerequisites'] != proof:
        raise ValueError('Comparator registration identity/input changed')
    verify_source_snapshot(read(verify_file_pin(reg['source_receipt'])), live=True)
    if a.replica not in cfg['replicas']: raise ValueError('Explicit fresh A/B replica required')
    rr = read(paths['reader_registration'])
    if rr['worker_registration'] != file_pin(paths['registration']) or rr['passed'] != rr['total'] or not all(rr['checks'].values()):
        raise ValueError('Pre-register separate saved reader before native work')
    verify_source_snapshot(read(verify_file_pin(rr['source_receipt'])), live=True)
    out = paths['output'] / a.replica
    if out.exists(): raise ValueError('Fresh native output required; never restart partial results')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    from mjlab.sim.sim import Simulation
    from supermimic_research.runtime.explicit_collision_backend import backend_scope
    from supermimic_research.runtime.full_rehash_scope import FullRehashVerifierScope, implementation_pins
    from supermimic_research.runtime.pen_live_packets import LivePenPackets
    from supermimic_research.runtime.pen_owned_probe_recorder import OwnedProbeRecorder
    from supermimic_research.runtime.pen_solve_exit_observer_v2 import SolveExitGraphObserverV2
    from supermimic_research.runtime.pen_termination_source_import import import_historical_state, verify_imported_capture
    from supermimic_research.runtime.shared_snapshot_transport import load_shared, tree_digest
    from supermimic_research.runtime import owned_pen_state as native
    from supermimic_research.learning.pen_matched_recovery import WorkCounter
    from supermimic_research.learning.world_model import raw_tree_equal
    from supermimic_research.mechanics.pen_solve_exit_epoch_v2 import review as review_epoch
    from supermimic_research.mechanics.warp_packet_decode import decode_saved_case
    differences = runpy.run_path(str(ROOT / 'tools/review_pen_solve_exit_saved.py'))['differences']
    class ImportedPackets(LivePenPackets):
        def restore(self, record, label):
            t = time.perf_counter(); before = counter.metadata(); old = load_shared(record)
            state, audit = import_historical_state(old, runtime.metadata(), profile.contract)
            restored = native.restore(runtime, state); after = native.capture(runtime, self.receipt)
            readback = verify_imported_capture(old, after, profile.contract)
            equal = {k: raw_tree_equal(state[k], after[k]) for k in ['native', 'python', 'rng']}
            item = dict(label=label, original_source=record, import_audit=audit, import_readback=readback, audit=restored,
                        complete_state_equal=equal, original_source_unmutated=tree_digest(old) == audit['original_tree_sha256'],
                        external_counter_unchanged=before == counter.metadata(), wall_s=time.perf_counter() - t)
            self.restores.append(item)
            if not all(equal.values()) or not item['original_source_unmutated'] or not item['external_counter_unchanged'] or not restored['all_array_bytes_equal'] or not restored['all_scalar_values_equal']:
                raise ValueError('Explicit imported source restore/readback discrepancy')
            return state
    class ExitRecorder(OwnedProbeRecorder):
        def begin(self, raw): self.exit_rows=[]; self.exit_counts=[]; super().begin(raw)
        def after_step(self):
            super().after_step(); self.exit_counts.append(observer.count())
            if observer.scope == 'PACKET_PLUS_SOLVE_EXIT': self.exit_rows.append(observer.read())
    out.mkdir(parents=True); receipt = freeze(out); plan = reg['schedules'][a.replica]
    write_json(out / 'intent.json', dict(registration=file_pin(paths['registration']), reader=file_pin(paths['reader_registration']), schedule=plan, native_cap=960))
    counter=WorkCounter(); packets=runtime=observer=profile=None; rows=[]; builds=[]; profiles=[]; verifiers=[]; expected_obs={}; traces={}; stage='setup'; setup_unknown=False; setup_extra=0
    original_step = Simulation.step
    def step(sim):
        if packets is not None and packets.recorder.enabled: packets.recorder.before_step()
        counter.native_call(original_step, sim)
        if packets is not None and packets.recorder.enabled: packets.recorder.after_step()
    Simulation.step = step
    try:
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True); torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        with backend_scope(Path(backend_pin['path'])) as collision, torch.no_grad(), torch.jit.optimized_execution(False):
            blocks = list(dict.fromkeys((d['teacher'], d['algorithm']) for d in plan))
            for teacher, algorithm in blocks:
                local = [d for d in plan if (d['teacher'], d['algorithm']) == (teacher, algorithm)]
                folder=out / teacher / algorithm; folder.mkdir(parents=True); stage=teacher+'_'+algorithm+'_setup'
                with TerminationProfile(identities[algorithm], installed) as profile:
                    try:
                        runtime=profile.create_runtime(Path(inputs['objective_config']['path']), 1); runtime.set_fixed_frame_features(False)
                        packets=ImportedPackets(runtime, receipt, counter, folder, 'TERMINATION_COMPARATOR_NEVER_FIT')
                        verifier=FullRehashVerifierScope('SINGLE_RESOLVE_FULL_REHASH', implementation_pins())
                        with verifier:
                            packets.restore(prep['rows'][local[0]['row_index']]['case']['source'], 'before_graph')
                            before,bp=packets.capture(folder, 'before_graph'); count0=counter.metadata(); t=time.perf_counter()
                            observer=SolveExitGraphObserverV2(runtime.env.sim, config_path=paths['observation_config'], installed_solver_pin=installed)
                            setup_unknown=True
                            try: observer.build()
                            finally:
                                after,ap=packets.capture(folder, 'after_graph'); delta=(after['native']['data']['arrays']['time'].astype(np.float64)-before['native']['data']['arrays']['time'].astype(np.float64))/float(before['native']['model']['arrays']['opt.timestep'][0]); estimate=int(np.rint(delta[0])); known=bool(np.all(abs(delta-estimate)<=1e-3) and estimate>=0); setup_extra += estimate if known else 0; setup_unknown=not known
                                b=dict(teacher=teacher, algorithm=algorithm, backend_id=profile.contract['backend_id'], before=bp, after=ap,
                                    wall_s=time.perf_counter()-t, counter_before=count0, counter_after=counter.metadata(), copy_events=observer.count(),
                                    clock_native_estimate=estimate if known else None, state_equal={k:raw_tree_equal(before[k],after[k]) for k in ['native','python','rng']})
                                builds.append(b); write_json(folder/'graph_build.json', b)
                            if not all(b['state_equal'].values()) or b['copy_events'] != 0 or count0 != counter.metadata() or b['clock_native_estimate'] != 0:
                                raise ValueError('Graph construction changed source/clock/counter')
                            packets.recorder=ExitRecorder(runtime)
                            for decl in local:
                                stage=decl['name']; qf=folder/stage; qf.mkdir(); old=prep['rows'][decl['row_index']]['case']; oz=dict(np.load(verify_file_pin(old['trace'])))
                                profile.verify(); observer.select(decl['scope']); restored=packets.restore(old['source'], stage)
                                obs={k:v.detach().cpu().clone() for k,v in runtime.observation().items()}; same_obs=old['name'] not in expected_obs or raw_tree_equal(expected_obs[old['name']],obs); expected_obs.setdefault(old['name'],obs); torch.save(obs,qf/'regenerated_observation.pt')
                                state,record=packets.capture(qf,'source'); n0=counter.native; events=observer.count()
                                try: case,_=packets.query(oz['actions20'][0].astype(np.float32),qf/'query',stage,state,record)
                                except BaseException:
                                    torch.save(dict(trace=packets.recorder.trace,contact_rows=packets.recorder.rows,addresses=packets.recorder.addresses,efc=packets.recorder.efc,offsets=packets.recorder.offsets,exit_rows=packets.recorder.exit_rows,exit_counts=packets.recorder.exit_counts),qf/'partial_NOT_RESUMABLE.pt'); raise
                                z=dict(np.load(verify_file_pin(case['trace']))); copies=observer.count()-events
                                checks=dict(imported_capture=all(verify_imported_capture(load_shared(old['source']),state,profile.contract)['readback_fields'].values()),
                                    source_native=raw_tree_equal(restored['native'],state['native']), observation=same_obs,
                                    actual_PD_exact=np.array_equal(z['post_ctrl'][:,case['aids']],oz['post_ctrl'][:,old['aids']]),
                                    native8=counter.native-n0==8, copy_events=copies==(8 if decl['scope']=='PACKET_PLUS_SOLVE_EXIT' else 0),
                                    exit_counts=packets.recorder.exit_counts==[events+(i+1 if decl['scope']=='PACKET_PLUS_SOLVE_EXIT' else 0) for i in range(8)], profile=profile.verify())
                                row=dict(declaration=decl, backend=profile.contract, original_case=old, case=file_pin(qf/'query/report.json'),
                                    regenerated_observation=file_pin(qf/'regenerated_observation.pt'), source_restore=packets.restores[-1], copy_events=copies,
                                    original_physical_valid=old['physical_audit']['valid'], new_physical_valid=case['physical_audit']['valid'],
                                    original_impulse_valid=old['physical_audit']['checks']['EFC_linear_impulse'], new_impulse_valid=case['physical_audit']['checks']['EFC_linear_impulse'],
                                    differences_from_original=differences(z,oz), checks=checks)
                                if decl['scope']=='PACKET_PLUS_SOLVE_EXIT':
                                    o={k:np.stack([s[k] for s in packets.recorder.exit_rows]) for k in observer.buffers}; np.savez_compressed(qf/'solve_exit.npz',**o)
                                    _,decoded=decode_saved_case(case,z,packets.model,packets.physical); epoch,arrays=review_epoch(z,o,state['native']['model']['arrays'],case,decoded,packets.model.opt.gravity,50)
                                    np.savez_compressed(qf/'balance.npz',**arrays); write_json(qf/'epoch_review.json',epoch); row.update(solve_exit=file_pin(qf/'solve_exit.npz'),balance=file_pin(qf/'balance.npz'),epoch_review=file_pin(qf/'epoch_review.json')); checks['epoch']=epoch['passed']==epoch['total'] and all(epoch['checks'].values())
                                key=(algorithm,decl['repeat'],old['name'])
                                if key in traces: row['differences_between_scopes']=differences(z,traces[key])
                                else: traces[key]=z
                                row.update(passed=sum(checks.values()),total=len(checks)); write_json(qf/'report.json',row); rows.append(dict(declaration=decl,report=file_pin(qf/'report.json')))
                                write_json(out/'work.json',dict(stage=stage,completed=len(rows),**counter.metadata()))
                                if not all(checks.values()) or counter.native>960: raise ValueError('Registered query contract/budget discrepancy')
                        write_json(folder/'verifier.json',verifier.audit()); verifiers.append(file_pin(folder/'verifier.json'))
                    finally:
                        try:
                            if observer is not None: observer.close(); observer=None
                        finally:
                            if runtime is not None: runtime.close(); runtime=None; packets=None
                write_json(folder/'termination_profile.json',profile.audit); profiles.append(file_pin(folder/'termination_profile.json')); profile=None
            checks=dict(all120=len(rows)==120, exact_schedule=[r['declaration'] for r in rows]==plan,
                native960=counter.native==counter.native_attempted==960 and setup_extra==0 and not setup_unknown,
                all4_graphs=len(builds)==4 and all(all(b['state_equal'].values()) and b['clock_native_estimate']==b['copy_events']==0 for b in builds),
                all4_profiles=len(profiles)==4 and all(read(verify_file_pin(p))['bindings_restored'] for p in profiles),
                all4_verifiers=len(verifiers)==4 and all(read(verify_file_pin(p))['validated'] and read(verify_file_pin(p))['bindings_restored'] for p in verifiers),
                no_updates=counter.optimizer==counter.optimizer_attempted==0)
            collision.verify(cold=True); verify_source_snapshot(receipt,live=True)
            write_json(out/'report.json',dict(schema='pen_termination_comparator:1',registration=file_pin(paths['registration']),reader_registration=file_pin(paths['reader_registration']),
                source_receipt=file_pin(out/'source_receipt.json'),replica=a.replica,prerequisites=proof,rows=rows,builds=builds,profiles=profiles,verifiers=verifiers,
                checks=checks,passed=sum(checks.values()),total=len(checks),**counter.metadata(),normalizer_fits=0,setup_extra_native=setup_extra,setup_native_unknown=setup_unknown,
                wall_s=time.perf_counter()-started,scope='Targeted changed-backend comparator;all masks/failures retained. Separate saved reader required;no physical backend/Actor/ego admission.'))
            if not all(checks.values()): raise ValueError('Final comparator accounting mismatch')
    except BaseException as error:
        failure=dict(stage=stage,type=type(error).__name__,message=str(error),rows=rows,missing=[d for d in plan if d['name'] not in {r['declaration']['name'] for r in rows}],builds=builds,
            **counter.metadata(),setup_extra_native=setup_extra,setup_native_unknown=setup_unknown,resumable=False,
            active_profile_audit=profile.audit if profile is not None else None)
        failure.update(recorded_control_native_integrations=counter.native,
            actual_native_integrations=counter.native+setup_extra if not setup_unknown else None,
            attempted_native_integrations=counter.native_attempted+setup_extra if not setup_unknown else None,
            completed_call_accounting_exact=not setup_unknown and counter.native==counter.native_attempted)
        write_json(out/'failure.json',failure)
        raise
    finally: Simulation.step=original_step
    print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[sum(checks.values()),len(checks)],**counter.metadata())),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',type=Path,required=True); parser.add_argument('--register',action='store_true'); parser.add_argument('--replica',choices=['A','B']); main(parser.parse_args())
