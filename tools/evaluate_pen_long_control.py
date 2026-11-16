"""Registered final long-PPO execution using the shared strict trace recorder."""
import argparse
import json
import subprocess
import sys
import time
import tomllib
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'tools')]
import numpy as np
import torch
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.learning.pen_w1_evaluation import actor_for_fresh_live_W1
from supermimic_research.runtime.controlled_evaluation import summarize_trace
from supermimic_research.runtime.pen_support_runtime import SupportPenRuntime
from supermimic_research.runtime.pen_executed_control_audit import audit_executed_control
from preview_pen_support_pilot import execute, render, write


def settings(path):
    cfg = tomllib.loads(path.read_text())
    if (cfg['schema'] != 1 or cfg['repetitions'] != 3
            or cfg['case_order'] != ['INITIAL', 'RESIDUAL', 'SUPPORT']
            or cfg['video_repetition'] != 0 or cfg['required_training_iterations'] != 2000
            or cfg['evaluation_features'] != 'LIVE'):
        raise ValueError('Registered final2000/LIVE/nine-case matrix required')
    return cfg, {k: (path.parent / cfg[k]).resolve() for k in [
        'training_report', 'objective_config', 'gates', 'protocol', 'registration',
        'startup_ledger', 'plant_context_report']}


def context_pins(report):
    if report['passed'] != report['total'] or report['role'] != 'CHECK_TRAINED_ACTOR_PHYSICAL_RESPONSE_NEVER_FIT':
        raise ValueError('Completed historical W1 plant witnesses required')
    return {label: next(s['source'] for s in report['sources']
                        if s['actor'] == label and s['phase'] == 10 and s['available'])
            for label in ['RESIDUAL', 'SUPPORT']}


def validate_final_training(report, objective_pin):
    if (report['schema'] != 'pen_geometry_support_long_control_training:2'
            or report['config'] != objective_pin or report['passed'] != 5 or report['total'] != 5
            or not all(report['checks'].values()) or report['actual_native_integrations'] != 65536000
            or report['optimizer_steps'] != 128000 or [a['arm'] for a in report['arms']] != ['RESIDUAL', 'SUPPORT']):
        raise ValueError('Complete equal-budget final long trial required')
    for arm in report['arms']:
        if (arm['iterations'], arm['actual_native_integrations'], arm['optimizer_steps'],
                arm['training_transitions']) != (2000, 32768000, 64000, 4096000):
            raise ValueError('Final per-arm work differs')


def initialize_final_actor(runtime, state, pin, *, plant_context_pin, label):
    expected = (0, 0, 0) if label == 'INITIAL' else (2000, 64000, 4096000)
    if (state['iteration'], state['optimizer_steps'], state['training_transitions']) != expected:
        raise ValueError('Final/initial checkpoint counters differ from declared case')
    if state['metadata']['arm'] != ('RESIDUAL' if label == 'INITIAL' else label):
        raise ValueError('Checkpoint arm differs from declared case')
    return actor_for_fresh_live_W1(runtime, state, pin, plant_context_pin)


def main(a):
    cp = a.config.resolve()
    cfg, paths = settings(cp)
    inputs = {k: file_pin(paths[k]) for k in ['objective_config', 'gates', 'protocol', 'startup_ledger', 'plant_context_report']}
    contexts = context_pins(json.loads(paths['plant_context_report'].read_text()))
    if a.register:
        folder = paths['registration'].parent
        if folder.exists() or not folder.is_relative_to(ROOT / 'runs'):
            raise ValueError('Fresh registration directory required')
        # No final training output is read during registration.
        folder.mkdir()
        sources = sorted((ROOT / 'src').rglob('*.py')) + [
            Path(__file__), ROOT / 'tools/preview_pen_support_pilot.py',
            ROOT / 'tools/render_mug_progress_video.py', cp, paths['gates'], paths['protocol']]
        receipt = freeze_sources(sources, folder / 'source_snapshot', ROOT,
                                 subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
        write(folder / 'source_receipt.json', receipt)
        write(paths['registration'], dict(schema='pen_long_control_evaluation_registration:1',
              config=file_pin(cp), inputs=inputs, contexts=contexts,
              future_training_report=str(paths['training_report']), source_receipt=file_pin(folder / 'source_receipt.json'),
              selection='Final2000 only;fixed repeat0 video;no early stop or checkpoint selection.'))
        print(json.dumps(dict(registration=file_pin(paths['registration']))), flush=True)
        return
    registration = json.loads(paths['registration'].read_text())
    if registration['config'] != file_pin(cp) or registration['inputs'] != inputs or registration['contexts'] != contexts:
        raise ValueError('Pre-execution registration changed')
    receipt = json.loads(verify_file_pin(registration['source_receipt']).read_text())
    verify_source_snapshot(receipt, live=True)
    out = a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT / 'runs'):
        raise ValueError('Fresh output directory required')
    if a.qualify_only or a.qualify_recording:
        startup = json.loads(paths['startup_ledger'].read_text())
        first = json.loads(verify_file_pin(startup['roots']['first_health']).read_text())
        checkpoints = [first['checkpoint']]
        labels = ['RESIDUAL']
    else:
        training = json.loads(paths['training_report'].read_text())
        validate_final_training(training, inputs['objective_config'])
        verify_source_snapshot(json.loads(verify_file_pin(training['source_receipt']).read_text()), live=True)
        checkpoints = [training['arms'][0]['initial']] + [arm['checkpoint'] for arm in training['arms']]
        labels = cfg['case_order']
    out.mkdir()
    write(out / 'source_receipt.json', receipt)
    from mjlab.sim.sim import Simulation
    original = Simulation.step
    observed = 0
    phase = 'construction'
    started = time.perf_counter()

    def step(sim):
        nonlocal observed
        original(sim)
        observed += sim.num_envs

    Simulation.step = step
    runtime = None
    try:
        torch.set_num_threads(4)
        with torch.no_grad(), torch.jit.optimized_execution(False):
            if a.qualify_only:
                runtime = SupportPenRuntime(paths['objective_config'], 1)
                state = torch.load(verify_file_pin(checkpoints[0]), map_location='cpu', weights_only=False)
                actor, audit = actor_for_fresh_live_W1(runtime, state, checkpoints[0], contexts['RESIDUAL'])
                if observed or runtime.owned_native_integrations:
                    raise ValueError('Unstepped actor admission qualification advanced physics')
                runtime.close()
                runtime = None
                result = dict(schema='pen_long_control_W1_admission:1', audit=audit,
                              actual_native_integrations=observed, optimizer_steps=0,
                              policy_performance_evaluated=False)
            elif a.qualify_recording:
                gates = tomllib.loads(paths['gates'].read_text())
                initializer = partial(actor_for_fresh_live_W1, plant_context_pin=contexts['RESIDUAL'])
                case = execute(checkpoints[0], paths['objective_config'], gates, out / 'FIRST_CHECKPOINT', actor_initializer=initializer)
                with np.load(verify_file_pin(case['trace'])) as z:
                    control_audit = audit_executed_control(dict(z), case)
                if control_audit['passed'] != control_audit['total'] or observed != case['native_integrations']:
                    raise ValueError('Actual recording QA failed')
                result = dict(schema='pen_long_control_recording_qualification:1', case=case,
                              executed_control_audit=control_audit, actual_native_integrations=observed,
                              optimizer_steps=0, performance_role='FIRST_CHECKPOINT_INSTRUMENTATION_QA_NOT_FINAL_POLICY')
            else:
                gates = tomllib.loads(paths['gates'].read_text())
                cases = []
                for repetition in range(3):
                    folder = out / f'REP{repetition}'
                    folder.mkdir()
                    for label, pin in zip(labels, checkpoints):
                        phase = f'REP{repetition}_{label}'
                        context = contexts['SUPPORT' if label == 'SUPPORT' else 'RESIDUAL']
                        initializer = partial(initialize_final_actor, plant_context_pin=context, label=label)
                        case = execute(pin, paths['objective_config'], gates, folder / label,
                                       actor_initializer=initializer)
                        case.update(name=phase, policy=label, repetition=repetition)
                        with np.load(verify_file_pin(case['trace'])) as z:
                            case['executed_control_audit'] = audit_executed_control(dict(z), case)
                        if case['executed_control_audit']['passed'] != case['executed_control_audit']['total']:
                            raise ValueError('Recorded transmitted control audit failed')
                        cases.append(case)
                        print(json.dumps(dict(name=phase, summary=case['summary'])), flush=True)
                if observed != sum(c['native_integrations'] for c in cases):
                    raise ValueError('Independent native integration counter differs')
                execution_s = time.perf_counter() - started
                aggregate = {label: dict(
                    durations_s=[c['summary']['metrics']['actual_duration_s'] for c in cases if c['policy'] == label],
                    qualified_count=sum(c['summary']['qualified'] for c in cases if c['policy'] == label))
                    for label in labels}
                common_s = min(c['summary']['metrics']['actual_duration_s'] for c in cases if c['policy'] != 'INITIAL')
                prefixes = {}
                for case in cases:
                    if case['policy'] == 'INITIAL':
                        continue
                    with np.load(verify_file_pin(case['trace'])) as z:
                        n = round(common_s / .0025)
                        trace = {k: z[k][:n] for k in z.files if k.startswith(('post_', 'pre_')) and k != 'pre_action_phase'}
                        trace['overflow'] = z['overflow'][:n]
                        reference = {k: z[k] for k in ['time_s', 'COM_world_m', 'quaternion_wxyz']}
                        prefixes[case['name']] = summarize_trace(trace, reference, .0025, False, gates)['metrics']
                phase = 'saved_pose_rendering'
                video = render(cases[:3], out, title='2000轮长训练：真实仿真执行')
                result = dict(schema='pen_long_control_fresh_W1_evaluation:1',
                    training=file_pin(paths['training_report']), cases=cases, aggregate=aggregate,
                    trained_common_horizon_s=common_s, common_horizon_metrics=prefixes,
                    rendering=video, execution_record_wall_s=execution_s,
                    render_wall_s=time.perf_counter() - started - execution_s,
                    actual_native_integrations=observed, optimizer_steps=0, scope=cfg['scope'])
        verify_source_snapshot(receipt, live=True)
        for pin in inputs.values():
            verify_file_pin(pin)
        result.update(config=file_pin(cp), registration=file_pin(paths['registration']),
                      source_receipt=file_pin(out / 'source_receipt.json'), total_wall_s=time.perf_counter() - started)
        write(out / 'report.json', result)
        print(json.dumps(dict(report=file_pin(out / 'report.json'), native_integrations=observed)), flush=True)
    except Exception as error:
        write(out / 'failure.json', dict(type=type(error).__name__, message=str(error), phase=phase,
                                        actual_native_integrations=observed, optimizer_steps=0))
        raise
    finally:
        if runtime is not None:
            runtime.close()
        Simulation.step = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--register', action='store_true')
    mode.add_argument('--qualify-only', action='store_true')
    mode.add_argument('--qualify-recording', action='store_true')
    args = parser.parse_args()
    if not args.register and args.output is None:
        parser.error('--output is required for execution')
    main(args)
