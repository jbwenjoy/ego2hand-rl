"""Actual GPU first-event latch qualification; zero physics or Adam work."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
import torch
import warp as wp
from supermimic_research.runtime.first_overflow_observer import FirstOverflowObserver
from supermimic_research.artifacts import file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.learning.pen_matched_recovery import write_metadata as write


def main(a):
    out = a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT / 'runs'):
        raise ValueError('Fresh GPU qualification required')
    out.mkdir()
    receipt = freeze_sources([Path(__file__), ROOT / 'src/supermimic_research/runtime/first_overflow_observer.py'],
        out / 'source_snapshot', ROOT, subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write(out / 'source_receipt.json', receipt)
    wp.init()
    stream = wp.Stream('cuda:0')
    ts = torch.cuda.ExternalStream(stream.cuda_stream, device=0)
    with torch.cuda.stream(ts), wp.ScopedStream(stream):
        raw = wp.zeros(2, dtype=int, device='cuda:0')
        v = {k: torch.arange(6, dtype=torch.float32, device='cuda:0').reshape(2, 3)
             for k in ['qpos', 'qvel', 'qacc_warmstart', 'ctrl']}
        v.update(overflow=wp.to_torch(raw), phase=torch.tensor([67, 68], device='cuda:0'),
                 time=torch.tensor([.2, .3], device='cuda:0'),
                 geom_xpos=torch.zeros(2, 4, 3, device='cuda:0'), geom_xmat=torch.zeros(2, 4, 3, 3, device='cuda:0'))
        saved = {k: t.clone() for k, t in v.items()}
        o = FirstOverflowObserver(v)
        o.before(); o.after(iteration=251, kind=1, step_call=1, forward_call=0)
        unchanged = {k: torch.equal(v[k], saved[k]) for k in v}
        clear = o.packet()
        mask = torch.tensor([0, 256], dtype=torch.int32, device='cuda:0')
        o.before()
        wp.copy(raw, wp.from_torch(mask))
        v['qpos'].add_(10)
        o.after(iteration=252, kind=2, step_call=508, forward_call=31)
        captured = v['qpos'].clone()
        for t in v.values(): t.zero_()
        o.before()
        mask.copy_(torch.tensor([3, 128], dtype=torch.int32, device='cuda:0'))
        wp.copy(raw, wp.from_torch(mask))
        o.after(iteration=253, kind=1, step_call=509, forward_call=31)
        packet = o.packet()
        checks = dict(actual_cuda_views=all(t.is_cuda for t in v.values()),
            raw_Warp_Torch_shared_ptr=raw.ptr == v['overflow'].data_ptr(),
            no_event_clear=not bool(clear['raw_mask'].any()),
            live_buffers_unchanged=all(unchanged.values()),
            raw_mask_not_bool=packet['raw_mask'].dtype == torch.int32,
            first_event_survives_reset=packet['raw_mask'].tolist() == [3, 256],
            correct_STEP_FORWARD_and_counters=packet['event'].tolist() == [[253, 1, 509, 31], [252, 2, 508, 31]],
            exact_pre_state=torch.equal(packet['pre']['qpos'][1], saved['qpos'][1].cpu()),
            exact_post_state=torch.equal(packet['post']['qpos'][1], captured[1].cpu()),
            pre_flag_preserved=packet['pre']['overflow'].tolist() == [0, 0])
    torch.save(packet, out / 'synthetic_NOT_PHYSICS_packet.pt')
    verify_source_snapshot(receipt, live=True)
    report = dict(schema='first_overflow_observer_GPU_qualification:1', checks=checks,
        passed=sum(checks.values()), total=len(checks), live_unchanged=unchanged,
        source_receipt=file_pin(out / 'source_receipt.json'), packet=file_pin(out / 'synthetic_NOT_PHYSICS_packet.pt'),
        actual_native_integrations=0, optimizer_steps=0,
        scope='Actual CUDA Warp/Torch stream-ordered synthetic events. Qualifies latch bytes/reset retention and read-only buffers, not physical replay identity or collision correctness.')
    write(out / 'report.json', report)
    print(json.dumps(dict(report=file_pin(out / 'report.json'), passed=report['passed'], total=report['total'])), flush=True)
    if not all(checks.values()): raise ValueError('GPU observer admission failed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); main(p.parse_args())
