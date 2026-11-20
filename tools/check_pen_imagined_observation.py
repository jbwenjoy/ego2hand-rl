"""Rebuild batched imagined-state Actor observations against native W1 recordings."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.controlled_pen_records import native_state55
from supermimic_research.mechanics.pen_imagined_observation import pen_imagined_observation
from supermimic_research.mechanics.pen_reward_chain import A,ACTION,CACHE,MOTION,ROBOT
from supermimic_research.runtime.shared_snapshot_transport import load_shared
from supermimic_research.strict_json import write_json


def read(pin):
    return json.loads(verify_file_pin(pin).read_bytes())


def main(out):
    out.mkdir(exist_ok=False,parents=True)
    records=[]; phases=(1,50,185,206,222,368)
    for arm,folder in [('residual','pen_scaled_train_v1'),('support','pen_scaled_support_matched_v1')]:
        run_pin=file_pin(ROOT/'runs'/folder/'report.json')
        evaluation=read(read(run_pin)['evaluations'][-1])
        case_pin=evaluation['cases'][0]['report']
        case=read(case_pin); initial=load_shared(case['initial'])
        static=initial['python']['tensors'];joint_ids=static[ACTION+(A('_target_ids'),)].long()
        bias,default_q,default_qd=(static[ROBOT+(A(key),)][:,joint_ids] for key in
                                   ('encoder_bias','default_joint_pos','default_joint_vel'))
        references=static[MOTION+(A('group'),A('joint_pos'))]
        links=static[MOTION+(A('group'),A('link_pos'))]
        poses=static[CACHE+(A('world_pose'),)][0]
        layout=case['model_layout']
        inputs=[]; recorded=[]; source_pins=[]
        for phase in phases:
            control=read(case['controls'][phase-1])
            pin=control['control_source']
            source=torch.load(verify_file_pin(pin),map_location='cpu',weights_only=False)
            dynamic=source['dynamic'];arrays=dynamic['arrays']
            assert phase==int(dynamic['phase'])
            state=native_state55(arrays['qpos'][0],arrays['qvel'][0],case['qids'],case['dofs'],
                                 layout['object_qposadr'],layout['object_dofadr'])
            # The previous executed/clipped action (or the initialized native
            # state at phase 1) is separately available without reading obs.
            if phase>1:
                prev=torch.tensor(read(case['controls'][phase-2])['applied_RAW20'])
            else:
                prev=static[ACTION+(A('_raw_actions'),)][0]
            torch.testing.assert_close(dynamic['last_raw'][0],prev,rtol=0,atol=0)
            inputs.append((torch.from_numpy(state),prev,references[phase],poses[phase],links[phase],dynamic['mount'][0]))
            recorded.append(torch.as_tensor(source['policy_observation'][0]).double())
            source_pins.append(pin)
        args=[torch.stack(column).double() for column in zip(*inputs)]
        actual=pen_imagined_observation(*args,bias.double(),default_q.double(),default_qd.double())
        expected=torch.stack(recorded)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)
        errors=abs(actual-expected)
        widths=[20,9,63,20,20,9,20]
        records.append(dict(arm=arm,run=run_pin,case=case_pin,control_sources=source_pins,
                            phases=phases,max_abs=float(errors.max()),
                            block_max_abs=[float(v.max()) for v in errors.split(widths,dim=-1)],
                            exact_previous_applied_raw=True))
    write_json(out/'report.json',dict(schema='pen_imagined_observation_native_parity:1',
        source=file_pin(ROOT/'src/supermimic_research/mechanics/pen_imagined_observation.py'),
        audit=file_pin(__file__),dtype='CPU float64 against saved native float32 Actor obs',
        columns=['reference_q','reference_pose9','reference_links','hand_q','hand_qd',
                 'current_pose9','previous_applied_raw'],atol=1e-6,rtol=0,rows=records))
    print([(r['arm'],r['max_abs'],r['block_max_abs']) for r in records])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    main(parser.parse_args().out.resolve())
