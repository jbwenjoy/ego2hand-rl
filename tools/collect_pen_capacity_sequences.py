"""Capture prospective geometry-capacity labels on the audited load-sequence schedule."""
import json,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import collect_pen_wrench_sequences as sequences
from supermimic_research.mechanics.controlled_pen_records import native_state55
from supermimic_research.mechanics.pen_capacity_trace import packet_capacity
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json


def main(out):
    out.mkdir(exist_ok=False)
    root=sequences.b.ROOT/'runs'
    template=read(file_pin(root/'pen_scaled_train_v1/eval_02781/repeat_0/report.json'))
    refs=file_pin(root/'pen_capacity_targets_v1/report.json')
    write_json(out/'intent.json',dict(reference_capacity=refs,source=file_pin(Path(__file__)),definition='Eight per-tick direction-support vectors for each 20ms control, with individual moving object BODY frame and COM. Arithmetic time average and final tick are separately retained, 1N shared normal budget. Geometry from native contact positions, normals, friction before considering realized force. No policy update.',split='Original prospective W1 sequence schedule: A/B/C/D TRAIN, VAL, TEST1/2/3. Actor families share a reference. Never fit on previously saved final evaluation packets.'))
    samples=[];original=sequences.b.decode_saved_case
    def decoded(case,z,model,physical):
        validated=original(case,z,model,physical)
        layout=case['model_layout']
        row=packet_capacity(z,model,layout,physical['geom_friction'])
        if row['maximum_containment_excess']>2e-5 or row['positive_gap_skipped_max_load_N']>1e-5:
            raise ValueError('Contact capacity inconsistent with realized native force')
        s=native_state55(z['pre_qpos'][0],z['pre_qvel'][0],template['qids'],template['dofs'],layout['object_qposadr'],layout['object_dofadr'])
        samples.append((s,z['post_ctrl'][0,template['aids']].copy(),row['capacity'],row['mask'],row['normal_load_N']))
        return validated
    with patch.object(sequences.b,'decode_saved_case',decoded):
        sequences.main(out/'collection')
    run=out/'collection/report.json';data=json.loads(run.read_bytes());index=0;items=[]
    for e in data['episodes']:
        with np.load(verify_file_pin(e['trajectory'])) as f:x=f['x']
        batch=samples[index:index+len(x)];index+=len(x)
        np.testing.assert_array_equal(x[:,:55],np.asarray([r[0] for r in batch]))
        np.testing.assert_array_equal(x[:,81:101],np.asarray([r[1] for r in batch]))
        path=out/f"{e['label']}_{e['seed']}.npz"
        np.savez_compressed(path,capacity=np.stack([r[2] for r in batch]),mask=np.stack([r[3] for r in batch]),normal_load_N=np.stack([r[4] for r in batch]))
        items.append(dict(**e,capacity=file_pin(path)))
    assert index==len(samples)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),collection=file_pin(run),episodes=items,controls=index,native=data['native']))
def read(pin):return json.loads(verify_file_pin(pin).read_bytes())
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
