"""Paired contact-landmark IK experiment on the original 101 timestamps."""
from pathlib import Path
import json,sys
from unittest.mock import patch
import numpy as np
import track_mug_tip_sequence as baseline
import screen_mug_sequence_wrench as screen
from supermimic_research.mechanics.mug_contact_sequence import solve_contact_sequence
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json

def main(out):
    out.mkdir(exist_ok=False);root=baseline.ROOT/'runs';parent=root/'mug_tip_sequence_v1/report.json'
    prev=json.loads(parent.read_bytes());intent=json.loads(verify_file_pin(prev['intent']).read_bytes())
    binding=json.loads(verify_file_pin(intent['binding']).read_bytes());source=json.loads(verify_file_pin(intent['reference']).read_bytes())
    with np.load(verify_file_pin(source['output'])) as f:poses=f['T_world_from_BODY']
    write_json(out/'intent.json',dict(parent=file_pin(parent),rules='All active nonpositive-gap hand/mug material points in initial mechanical grasp. 0.3 mm position scale per hand body, 10 mm human tip scale; same joint regularization, time, preload and no-aid native test.'))
    def solve(model,q,targets):
        ids={model.geom(n).id for n in binding['hand_collision_names']}
        seq,tips,ok,err,names=solve_contact_sequence(model,q,targets,poses,binding['mug_body_id'],ids)
        np.savez_compressed(out/'contact_errors.npz',error_m=err,names=names)
        return seq,tips,ok
    with patch.object(baseline,'solve_trajectory',solve):
        baseline.main(out/'contact_ik')
    screen.main(out/'contact_ik',out/'wrench')
    new=json.loads((out/'contact_ik/report.json').read_bytes())
    with np.load(verify_file_pin(new['arms']['constant']['trace'])) as f,np.load(verify_file_pin(prev['arms']['constant']['trace'])) as g:
        np.testing.assert_array_equal(f['qpos'],g['qpos'])
    rows={}
    for label,pin in [('tip_only',prev['arms']['retarget']['trace']),('contact_ik',new['arms']['retarget']['trace'])]:
        with np.load(verify_file_pin(pin)) as f:
            j=np.flatnonzero(f['com_error_m']>.01);k=np.flatnonzero(f['angle_deg']>10)
            rows[label]=dict(first_gt_1cm_s=float((j[0]+1)*.02) if j.size else None,first_gt_10deg_s=float((k[0]+1)*.02) if k.size else None)
    with np.load(out/'contact_errors.npz') as f: summary=dict(points=len(f['names']),rms_mm=1000*np.sqrt(np.mean(f['error_m']**2)),max_mm=1000*f['error_m'].max())
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),dynamic=file_pin(out/'contact_ik/report.json'),screen=file_pin(out/'wrench/report.json'),material_points=summary,thresholds=rows,constant_native_replay_exact=True))
    print(summary,rows)
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
