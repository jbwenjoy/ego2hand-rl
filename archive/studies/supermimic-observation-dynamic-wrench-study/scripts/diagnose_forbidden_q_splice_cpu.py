"""Saved-only counterexample to a warm-q0/old-frame1 splice; no trajectory export."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
FULL=Path('D:/workspace/supermimic-observation-retarget-study/outputs/native_contact_refined_geometry_v1/geometric_retarget.npz')
WARM=Path('D:/workspace/supermimic-observation-contact-intent-study/outputs/frame0_h123_seeded_baseline_v3/initialization_candidate.npz')
def pin(p):return {'path':str(p),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'proposals/short_dynamic_tracking_v1/splice_counterexample.json');a=ap.parse_args()
    if pin(FULL)['sha256']!='539d6247576b9b2e3fb1b143cf17b6ad507e3baea73c5fafcda4e221d9230c8f':raise ValueError('Full geometry drift')
    if pin(WARM)['sha256']!='57135b4c607f48d844e3977b56065dc6f1b4c8991626fc033651c6d136b89406':raise ValueError('Warm geometry drift')
    with np.load(FULL,allow_pickle=False) as z:old0=z['joint_pos_rad'][0].copy();old1=z['joint_pos_rad'][1].copy()
    with np.load(WARM,allow_pickle=False) as z:q0=z['joint_pos_rad'][0].copy()
    delta=old1-q0;j=int(np.argmax(np.abs(delta)))
    result={'status':'READONLY_FORBIDDEN_SPLICE_COUNTEREXAMPLE','inputs':{'full539d':pin(FULL),'warmH123baseline':pin(WARM)},'max_same_frame_q0_difference_rad':float(np.abs(q0-old0).max()),'warm0_to_oldframe1_delta_rad':delta.tolist(),'implicit_q_goal_FD_speed_rad_s':(delta/.020).tolist(),'maximum_abs_q_jump_rad':float(np.abs(delta).max()),'maximum_abs_implicit_q_speed_rad_s':float(np.abs(delta/.020).max()),'joint_index':j,'joint_name':f'robot/right_finger{j//4+1}_joint{j%4+1}','reference_target_rate_threshold_rad_s':15.,'source_code':pin(__file__),'physics_steps':0,'new_geometry_queries':0,'GPU_used':False,'scope':'This is an implicit q-goal discontinuity counterexample only. Actual u is separate; C2 was not applied here and no control/physical failure is measured. No hybrid trajectory is constructed or exported.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('maximum_abs_q_jump_rad','maximum_abs_implicit_q_speed_rad_s','joint_name','scope')},indent=2))
if __name__=='__main__':main()
