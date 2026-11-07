Replication joint-limit diagnosis: all nine cases

The frozen allowance remains 1 mrad. No physics or policy was rerun.
case_peaks.csv has every case, including both fully passing cases.
joint_violations.csv has every joint/side exceeding 1 mrad, with first and peak q/qd/target/torque.
report.json preserves model/input hashes and contiguous sampled violation windows.
all_case_joint_traces.npz preserves every sampled joint for all nine cases.

target[k] belongs to the previous [k-1,k] control interval. Outward qd/torque use the offending-bound sign.
A positive target margin is inside the hard range; tiny negative values are retained as recorded.
All maxima are from 20 ms logs, not unobserved 5 ms or continuous extrema.
Bounded targets and measured joint-state overshoot are distinct. This is not hardware safety evidence.
A future velocity-aware feasible controller requires a separate condition; current penalty-only must stay unchanged.

Reproduce independently:
"D:\workspace\dex-supermimic-research\.venv\Scripts\python.exe" "D:\workspace\supermimic-chord-study\outputs\replication_joint_limit_diagnostics_v1\reproduce.py" --output NEW_DIRECTORY

A1L_seed1_primary_equal_updates_2000: state max 0.825029 mrad, right_finger5_joint3 upper frame 97 t=1.94s; joint_gate=True, task=True
A2_seed1_primary_equal_updates_2000: state max 3.305249 mrad, right_finger3_joint1 upper frame 95 t=1.90s; joint_gate=False, task=False
A1L_seed2_primary_equal_updates_2000: state max 1.945835 mrad, right_finger5_joint3 upper frame 92 t=1.84s; joint_gate=False, task=False
A2_seed2_primary_equal_updates_2000: state max 1.456075 mrad, right_finger5_joint1 upper frame 97 t=1.94s; joint_gate=False, task=False
A2_seed3_primary_equal_updates_2000: state max 2.063614 mrad, right_finger5_joint3 upper frame 87 t=1.74s; joint_gate=False, task=False
A1L_seed3_primary_equal_updates_2000: state max 15.202575 mrad, right_finger3_joint1 upper frame 135 t=2.70s; joint_gate=False, task=False
A2_seed1_pre1200_availability_with_2000_cap: state max 1.747232 mrad, right_finger5_joint3 upper frame 105 t=2.10s; joint_gate=False, task=False
A2_seed2_pre1200_availability_with_2000_cap: state max 0.228934 mrad, right_finger5_joint1 upper frame 105 t=2.10s; joint_gate=True, task=True
A2_seed3_pre1200_availability_with_2000_cap: state max 2.135139 mrad, right_finger5_joint3 upper frame 63 t=1.26s; joint_gate=False, task=False
