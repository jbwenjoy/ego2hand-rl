"""Expose a 50Hz controller hook while preserving the frozen native loop/gates."""
from pathlib import Path
import hashlib,json,difflib
H=Path(__file__).resolve().parent;P=H.parent
base=P/'functional_tripod_dynamics_v1/run_functional_tripod_dynamics_v3.py'
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
assert pin(base)['sha256']=='d433a51df307bf45b40e70bc2c74776654eb2ece65bd2de67ae8163c201fd960'
source=base.read_text(encoding='utf8');original=source[source.index('def evaluate_case('):source.index('\ndef dt_pair_summary(')]
changes=[
('def evaluate_case(m,variant,dt,case_dir,phone,pg,table,pads,mujoco,np,counts):','def evaluate_case_controlled(m,variant,dt,case_dir,phone,pg,table,pads,mujoco,np,counts,controller):'),
('    for tick in range(steps):\n        mujoco.mj_step(m,d);',
'''    # A declared control hook is the sole new mutation of d.ctrl. No state/force resets.
    stride=round(.02/dt)
    if stride not in (64,128) or abs(stride*dt-.02)>1e-15:raise ValueError('Integer20ms control stride')
    control_records=[];pre_ctrl=[]
    sig=mujoco.mjtState.mjSTATE_INTEGRATION;size=mujoco.mj_stateSize(m,sig)
    if int(sig)!=16383 or size!=300:raise ValueError('Frozen integration receipt shape')
    for tick in range(steps):
        if tick%stride==0:
            before=np.empty(size);after=np.empty(size);mujoco.mj_getState(m,d,before,sig)
            requested=np.asarray(controller(tick,dt,m,d),dtype=np.float64)
            mujoco.mj_getState(m,d,after,sig)
            counts['controller_receipt_reads']=counts.get('controller_receipt_reads',0)+2
            if before.tobytes()!=after.tobytes():raise ValueError('Controller mutated native integration state')
            if requested.shape!=(20,) or not np.isfinite(requested).all() or np.any(requested<m.actuator_ctrlrange[:,0]) or np.any(requested>m.actuator_ctrlrange[:,1]):raise ValueError('Finite legal absolute command, no clamp')
            ctrl=requested.copy();d.ctrl[:]=ctrl
            control_records.append(dict(pre_tick=tick,time_s=tick*dt,ctrl20_F64=ctrl.tolist()))
            counts['controller_calls']=counts.get('controller_calls',0)+1
        pre_ctrl.append(ctrl.copy())
        mujoco.mj_step(m,d);'''),
('    np.savez_compressed(case_dir/\'trace.npz\',**r,ctrl20=ctrl,ctrl_margin20x2=np.asarray(variant[\'ctrl_margin20x2\']),initial_qpos27=q0,initial_COM_world3=COM0,',
 '    write(case_dir/\'control_schedule.json\',control_records)\n    np.savez_compressed(case_dir/\'trace.npz\',**r,pre_executed_ctrl20=np.asarray(pre_ctrl[:completed]),ctrl20=ctrl,ctrl_margin20x2=np.asarray(variant[\'ctrl_margin20x2\']),initial_qpos27=q0,initial_COM_world3=COM0,'),
('        no_external_forces_equality_mocap_or_pose_reset=True,static_zero_acceleration_entry_required=False)',
'''        no_external_forces_equality_mocap_or_pose_reset=True,static_zero_acceleration_entry_required=False,
        control_schedule=pin(case_dir/'control_schedule.json'),control_scope='Planned50Hz controller only; trace pre_executed_ctrl20 is actual perPRE-tick command. Legacy trace ctrl20 is last command and ctrl_margin20x2 describes initial source only. Raw integration receipts protect callback state; no full-MjData certificate.',
        native_loop_and_physical_success_gates_from='frozen functional_tripod_dynamics_v3, unchanged after insertion of explicit control hook and schedule fields')''')]
new=original
for a,b in changes:
    assert new.count(a)==1,a[:100];new=new.replace(a,b,1)
reverse=new
for a,b in reversed(changes):assert reverse.count(b)==1;reverse=reverse.replace(b,a,1)
assert reverse==original
header='''"""Scheduled-control native loop derived from frozen v3; no backend on import."""
from pathlib import Path
import hashlib,importlib.util
BASE=Path(__file__).resolve().parents[1]/'functional_tripod_dynamics_v1/run_functional_tripod_dynamics_v3.py'
if hashlib.sha256(BASE.read_bytes()).hexdigest()!='d433a51df307bf45b40e70bc2c74776654eb2ece65bd2de67ae8163c201fd960':raise ValueError('Frozen evaluator changed')
_sp=importlib.util.spec_from_file_location('policy_eval_frozen_native_gates',BASE);frozen=importlib.util.module_from_spec(_sp);_sp.loader.exec_module(frozen)
DURATION,GATES,OLD_GATES=frozen.DURATION,frozen.GATES,frozen.OLD_GATES
pin,write,contact_snapshot=frozen.pin,frozen.write,frozen.contact_snapshot
'''
target=H/'controlled_native_kernel_v1.py'
with target.open('x',encoding='utf8',newline='\n') as f:f.write(header+'\n'+new)
diff=H/'controlled_native_kernel_v1.diff'
with diff.open('x',encoding='utf8') as f:f.write(''.join(difflib.unified_diff(original.splitlines(True),new.splitlines(True),fromfile='frozen_evaluate_case',tofile='controlled_evaluate_case')))
manifest=dict(schema='scheduled_control_frozen_native_kernel_derivation:1',generator=pin(__file__),parent=pin(base),kernel=pin(target),diff=pin(diff),
    exact_original_function_recovered_by_reversing_only_declared_insertions=True,native_step_force_POST_observer_gates_and_dt_metrics_unchanged=True,
    named_changes=['20ms legal-control callback, guarded full300 integration receipt before/after callback','perPRE actual ctrl and command schedule saved; legacy final ctrl field explicitly scoped','new kernel name/explicit controller argument and summary scope'],actual_native_or_NN_execution=0)
with (H/'kernel_derivation_v1.json').open('x',encoding='utf8') as f:json.dump(manifest,f,indent=2)
print(json.dumps(dict(kernel=pin(target),derivation=pin(H/'kernel_derivation_v1.json'))))
