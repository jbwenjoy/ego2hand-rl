"""Integrate source-review corrections; never regenerate directions or physics."""
from __future__ import annotations
import copy
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'proposals/current104_samewarp_physical_fd_v1'
OUT=ROOT/'proposals/current104_samewarp_physical_fd_v2'


def pin(path):
    p=Path(path).resolve(strict=True);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return {'path':str(p),'sha256':h.hexdigest()}


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    old=json.loads((OLD/'protocol.json').read_text(encoding='utf-8'))
    correction=json.loads((OLD/'CORRECTION_v1.json').read_text(encoding='utf-8'))
    dp=ROOT/'configs/current104-physical-directions-CPU-v1.json'
    direction=json.loads(dp.read_text(encoding='utf-8'))
    assert pin(OLD/'protocol.json')==correction['applies_to']
    assert direction['seed']==old['directions']['seed'] and direction['path_order']==old['selection']['trajectory_order']
    assert direction['phase_order']==old['selection']['phase_k'] and direction['direction_count']==8
    assert direction['epsilons']==[.002,.001] and direction['guard_ulps']==8
    p=copy.deepcopy(old)
    p['schema']='current104_same_Warp_physical_FD_proposal:2'
    p['status']='DRAFT_NOT_AUTHORIZED_PHYSICS_PENDING_CPU_DIRECTIONS_PREPARED'
    p['created_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    p['supersedes_for_future_execution_config']={'v1':pin(OLD/'protocol.json'),'mandatory_correction':pin(OLD/'CORRECTION_v1.json'),'old_bytes_preserved':True}
    p['implementation']['WM_comparison_default']='Separate fresh CPU-only consumer after physical GPUworker terminates and releasesGPU. Preserve strict104loader CUDA-uninitialized/CPUinput guards; never weaken them.'
    p['implementation']['direction_preparation']='Already executed pure NumPy input-domain preparation only; do not regenerate directions, pick a newseed or claim physical qualification.'
    p['inputs']['direction_preparation']={
        'plan':pin(dp),
        'module':pin(ROOT/'src/rough_prefix/interior_directions.py'),
        'CPU_contract':pin(ROOT/'outputs/interior_directions_CPU_v1/report.json'),
        'report':pin(ROOT/'outputs/current104_physical_directions_CPU_v1/report.json'),
        'targets_and_vectors':pin(ROOT/'outputs/current104_physical_directions_CPU_v1/directions_and_FP32_targets.npz'),
        'independent40':pin(ROOT/'outputs/samewarp_physical_FD_independent_review_v1/directions_review.json')}
    p['inputs']['physical_interface_review']=pin(ROOT/'outputs/current104_physical_probe_interface_review_v1/report.json')
    p['source_authorities']['installed_Warp_integration_serializer']=pin(Path('D:/workspace/dex-supermimic-research/.venv/Lib/site-packages/mujoco_warp/_src/support.py'))
    c=p['counts_maximum'];c.pop('startup_forwards');c.pop('total_physics_substeps');c.pop('aggregate_simulated_time_s')
    c.update(scheduled_donor_and_branch_integration_substeps_max=2480,
             scheduled_aggregate_simulated_time_s=12.4,logical_checked_donor_resets=4,
             reset_startup_forward_lower_bound=8,
             constructor_and_graphwarmup_forward_or_step_calls='PENDING actual separate count; not assumed zero or included in scheduled2480/620',
             wholeprocess_forward_or_step_exacttotal='NOT_STATICALLY_ASSERTED')
    p['snapshot_contract']['required_integration_channels'].append('history (installed HISTORY field)')
    p['snapshot_contract']['plugin_qualification']='Assert current bound MJB nplugin==0 before anyphysics. Installed Warp plugin state unsupported; plugin-bearing model is a setup rejection, never silently omit it.'
    p['directions']['empty_or_numerically_zero']='If free coordinates are zero, every direction is explicit NOT_RUN/no redraw. If free>0 but any masked Gaussian norm is exactlyzero (including numeric underflow), pinned module raises and the preparation/attempt fails closed; it is not an omitted direction or a redraw. This proposal imports the already saved artifact rather than executing the module again.'
    p['directions']['primary_comparison']='Physical central slope denominator remains2*epsilon. Primary WM AD projection uses the saved effectiveFP32vector=(uplus-u_minus)/(2epsilon); nominal-unit projection is also retained under an explicit secondary label, never choose which fits physics better.'
    p['directions']['additional_diagnostics']='Save midpoint shift (uplus+uminus)/2-center, effectivevector norm, maskedvector matrix rank/singularvalues. Eight directions share a state and may have rank<8; neither rows nor rank create independent training/demo N.'
    p['sensor_force_and_time']['postforward_contact_cache']='A forward call need not invalidate the ContactSensor lazy cache, so postforward reader.contact is not automatically fresh20ms truth; preserve exact native reward_snapshot instead.'
    p['sensor_force_and_time']['self15_truth_policy']='Freshdonor reward_snapshot.contact is the fixed primary truth for that newdonor, with original found bits saved. Save each of3baseline labels and all exact-bit disagreement masks. No majority vote, bestlabel, hand-object substitution or force-selected label. Baseline label variation is reported separately from the continuous state/object-reward replay gate; no binary-contact derivative qualification.'
    p['cost_and_lifecycle']['setup_call_accounting']='Scheduled2480 are donor+branch integrating substeps after checked initialization, not wholeprocess exactstep total. Record all constructor/reset/graphwarmup forward or step calls separately, plus4logical resets and at least8resetstartup forwards; no exact624forward claim.'
    p['future_runplan_requirements']=['explicit implementation/supervisor/checker sourcepins and complete dependencyclosure before/after, metadata archives',
        'bind this integratedV2 plus existingdirection plan/report/NPZ unchanged',
        'GPU_run_authorized false until separate Rootauthorization; no autoexecution',
        'stageGPU physicalcapture and stageCPU frozenWMconsumer have separate process identities/costs and immutable datajoin',
        'new proposal alone is not reset/snapshot/nativeparity/physicalgradient qualification']
    p['generator']=pin(__file__)
    write(OUT/'protocol.json',p)
    write(OUT/'READY.json',{'schema':'same_Warp_FD_integrated_proposal_preparation:2','status':p['status'],
          'proposal':pin(OUT/'protocol.json'),'generator':pin(__file__),'historical_v1':pin(OLD/'protocol.json'),
          'mandatory_correction':pin(OLD/'CORRECTION_v1.json'),'GPU_run_authorized':False,
          'GPU_used':False,'physics_steps':0,'WM_forward_calls':0,'direction_regeneration_calls':0,
          'Torch_imported':'torch' in sys.modules,'no_solver_or_geometry_calls':True})
    print(json.dumps({'proposal':pin(OUT/'protocol.json'),'READY':pin(OUT/'READY.json')},indent=2))


if __name__=='__main__':main()
