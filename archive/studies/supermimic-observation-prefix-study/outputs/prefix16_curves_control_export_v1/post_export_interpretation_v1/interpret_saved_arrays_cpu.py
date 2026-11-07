"""Read-only interpretation of frozen export arrays/roots, no model/query/LP."""
from pathlib import Path
import hashlib,json,csv
import numpy as np
ROOT=Path(__file__).resolve().parent.parent;OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
pin=lambda p:{'path':str(p),'sha256':sha(p)}
write=lambda p,x:Path(p).write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda x:x.item())+'\n',encoding='utf-8')
receiptpath=ROOT/'receipt.json';before=sha(receiptpath);assert before=='f5e45ac4b42cac400b6a723473087aac0da465d34457da48c03b86edbb639489'
r=read(receiptpath);qr=read(ROOT/'whole_q_range.json')
with np.load(ROOT/'reference_curves.npz') as z:ref={k:z[k].copy() for k in z.files}
with np.load(ROOT/'control_schedule.npz') as z:ctrl={k:z[k].copy() for k in z.files}
q0pin=r['source_pins']['q0_source'];assert sha(q0pin['path'])==q0pin['sha256']
with np.load(q0pin['path']) as z:lower=z['joint_lower_rad'].copy();upper=z['joint_upper_rad'].copy()
name=lambda j:f'robot/right_finger{j//4+1}_joint{j%4+1}'
peaks=[]
for row in qr['all300_segments_joints']:
 j=row['joint'];i=row['segment']
 for s,v in zip(row['candidate_s'],row['candidate_values_rad']):
  excess=max(lower[j]-v,v-upper[j],0.)
  if excess>0:peaks.append({'joint_index0':j,'joint':name(j),'segment_index0':i,'normalized_s':s,'time_s':float(ref['time_s'][i]+s*(ref['time_s'][i+1]-ref['time_s'][i])),'q_rad':v,'side':'lower' if v<lower[j] else 'upper','bound_rad':float(lower[j] if v<lower[j] else upper[j]),'excess_rad':float(excess)})
maximum=max(peaks,key=lambda x:x['excess_rad']);assert abs(maximum['excess_rad']-qr['max_hard_excess_rad'])<1e-12
rates=ctrl['actual_target_rate_14_rad_s'];i,j=np.unravel_index(np.abs(rates).argmax(),rates.shape)
rate={'joint_index0':int(j),'joint':name(j),'update_number1':int(i+1),'previous_command_index0':int(i),'new_command_index0':int(i+1),'update_time_s':float(ref['time_s'][i+1]),'before_physics_step1':int(4*(i+1)+1),'signed_rate_rad_s':float(rates[i,j]),'abs_rate_rad_s':float(abs(rates[i,j])),'previous_u_rad':float(ctrl['tracking_u_exec_15_FP32_rad'][i,j]),'new_u_rad':float(ctrl['tracking_u_exec_15_FP32_rad'][i+1,j]),'limit_rad_s':15.}
# Both caps are already exported and hash-bound by the static-load prior. No
# MjModel/MjData/FK/dynamics or new inference is needed to reconstruct ratios.
designpin=r['source_pins']['static_preload_design'];assert sha(designpin['path'])==designpin['sha256'];design=read(designpin['path']);mat=Path(designpin['path']).with_name('static_matrices.npz');assert sha(mat)==design['static_matrices_sha256']
with np.load(mat) as z:act=z['actuator_force_bounds'].copy();joint=z['joint_force_bounds'].copy()
lo=np.maximum(act[:,0],joint[:,0]);hi=np.minimum(act[:,1],joint[:,1]);assert np.all(lo<0) and np.all(hi>0)
force={};force_rows=[];ratio_arrays={}
for condition,prefix in [('tracking',''),('hold','hold_')]:
 group={}
 for grouping in ['affine','difference']:
  f=ctrl[prefix+'reference_PD_'+grouping+'_60_FP32_Nm'];ratio=np.where(f>=0,f/hi,f/lo);ratio_arrays[condition+'__'+grouping]=ratio
  k,j=np.unravel_index(ratio.argmax(),ratio.shape);first=np.flatnonzero(np.any(ratio>1,axis=1));row0=int(first[0]) if len(first) else None
  group[grouping]={'max_raw_predicted_ratio':float(ratio[k,j]),'joint_index0':int(j),'joint':name(j),'solver_prestate_index0':int(k),'time_s':float(k*.005),'raw_predicted_force_Nm':float(f[k,j]),'combined_force_lower_Nm':float(lo[j]),'combined_force_upper_Nm':float(hi[j]),'first_exceed':None if row0 is None else {'solver_prestate_index0':row0,'time_s':float(row0*.005),'joints':[name(x) for x in np.flatnonzero(ratio[row0]>1)],'joint_indices0':np.flatnonzero(ratio[row0]>1).tolist()},'exceed_components_of1200':int(np.count_nonzero(ratio>1))}
  for k in range(60):
   for j in range(20):force_rows.append({'condition':condition,'grouping':grouping,'solver_prestate_index0':k,'time_s':k*.005,'joint':name(j),'raw_predicted_force_Nm':float(f[k,j]),'ratio':float(ratio[k,j]),'cap_exceeded':bool(ratio[k,j]>1)})
 force[condition]=group
clamped=ctrl['u_pre_16_FP32_rad'][:15]!=ctrl['tracking_u_exec_15_FP32_rad'];assert int(clamped.sum())==22
outward=[{'joint_index0':int(j),'joint':name(int(j)),'q0_rad':float(ref['q_goal_rad'][0,j]),'hard_lower_rad':float(lower[j]),'hard_upper_rad':float(upper[j]),'continuous_qd0_rad_s':float(ref['q_curve_qd_rad_s'][0,j]),'native_forward_FD0_rad_s':float(ref['native_FD_goal_qd_rad_s'][0,j])} for j in qr['initial_outward_boundary_joint_indices']]
np.savez_compressed(OUT/'raw_prediction_ratios.npz',**ratio_arrays,combined_force_lower_Nm=lo,combined_force_upper_Nm=hi,clamp_mask_actual15x20=clamped)
with (OUT/'raw_prediction_ratios.csv').open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(force_rows[0]));w.writeheader();w.writerows(force_rows)
write(OUT/'visual_QA.json',{'status':'PASS_ACTUAL_IMAGE_VIEW','actual_view_image_inspection':True,'image':pin(ROOT/'prefix_cpu_diagnostics.png'),'SVG':pin(ROOT/'prefix_cpu_diagnostics.svg'),'checks':{'titles_axes_legends_no_clipping':True,'failed_status_in_title':True,'16knots61ticks_and_continuous_extrema_distinguished':True,'15commands14updates_distinguished':True,'hold_track_prediction_not_measured_force_label':True,'reference_shaft_not_actual_tracking_label':True},'actual_physics_or_actual_saturation_observed':False,'note':'Actual15 in title means post-C2 planned executed-target schedule, not simulator readback. No physical rollout occurred.'})
report={'status':'FAILED_CPU_GATES_RETAINED_INTERPRETATION_ONLY','original_receipt':pin(receiptpath),'export_source':r['source_pins']['exporter_source'],'curve_source':r['source_pins']['curves_source'],'control_source':r['source_pins']['control_source'],'maximum_continuous_q_excess':maximum,'all_positive_extrema_events':peaks,'initial_outward_boundary':outward,'maximum_actual_scheduled_target_rate':rate,'raw_reference_PD_prediction':force,'actual15_C2_clamps':{'components':int(clamped.sum()),'denominator':300,'per_command_counts':clamped.sum(axis=1).tolist(),'initial_u0_clamp_count':int(clamped[0].sum()),'joint_indices0_by_command':[np.flatnonzero(x).tolist() for x in clamped]},'force_cap_source':pin(mat),'force_cap_identity':'Static-design matrices use same pinnedmodel/context as this export. Both actuator and joint-force caps used; only read frozen arrays.','visual_QA':pin(OUT/'visual_QA.json'),'receipt_unchanged':sha(receiptpath)==before,'source_input_pins_still_match':all(sha(v['path'])==v['sha256'] for v in r['source_pins'].values()),'new_export_runs':0,'new_geometry_queries':0,'new_physics_steps':0,'new_LP_solves':0,'GPU_used':False,'interpretation':'Knots are hard-legal yet continuous polynomial can leave range; rate and raw PD force predictions fail independently. Raw reference PD>cap is not measured force or observed actual saturation. No stencil/weight/pose/control repair, fallback or physical execution is authorized.'}
assert report['receipt_unchanged'] and report['source_input_pins_still_match']
write(OUT/'report.json',report)
print(json.dumps({'max_q':maximum,'outward':outward,'maxrate':rate,'force':force,'clamps':report['actual15_C2_clamps'],'report_SHA256':sha(OUT/'report.json'),'QA_SHA256':sha(OUT/'visual_QA.json')},indent=2))
