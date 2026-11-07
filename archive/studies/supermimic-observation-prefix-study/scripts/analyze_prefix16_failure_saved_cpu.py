"""Saved candidate/contact/report analysis only; no geometric or solver query."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,default=ROOT/'outputs/prefix16_geometry_v1');ap.add_argument('--output',type=Path,default=ROOT/'outputs/prefix16_failure_review_v1');a=ap.parse_args();src=a.input;out=a.output;out.mkdir(parents=True,exist_ok=False)
    # These are the ONLY scientific inputs; no oldmodel/reference/curve/query.
    candidate=src/'candidate.npz';contactsfile=src/'native_contacts.json';reportfile=src/'report.json'
    z=np.load(candidate,allow_pickle=False);contacts=read(contactsfile);old=read(reportfile)
    if sha(candidate)!='1e67d99b25b6d991fa8e5209f521ea5c269b3212a45b054fe7353ab5b7277ae4' or old['candidate_sha256']!=sha(candidate):raise ValueError('Not frozenformal candidate')
    t=z['time_s'];Q=z['q_goal_rad'];residual=z['residual'];pairs=z['native_pair_indices'];kind=z['native_pair_kind'];mask=z['native_generated_pair_mask'];distance=z['native_selected_contact_distance_m'];ties=z['native_contact_tie_count_by_pair']
    P=len(pairs);L=45+P+18+40;local=residual[:16*L].reshape(16,L);J=csr_matrix((z['jacobian_CSR_data'],z['jacobian_CSR_indices'],z['jacobian_CSR_indptr']),shape=tuple(z['jacobian_shape']))
    groups={'bone_direction':(0,45),'native_overlap':(45,45+P),'material_position':(45+P,45+P+9),'material_normal':(45+P+9,45+P+18),'q0_prior':(45+P+18,45+P+38),'nominal_prior':(45+P+38,L)}
    by_frame={name:.5*np.sum(local[:,start:stop]**2,axis=1) for name,(start,stop) in groups.items()}
    by_frame['adjacent_q']=np.r_[0.,.5*np.sum(residual[16*L:16*L+300].reshape(15,20)**2,axis=1)]
    by_frame['second_q']=np.r_[0.,.5*np.sum(residual[16*L+300:].reshape(14,20)**2,axis=1),0.]
    cost=float(.5*residual@residual);costs={name:{'cost':float(v.sum()),'fraction_percent':float(100*v.sum()/cost)} for name,v in by_frame.items()}
    native_min={};names={}
    for c in contacts:
        key=(c['frame'],tuple(sorted(c['geom'])));native_min[key]=min(native_min.get(key,np.inf),c['distance_m']);names[tuple(sorted(c['geom']))]=c['names']
    checks={'all16_shapes':Q.shape==(16,20) and distance.shape==(16,P),'cost_matches_saved_report':abs(cost-old['cost'])<1e-10,'cost_groups_sum':abs(sum(c['cost'] for c in costs.values())-cost)<1e-10,'q0_hard_bitexact':np.array_equal(Q[0],z['q0_fixed_rad']),'knots_hardrange_zero_excess':np.maximum(np.maximum(z['joint_lower_rad']-Q,Q-z['joint_upper_rad']),0).max()==0,'all_absent_distances_remain_NaN':np.isnan(distance[~mask]).all()}
    rows=[];overlap_pairs={};maxdepth=[]
    for frame in range(16):
        frame_records=[c for c in contacts if c['frame']==frame]
        for p in np.flatnonzero(mask[frame]):
            key=(frame,tuple(pairs[p]));checks[f'frame{frame}_saved_pairmin_matches_raw']=checks.get(f'frame{frame}_saved_pairmin_matches_raw',True) and key in native_min and abs(native_min[key]-distance[frame,p])<1e-12
            pair=tuple(pairs[p]);value=overlap_pairs.setdefault(pair,{'geom':list(map(int,pair)),'names':names.get(pair),'kind':str(kind[p]),'generated_frames':[],'frames_over3mm':[],'max_depth_m':0.,'cost':0.})
            value['generated_frames'].append(frame);depth=max(-float(distance[frame,p]),0.);value['max_depth_m']=max(value['max_depth_m'],depth);value['cost']+=.5*max((depth-.0005)/.001,0.)**2
            if depth>.003:value['frames_over3mm'].append(frame)
        ho=mask[frame]&(kind=='hand_object');hs=mask[frame]&(kind=='hand_self');depth=float(np.maximum(-distance[frame,ho],0).max()) if ho.any() else None;maxdepth.append(depth)
        row={'frame':frame,'time_s':float(t[frame]),'hand_object_max_depth_m':depth,'hand_self_max_depth_m':float(np.maximum(-distance[frame,hs],0).max()) if hs.any() else None,'HO_over3mm':bool(depth is not None and depth>.003),'tied_generated_pairs':int(np.count_nonzero(mask[frame]&(ties[frame]>1))),'native_raw_contacts':len(frame_records),'unknown_catalog_pairs':int((~mask[frame]).sum()),'bone_mean_deg':float(z['bone_angle_error_deg'][frame].mean()),'bone_max_deg':float(z['bone_angle_error_deg'][frame].max())}
        for finger in range(3):row[f'f{finger+1}_material_position_error_m']=float(z['anchor_position_error_m'][frame,finger]);row[f'f{finger+1}_material_normal_error_deg']=float(z['anchor_normal_error_deg'][frame,finger])
        for name,v in by_frame.items():row[name+'_cost']=float(v[frame])
        rows.append(row)
    overlap_pairs=sorted(overlap_pairs.values(),key=lambda x:x['max_depth_m'],reverse=True)
    worst=min(contacts,key=lambda c:c['distance_m']);worst_frame=worst['frame'];worst_pair=tuple(sorted(worst['geom']));pairindex=next(i for i,p in enumerate(pairs) if tuple(p)==worst_pair)
    worst_records=[c for c in contacts if c['frame']==worst_frame and tuple(sorted(c['geom']))==worst_pair]
    collision_gradient={};gradient_norms={}
    for name,(start,stop) in groups.items():
        ids=np.concatenate([np.arange(frame*L+start,frame*L+stop) for frame in range(16)]);collision_gradient[name]=np.asarray(J[ids].T@residual[ids]);gradient_norms[name]=float(np.max(np.abs(collision_gradient[name])))
    opposition={}
    for f in (worst_frame,15):
        block=slice((f-1)*20,f*20);g=collision_gradient['native_overlap'][block];h=collision_gradient['material_position'][block];opposition[str(f)]=float(g@h/(np.linalg.norm(g)*np.linalg.norm(h)))
    interval_speed=np.diff(Q,axis=0)/.02;idx=np.unravel_index(np.argmax(np.abs(interval_speed)),interval_speed.shape);qd=np.gradient(Q,.02,axis=0,edge_order=2)
    lower_gap=Q[0]-z['joint_lower_rad'];upper_gap=z['joint_upper_rad']-Q[0]
    near_lower=np.flatnonzero(lower_gap<1e-12);near_upper=np.flatnonzero(upper_gap<1e-12)
    outward=[{'joint_index':int(j),'side':'lower','margin_rad':float(lower_gap[j]),'designed_endpoint_qd_rad_s':float(qd[0,j])} for j in near_lower if qd[0,j]<0]+[{'joint_index':int(j),'side':'upper','margin_rad':float(upper_gap[j]),'designed_endpoint_qd_rad_s':float(qd[0,j])} for j in near_upper if qd[0,j]>0]
    # Verify group semantics directly from saved geometric features.
    checks['anchor_position_weight_scale_matches']=np.max(np.abs(local[:,45+P:45+P+9].reshape(16,3,3)-(z['anchor_material_points_w']-z['anchor_goal_points_w'])/.001))<1e-12
    checks['normal_weight_dimensionless_matches']=np.max(np.abs(local[:,45+P+9:45+P+18].reshape(16,3,3)-.25*(z['anchor_material_normals_w']-z['anchor_goal_hand_normals_w'])))<1e-12
    bones=np.array([(1+f*4+b,2+f*4+b) for f in range(5) for b in range(3)]);v=z['fk_hand_points_w'][:,bones[:,1]]-z['fk_hand_points_w'][:,bones[:,0]];v/=np.linalg.norm(v,axis=-1,keepdims=True)
    checks['bone_residual_is_unitdirection_not_angle']=np.max(np.abs(local[:,:45].reshape(16,15,3)-(v-z['desired_unit_bone_directions_w'])))<1e-12
    checks['optimizer_only_ftol_not_feasibility']=old['optimizer_success'] is True and old['optimizer_status']==2 and old['optimality']>1e-8
    report={'status':'SAVED_ONLY_PREFIX_CONSTRAINT_FAILURE_REVIEW_COMPLETE' if all(checks.values()) else 'SAVED_REVIEW_CHECK_FAIL','checks':{k:bool(v) for k,v in checks.items()},'inputs':{'candidate':pin(candidate),'native_contacts':pin(contactsfile),'report':pin(reportfile)},'optimizer_termination':{k:old[k] for k in ('optimizer_success','optimizer_status','optimizer_message','nfev','njev','optimality','cost')},'cost_components':costs,'worst_overlap':{'frame':worst_frame,'time_s':float(t[worst_frame]),'geom':worst['geom'],'names':worst['names'],'depth_m':-worst['distance_m'],'native_records_at_pair':worst_records,'saved_pair_tie_count':int(ties[worst_frame,pairindex]),'anchor_position_errors_m':z['anchor_position_error_m'][worst_frame].tolist(),'anchor_normal_errors_deg':z['anchor_normal_error_deg'][worst_frame].tolist(),'bone_mean_deg':float(z['bone_angle_error_deg'][worst_frame].mean()),'anchor_order':'Frozen producerH123 orderf1/f2/f3. Priorverifiedmaterialcontract usesf1link4; thisreview doesnotread/reselectthatbinding.'},'overlap_pair_summary':overlap_pairs,'frames_over3mm':[r['frame'] for r in rows if r['HO_over3mm']],'tied_pair_frames':np.flatnonzero(np.any(mask&(ties>1),axis=1)).tolist(),'tied_generated_pair_slots':int(np.count_nonzero(mask&(ties>1))),'saved_local_gradient_algebra':{'inf_norms_by_group':gradient_norms,'collision_vs_material_position_cosine_by_frame':opposition,'raw_total_gradient_inf':float(np.max(np.abs(J.T@residual))),'scope':'Algebra on savedselectedbranch J only. Native ties/numerical derivative qualification remain unresolved; not a global physicalgradient or feasibility proof.'},'continuity_diagnostics':{'maximum_abs_q_knot_FD_rad_s':float(np.max(np.abs(interval_speed))),'interval':[int(idx[0]),int(idx[0]+1)],'joint_index':int(idx[1]),'joint_name':f'robot/right_finger{idx[1]//4+1}_joint{idx[1]%4+1}','endpoint_continuous_FD_qd0_rad_s':qd[0].tolist(),'near_bound_outward_endpoint_derivative':outward,'q_is_not_u':'Actual u/C2/PDdamping schedule not in theseinputs. This q-FD number is not a measuredtarget-rate failure; futureactualu mustmeet frozen15rad/s criterion.','curve_scope':'Knothardbounds andsmallsoftdifferences do not certify interpolatedrange/velocity/acceleration. Nearboundoutward derivative flagsneed exactcurve rootcheck, not a replacementstencil orclamp.'},'prospective_constraint_review':['First separate feasibility from target-fit objective. Preserve3mm qualification bound as a hardgate; penaltyminimum isnot an inequality certificate.','A universalhard.5mm planningconstraint conflicts with fixedq0 whoseexistingdepth2.183mm alreadyexceeds.5mm; do not silently changeq0/threshold to makeit fit.','Joint/target velocity andcontinuousrange constraints must beexplicit. Weight.03 onrawΔq/Δ²q isnot a rate/acceleration bound, andnot divided bydt in thefrozenexperiment.','Hardcollision methods need a validated distance/contactdiscovery interface forabsentpairs andtie-aware/branch-aware constraint derivatives. UNKNOWNgaps cannot befilled0; changingCCDopts isnot automaticallyhigherprecision.','A point anchor doesnot constrain all descendantcolliders or correctsurfaceorientation. Handself/HO fullgeometry constraints remainindependent of materialfit.','Fixed objectlocal/handmaterial sticking through16frames is an uncertain contactmode hypothesis; no knownforce/contactlabels. A prospective release/slide hypothesis requiresnewdeclaredprotocol, not online nearest-anchor reselection.','If this constrainedinitialization family is infeasible oroptimizerfails, reportthatfamily/searchfailure rather than motionimpossible. Fullmotion requiresanewcontinuouswholetrajectory; neverappend theold151qtail.','No newoptimization/physics is proposed forthisturn. Preserve thisfailedcandidate and theearlierframe0restPASS independently.'],'physics_steps':0,'geometry_queries':0,'optimization_calls':0,'GPU_used':False,'new_contact_labels':False,'source_sha256':sha(__file__)}
    with (out/'all_frames.csv').open('w',newline='',encoding='utf-8-sig') as f:writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(out/'report.json',report)
    plt.rcParams.update({'svg.fonttype':'none','font.size':10});fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True,layout='constrained')
    axes[0].plot(t,np.array(maxdepth)*1000,'o-',label='Saved native HO maxdepth');axes[0].axhline(3,color='crimson',linestyle='--',label='Frozen3mm gate');axes[0].axhline(.5,color='gray',linestyle=':',label='Soft.5mm allowance');axes[0].set_ylabel('Depth (mm)');axes[0].legend(fontsize=8)
    for i in range(3):axes[1].plot(t,z['anchor_position_error_m'][:,i]*1000,'o-',label=f'F{i+1} materialpoint error');axes[2].plot(t,z['anchor_normal_error_deg'][:,i],'o-',label=f'F{i+1} normal error')
    axes[1].set_ylabel('Materialpoint error (mm)');axes[2].set_ylabel('Normal error (deg)');axes[1].legend(fontsize=8);axes[2].legend(fontsize=8);axes[2].set_xlabel('Geometric reference time (s)')
    for ax in axes:ax.grid(alpha=.2);ax.set_xlim(0,.3)
    fig.suptitle('Saved prefix16: point fitting does not enforce whole-hand nonpenetration\nftol termination is not geometry/velocity/dynamics qualification',fontsize=12)
    for ext in ('png','svg'):fig.savefig(out/('saved_tradeoffs.'+ext),dpi=180)
    plt.close(fig)
    print(json.dumps({'status':report['status'],'checks':len(checks),'failed':[k for k,v in checks.items() if not v],'worst':{k:report['worst_overlap'][k] for k in ('frame','names','depth_m','saved_pair_tie_count')},'frames_over3mm':report['frames_over3mm'],'costs':costs,'continuity':report['continuity_diagnostics']},indent=2));raise SystemExit(not all(checks.values()))

if __name__=='__main__':main()
