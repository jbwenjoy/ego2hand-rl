"""Explain observed side diagnostics; no source changes, solver or dynamics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import json,sys
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path('D:/workspace/supermimic-observation-retarget-study');OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from contact_penetration_geometry_nominal import ContactGeometry

def main():
    a=np.load(OUT/'reconstructed_arrays.npz',allow_pickle=False);c=np.load(ROOT/'outputs/contact_refined_nominal_geometry_v1/geometric_retarget.npz',allow_pickle=False)
    d=a['pair_signed_distance_m'];w=a['pair_witness_w'];err=np.abs(np.linalg.norm(w[...,3:]-w[...,:3],axis=-1)-np.abs(d));bad=np.argwhere((d>=0)&(err>1e-6))
    g=ContactGeometry('D:/workspace/supermimic-chord-study/outputs/rotating_reference_warp_a0/model.mjb','D:/workspace/supermimic-chord-study/outputs/a1l_paired_v2_clip007_2000iter_seed1/compiled/fixed_physics_context.npz')
    anomalies=[{'frame':int(f),'pair_index':int(k),'names':[g.names[i] for i in g.pairs[k]],'distance_m':float(d[f,k]),'segment_length_m':float(np.linalg.norm(w[f,k,3:]-w[f,k,:3])),'error_m':float(err[f,k]),'qualification':'Not an attraction or a certified positivegap oracle; residue0 retains the declared nonattraction scope.'} for f,k in bad]
    side=json.loads((OUT/'side_geometry_queries.json').read_text());frame=min(side,key=lambda x:x['hand_mount_unfiltered']['signed_min_m'])['frame']
    g.set_object_pose(c['mapped_object_pose_w'][frame]);g.robot.forward(c['joint_pos_rad'][frame]);mount=g.model.geom('robot/right_mount_collision').id;mountrows=[]
    for h in sorted(set(g.pairs[:,0])):
        seg=np.zeros(6);dist=mujoco.mj_geomDistance(g.model,g.data,int(h),mount,1.,seg)
        if dist<0:mountrows.append({'frame':frame,'hand_geom':g.names[h],'mount_geom':g.names[mount],'signed_distance_m':float(dist),'negative_witness_error_m':float(abs(np.linalg.norm(seg[3:]-seg[:3])-abs(dist))),'hand_contype':int(g.model.geom_contype[h]),'hand_conaffinity':int(g.model.geom_conaffinity[h]),'mount_contype':int(g.model.geom_contype[mount]),'mount_conaffinity':int(g.model.geom_conaffinity[mount])})
    contacts=[json.loads(line) for line in (OUT/'native_contacts.jsonl').read_text().splitlines()]
    worst={kind:min((x for x in contacts if x['category']==kind),key=lambda x:x['distance_m']) for kind in ['hand_self','hand_object']}
    report={'status':'SIDE_DIAGNOSTICS_RECORDED_NO_PHYSICAL_CLEARANCE_CLAIM','nonnegative_pair_query_anomalies':anomalies,'worst_native_contacts':worst,'mount_unfiltered_negative_pairs_at_worst_frame':mountrows,'mount_scope':'Mount collision is disabled in frozen model. Direct unfiltered mesh overlap may include expected rigid assembly overlap and is not a native physics contact. Preserve values and masks; do not write missingnativehits as zero clearance.','table_scope':'No table geom exists; terrainplane is independently named and recorded.','dynamics_steps':0,'physical_reference_qualified':False}
    (OUT/'side_diagnostic_detail.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
