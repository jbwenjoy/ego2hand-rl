"""Additional analytic surface checks without simulator queries or input edits."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
import mujoco
from pen_outer_surface import PenExterior
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/contact_intent_v2'
def main():
    report=json.loads((OUT/'report.json').read_text());model=Path('D:/workspace/supermimic-chord-study/outputs/rotating_reference_warp_a0/model.mjb');context=Path('D:/workspace/supermimic-chord-study/outputs/a1l_paired_v2_clip007_2000iter_seed1/compiled/fixed_physics_context.npz')
    forbidden={k:0 for k in ['mj_geomDistance','mj_step','mj_forward','mj_collision','mj_contactForce']}
    def deny(k):
        def f(*a,**kw):forbidden[k]+=1;raise AssertionError('Forbidden '+k)
        return f
    for k in forbidden:setattr(mujoco,k,deny(k))
    pen=PenExterior(model,context);z=np.load(OUT/'surface_proximity_151.npz',allow_pickle=False);checks={};rows=[]
    for name in ['source_original','source_rigid_mapped','morphology_normalized_target','robot_candidate_FK']:
        point=z[name+'__points_o'];closest=z[name+'__closest_o_m'];normal=z[name+'__outward_normal_o'];distance=z[name+'__signed_distance_m'];on=pen.query(closest)['unsigned_distance_m'];normerror=np.abs(np.linalg.norm(normal,axis=-1)-1).max();checks[name+'_closest_point_on_true_exterior']=float(on.max())<1e-12;checks[name+'_outward_unit_norm']=float(normerror)<1e-12
        # All three prespecified actual frames and all21 points are retained.
        # Nondifferentiable ties/radial-center cases are reported separately.
        sampled=point[[0,75,150]].reshape(-1,3);q=pen.query(sampled);fd=np.empty_like(sampled)
        for axis in range(3):
            a=sampled.copy();b=sampled.copy();a[:,axis]+=1e-7;b[:,axis]-=1e-7;fd[:,axis]=(pen.query(a)['signed_distance_m']-pen.query(b)['signed_distance_m'])/(2e-7)
        unique=(q['tie_count']==1)&~q['axis_radial_normal_ambiguity'];error=np.max(np.abs(fd[unique]-q['outward_normal_o'][unique])) if unique.any() else None;checks[name+'_unique_surface_normal_FD']=bool(error is not None and error<1e-5)
        rows.append({'variant':name,'all151_points':int(point.shape[0]*point.shape[1]),'closest_surface_residual_max_m':float(on.max()),'normal_unit_error_max':float(normerror),'FD_frames':[0,75,150],'FD_points_all':len(sampled),'unique_FD_points':int(unique.sum()),'nonsmooth_or_radial_ambiguous_points':np.flatnonzero(~unique).tolist(),'unique_FD_max_abs':float(error) if error is not None else None})
    v1=np.load(ROOT/'outputs/contact_intent_v1/surface_proximity_151.npz',allow_pickle=False);checks['v1_v2_same_numerical_arrays']=set(v1.files)==set(z.files) and all(np.array_equal(v1[k],z[k]) for k in z.files);checks['frame0_frozen_pin_unchanged']=hashlib.sha256((OUT/'frame0_rankings.json').read_bytes()).hexdigest()=='03a22cbdb2b9b74e6582bc3acc83843d007f9c0a07f0537437e158c326391e33';checks['no_forbidden_API']=not any(forbidden.values())
    result={'status':'passed_analytic_surface_consistency_checks' if all(checks.values()) else 'failed_analytic_surface_consistency_checks','checks':checks,'rows':rows,'forbidden_calls':forbidden,'source_report_sha256':hashlib.sha256((OUT/'report.json').read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'physical_reference_qualified':False,'scope':'Distance/nearest-exterior/normal consistency for this validated template and diagnostic poses, not observed contacts/forces or a global dynamical qualification.'};(OUT/'surface_checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2));raise SystemExit(not all(checks.values()))
if __name__=='__main__':main()
