"""Closed-form axial-moment upper bound, no force optimization or new query."""
from pathlib import Path
import json,hashlib
import numpy as np

P=Path(__file__).resolve().parent
def pin(f):
    f=Path(f).resolve();b=f.read_bytes()
    return {'path':str(f),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def main():
    dest=P/'lever_bound_addendum_v1.json'
    if dest.exists():raise FileExistsError(dest)
    prior=json.loads((P/'report.json').read_text(encoding='utf-8'))
    src=prior['inputs']['native_report'];assert pin(src['path'])==src
    r=json.loads(Path(src['path']).read_text(encoding='utf-8'))
    c=next(s for s in r['seed_results'] if s['seed']==0)['shortlist'][0]
    n=c['native'];points=np.array(n['raw']['contact_rows24'])[:,6:9]
    COM=np.array(n['raw']['phone_COM_world_m']);origin=points[0]
    ell=points[1:].mean(axis=0)-origin;ell/=np.linalg.norm(ell)
    d=points-origin;perp=d-(d@ell)[:,None]*ell
    radii=np.linalg.norm(perp,axis=1);mu=.7;cap=3.;force_per_Fn=np.sqrt(1+mu*mu)
    groups=[x['category'] for x in n['native_contact_details']]
    group_r={g:float(radii[np.array(groups)==g].max()) for g in sorted(set(groups))}
    bound=sum(cap*force_per_Fn*x for x in group_r.values())
    tau=abs(float(ell@np.cross(COM-origin,.2*np.array([0.,0.,-9.81]))))
    out={'schema':'saved_contact_line_axial_moment_bound:7','source':pin(__file__),
        'geometry_analysis':pin(P/'report.json'),'native_report':src,
        'line':'thumb point to centroid of both native index points',
        'formula':'abs(sum tau_line)<=sum_finger cap_f*sqrt(1+mu^2)*max_perp_radius_f',
        'force_norm_reason':'Each native four-ray vertex has norm sqrt(1+mu^2) per unit Fn; nonnegative combinations obey triangle bound.',
        'perp_radii_m':radii.tolist(),'group_max_perp_radius_m':group_r,
        'index_pair_distance_m':float(np.linalg.norm(points[1]-points[2])),
        'absolute_gravity_axial_moment_Nm':tau,'capped_optimistic_contact_axial_moment_upper_Nm':bound,
        'gravity_moment_over_bound':tau/bound,
        'minimum_total_normal_N_if_all_at_max_radius_and_no_other_constraint':tau/(force_per_Fn*radii.max()),
        'scope':'Only fixed native three-point ideal forces. Does not construct equilibrium or certify new surface patches; force/moment realization and new geometry remain unknown.',
        'new_LP_native_FK_query_step_NN_GPU_calls':0}
    dest.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report':pin(dest),'gravity_tau_Nm':tau,'capped_bound_Nm':bound,'ratio':tau/bound,
        'normal_lower_bound_N':out['minimum_total_normal_N_if_all_at_max_radius_and_no_other_constraint']}))

if __name__=='__main__':main()
