"""All four common-A line angles, not a native-contact/force-closure solve."""
from pathlib import Path
import hashlib,json,math,sys
import numpy as np

HERE=Path(__file__).resolve().parent
P=HERE.parent.parent
def pin(f):
    f=Path(f).resolve();b=f.read_bytes()
    return {'path':str(f),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(f):return json.loads(Path(f).read_text(encoding='utf-8'))
def description(n,direction):
    n=np.asarray(n);direction=np.asarray(direction)
    cosine=float(np.clip((n@direction)/(np.linalg.norm(n)*np.linalg.norm(direction)),-1,1))
    angle=math.acos(cosine)
    return {'angle_deg':math.degrees(angle),'cos_angle':cosine,
        'tan_mu_required_for_this_line_direction':math.tan(angle) if cosine>0 else None,
        'finite_compressive_line_direction':cosine>0,
        'outside_outer_circular_mu0p7':bool(cosine<=0 or math.tan(angle)>.7)}
def main():
    dest=HERE/'antipodal_geometry_v1.json'
    if dest.exists():raise FileExistsError(dest)
    rule=read(HERE/'antipodal_rules_v1.json')
    for rec in [rule['source'],*rule['inputs'].values()]:assert pin(rec['path'])==rec
    active=read(rule['inputs']['active_bound_report']['path'])
    anchor=read(rule['inputs']['surface_anchors']['path']);ego=read(rule['inputs']['original_ego_targets']['path'])
    origin=[]
    for i,row in enumerate(anchor['anchors']):
        B=np.array(row['surfaces']['phone']['projected_point_phone_body_m'])
        nO=np.array(row['surfaces']['phone']['outward_normal_phone_body'])
        old=np.array(ego['preparation'][i]['human_scan_target_object_m'])
        origin.append({'finger':row['finger'],'fixed_CAD_phone_material_body_m':B.tolist(),
            'CAD_phone_outward_normal_body':nO.tolist(),'original_ego_scan_material_body_m':old.tolist(),
            'CAD_minus_ego_body_mm':((B-old)*1000).tolist(),'distance_CAD_to_ego_material_mm':1000*float(np.linalg.norm(B-old))})
    results=[]
    for row in active['cases']:
        rec=row['case_pin'];assert pin(rec['path'])==rec;c=read(rec['path']);k=c['candidate']['kinematics']
        A=np.array(k['A_hand_world_m']);nH=np.array(k['n_hand_world']);ell=A[1]-A[0];ell/=np.linalg.norm(ell)
        results.append({'seed':row['seed'],'rank':row['rank'],'source_case':rec,'common_A_world_m':A.tolist(),
            'line_thumb_to_index_world':ell.tolist(),'line_length_m':float(np.linalg.norm(A[1]-A[0])),
            'thumb':description(nH[0],ell),'index':description(nH[1],-ell),
            'current_native_generated_contacts':c['native_probe']['native_contact_count'],
            'current_native_phone_hand_contacts':c['native_probe']['contact_design']['phone_hand_contact_count'],
            'native7_or_whole_hand_antipodality_evaluated':False})
    B=np.array([r['fixed_CAD_phone_material_body_m'] for r in origin]);ellB=B[1]-B[0];ellB/=np.linalg.norm(ellB)
    material_pair={'line_thumb_to_index_body':ellB.tolist(),
        'thumb':description(-np.array(origin[0]['CAD_phone_outward_normal_body']),ellB),
        'index':description(-np.array(origin[1]['CAD_phone_outward_normal_body']),-ellB)}
    out={'schema':'saved_common_A_antipodal_direction_description:1','rules':pin(HERE/'antipodal_rules_v1.json'),
        'source':pin(__file__),'inputs':rule['inputs'],'outer_circular_mu':.7,'outer_cone_angle_deg':math.degrees(math.atan(.7)),
        'common_A_results_all4':results,'fixed_phone_material_pair_body':material_pair,'material_origin_comparison':origin,
        'scope':'Only selected two common-A points and their hand outward force normals. Outside even the outer circular cone excludes that co-linear compressive direction in a two-point antipodal construction; inside the outer circle is not native4ray qualification. Does not evaluate all generated native contacts, arbitrary hand force closure, actual gravity support or 2s holding. Two torsionless3D point contacts are not a full6D force-closure certificate in any case. Original ego points have no measured contact normal/force/gravity truth here.',
        'new_model_FK_LP_step_NN_GPU_calls':0}
    assert not any(n in sys.modules for n in ('mujoco','torch','warp','mujoco_warp'))
    dest.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'report':pin(dest),'outer_angle':out['outer_cone_angle_deg'],'all4':results,
        'origins':origin,'material_pair':material_pair}))
if __name__=='__main__':main()
