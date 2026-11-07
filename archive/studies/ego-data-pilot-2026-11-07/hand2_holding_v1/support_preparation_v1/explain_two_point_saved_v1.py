"""Saved operands only: necessary two-point wrench obstruction; no LP/FK."""
import hashlib,json,math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
def pin(p):
    p=p.resolve(strict=True);b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def cross(a,b):
    return [math.fsum([a[1]*b[2],-a[2]*b[1]]),math.fsum([a[2]*b[0],-a[0]*b[2]]),math.fsum([a[0]*b[1],-a[1]*b[0]])]
def dot(a,b):return math.fsum(x*y for x,y in zip(a,b))

raw=HERE/'actual_v1/support_operands_and_controls.npz'
lpfile=HERE/'actual_v1/LP_result.json';lp=json.loads(lpfile.read_text())
with np.load(raw,allow_pickle=False) as a:
    p1,p2=a['object_contact_points_world'].tolist();com=a['object_COM_world'].tolist()
line=[b-a for a,b in zip(p1,p2)];length=math.sqrt(dot(line,line));line=[v/length for v in line]
gravity=[float(lp['mass_kg'])*v for v in lp['gravity_world_m_s2']]
tau=cross([b-a for a,b in zip(p1,com)],gravity);projection=dot(line,tau)
allowed=math.fsum(abs(v) for v in line)*lp['moment_residual_tolerance_Nm']
out={'schema':'fixed_two_point_support_necessary_wrench_obstruction:1',
     'source':pin(Path(__file__)),'inputs':{'operands':pin(raw),'LP_result':pin(lpfile)},
     'object_contact_points_world_m':[p1,p2],'object_COM_world_m':com,
     'line_unit_p1_to_p2_world':line,'contact_separation_m':length,
     'gravity_force_world_N':gravity,'gravity_moment_about_p1_world_Nm':tau,
     'gravity_moment_projected_on_contact_line_Nm':projection,
     'absolute_unsupported_line_torque_Nm':abs(projection),
     'existing_LP_component_moment_tolerance_Nm':lp['moment_residual_tolerance_Nm'],
     'maximum_projected_residual_from_existing_component_tolerance_Nm':allowed,
     'exceeds_original_LP_component_tolerance_projection':abs(projection)>allowed,
     'identity':'For arbitrary f1,f2 at p1,p2, line dot ((p1-p1) cross f1 + (p2-p1) cross f2)=0 exactly in the fixed point model.',
     'conclusion':'The saved fixed two-point model cannot balance the nonzero gravity moment along its contact line, even with unlimited friction or actuator torque. This is not attributed to the conservative design cone or torque bounds.',
     'limits':'Saved F64 nominal geometry and assumed mass/gravity. Finite mesh patches, torsion, different contacts, controlled closure or changed object/hand pose are outside this impossibility statement. No real-object or holding impossibility claim.',
     'new_LP_or_FK_or_physics_calls':0}
dest=HERE/'support_explanation_v1.json'
with dest.open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'projection_Nm':projection,'report':pin(dest)}))
