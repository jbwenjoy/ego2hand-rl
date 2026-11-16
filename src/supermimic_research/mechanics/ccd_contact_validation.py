"""Signed distance/normal/witness numerical criteria for isolated CCD queries."""
import numpy as np


def evaluate(contact, mask, distance, witness, geoms, limits):
    count=len(contact['dist']);scope='OVERLAP' if distance < -limits['boundary_m'] else 'SEPARATED' if distance > limits['boundary_m'] else 'BOUNDARY'
    finite=bool(np.isfinite(distance) and np.isfinite(witness).all() and all(np.isfinite(v).all() for v in contact.values()))
    metrics={}
    if count==1:
        frame=contact['frame'][0].astype(np.float64)
        metrics['frame_orthogonality']=float(np.max(np.abs(frame@frame.T-np.eye(3))))
        if abs(distance)>limits['boundary_m']:
            normal=frame[0];reference=(witness[3:]-witness[:3])/distance
            cosine=np.dot(normal,reference)/(np.linalg.norm(normal)*np.linalg.norm(reference))
            metrics.update(distance_error_m=float(abs(float(contact['dist'][0])-distance)),
                normal_angle_deg=float(np.degrees(np.arccos(np.clip(cosine,-1,1)))),
                midpoint_error_m=float(np.linalg.norm(contact['pos'][0].astype(np.float64)-.5*(witness[:3]+witness[3:]))))
    checks=dict(no_overflow=not any(mask),finite=finite,one_contact_max=count<=1,
        overlap_has_contact=scope!='OVERLAP' or count==1,separated_no_contact=scope!='SEPARATED' or count==0,
        correct_contact_identity=count==0 or (contact['worldid'].tolist()==[31] and contact['geom'].tolist()==[geoms]),
        distance=scope!='OVERLAP' or (count==1 and metrics.get('distance_error_m',np.inf)<=limits['distance_error_m']),
        normal=scope!='OVERLAP' or (count==1 and metrics.get('normal_angle_deg',np.inf)<=limits['normal_angle_deg']),
        midpoint=scope!='OVERLAP' or (count==1 and metrics.get('midpoint_error_m',np.inf)<=limits['midpoint_error_m']),
        frame=count==0 or metrics.get('frame_orthogonality',np.inf)<=1e-4)
    return dict(boundary_scope=scope,metrics=metrics,checks=checks,passed=sum(checks.values()),total=len(checks))
