"""Independent saved-contact semantics only; no native/model/backend imports.

Generated candidates are classified before force qualification. No query,
positive-gap, contact persistence, new wrench, label, or causal claim is made.
The source-captured world HO wrench/solver pose/COM may be carried unchanged by
an adapter; this module never recomputes them. All contact indices are original.
"""
import re
import numpy as np

HAND_RULE=r'robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)'
FN_THRESHOLD_N=1e-6


def integer(v,label,minimum=None):
    if isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)):
        raise ValueError('Integer '+label+' required, not bool/float')
    v=int(v)
    if minimum is not None and v<minimum:raise ValueError(label+' range')
    return v


def finite(a,shape,label):
    a=np.asarray(a)
    if a.shape!=shape or a.dtype.kind not in 'fiu' or not np.isfinite(a).all():
        raise ValueError('Finite '+label+' '+str(shape)+' required')
    return a


def catalog_sets(native_ngeom,native_object_body_id,catalog):
    n=integer(native_ngeom,'native_ngeom',1);obj=integer(native_object_body_id,'object body',0)
    if len(catalog)!=n:raise ValueError('Complete native catalog required')
    table={};hand=set();objects=set()
    for row in catalog:
        gid=integer(row['id'],'geom ID',0);bid=integer(row['native_body_id'],'native body ID',0)
        ct=integer(row['contype'],'contype',0);ca=integer(row['conaffinity'],'conaffinity',0)
        if gid>=n or gid in table or type(row['name']) is not str:raise ValueError('Geom ID/name catalog')
        table[gid]=row
        if re.fullmatch(HAND_RULE,row['name']) and (ct or ca):hand.add(gid)
        if bid==obj:objects.add(gid)
    if set(table)!=set(range(n)) or not hand or not objects or hand&objects:raise ValueError('Native hand/object catalog binding')
    return {'catalog':table,'hand':frozenset(hand),'object':frozenset(objects),'native_ngeom':n}


def pair_classification(geom,sets):
    pair=np.asarray(geom)
    if pair.shape!=(2,) or pair.dtype.kind not in 'iu':raise ValueError('Ordered native geom pair')
    a,b=map(int,pair)
    if a not in sets['catalog'] or b not in sets['catalog']:raise ValueError('Unknown native geom ID; do not use FK-local/mocap')
    h,o=sets['hand'],sets['object'];hashand=a in h or b in h;hasobject=a in o or b in o
    category='hand_object' if hashand and hasobject else 'hand_self' if a in h and b in h else 'other'
    # Exactly the frozen helper's object_environment predicate, named narrowly.
    return {'ordered_geom_pair':[a,b],'category':category,'object_nonhand':hasobject and not hashand,
       'hand_other':hashand and not hasobject and not(a in h and b in h),
       'both_object':a in o and b in o,'native_geom_names':[sets['catalog'][a]['name'],sets['catalog'][b]['name']],
       'native_body_ids':[sets['catalog'][a]['native_body_id'],sets['catalog'][b]['native_body_id']]}


def inspect_contact(original_index,contact,raw_wp,nefc,sets):
    """One saved row. Raw WP addresses, not reordered host addresses, qualify force.

    force6 is original decoded contact-coordinate [N,N,N,Nm,Nm,Nm]. No independent
    pyramid decode: per-contact dim is not completely serialized in this ABI.
    Sensor-only and invalid-address records remain generated candidates.
    """
    ci=integer(original_index,'original contact index',0);nefc=integer(nefc,'nefc',0)
    cat=pair_classification(contact['geom'],sets)
    rawpair=pair_classification(raw_wp['geom'],sets)
    if cat['ordered_geom_pair']!=rawpair['ordered_geom_pair']:raise ValueError('Contact index/native pair order differs')
    typ=integer(raw_wp['type'],'WP ContactType',0);world=integer(raw_wp['worldid'],'worldid',0)
    hostadr=integer(contact['efc_address'],'host efc address')
    adr=np.asarray(raw_wp['efc_address'])
    if adr.ndim!=1 or not len(adr) or adr.dtype.kind not in 'iu':raise ValueError('Raw WP efc slot vector')
    f=finite(contact['force6'],(6,),'original decoded force6').astype('f8')
    finite(contact['pos'],(3,),'world contact position');finite(contact['frame'],(3,3),'contact-frame rows')
    d=float(contact['dist'])
    if not np.isfinite(d):raise ValueError('Finite generated distance')
    valid=adr>=0;legal=bool(((adr==-1)|((adr>=0)&(adr<nefc))).all())
    has_constraint=bool(typ&1);has_sensor=bool(typ&2)
    # No force/amplitude gate is used for geometric classification.
    eligible=bool(has_constraint and world==0 and valid.any() and legal and hostadr>=0)
    reasons=[]
    if not has_constraint:reasons.append('NO_CONSTRAINT_TYPE_BIT')
    if world!=0:reasons.append('NONZERO_WORLD')
    if not valid.any():reasons.append('NO_ACTIVE_RAW_EFC_SLOT')
    if not legal:reasons.append('RAW_EFC_SLOT_OUT_OF_RANGE')
    if hostadr<0:reasons.append('NEGATIVE_HOST_EFC_ADDRESS')
    return {**cat,'original_contact_index':ci,'generated_candidate':True,'dist_m':d,'observed_candidate_depth_m':max(0.,-d),
       'raw_WP_type':typ,'constraint_type_bit':has_constraint,'sensor_type_bit':has_sensor,'raw_worldid':world,
       'raw_efc_slots':adr.tolist(),'raw_efc_slot_legal_mask':((adr==-1)|((adr>=0)&(adr<nefc))).tolist(),
       'raw_active_slot_mask':valid.tolist(),'host_efc_address_reordered':hostadr,
       'decoded_normal_force_N':float(f[0]),'decoded_force6_contact_coordinates':f.tolist(),
       'force_evidence_eligible':eligible,'secondary_positive_force':bool(eligible and f[0]>FN_THRESHOLD_N),
       'force_ineligible_reasons':reasons,'independent_pyramid_decode_qualified':False,
       'solver_phase':'just_completed_solver_preintegration; separate from postintegration q/qvel/time',
       'global_clearance':'UNKNOWN'}


def generated_summary(rows):
    """Empty candidates have observed depth0, not a clearance certificate."""
    summary={}
    selectors={'all':lambda r:True,'hand_object':lambda r:r['category']=='hand_object',
       'hand_self':lambda r:r['category']=='hand_self','other':lambda r:r['category']=='other',
       'object_nonhand':lambda r:r['object_nonhand'],'hand_other':lambda r:r['hand_other']}
    for name,select in selectors.items():
        selected=[r for r in rows if select(r)];d=[r['dist_m'] for r in selected]
        summary[name]={'generated_candidate_count':len(d),'minimum_generated_distance_m':min(d) if d else None,
           'observed_candidate_depth_m':max([0.]+[-x for x in d]),'generated_candidate_absent':not bool(d),
           'secondary_positive_force_count':sum(r['secondary_positive_force'] for r in selected),
           'global_clearance':'UNKNOWN'}
    return summary


def zero_descriptors(minus_u,plus_u,minus_target,plus_target,minus_prediction,plus_prediction,joint_index):
    """Executed-control delta and endpoint encodings only; not force/causal labels."""
    q=integer(joint_index,'declared joint',0)
    if q>=20:raise ValueError('Joint coordinate20')
    arrays=[]
    for value,shape,label in [(minus_u,(20,),'minus u'),(plus_u,(20,),'plus u'),(minus_target,(58,),'minus target'),
        (plus_target,(58,),'plus target'),(minus_prediction,(58,),'minus prediction'),(plus_prediction,(58,),'plus prediction')]:
        a=finite(value,shape,label)
        if a.dtype!=np.float32:raise ValueError('Original FP32 '+label+' required')
        arrays.append(a)
    um,up,tm,tp,pm,pp=arrays;du=up.astype('f8')-um.astype('f8')
    return {'actual_delta_u20_rad_F64':du,'actual_coordinate_delta_rad':float(du[q]),'zero_actual_coordinate_delta':bool(du[q]==0),
       'zero_actual_all20_delta':bool((du==0).all()),'control_endpoint_bits_equal20':um.view('u4')==up.view('u4'),
       'control_endpoints_all_bits_equal':bool(np.array_equal(um.view('u4'),up.view('u4'))),
       'both_absolute_controls_all_numeric_zero':bool((um==0).all() and (up==0).all()),
       'physical_encoded_numeric_zero58':tp.astype('f8')-tm.astype('f8')==0,
       'physical_endpoint_bits_equal58':tp.view('u4')==tm.view('u4'),
       'model_encoded_numeric_zero58':pp.astype('f8')-pm.astype('f8')==0,
       'model_endpoint_bits_equal58':pp.view('u4')==pm.view('u4')}
