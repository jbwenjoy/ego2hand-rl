"""DRAFT: single-state static mechanics/geometry interface, no optimizer.

Explicit construction/evaluate calls do CPU FK/RNE/Jacobians and checked convex
queries. Import does not load models/data. No mj_forward, native distance,
constraint-force solve, integration, trainer or GPU. Thresholds belong to the
future protocol, passed explicitly to constraint_vector().
"""
from pathlib import Path
import importlib.util,json,hashlib,sys,time
import numpy as np

PILOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PILOT/'geometry_query_v2'))
from convex_query import ConvexPolytope,signed_origin_polytope
from query_interface_v2 import support_features


class UnknownGeometry(RuntimeError):pass
class StaticBudgetExceeded(RuntimeError):pass


def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def _load_source(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class StaticRetargetEvaluator:
    """Own one native model/data, exact byte-cache, no implicit force model."""
    def __init__(self,model,initial_state,anchor_entries,human_tip_targets_object,
                 material_contact_points_object,*,max_heavy_queries=10000,
                 deadline_monotonic=None,max_pair_points=4_000_000):
        import mujoco
        from scipy.spatial.transform import Rotation
        self.mj=mujoco;self.Rotation=Rotation;self.m=model;self.d=mujoco.MjData(model)
        if (model.nq,model.nv,model.nu,model.neq,model.nmocap,model.ntendon,model.nflex)!=(27,26,20,0,0,0,0):raise ValueError('Explicit rigid fixed-wrist20+freephone layout only')
        if any(mujoco.get_mjcb_passive() is not None for _ in [0]):raise ValueError('No opaque passive callback')
        if np.any(model.actuator_trntype!=int(mujoco.mjtTrn.mjTRN_JOINT)) or not np.array_equal(model.actuator_trnid[:,0],np.arange(20)) or not np.array_equal(model.actuator_gear,np.c_[np.ones(20),np.zeros((20,5))]):raise ValueError('Direct unit joint actuators only')
        if np.any(model.actuator_gaintype!=int(mujoco.mjtGain.mjGAIN_FIXED)) or np.any(model.actuator_biastype!=int(mujoco.mjtBias.mjBIAS_AFFINE)) or np.any(model.actuator_dyntype!=int(mujoco.mjtDyn.mjDYN_NONE)):raise ValueError('Direct affine fixed-gain PD only')
        self.initial={k:np.asarray(v).copy() for k,v in initial_state.items()};self.pid=model.body('right_palm_link').id;self.oid=model.body('hot3d_phone24').id;self.phone=model.geom('phone24_collision_convex').id;self.table=model.geom('stage1_single_table').id
        self.Rp=self.initial['T_world_from_palm'][:3,:3].copy();self.pp=self.initial['T_world_from_palm'][:3,3].copy()
        self.qlo=model.jnt_range[:20,0].copy();self.qhi=model.jnt_range[:20,1].copy()
        self.tips=[model.site(f'right_finger{i}_tip').id for i in range(1,6)];self.pads=[model.geom(f'right_finger{i}_tip_sensor_col').id for i in (1,2)]
        self.anchors=np.array([a['anchor_geom_local_m'] for a in anchor_entries]);self.anchor_targets=np.array([a['target_projected_phone_convex_object_m'] for a in anchor_entries]);self.human_tips=np.asarray(human_tip_targets_object,np.float64).copy();self.material=np.asarray(material_contact_points_object,np.float64).copy()
        if self.anchors.shape!=(2,3) or self.human_tips.shape!=(5,3) or self.material.shape!=(2,3) or not np.isfinite(self.material).all() or np.linalg.norm(self.material[1]-self.material[0])<=1e-12:raise ValueError('Finitefixed2anchors/materialpoints and5tip targets')
        self.hand=[g for g in range(model.ngeom) if (model.geom_contype[g] or model.geom_conaffinity[g]) and model.geom_bodyid[g]!=self.oid and g!=self.table]
        self.phone_pairs=self.hand+[self.table];self.local={};self.local_shapes={}
        for g in self.phone_pairs+[self.phone]:
            kind=int(model.geom_type[g])
            if kind==7:
                mid=int(model.geom_dataid[g]);a=int(model.mesh_vertadr[mid]);n=int(model.mesh_vertnum[mid]);self.local[g]=model.mesh_vert[a:a+n].astype(np.float64).copy()
            elif kind==6:
                from itertools import product
                self.local[g]=np.array(list(product(*[(-x,x) for x in model.geom_size[g]])))
            elif kind!=3:raise ValueError('Unsupportedactivecollider')
        geomR=Rotation.from_quat(model.geom_quat[self.phone][[1,2,3,0]]).as_matrix();self.phone_vertices=self.local[self.phone]@geomR.T+model.geom_pos[self.phone];self.poly=ConvexPolytope(self.phone_vertices);self.phone_box=np.array([self.phone_vertices.min(0),self.phone_vertices.max(0)])
        self.max_heavy=int(max_heavy_queries);self.deadline=deadline_monotonic;self.max_pair_points=int(max_pair_points);self.query_cache={};self.token=0;self.last_state=None
        self.counts={k:0 for k in ['evaluations','mj_kinematics','mj_comPos','mj_comVel','mj_rne','mj_passive','mj_transmission','anchor_mj_jac','actual_pair_mj_jac','heavy_queries_attempted','heavy_queries_returned','analytic_capsule_capsule','query_cache_hits','AABB_certificates','unknown_queries','closest_pair_unique','closest_pair_nonunique','closest_pair_nonpositive']}
        mus=[]
        for g in self.pads:
            ids=[i for i in range(model.npair) if {int(model.pair_geom1[i]),int(model.pair_geom2[i])}=={g,self.phone}]
            if len(ids)!=1:raise ValueError('Uniqueexplicitpadpairrequired')
            i=ids[0]
            if int(model.pair_dim[i])!=3 or model.pair_friction[i,0]!=model.pair_friction[i,1]:raise ValueError('Condim3isotropicslidingpairrequired')
            mus.append(float(model.pair_friction[i,0]))
        if mus[0]!=mus[1] or mus[0]<=0:raise ValueError('Samepositivemuonbothpads')
        self.mu_actual=mus[0]
        self.parameter_digest=self._parameter_digest();self.self_pairs,self.filtered_self=self._self_eligibility()
        self.pair_helper=_load_source(PILOT/'hand2_holding_v1/support_preparation_v1/prepare_support_v1.py','frozen_pairing_helper_v3')
        self.segment_helper=_load_source(PILOT/'retarget_geometry_v2/solve_contact_first_v2.py','frozen_segment_helper_v3')

    @classmethod
    def from_existing_pilot(cls,*,max_heavy_queries=10000,deadline_monotonic=None):
        """Explicit loader; no caller authorization implied, no execution on import."""
        import mujoco
        support=json.loads((PILOT/'hand2_holding_v1/support_preparation_v1/actual_v1/report.json').read_text(encoding='utf8'))
        for record in support['input_pins'].values():
            if pin(record['path'])!=record:raise ValueError('Changedsupportinput')
        m=mujoco.MjModel.from_binary_path(support['input_pins']['compiled_model']['path'])
        initial_loader=_load_source(PILOT/'hand2_stage1/scene_loader_v1.py','initial_loader_v3');initial=initial_loader.initial_state()
        prep=json.loads((PILOT/'retarget_geometry_v2/fixed_anchors_and_targets.json').read_text(encoding='utf8'))['preparation'];targets=json.loads((PILOT/'retarget_geometry_v1/frame42_targets.json').read_text(encoding='utf8'))
        operands_path=support['outputs']['support_operands_and_controls.npz'];assert pin(operands_path['path'])==operands_path
        with np.load(operands_path['path'],allow_pickle=False) as z:q=z['qpos27'].copy();points=z['object_contact_points_world'].copy()
        from scipy.spatial.transform import Rotation
        R=Rotation.from_quat(q[23:27][[1,2,3,0]]).as_matrix();material=(points-q[20:23])@R
        return cls(m,initial,prep,[a['reference_tip_object_m'] for a in targets['mapping']],material,max_heavy_queries=max_heavy_queries,deadline_monotonic=deadline_monotonic)

    def set_budget(self,*,deadline_monotonic,max_additional_heavy_queries):
        """Cumulativecounters stay intact; newexplicitphase/seedbudget."""
        self.deadline=deadline_monotonic;self.max_heavy=self.counts['heavy_queries_attempted']+int(max_additional_heavy_queries)

    def _parameter_digest(self):
        names=['jnt_range','actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange','actuator_gear','dof_armature','dof_damping','dof_frictionloss','body_pos','body_quat','body_mass','body_inertia','body_ipos','body_iquat']
        return hashlib.sha256(b''.join(np.ascontiguousarray(getattr(self.m,n)).tobytes() for n in names)).hexdigest()

    def _budget(self):
        if self.deadline is not None and time.perf_counter()>=self.deadline:raise StaticBudgetExceeded('Deadline')
        if self.counts['heavy_queries_attempted']>=self.max_heavy:raise StaticBudgetExceeded('Heavyquerybudget')

    def _self_eligibility(self):
        m=self.m;explicit={tuple(sorted([int(a),int(b)])) for a,b in zip(m.pair_geom1,m.pair_geom2)};excluded=set(map(int,m.exclude_signature));disable=bool(int(m.opt.disableflags)&int(self.mj.mjtDisableBit.mjDSBL_FILTERPARENT));yes=[];no=[]
        for i,g in enumerate(self.hand):
            for h in self.hand[i+1:]:
                b1,b2=int(m.geom_bodyid[g]),int(m.geom_bodyid[h]);w1,w2=int(m.body_weldid[b1]),int(m.body_weldid[b2]);pw1=int(m.body_weldid[m.body_parentid[w1]]);pw2=int(m.body_weldid[m.body_parentid[w2]]);reason=None
                if (g,h) not in explicit:
                    if b1==b2 or w1==w2:reason='samebody/weld'
                    elif not disable and w1 and w2 and (w1==pw2 or w2==pw1):reason='weldparent'
                    elif ((min(b1,b2)<<16)|max(b1,b2)) in excluded:reason='exclude_signature'
                    elif not ((int(m.geom_contype[g])&int(m.geom_conaffinity[h])) or (int(m.geom_contype[h])&int(m.geom_conaffinity[g]))):reason='geom_mask'
                (yes if reason is None else no).append({'geoms':(g,h),'names':(m.geom(g).name,m.geom(h).name),'filter_reason':reason})
        return yes,no

    @staticmethod
    def _bounds(s):
        if s['kind']=='capsule':return np.array([np.minimum(s['a'],s['b'])-s['r'],np.maximum(s['a'],s['b'])+s['r']])
        return np.array([s['v'].min(0),s['v'].max(0)])

    @staticmethod
    def _aabb_lower(a,b):return float(np.linalg.norm(np.maximum(np.maximum(a[0]-b[1],b[0]-a[1]),0)))

    def _shape(self,g,R,p):
        d,m=self.d,self.m;gr=d.geom_xmat[g].reshape(3,3);gp=d.geom_xpos[g]
        if int(m.geom_type[g])==3:return {'kind':'capsule','a':(gp-m.geom_size[g,1]*gr[:,2]-p)@R,'b':(gp+m.geom_size[g,1]*gr[:,2]-p)@R,'r':float(m.geom_size[g,0])}
        return {'kind':'convex','v':(self.local[g]@gr.T+gp-p)@R}

    def _shape_world(self,g):
        d,m=self.d,self.m;gr=d.geom_xmat[g].reshape(3,3);gp=d.geom_xpos[g]
        if int(m.geom_type[g])==3:return {'kind':'capsule','a':gp-m.geom_size[g,1]*gr[:,2],'b':gp+m.geom_size[g,1]*gr[:,2],'r':float(m.geom_size[g,0])}
        return {'kind':'convex','v':self.local[g]@gr.T+gp}

    @staticmethod
    def _shape_key(s):return (s['kind'],(np.r_[s['a'],s['b'],s['r']] if s['kind']=='capsule' else s['v']).tobytes())

    def _exact_pair(self,a,b,*,phone=False):
        self._budget();key=(self._shape_key(a),self._shape_key(b),phone)
        if key in self.query_cache:self.counts['query_cache_hits']+=1;return self.query_cache[key]
        if a['kind']==b['kind']=='capsule':
            gap=self.segment_helper.segment_distance(a['a'],a['b'],b['a'],b['b'],np)[0]-a['r']-b['r'];guard=256*np.finfo(float).eps;self.counts['analytic_capsule_capsule']+=1;q={'signed_translation_margin_m':gap,'roundoff_guard_m':guard,'margin_roundoff_guarded_interval_m':[gap-guard,gap+guard],'query_kind':'analytic_capsule_capsule'}
        else:
            self.counts['heavy_queries_attempted']+=1
            try:
                if a['kind']=='convex':
                    poly=self.poly if phone else ConvexPolytope(a['v']);q=poly.capsule(b['a'],b['b'],b['r']) if b['kind']=='capsule' else poly.convex(b['v'],max_pair_points=self.max_pair_points)
                    if phone:q.update(support_features(q,self.phone_vertices,a=b['a'],b=b['b'],radius=b['r']) if b['kind']=='capsule' else support_features(q,self.phone_vertices,hand_vertices=b['v']))
                else:q=ConvexPolytope(b['v']).capsule(a['a'],a['b'],a['r'])
                self.counts['heavy_queries_returned']+=1
            except Exception as e:self.counts['unknown_queries']+=1;raise UnknownGeometry(f'{type(e).__name__}: {e}')
        self.query_cache[key]=q;return q

    def _describe_pair(self,a,b,*,phone=False,force_exact=False):
        sep=self._aabb_lower(self._bounds(a),self._bounds(b));guard=256*np.finfo(float).eps*max(1.,float(abs(self._bounds(a)).max()),float(abs(self._bounds(b)).max()))
        if not force_exact and sep>.0005:
            self.counts['AABB_certificates']+=1;return {'status':'DISJOINT_AABB_CERTIFICATE','lower_bound_m':sep-guard,'exact_gap_m':None,'guard_m':guard,'not_exact_gap':True}
        q=self._exact_pair(a,b,phone=phone)
        return {'status':'EXACT_NUMERICAL_IDEAL_CONVEX','lower_bound_m':float(q['margin_roundoff_guarded_interval_m'][0]),'upper_bound_m':float(q['margin_roundoff_guarded_interval_m'][1]),'exact_gap_m':float(q['signed_translation_margin_m']),'guard_m':float(q['roundoff_guard_m']),'query_phone_frame':q}

    @staticmethod
    def _world_features(q,R,p):
        out={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in q.items()};out['query_input_frame']='phone_body';out['reported_support_frame']='world'
        for key in ['normal_world','translation_normal_world','closest_unrounded_boundary_world']:
            if key in q:out[key]=R@q[key]
        for key in ['object_support_points_world_m','hand_support_points_world_m']:
            if key in q:out[key]=q[key]@R.T+p
        if 'normal_world'in q:
            offset=float((R@q['normal_world'])@p)
            for key in ['object_plane_max_projection_m','hand_plane_min_projection_m']:
                if key in q:out[key]=float(q[key])+offset
        return out

    def _unique_pair(self,feature):
        gap=feature['signed_gap_m'];Ps=feature['object_support_points_world_m'];Hs=feature['hand_support_points_world_m'];n=feature['normal_world']
        if gap<=0:self.counts['closest_pair_nonpositive']+=1;return {'status':'NONPOSITIVE_GAP_CONTACT_MODEL_UNAVAILABLE','gap_m':float(gap),'closest_pair':None}
        axis=np.eye(3)[np.argmin(abs(n))];t1=np.cross(n,axis);t1/=np.linalg.norm(t1);T=np.stack([t1,np.cross(n,t1)]);center=Ps[0];scale=max(.01,float(abs(np.r_[Ps-center,Hs-center]).max()));A=np.zeros((4,len(Ps)+len(Hs)));A[0,:len(Ps)]=1;A[1,len(Ps):]=1;A[2:,:len(Ps)]=-T@(Ps-center).T/scale;A[2:,len(Ps):]=T@(Hs-center).T/scale;rank=int(np.linalg.matrix_rank(A))
        if rank<len(Ps)+len(Hs):self.counts['closest_pair_nonunique']+=1;return {'status':'NON_UNIQUE_SUPPORT_PAIR_NO_CENTROID_OR_REPRESENTATIVE','rank':rank,'unknown_count':len(Ps)+len(Hs),'closest_pair':None}
        try:pair=self.pair_helper.closest_support_pair(Ps,Hs,n,gap)
        except (ValueError,ArithmeticError) as e:return {'status':'PAIR_RECONSTRUCTION_UNKNOWN','error':str(e),'closest_pair':None}
        self.counts['closest_pair_unique']+=1;return {'status':'UNIQUE_CLOSEST_PAIR','rank':rank,'unknown_count':len(Ps)+len(Hs),'closest_pair':pair}

    def _jacobians(self,points,bodies,kind):
        result=[]
        for point,body in zip(points,bodies):
            jp=np.zeros((3,26));self.mj.mj_jac(self.m,self.d,jp,None,np.asarray(point,np.float64),int(body));self.counts[kind]+=1
            if np.any(jp[:,20:]!=0):raise ValueError('Handpointdepends onphone DOFs')
            result.append(jp)
        return np.asarray(result)

    def evaluate(self,q20,phone_translation_palm_m,phone_rotation_vector_rad,*,exact_all=False):
        """Return full current-state operands; missingpair is explicit, neverzero."""
        self._budget();q=np.asarray(q20,np.float64);t=np.asarray(phone_translation_palm_m,np.float64);rv=np.asarray(phone_rotation_vector_rad,np.float64)
        if q.shape!=(20,) or t.shape!=(3,) or rv.shape!=(3,) or not np.isfinite(np.r_[q,t,rv]).all():raise ValueError('Finite20/3/3state')
        if np.any(q<self.qlo) or np.any(q>self.qhi):raise ValueError('Truehardjointlimits')
        R=self.Rp@self.Rotation.from_rotvec(rv).as_matrix();p=self.pp+self.Rp@t;d,m=self.d,self.m;d.qpos[:]=self.initial['qpos'];d.qpos[:20]=q;d.qpos[20:23]=p;quat=self.Rotation.from_matrix(R).as_quat();d.qpos[23:27]=quat[[3,0,1,2]];d.qvel[:]=0;d.ctrl[:]=0;d.time=0
        for name in ['mj_kinematics','mj_comPos','mj_comVel']:
            getattr(self.mj,name)(m,d);self.counts[name]+=1
        if not np.array_equal(d.xpos[self.pid],self.pp) or not np.array_equal(d.xmat[self.pid].reshape(3,3),self.Rp):raise ValueError('Fixedwristchanged')
        bias=np.empty(26);self.mj.mj_rne(m,d,0,bias);self.counts['mj_rne']+=1
        self.mj.mj_passive(m,d);self.counts['mj_passive']+=1;self.mj.mj_transmission(m,d);self.counts['mj_transmission']+=1
        if not np.isfinite(bias).all() or not np.isfinite(d.qfrc_passive).all() or not np.allclose(d.actuator_length,q,rtol=0,atol=1e-14):raise ValueError('Staticmechanicsinvalid')
        self.token+=1;self.counts['evaluations']+=1;anchors=np.array([self.anchors[i]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g] for i,g in enumerate(self.pads)]);bodies=[int(m.geom_bodyid[g]) for g in self.pads];anchorJ=self._jacobians(anchors,bodies,'anchor_mj_jac')
        shapes={g:self._shape(g,R,p) for g in self.phone_pairs};phone_shape={'kind':'convex','v':self.phone_vertices};phone_rows=[];self_rows=[];unknown=[];contacts=[]
        for g in self.phone_pairs:
            try:row=self._describe_pair(phone_shape,shapes[g],phone=True,force_exact=exact_all or g in self.pads)
            except UnknownGeometry as e:row={'status':'UNKNOWN','error':str(e)};unknown.append({'pair':'phone','geom':g,'error':str(e)})
            phone_rows.append({'geom_id':g,'name':m.geom(g).name,'is_pad':g in self.pads,**row})
            if g in self.pads and row['status']!='UNKNOWN':
                feat=self._world_features(row['query_phone_frame'],R,p);paired=self._unique_pair(feat);actualJ=None
                if paired['closest_pair'] is not None:actualJ=self._jacobians([paired['closest_pair']['hand_point_world_m']],[int(m.geom_bodyid[g])],'actual_pair_mj_jac')[0]
                contacts.append({'geom_id':g,'body_id':int(m.geom_bodyid[g]),'features':feat,'pairing':paired,'actual_hand_point_J3x26':actualJ,'force_on_object_normal_world':-feat['normal_world']})
        self_world_shapes={g:self._shape_world(g) for g in self.hand}
        for pair in self.self_pairs:
            g,h=pair['geoms']
            try:row=self._describe_pair(self_world_shapes[g],self_world_shapes[h],force_exact=exact_all)
            except UnknownGeometry as e:row={'status':'UNKNOWN','error':str(e)};unknown.append({'pair':'self','geoms':(g,h),'error':str(e)})
            self_rows.append({**pair,**row})
        gain=m.actuator_gainprm[:,0].copy();offset=m.actuator_biasprm[:,0]+m.actuator_biasprm[:,1]*d.actuator_length+m.actuator_biasprm[:,2]*d.actuator_velocity;lower=np.maximum.reduce([gain*m.actuator_ctrlrange[:,0]+offset,m.actuator_forcerange[:,0],m.jnt_actfrcrange[:20,0]]);upper=np.minimum.reduce([gain*m.actuator_ctrlrange[:,1]+offset,m.actuator_forcerange[:,1],m.jnt_actfrcrange[:20,1]])
        COM=d.xipos[self.oid].copy();material_world=self.material@R.T+p;axis=material_world[1]-material_world[0];axis/=np.linalg.norm(axis);line_torque=float(axis@np.cross(COM-material_world[0],float(m.body_mass[self.oid])*m.opt.gravity))
        unique=len(contacts)==2 and all(c['pairing']['status']=='UNIQUE_CLOSEST_PAIR' for c in contacts)
        support={'available':unique,'reason':'Bothcurrentpairsunique' if unique else 'Nonpositive/nonunique/missingpair; no inventedzero support','object_COM':COM,'mass':float(m.body_mass[self.oid]),'mu_actual':self.mu_actual,'gravity_world':m.opt.gravity.copy(),'robot_gravity_bias_tau':bias[:20].copy(),'torque_lower':lower,'torque_upper':upper,'normal_cap_source':'Futureprotocol, notselectedhere'}
        if unique:support.update(contact_points_world=np.array([c['pairing']['closest_pair']['object_point_world_m'] for c in contacts]),hand_points_world=np.array([c['pairing']['closest_pair']['hand_point_world_m'] for c in contacts]),normals_force_on_object=np.array([c['force_on_object_normal_world'] for c in contacts]),J_robot_contact_3x20=np.array([c['actual_hand_point_J3x26'][:,:20] for c in contacts]))
        out={'state_token':self.token,'q20_rad':q.copy(),'qpos27':d.qpos.copy(),'phone_translation_palm_m':t.copy(),'phone_rotvec_rad':rv.copy(),'phone_R_world':R,'phone_p_world_m':p,'anchors_world_m':anchors,'anchor_targets_world_m':self.anchor_targets@R.T+p,'tips_world_m':d.site_xpos[self.tips].copy(),'human_tip_targets_world_m':self.human_tips@R.T+p,'anchor_J2x3x26':anchorJ,'qfrc_bias26_Nm':bias,'qfrc_passive26_Nm':d.qfrc_passive.copy(),'actuator_length20':d.actuator_length.copy(),'actuator_velocity20':d.actuator_velocity.copy(),'PD_gain20':gain,'PD_offset20':offset,'reachable_torque_lower20_Nm':lower,'reachable_torque_upper20_Nm':upper,'phone_geometry_rows':phone_rows,'self_geometry_rows':self_rows,'contacts':contacts,'unknown':unknown,'support_operands':support,'fixed_material_points_world_m':material_world,'fixed_material_line_axis_world':axis,'fixed_material_line_gravity_torque_Nm':line_torque,'fixed_material_line_scope':'Smooth necessary condition using oldobjectmaterialpoints, not claimedactualcontactline atnewstate.','frictionloss_help_used':False,'counts':dict(self.counts)}
        self.last_state=out
        if self._parameter_digest()!=self.parameter_digest:raise ValueError('Modelparametersmutated')
        return out

    @staticmethod
    def support_lp_kwargs(state):
        support=state['support_operands']
        if not support['available']:raise UnknownGeometry(support['reason'])
        keys=['contact_points_world','normals_force_on_object','object_COM','mass','mu_actual','J_robot_contact_3x20','robot_gravity_bias_tau','torque_lower','torque_upper','gravity_world']
        return {key:support[key] for key in keys}

    @staticmethod
    def constraint_vector(state,*,pad_gap_window_m,nonpad_buffer_m,self_buffer_m):
        """Raw meters; caller explicitly selects search/finalwindows inprotocol."""
        if state['unknown']:raise UnknownGeometry('UnknowncannotbeaconstraintPASS')
        lo,hi=map(float,pad_gap_window_m);values=[];labels=[]
        for row in state['phone_geometry_rows']:
            if row['is_pad']:values.extend([row['lower_bound_m']-lo,hi-row['upper_bound_m']]);labels.extend([row['name']+'_lower',row['name']+'_upper'])
            else:values.append(row['lower_bound_m']-float(nonpad_buffer_m));labels.append(row['name']+'_nonpenetration')
        for row in state['self_geometry_rows']:values.append(row['lower_bound_m']-float(self_buffer_m));labels.append('/'.join(row['names'])+'_self_nonpenetration')
        return {'values_m':np.asarray(values),'labels':labels,'all_satisfied':bool(np.all(np.asarray(values)>=0)),'unknown':False}


def declaration():
    return {'status':'DRAFT_SINGLE_STATE_INTERFACE_NO_OPTIMIZER_OR_RUN_AUTHORIZATION','entry':'StaticRetargetEvaluator.from_existing_pilot();evaluate(q20,t_palm,rotvec)','model_namespace':'ExactownfixedHand2+phone, notcheckpointresume','no_import_model_data_read':True,'no_native_distance_integration_training_GPU':True,'all_self_constraints_each_evaluation':True,'actualclosest_pair_policy':'Positivegapandfullcolumnrankrequired; nonunique/nonpositive explicitlyunavailable. No centroid/previousposecontacts.'}
