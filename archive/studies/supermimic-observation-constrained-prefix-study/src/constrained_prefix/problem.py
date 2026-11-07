"""Unchanged prefix16 fitting plus sampled generated-overlap constraints.

The 61x2 operational inequalities bound generated native overlap. Missing
contacts remain UNKNOWN gap; no global clearance or dynamic guarantee follows.
No optimizer or LP is called in this module.
"""
from contextlib import contextmanager
from pathlib import Path
import importlib.util
import sys
import time
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation

PARENT_ROOT=Path('D:/workspace/supermimic-observation-prefix-study')
sys.path.insert(0,str(PARENT_ROOT/'src'))
from prefix_study.problem import Prefix16GeometryProblem
from prefix_study.inputs import checked_protocol

PLANNING_DEPTH_BOUND_M=.002999
CATEGORY_ORDER=('hand_object','hand_self')
OBJECT_CURVE_SOURCE=PARENT_ROOT/'src/prefix_curves.py'


def _load_curves():
    name='constrained_prefix_frozen_object_curves'
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,OBJECT_CURVE_SOURCE)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module


class ConstrainedPrefixGeometryProblem:
    """340 B-spline variables; source64 P0/q0 fixed and never projected."""
    def __init__(self,basis,*,parent_protocol=None,event_sink=None,deadline=None):
        self.stage_name='construction';self.event_sink=event_sink;self.deadline=deadline
        self.events=[];self.stage_counts={};self.fit_requests=0;self.constraint_requests=0
        self.constraint_full_evaluations=0;self.constraint_cache_hits=0;self.sample_cache_hits=0
        self.parent=Prefix16GeometryProblem(checked_protocol() if parent_protocol is None else parent_protocol,event_sink=self._event,deadline=deadline)
        self.basis=basis;self.q0=self.parent.q0.copy();self.x0=np.tile(self.q0,17)
        self.fit_time=np.arange(16)*.02;self.geometry_time=np.arange(61)*.005
        self.B16=np.asarray(basis.evaluate(self.fit_time),dtype=float)
        self.B61=np.asarray(basis.evaluate(self.geometry_time),dtype=float)
        for B,n in ((self.B16,16),(self.B61,61)):
            if B.shape!=(n,18) or not np.isfinite(B).all() or np.max(np.abs(B.sum(axis=1)-1))>1e-12:raise ValueError('Expected finite partition-of-unity cubic basis')
            if not np.array_equal(B[0],np.r_[1.,np.zeros(17)]):raise ValueError('P0 must be exact clamped q0')
        self.fit_chain=sparse.kron(sparse.csr_matrix(self.B16[1:,1:]),sparse.eye(20),format='csr')
        self.curves=_load_curves()
        pin=self.parent.protocol['inputs']['rawFD_design_arrays']
        fields=('time_s','object_pose_mapped_w','raw_v_world_m_s','raw_a_world_m_s2','raw_omega_world_rad_s','raw_alpha_world_rad_s2')
        with np.load(pin['path'],allow_pickle=False) as z:self.object_raw={k:np.asarray(z[k][:16],dtype=float).copy() for k in fields}
        raw=self.object_raw
        if not np.array_equal(raw['time_s'],self.fit_time) or not np.array_equal(raw['object_pose_mapped_w'],self.parent.pose):raise ValueError('Frozen object knots/rawFD identity differs')
        object_curve=self.curves.build_prefix_curves(self.fit_time,np.tile(self.q0,(16,1)),self.parent.pose,raw['raw_v_world_m_s'],raw['raw_a_world_m_s2'],raw['raw_omega_world_rad_s'],raw['raw_alpha_world_rad_s2'])
        self.object_samples=object_curve.sample(self.geometry_time)
        R=self.object_samples['object_rotation_w'];xyzw=Rotation.from_matrix(R).as_quat()
        quat=xyzw[:,[3,0,1,2]]
        for i in range(1,61):
            if quat[i]@quat[i-1]<0:quat[i]*=-1
        self.pose61=np.c_[self.object_samples['object_COM_world_m'],quat]
        self.object_knot_roundoff={'position_max_m':float(np.max(np.abs(self.pose61[::4,:3]-self.parent.pose[:,:3]))),'rotation_matrix_max':float(np.max(np.abs(R[::4]-self.curves.quaternion_matrices(self.parent.pose[:,3:]))))}
        if self.object_knot_roundoff['position_max_m']>1e-10 or self.object_knot_roundoff['rotation_matrix_max']>1e-10:raise ValueError('Object quintic knot mismatch')
        self._sample_cache=[None]*61;self._constraint_x=None;self._constraint_cache=None
        self.first_fit_x=None;self.first_constraint_x=None

    def _event(self,event):
        event=dict(event);event['owner']='fit16';event['stage']=self.stage_name+'/'+event['stage']
        self.events.append(event);self.stage_counts[event['stage']]=self.stage_counts.get(event['stage'],0)+1
        if self.event_sink:self.event_sink(event)

    @contextmanager
    def stage(self,name):
        old=self.stage_name;self.stage_name=name
        with self.parent.stage(name):
            try:yield
            finally:self.stage_name=old

    def controls_from_x(self,x):
        x=np.asarray(x,dtype=float)
        if x.shape!=(340,) or not np.isfinite(x).all():raise ValueError('Expected finite control-major 340 variables')
        return np.vstack((self.q0,x.reshape(17,20)))

    def sample_q(self,x,B):
        P=self.controls_from_x(x)
        # Affine grouping preserves an exactly repeated source64 q0. Equivalent
        # to B@P in real arithmetic; never clips q0 or any variable.
        Q=self.q0+B[:,1:]@(P[1:]-self.q0)
        if not np.array_equal(Q[0],self.q0):raise ValueError('Hard q0 changed')
        return Q

    def q_from_x(self,x):return self.sample_q(x,self.B16)

    def fit(self,x):
        self.fit_requests+=1
        if self.deadline:self.deadline()
        x=np.asarray(x,dtype=float);Q=self.q_from_x(x)
        if self.first_fit_x is None:self.first_fit_x=x.copy()
        r,J,local=self.parent.evaluate(Q[1:].ravel())
        return r,(J@self.fit_chain).tocsr(),local

    def _sample(self,index,q):
        if self.deadline:self.deadline()
        cached=self._sample_cache[index]
        if cached is not None and np.array_equal(cached['q'],q):self.sample_cache_hits+=1;return cached
        g=self.parent.g;g.set_object_pose(self.pose61[index]);before=g.collision_query_count;e=g.evaluate(q)
        increment=g.collision_query_count-before
        if increment not in (0,1):raise ValueError('Unexpected native collision count increment')
        if increment:
            event={'query_index':g.collision_query_count,'owner':'constraint61','stage':self.stage_name,'sample':index,'time_s':float(self.geometry_time[index]),'monotonic_s':time.perf_counter(),'scope':'CPU mj_collision geometry only, no dynamics'}
            self.events.append(event);self.stage_counts[self.stage_name]=self.stage_counts.get(self.stage_name,0)+1
            if self.event_sink:self.event_sink(event)
        depth=np.zeros(2);jq=np.zeros((2,20));selected=np.full(2,-1,dtype=int);ties=[];contact_ties=[];generated=np.zeros(2,dtype=bool)
        for category,kind in enumerate(CATEGORY_ORDER):
            rows=np.flatnonzero(e.generated_mask&(g.pair_kind==kind))
            if not len(rows):ties.append([]);contact_ties.append([]);continue
            generated[category]=True
            distances=e.signed_distance_m[rows];row=int(rows[int(np.argmin(distances))]);selected[category]=row
            tied=rows[np.abs(distances-e.signed_distance_m[row])<=1e-9]
            ties.append(tied.tolist())
            contact_ties.append([{'pair_row':int(c[0]),'native_contact_index':int(c[1]),'distance_m':float(c[4])} for c in e.native_contact_records if g.pair_kind[c[0]]==kind and abs(c[4]-e.signed_distance_m[row])<=1e-9])
            if e.signed_distance_m[row]<0:depth[category]=-e.signed_distance_m[row];jq[category]=-e.selected_distance_jacobian[row]
        native=[]
        for contact_index,c in enumerate(g.data.contact):
            a,b=map(int,c.geom);obj=a in g.object_ids or b in g.object_ids;ha=a in g.hand_ids;hb=b in g.hand_ids
            kind='hand_object' if obj and (ha or hb) else 'hand_self' if ha and hb else 'object_environment' if obj else 'other'
            native.append({'sample':index,'time_s':float(self.geometry_time[index]),'contact_index':contact_index,'category':kind,'geom':[a,b],'names':[g.model.geom(a).name,g.model.geom(b).name],'distance_m':float(c.dist),'position_w':c.pos.tolist(),'frame_rows_geom0_to_geom1_w':c.frame.reshape(3,3).tolist(),'effective_condim':int(c.dim),'friction5':c.friction.tolist(),'phase':'native_CPU_geometry_not_solved_force'})
        result={'q':q.copy(),'evaluation':e,'operational_depth_m':depth,'depth_jac_q':jq,'selected_pair_row':selected,'category_generated':generated,'category_tied_pair_rows':ties,'category_global_tied_contacts':contact_ties,'native_records':native}
        self._sample_cache[index]=result;return result

    def nonlinear(self,x):
        """Return depth/1mm and Jacobian, upper bound 2.999 in SLSQP.

        A category without generated contacts has operational depth zero and a
        constant zero Jacobian. Its unknown physical gap is NOT assigned zero.
        """
        self.constraint_requests+=1
        if self.deadline:self.deadline()
        x=np.asarray(x,dtype=float);Q=self.sample_q(x,self.B61)
        if self.first_constraint_x is None:self.first_constraint_x=x.copy()
        if self._constraint_x is not None and np.array_equal(x,self._constraint_x):self.constraint_cache_hits+=1;return self._constraint_cache
        self.constraint_full_evaluations+=1;local=[self._sample(i,q) for i,q in enumerate(Q)]
        values=np.array([a['operational_depth_m'] for a in local]).ravel()/.001
        blocks=[sparse.kron(sparse.csr_matrix(self.B61[i:i+1,1:]),sparse.csr_matrix(a['depth_jac_q']/.001),format='csr') for i,a in enumerate(local)]
        J=sparse.vstack(blocks,format='csr')
        if values.shape!=(122,) or J.shape!=(122,340) or not np.isfinite(values).all() or not np.isfinite(J.data).all():raise ValueError('Invalid sampled constraint/Jacobian')
        self._constraint_x=x.copy();self._constraint_cache=(values,J,local);return self._constraint_cache

    def counters(self):
        return {'fit_requests':self.fit_requests,'fit_parent':self.parent.counters(),'constraint_requests':self.constraint_requests,'constraint_full_evaluations':self.constraint_full_evaluations,'constraint_cache_hits':self.constraint_cache_hits,'sample_cache_hits':self.sample_cache_hits,'actual_collision_calls':self.parent.g.collision_query_count,'collision_calls_by_stage':dict(self.stage_counts),'events_recorded':len(self.events)}

    def diagnostic_arrays(self,x):
        r,J,fit=self.fit(x);depth,C,local=self.nonlinear(x);Q=self.q_from_x(x);Q61=self.sample_q(x,self.B61)
        arrays=self.parent.diagnostic_arrays(Q[1:].ravel())
        # Replace the parent300-variable optimizer fields by the current340
        # control variables; parent knot-J remains explicitly named below.
        for key in ('jacobian_CSR_data','jacobian_CSR_indices','jacobian_CSR_indptr','jacobian_shape'):
            arrays['parent_knot_'+key]=arrays.pop(key)
        arrays.update(control_points_rad=self.controls_from_x(x),optimizer_x=np.asarray(x).copy(),requested_solver_x0=self.x0.copy(),initial_guess_q_goal_rad=self.q_from_x(self.x0),first_evaluated_x=self.first_fit_x,
                      basis16=self.B16,basis61=self.B61,planned_time_s=self.geometry_time,planned_q_goal_rad=Q61,planned_object_pose_w=self.pose61,
                      operational_generated_maxdepth_m=depth.reshape(61,2)*.001,category_order=np.array(CATEGORY_ORDER),category_generated_mask=np.array([a['category_generated'] for a in local]),category_selected_pair_row=np.array([a['selected_pair_row'] for a in local]),
                      planned_native_selected_contact_distance_m=np.array([a['evaluation'].signed_distance_m for a in local]),planned_native_generated_pair_mask=np.array([a['evaluation'].generated_mask for a in local]),planned_native_contact_count_by_pair=np.array([a['evaluation'].contact_count for a in local]),planned_native_contact_tie_count_by_pair=np.array([a['evaluation'].tie_count for a in local]),planned_native_selected_contact_index=np.array([a['evaluation'].selected_contact_index for a in local]),
                      planning_depth_bound_m=np.array(PLANNING_DEPTH_BOUND_M),constraint_scaled_mm=depth,constraint_J_CSR_data=C.data,constraint_J_CSR_indices=C.indices,constraint_J_CSR_indptr=C.indptr,constraint_J_shape=np.array(C.shape),
                      jacobian_CSR_data=J.data,jacobian_CSR_indices=J.indices,jacobian_CSR_indptr=J.indptr,jacobian_shape=np.array(J.shape))
        return arrays

    def diagnostic_summary(self,x):
        r,J,fit=self.fit(x);depth,C,local=self.nonlinear(x);Q=self.q_from_x(x)
        rows=[]
        for i,a in enumerate(local):
            e=a['evaluation'];rows.append({'sample':i,'time_s':float(self.geometry_time[i]),'operational_generated_maxdepth_m':a['operational_depth_m'].tolist(),'category_generated':a['category_generated'].tolist(),'selected_pair_row':a['selected_pair_row'].tolist(),'selected_pair_geoms':[self.parent.g.pairs[k].tolist() if k>=0 else None for k in a['selected_pair_row']],'category_tied_pair_rows':a['category_tied_pair_rows'],'category_global_tied_contacts':a['category_global_tied_contacts'],'selected_contact_ties':[int(e.tie_count[k]) if k>=0 else None for k in a['selected_pair_row']],'missing_catalogue_pairs_UNKNOWN':int((~e.generated_mask).sum()),'native_contact_count':e.native_contact_count})
        parent=self.parent.diagnostic_summary(Q[1:].ravel())
        return {'cost':float(.5*r@r),'residual_rows':len(r),'free_variables':340,'control_points':18,'fit_frames':16,'geometry_samples':61,'nonlinear_scalar_constraints':122,'fit_jacobian_nnz':int(J.nnz),'constraint_jacobian_nnz':int(C.nnz),'hardq0_bitexact':bool(np.array_equal(Q[0],self.q0)),'q_goal_hard_excess_max_rad':parent['q_goal_hard_excess_max_rad'],'planning_depth_bound_m':PLANNING_DEPTH_BOUND_M,'planning_numeric_margin_m':.000001,'qualification_depth_bound_m':.003,'sampled_generated_overlap_constraints_pass':bool(np.all(depth<=PLANNING_DEPTH_BOUND_M/.001)),'maximum_generated_overlap_m':float(depth.max()*.001),'planned61':rows,'fit16':parent,'object_quintic_knot_roundoff':self.object_knot_roundoff,'counts':self.counters(),'basis_grouping':'q0+B_free@(P_free-q0); constant source64 q0 exact, no clipping','constraint_scope':'61 sampled operational generated-overlap bounds only. Absent gap UNKNOWN; no continuous clearance, global differentiability, physical force, grasp or dynamic qualification. Ties select exact min then lowest catalog row/native index; diagnostics retain tolerance ties.','physical_reference_qualified':False,'known_contact_labels':False,'GPU_used':False,'physics_steps':0}

    def native_records(self,x):return {'fit16':self.parent.native_records(self.q_from_x(x)[1:].ravel()),'planned61':[record for a in self.nonlinear(x)[2] for record in a['native_records']]}
