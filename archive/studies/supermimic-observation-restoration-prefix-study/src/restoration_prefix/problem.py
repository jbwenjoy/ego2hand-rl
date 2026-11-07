"""New R/F adapter around immutable e0edf geometry and145117 basis sources."""
from pathlib import Path
import hashlib,sys
import numpy as np
from scipy import sparse
from .math import finite,primal_audit,restoration_box

FROZEN_ROOT=Path('D:/workspace/supermimic-observation-constrained-prefix-study')
sys.path.insert(0,str(FROZEN_ROOT/'src'))
from constrained_prefix.problem import ConstrainedPrefixGeometryProblem
import bspline_control


def signature(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.ascontiguousarray(a);h.update(a.dtype.str.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def sparse_signature(matrix):
    matrix=sparse.csr_matrix(matrix)
    return signature(np.asarray(matrix.shape),matrix.data,matrix.indices,matrix.indptr)


def load_linear_contract(path):
    fields=('x0','lower','upper','b_ub','b_centered','A_CSR_data','A_CSR_indices','A_CSR_indptr','A_shape')
    with np.load(path,allow_pickle=False) as z:data={k:z[k].copy() for k in fields}
    A=sparse.csr_matrix((data.pop('A_CSR_data'),data.pop('A_CSR_indices'),data.pop('A_CSR_indptr')),shape=data.pop('A_shape'))
    if A.shape!=(5960,340) or not np.isfinite(A.data).all():raise ValueError('Expected original0798 linear matrix')
    for key in ('x0','lower','upper'):data[key]=finite(data[key],(340,))
    for key in ('b_ub','b_centered'):data[key]=finite(data[key],(5960,))
    if np.any(data['lower']>data['upper']):raise ValueError('Empty original coefficient box')
    if np.max(np.abs(A@data['x0']+data['b_centered']-data['b_ub']))>1e-12:raise ValueError('Saved absolute/centered matrix identity')
    return {'A_ub':A,**data}


class RestorationPrefixProblem:
    def __init__(self,seed_candidate,linear_contract,*,parent_protocol=None,event_sink=None,deadline=None):
        self.geometry=ConstrainedPrefixGeometryProblem(bspline_control.make_basis(),parent_protocol=parent_protocol,event_sink=event_sink,deadline=deadline)
        self.contract=load_linear_contract(linear_contract);self.deadline=deadline
        fields=('optimizer_x','control_points_rad','q0_fixed_rad','time_s','q_goal_rad','mapped_object_pose_w','planned_object_pose_w','retarget_variant_id')
        with np.load(seed_candidate,allow_pickle=False) as z:seed={k:z[k].copy() for k in fields}
        self.seed=finite(seed['optimizer_x'],(340,));self.q0=self.geometry.q0
        if not np.array_equal(self.contract['x0'],self.geometry.x0) or not np.array_equal(seed['q0_fixed_rad'],self.q0):raise ValueError('Seed/source64 q0 changed')
        if not np.array_equal(seed['control_points_rad'],self.geometry.controls_from_x(self.seed)) or not np.array_equal(seed['q_goal_rad'],self.geometry.q_from_x(self.seed)):raise ValueError('Only frozen64aa terminal coefficients as seed')
        if not np.array_equal(seed['time_s'],self.geometry.fit_time) or not np.array_equal(seed['mapped_object_pose_w'],self.geometry.parent.pose) or not np.array_equal(seed['planned_object_pose_w'],self.geometry.pose61):raise ValueError('Seed time/mapped object identity differs')
        if str(seed['retarget_variant_id'])!='H123_cubic_bspline_constrained_prefix16_v1':raise ValueError('Wrong seed retarget variant')
        self.seed_read_fields=list(fields);self.R_lower,self.R_upper=restoration_box(self.seed,self.contract['lower'],self.contract['upper'])
        self.requests=0;self.full_evaluations=0;self.cache_hits=0;self.forced_constraint_refreshes=0
        self._x=None;self._state=None

    def stage(self,name):return self.geometry.stage(name)

    def evaluate(self,x,*,force_constraint_refresh=False):
        self.requests+=1
        if self.deadline:self.deadline()
        x=finite(x,(340,))
        if force_constraint_refresh:
            # A new122-row raw decision is based on 61 fresh native collisions,
            # not the artificial R slack or optimizer cached constraints. No
            # source/model/geometry definition is changed by clearing caches.
            self.geometry._constraint_x=None;self.geometry._constraint_cache=None
            self.geometry._sample_cache=[None]*61
            self.forced_constraint_refreshes+=1
        if not force_constraint_refresh and self._x is not None and np.array_equal(x,self._x):self.cache_hits+=1;return self._state
        self.full_evaluations+=1
        before=self.geometry.parent.g.collision_query_count
        r,J,fit=self.geometry.fit(x);after_fit=self.geometry.parent.g.collision_query_count
        depth,C,samples=self.geometry.nonlinear(x);after=self.geometry.parent.g.collision_query_count
        if force_constraint_refresh and after-after_fit!=61:raise ValueError('Forced raw check did not execute all61 native collision queries')
        selected=np.array([a['selected_pair_row'] for a in samples]);generated=np.array([a['evaluation'].generated_mask for a in samples]);contact_index=np.array([a['evaluation'].selected_contact_index for a in samples]);ties=np.array([a['evaluation'].tie_count for a in samples])
        branches=[]
        for i,a in enumerate(samples):
            for category,kind in enumerate(('hand_object','hand_self')):
                row=int(a['selected_pair_row'][category]);e=a['evaluation']
                branches.append({'sample':i,'category':kind,'category_generated':bool(a['category_generated'][category]),'depth_mm':float(a['operational_depth_m'][category]*1000),'selected_pair_row':row,'selected_pair_geom':self.geometry.parent.g.pairs[row].tolist() if row>=0 else None,'selected_native_contact_index':int(e.selected_contact_index[row]) if row>=0 else None,'tied_pair_rows':a['category_tied_pair_rows'][category],'global_tied_contacts':a['category_global_tied_contacts'][category]})
        record={'original_fit_cost':float(.5*r@r),'max_generated_depth_mm':float(depth.max()),'max_positive_planning_violation_mm':float(max(depth.max()-2.999,0.)),'sum_positive_planning_violation_mm':float(np.maximum(depth-2.999,0).sum()),
                'signature':{'fit_J_CSR':sparse_signature(J),'constraint_J_CSR':sparse_signature(C),'active_pair_contact_tie_structure':signature(generated,selected,contact_index,ties)},
                'generated_pair_count':int(generated.sum()),'pair_time_tie_slots':int((ties>1).sum()),'category_branches':branches,
                'geometry_semantics':'Sampled native generated-overlap only; absent gaps remain UNKNOWN. Signatures are exact saved numerical branch/J descriptions, not global derivative qualification.'}
        state={'generation':self.full_evaluations,'x':x.copy(),'r':r,'J':J,'depth_mm':depth,'C':C,'fit':fit,'samples':samples,'cost':float(.5*r@r),'record':record,
               'creation_query_delta':{'fit16':after_fit-before,'constraint61':after-after_fit,'total_shared_g':after-before,'force61_requested':force_constraint_refresh}}
        self._x=x.copy();self._state=state;return state

    def primal(self,state,*,R_trust=False):
        return primal_audit(state['x'],state['depth_mm'],self.contract,trust_lower=self.R_lower if R_trust else None,trust_upper=self.R_upper if R_trust else None)

    def diagnostic_arrays(self,state):
        x=state['x'];arrays=self.geometry.diagnostic_arrays(x)
        arrays.update(restoration_seed_optimizer_x=self.seed,R_trust_lower=self.R_lower,R_trust_upper=self.R_upper)
        return arrays

    def counters(self):
        return {'state_requests':self.requests,'full_state_evaluations':self.full_evaluations,'state_cache_hits':self.cache_hits,'forced_61_constraint_refreshes':self.forced_constraint_refreshes,'geometry':self.geometry.counters(),'scope':'geometry.actual_collision_calls is one sharedg total; do not add parent sharedcounter again.'}
