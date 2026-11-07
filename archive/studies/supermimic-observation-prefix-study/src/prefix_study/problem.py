"""Coupled prefix16 geometric residual and analytic sparse Jacobian.

Weights multiply residuals directly. q0 is not among the300 variables.
No dynamics, force solve, geomDistance, oldq splice or material reselection.
"""
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import sys,time
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation
from .inputs import checked_protocol,read_prefix_inputs,WEIGHTS
from .io import read


class Prefix16GeometryProblem:
    def __init__(self,protocol,*,event_sink=None,deadline=None,read_array=None):
        self.protocol=checked_protocol(protocol) if isinstance(protocol,(str,Path)) else protocol
        self.stage_name='construction';self.event_sink=event_sink;self.deadline=deadline
        self.events=[];self.stage_counts={};self.full_evaluations=0;self.requests=0;self.local_cache_hits=0;self.global_cache_hits=0
        pins=self.protocol['inputs']
        sys.path.insert(0,str(Path(pins['native_filtered_geometry']['path']).parent))
        from native_filtered_contact_geometry import NativeFilteredContactGeometry
        from retarget_palmroot_cpu import BONES
        sys.path.insert(0,str(Path(pins['material_geometry']['path']).parent))
        from material_anchors import MaterialAnchor,MaterialAnchorGeometry
        self.BONES=BONES;self.g=NativeFilteredContactGeometry(pins['model']['path'],pins['actual_context']['path']);self.robot=self.g.robot
        # Only collision geometry/material changes in this CPU host instance.
        # It is NOT an actual-inertia host suitable for a future Warp put_model.
        with np.load(pins['actual_context']['path'],allow_pickle=False) as z:self.g.model.geom_friction[:]=z['geom_friction'][0]
        self.data=read_prefix_inputs(self.protocol,self.robot,read_array=read_array)
        self.q0=self.data['q0'];self.pose=self.data['mapped_pose'];self.x0=np.tile(self.q0,15)
        self.lower=np.tile(self.robot.lower,15);self.upper=np.tile(self.robot.upper,15)
        self.material=MaterialAnchorGeometry(self.g);self.bindings=read(pins['frozen_material_anchors']['path'])
        if self.bindings['hypothesis_id']!='H123' or self.bindings['known_contacts_or_forces'] is not False:raise ValueError('Only the frozenH123 unknown-contact hypothesis')
        self.anchors=[]
        for entry in self.bindings['frozen_material_anchors']:
            a=dict(entry)
            for key in ('point_geom_local','support_normal_geom_local'):
                a[key]=np.array(a[key],dtype=float);a[key].setflags(write=False)
            anchor=MaterialAnchor(**a)
            if anchor.geom_id not in self.g.hand_ids or self.g.model.geom(anchor.geom_id).name!=anchor.geom_name or int(self.g.model.geom_bodyid[anchor.geom_id])!=anchor.body_id:raise ValueError('Frozen material geometry mismatch')
            if abs(np.linalg.norm(anchor.support_normal_geom_local)-1)>1e-8:raise ValueError('Frozen material normal invalid')
            self.anchors.append(anchor)
        if [a.finger for a in self.anchors]!=[1,2,3]:raise ValueError('H123 material order changed')
        R=Rotation.from_quat(self.pose[:,[4,5,6,3]]).as_matrix();R0=R[0];p0=self.pose[0,:3]
        hypotheses=self.bindings['observation_hypotheses']
        object_points=np.array([R0.T@(np.array(h['object_surface_closest_w'])-p0) for h in hypotheses])
        object_normals=np.array([R0.T@np.array(h['outward_normal_w']) for h in hypotheses])
        if np.max(np.abs(object_points-np.array([h['object_surface_closest_o'] for h in hypotheses])))>1e-12 or np.max(np.abs(object_normals-np.array([h['outward_normal_o'] for h in hypotheses])))>1e-12:raise ValueError('ObjectBODY local anchor conversion failed')
        self.object_points_body=object_points;self.object_normals_body=object_normals
        self.surface_points=np.einsum('tij,aj->tai',R,object_points)+self.pose[:,None,:3]
        self.surface_normals=np.einsum('tij,aj->tai',R,object_normals)
        self.anchor_goals=self.surface_points-.0003*self.surface_normals;self.normal_goals=-self.surface_normals
        if np.max(np.abs(self.anchor_goals[0]-self.bindings['target_material_points_w']))>1e-12 or np.max(np.abs(self.normal_goals[0]-self.bindings['target_hand_normals_w']))>1e-12:raise ValueError('Parentframe0 targets changed')
        for h in hypotheses:
            if h['confidence'] is not None or h['is_known_contact_label'] is not False:raise ValueError('Do not invent contactconfidence')
        self.local_rows=45+len(self.g.pairs)+18+40
        self.rows=16*self.local_rows+15*20+14*20
        self._local_cache=[None]*16;self._last_x=None;self._last=None
        self.first_evaluated_x=None

    @contextmanager
    def stage(self,name):
        before=self.stage_name;self.stage_name=name
        try:yield
        finally:self.stage_name=before

    def q_from_x(self,x):
        x=np.asarray(x,dtype=np.float64)
        if x.shape!=(300,) or not np.isfinite(x).all():raise ValueError('Expected finite freevariable shape(300,)')
        return np.vstack((self.q0,x.reshape(15,20)))

    def _local(self,frame,q):
        if self.deadline:self.deadline()
        cached=self._local_cache[frame]
        if cached is not None and np.array_equal(cached['q'],q):self.local_cache_hits+=1;return cached
        g=self.g;g.set_object_pose(self.pose[frame]);before=g.collision_query_count;e=g.evaluate(q);after=g.collision_query_count
        if after-before not in (0,1):raise ValueError('Unexpected collisioncount increment')
        if after>before:
            event={'query_index':after,'stage':self.stage_name,'frame':frame,'full_evaluation':self.full_evaluations,'monotonic_s':time.perf_counter(),'scope':'CPU mj_collision geometry only, not physics'}
            self.events.append(event);self.stage_counts[self.stage_name]=self.stage_counts.get(self.stage_name,0)+1
            if self.event_sink:self.event_sink(event)
        bone,boneJ=self.robot.bone_features(q)
        features=[self.material.oriented_features(q,a) for a in self.anchors]
        points=np.array([f[0] for f in features]);normals=np.array([f[2] for f in features])
        r=np.r_[(bone-self.data['desired_bones'][frame]).ravel(),e.residual,
                 ((points-self.anchor_goals[frame])/.001).ravel(),(.25*(normals-self.normal_goals[frame])).ravel(),
                 .03*(q-self.q0),.001*(q-self.robot.nominal)]
        J=np.vstack((boneJ,e.residual_jacobian,np.vstack([f[1]/.001 for f in features]),np.vstack([.25*f[3] for f in features]),.03*np.eye(20),.001*np.eye(20)))
        native=[]
        for index,c in enumerate(g.data.contact):
            a,b=map(int,c.geom);og=a in g.object_ids or b in g.object_ids;ha=a in g.hand_ids;hb=b in g.hand_ids
            kind='hand_object' if og and (ha or hb) else 'hand_self' if ha and hb else 'object_environment' if og else 'other'
            native.append({'frame':frame,'contact_index':index,'category':kind,'geom':[a,b],'names':[g.model.geom(a).name,g.model.geom(b).name],'distance_m':float(c.dist),'position_w':c.pos.tolist(),'frame_rows_geom0_to_geom1_w':c.frame.reshape(3,3).tolist(),'effective_condim':int(c.dim),'friction5':c.friction.tolist(),'phase':'native_CPU_geometry_not_solved_force'})
        fk=self.robot.forward(q)[0].copy()
        diag={'q':q.copy(),'residual':r,'jacobian':J,'native':e,'native_records':native,'fk':fk,'anchor_points_w':points,'anchor_normals_w':normals,'bone_angle_error_deg':np.rad2deg(np.arccos(np.clip(np.sum(bone*self.data['desired_bones'][frame],axis=1),-1,1))),
              'anchor_position_error_m':np.linalg.norm(points-self.anchor_goals[frame],axis=1),'anchor_normal_error_deg':np.rad2deg(np.arccos(np.clip(np.sum(normals*self.normal_goals[frame],axis=1),-1,1)))}
        self._local_cache[frame]=diag;return diag

    def evaluate(self,x):
        self.requests+=1
        if self.deadline:self.deadline()
        x=np.asarray(x,dtype=float);Q=self.q_from_x(x)
        if self.first_evaluated_x is None:self.first_evaluated_x=x.copy()
        if self._last_x is not None and np.array_equal(self._last_x,x):self.global_cache_hits+=1;return self._last
        self.full_evaluations+=1;local=[self._local(f,Q[f]) for f in range(16)]
        residual=np.r_[np.concatenate([a['residual'] for a in local]),(.03*(Q[1:]-Q[:-1])).ravel(),(.03*(Q[2:]-2*Q[1:-1]+Q[:-2])).ravel()]
        ii=[];jj=[];vv=[]
        def add(block,row,frame):
            if frame==0:return  # Fixed q0 has no variable columns.
            a,b=np.nonzero(block);ii.extend((a+row).tolist());jj.extend((b+(frame-1)*20).tolist());vv.extend(block[a,b].tolist())
        for frame,a in enumerate(local):add(a['jacobian'],frame*self.local_rows,frame)
        row=16*self.local_rows
        for frame in range(1,16):
            add(.03*np.eye(20),row,frame);add(-.03*np.eye(20),row,frame-1);row+=20
        for frame in range(1,15):
            add(.03*np.eye(20),row,frame+1);add(-.06*np.eye(20),row,frame);add(.03*np.eye(20),row,frame-1);row+=20
        J=sparse.coo_matrix((vv,(ii,jj)),shape=(self.rows,300)).tocsr()
        if residual.shape!=(self.rows,) or row!=self.rows or not np.isfinite(residual).all() or not np.isfinite(J.data).all():raise ValueError('Invalid coupled residual/CSR')
        self._last_x=x.copy();self._last=(residual,J,local)
        return self._last

    def counters(self):
        return {'evaluation_requests':self.requests,'full_evaluations':self.full_evaluations,'global_cache_hits':self.global_cache_hits,'frame_cache_hits':self.local_cache_hits,'actual_collision_calls':self.g.collision_query_count,'collision_calls_by_stage':dict(self.stage_counts),'events_recorded':len(self.events)}

    def diagnostic_arrays(self,x):
        r,J,local=self.evaluate(x);Q=self.q_from_x(x)
        return {'time_s':self.data['time_s'],'q_goal_rad':Q,'mapped_object_pose_w':self.pose,'q0_fixed_rad':self.q0,'initial_guess_q_goal_rad':self.q_from_x(self.x0),'requested_solver_x0':self.x0,'first_evaluated_x':self.first_evaluated_x,
                'morphology_normalized_target_points_w':self.data['normalized_target_points_w'],'source_to_fixed_wrist_R':self.data['R'],'source_to_fixed_wrist_translation_m':self.data['translation'],'desired_unit_bone_directions_w':self.data['desired_bones'],
                'fk_hand_points_w':np.array([a['fk'] for a in local]),'bone_angle_error_deg':np.array([a['bone_angle_error_deg'] for a in local]),'anchor_position_error_m':np.array([a['anchor_position_error_m'] for a in local]),'anchor_normal_error_deg':np.array([a['anchor_normal_error_deg'] for a in local]),'anchor_goal_points_w':self.anchor_goals,'anchor_goal_hand_normals_w':self.normal_goals,'anchor_material_points_w':np.array([a['anchor_points_w'] for a in local]),'anchor_material_normals_w':np.array([a['anchor_normals_w'] for a in local]),'object_anchor_points_body_m':self.object_points_body,'object_anchor_normals_body':self.object_normals_body,
                'native_selected_contact_distance_m':np.array([a['native'].signed_distance_m for a in local]),'native_generated_pair_mask':np.array([a['native'].generated_mask for a in local]),'native_contact_count_by_pair':np.array([a['native'].contact_count for a in local]),'native_contact_tie_count_by_pair':np.array([a['native'].tie_count for a in local]),'native_pair_indices':self.g.pairs,'native_pair_kind':self.g.pair_kind,'joint_lower_rad':self.robot.lower,'joint_upper_rad':self.robot.upper,'residual':r,'jacobian_CSR_data':J.data,'jacobian_CSR_indices':J.indices,'jacobian_CSR_indptr':J.indptr,'jacobian_shape':np.array(J.shape)}

    def diagnostic_summary(self,x):
        r,J,local=self.evaluate(x);Q=self.q_from_x(x);frames=[]
        for i,a in enumerate(local):
            e=a['native'];entry={'frame':i,'time_s':float(self.data['time_s'][i]),'bone_mean_deg':float(a['bone_angle_error_deg'].mean()),'bone_max_deg':float(a['bone_angle_error_deg'].max()),'tip_error_mean_m':float(np.linalg.norm(a['fk'][[4,8,12,16,20]]-self.data['normalized_target_points_w'][i,[4,8,12,16,20]],axis=1).mean()),'anchor_position_error_m':a['anchor_position_error_m'].tolist(),'anchor_normal_error_deg':a['anchor_normal_error_deg'].tolist(),'native_contact_count':e.native_contact_count,'tied_generated_pairs':int(np.count_nonzero(e.generated_mask&(e.tie_count>1))),'missing_catalogue_pairs_UNKNOWN':int((~e.generated_mask).sum())}
            for kind in ('hand_object','hand_self'):
                mask=e.generated_mask&(self.g.pair_kind==kind);entry[kind]={'generated_pairs':int(mask.sum()),'max_depth_m':float(np.maximum(-e.signed_distance_m[mask],0).max()) if mask.any() else None}
            frames.append(entry)
        return {'frames':frames,'cost':float(.5*r@r),'residual_rows':len(r),'free_variables':300,'jacobian_nnz':int(J.nnz),'hardq0_bitexact':bool(np.array_equal(Q[0],self.q0)),'q_goal_hard_excess_max_rad':float(np.maximum(np.maximum(self.robot.lower-Q,Q-self.robot.upper),0).max()),'input_checks':self.data['checks'],'input_array_read_log':self.data['read_log'],'counts':self.counters(),'geometry_metadata':self.g.metadata,'host_model_modified_fields':['geom_size','geom_pos','geom_friction'],'host_model_suitable_for_Warp_upload':False,'contacts_known':False,'physical_reference_qualified':False,'qd_or_u_exported':False,'scope':'Geometry knots only; continuouscurve/derivative/controller/init/physics qualification remain PENDING. Not a global smoothness/contactforce/grasp certificate.'}

    def native_records(self,x):return [r for a in self.evaluate(x)[2] for r in a['native_records']]
