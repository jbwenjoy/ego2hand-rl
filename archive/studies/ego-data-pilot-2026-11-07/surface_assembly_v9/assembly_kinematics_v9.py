"""SOURCE DRAFT: cheap same-model assembly FK; no dynamics or collision query.

Use AssemblyKinematics(existing_native_or_geometry_evaluator.geometry). Each
query owns its data and performs one mj_kinematics plus one mj_comPos. Fixed
filled-triangle material anchors/normals come from the immutable307e91 source.
Ten raw equations contain6 positional [m] and4 tangent-normal [unitless] terms.
"""
from pathlib import Path
import hashlib
import json

PILOT=Path(__file__).resolve().parent.parent
ANCHOR_PATH=PILOT/'contact_patch_v7/paired_surface_anchors_v1.json'
ANCHOR_SHA='307e91a728c14fc4f799465649bc11f71c1b0be0971a3ec289d2a1acfaf565bb'
COMPRESSION_M=.000025


def pin(path):
    path=Path(path).resolve();b=path.read_bytes()
    return {'path':str(path),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


class AssemblyKinematics:
    def __init__(self,geometry_evaluator,*,anchors_path=ANCHOR_PATH,qrest12=None):
        import mujoco
        import numpy as np
        from scipy.spatial.transform import Rotation
        self.mj,self.np,self.Rotation=mujoco,np,Rotation
        self.geometry=geometry_evaluator;self.m=geometry_evaluator.m
        self.Rp=geometry_evaluator.Rp.copy();self.pp=geometry_evaluator.pp.copy()
        self.initial_qpos=geometry_evaluator.initial['qpos'].copy()
        self.pid,self.oid=geometry_evaluator.pid,geometry_evaluator.oid
        self.pads=list(geometry_evaluator.pads)
        record=pin(anchors_path)
        if record['sha256']!=ANCHOR_SHA:raise ValueError('Fixed actual307e91 triangle anchors required')
        a=json.loads(Path(anchors_path).read_text(encoding='utf8'))
        if (a['schema']!='saved_filled_triangle_paired_surface_anchors:7'
                or not a['checks'] or any(v is not True for v in a['checks'].values())):
            raise ValueError('Completed original anchor contract')
        rows=a['anchors']
        if (len(rows)!=2 or [v['finger'] for v in rows]!=['thumb','index']
                or [v['hand_geom_id'] for v in rows]!=self.pads
                or any(v['phone_geom_id']!=geometry_evaluator.phone for v in rows)):
            raise ValueError('Actual thumb/index and phone geom identities')
        source_qpos=np.asarray(a['source_pose']['qpos27'],dtype=np.float64)
        if source_qpos.shape!=(27,) or not np.isfinite(source_qpos).all():raise ValueError('Original source qpos27')
        self.qrest12=source_qpos[8:20].copy()
        if qrest12 is not None and not self._bits(np.asarray(qrest12,dtype=np.float64),self.qrest12):
            raise ValueError('Remaining12 joints must be original source values')
        self.hand_points=np.array([v['surfaces']['hand']['projected_point_geom_local_m'] for v in rows])
        self.phone_points=np.array([v['surfaces']['phone']['projected_point_phone_body_m'] for v in rows])
        self.hand_normals=np.array([v['surfaces']['hand']['outward_normal_geom_local'] for v in rows])
        self.phone_normals=np.array([v['surfaces']['phone']['outward_normal_phone_body'] for v in rows])
        for value in (self.hand_points,self.phone_points,self.hand_normals,self.phone_normals):
            if value.shape!=(2,3) or not np.isfinite(value).all():raise ValueError('Finite actual two-pair operands')
        if max(np.max(abs(np.linalg.norm(v,axis=1)-1)) for v in (self.hand_normals,self.phone_normals))>1e-8:
            raise ValueError('Original unit outward normals')
        tangent=[]
        for i,row in enumerate(rows):
            tri=np.asarray(row['surfaces']['phone']['source_triangle_vertices_phone_body_m'],dtype=np.float64)
            n=self.phone_normals[i];n_unit=n/np.linalg.norm(n)
            edge=tri[1]-tri[0]
            # Roundoff-only orthogonalization of a basis, not asset/R repair.
            edge=edge-n_unit*(edge@n_unit)
            length=float(np.linalg.norm(edge))
            if not length>0:raise ValueError('Nondegenerate original first directed triangle edge')
            t1=edge/length;t2=np.cross(n_unit,t1);t2/=np.linalg.norm(t2)
            tangent.append([t1,t2])
        self.phone_tangents_body_rows=np.asarray(tangent)
        self.anchors=a;self.anchor_pin=record
        if (self.m.nq,self.m.nv,self.m.nu,self.m.neq,self.m.nmocap)!=(27,26,20,0,0):
            raise ValueError('Original fixed-wrist20/free-phone layout')
        self.d=mujoco.MjData(self.m)
        self.counts={'MJB_reads':0,'owned_MjData':1,'query_attempted':0,'query_returned':0,
            'mj_kinematics':0,'mj_comPos':0,'jacobian_calls':0,'mj_comVel':0,
            'mj_rne':0,'mj_passive':0,'geometry_queries':0,'mj_forward':0,'LP_calls':0,
            'integration_steps':0,'learning_GPU':0}
        self.model_digest=self._model_digest()

    def _bits(self,a,b):
        return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

    def _model_digest(self):
        h=hashlib.sha256()
        for name in ('body_parentid','body_pos','body_quat','body_jntadr','body_jntnum',
                     'body_ipos','body_iquat','jnt_type','jnt_pos','jnt_axis','jnt_range',
                     'geom_pos','geom_quat','geom_bodyid','actuator_gainprm','actuator_biasprm',
                     'actuator_gear','actuator_ctrlrange','actuator_forcerange','jnt_actfrcrange'):
            a=self.np.ascontiguousarray(getattr(self.m,name));h.update((name+a.dtype.str+str(a.shape)).encode());h.update(a.tobytes())
        return h.hexdigest()

    def audit_model(self):
        if self._model_digest()!=self.model_digest:raise ValueError('Original model or wrist changed')
        return {'passed':True,'model_digest':self.model_digest,'counts':dict(self.counts)}

    def query(self,x14,qrest12=None):
        np,mj,m,d=self.np,self.mj,self.m,self.d
        self.counts['query_attempted']+=1
        x=np.asarray(x14,dtype=np.float64)
        rest=self.qrest12 if qrest12 is None else np.asarray(qrest12,dtype=np.float64)
        if x.shape!=(14,) or not np.isfinite(x).all() or not self._bits(rest,self.qrest12):
            raise ValueError('Finiteq8/t3/rotvec3 and fixed original qrest12')
        q=np.r_[x[:8],rest]
        if np.any(q<m.jnt_range[:20,0]) or np.any(q>m.jnt_range[:20,1]):
            raise ValueError('Original true hard joint ranges')
        t,rv=x[8:11],x[11:14]
        requested_R=self.Rp@self.Rotation.from_rotvec(rv).as_matrix()
        d.qpos[:]=self.initial_qpos;d.qpos[:20]=q;d.qpos[20:23]=self.pp+self.Rp@t
        d.qpos[23:27]=self.Rotation.from_matrix(requested_R).as_quat()[[3,0,1,2]]
        d.qvel[:]=0;d.ctrl[:]=0;d.act[:]=0;d.qacc_warmstart[:]=0
        d.qfrc_applied[:]=0;d.xfrc_applied[:]=0;d.time=0
        mj.mj_kinematics(m,d);self.counts['mj_kinematics']+=1
        mj.mj_comPos(m,d);self.counts['mj_comPos']+=1
        if not self._bits(d.xpos[self.pid],self.pp) or not self._bits(d.xmat[self.pid].reshape(3,3),self.Rp):
            raise ValueError('Original fixed wrist/palm changed')
        GR=np.asarray([d.geom_xmat[g].reshape(3,3) for g in self.pads])
        A=np.asarray([GR[i]@self.hand_points[i]+d.geom_xpos[g] for i,g in enumerate(self.pads)])
        R=d.xmat[self.oid].reshape(3,3).copy();p=d.xpos[self.oid].copy()
        B=self.phone_points@R.T+p
        nH=np.einsum('nij,nj->ni',GR,self.hand_normals)
        nO=self.phone_normals@R.T
        Trows=self.phone_tangents_body_rows@R.T
        closure=B-A+COMPRESSION_M*nH
        normal_eq=np.einsum('nki,ni->nk',Trows,nH)
        dots=np.einsum('ni,ni->n',nH,nO)
        if not np.isfinite(np.r_[A.ravel(),B.ravel(),nH.ravel(),nO.ravel(),closure.ravel(),normal_eq.ravel()]).all():
            raise ValueError('Finite FK assembly operands')
        self.counts['query_returned']+=1
        return {'schema':'fixed_original_surface_assembly_kinematics:9',
            'x14':x.copy(),'q20':q,'qrest12':rest.copy(),'qpos27':d.qpos.copy(),
            'phone_t_palm3':t.copy(),'phone_rotvec3':rv.copy(),
            'phone_p_world_m':p,'phone_R_world':R,'phone_COM_world_m':d.xipos[self.oid].copy(),
            'A_hand_world_m':A,'B_phone_world_m':B,'n_hand_world':nH,'n_phone_world':nO,
            'phone_tangents_world2x2x3':Trows,'phone_tangent_columns_world_2x3x2':Trows.transpose(0,2,1).copy(),
            'closure2x3':closure,'normal_eq2x2':normal_eq,
            'eq10':np.r_[closure.ravel(),normal_eq.ravel()],
            'eq10_units':['m']*6+['dimensionless']*4,
            'normal_dot2':dots,'antidot2':-dots,'antiparallel_branch2':dots<0,
            'antidot_branch_rule':'antidot2 must be strictly positive (normal_dot2<0); projection eq alone also admits same-direction normals.',
            'compression_target_m':COMPRESSION_M,'anchor_pin':self.anchor_pin,
            'tangent_basis_rule':'Original phone triangle edge v1-v0, projected off saved nO for numerical orthogonality, t2=nO_unit cross t1. Only basis arithmetic; original anchor/normals/model are not repaired.',
            'counts':dict(self.counts),'full_geometry_or_support_or_holding_qualified':False,
            'scope':'Pure fixed-material assembly FK; no collision, force/dynamics, native contact or global geometry certificate.'}
