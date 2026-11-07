"""Finite three-tip functional geometry proposals; no native forward/steps or NN.

Metadata default is read/hash-only. --run explicitly owns one fixed MJB, cheap
FK and frozen exact-convex static queries. All4 entries, including failure, remain.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,sys,time
import numpy as np

H=Path(__file__).resolve().parent;P=H.parent
PADS=[15,28,41]
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def checked(r):
    assert pin(r['path'])==r,'Changed pinned input '+r['path'];return Path(r['path'])
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,(np.bool_,)):return bool(x)
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(clean(x),f,indent=2,allow_nan=False)
def module(path,name):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
class BudgetStop(RuntimeError):pass

class Tripod:
    def __init__(self,cfg,out):
        import mujoco
        from scipy.spatial import ConvexHull
        from scipy.spatial.transform import Rotation
        self.mj=mujoco;self.Rotation=Rotation;self.Hull=ConvexHull;self.cfg=cfg;self.out=out
        G=module(checked(cfg['source_pins']['static_geometry']),'tripod_frozen_static_geometry')
        self.G=G;self.E=G.StaticRetargetEvaluator.from_existing_pilot(max_heavy_queries=cfg['budget']['max_total_heavy_queries'])
        self.m=self.E.m;self.d=mujoco.MjData(self.m);self.pp=self.E.pp.copy();self.Rp=self.E.Rp.copy()
        assert (self.m.nq,self.m.nv,self.m.nu)==(27,26,20) and self.E.phone==71 and len(self.E.phone_pairs)==22 and len(self.E.self_pairs)==189
        self.counts=dict(MJB_reads=1,owned_MjData=2,cheap_FK=0,cheap_comPos=0,full_geometry_evaluations=0,
            entries_attempted=0,entries_returned=0,query_cache_prunes=0,mj_forward=0,integration_steps=0,LP_calls=0,NN_training_GPU=0)
        self.geom_local={};self.triangles={};self.equations={}
        for gid in PADS+[71]:
            V=self.E.local[gid].copy();h=ConvexHull(V);self.geom_local[gid]=V;self.triangles[gid]=h.simplices.copy();self.equations[gid]=h.equations.copy()
        ph=ConvexHull(self.E.phone_vertices);self.phoneV=self.E.phone_vertices.copy();self.phoneT=ph.simplices.copy();self.phoneEq=ph.equations.copy()
        self.projection=module(checked(cfg['source_pins']['convex_query']),'tripod_frozen_projection')
        mus=[]
        for gid in PADS:
            ii=[i for i in range(self.m.npair) if set(map(int,[self.m.pair_geom1[i],self.m.pair_geom2[i]]))=={gid,71}]
            assert len(ii)==1 and self.m.pair_dim[ii[0]]==3
            mus.append(self.m.pair_friction[ii[0],:2].copy())
        assert np.array_equal(np.array(mus),np.full((3,2),.7))
        self.model_initial=self.E._parameter_digest();self.static_inputs={}
        for key in ['jnt_range','actuator_ctrlrange','actuator_forcerange','jnt_actfrcrange','actuator_gainprm','actuator_biasprm','body_ipos','body_mass','body_inertia']:
            self.static_inputs[key]=np.asarray(getattr(self.m,key)).copy()
        with (out/'compiled_geometry_and_limits.npz').open('xb') as f:
            np.savez_compressed(f,**self.static_inputs,**{f'geom{g}_vertices':self.geom_local[g] for g in PADS+[71]},
                **{f'geom{g}_triangles':self.triangles[g] for g in PADS+[71]},phone_body_vertices=self.phoneV,phone_body_triangles=self.phoneT)
        self.entry=None;self.full_cache={};self.global_deadline=time.perf_counter()+cfg['budget']['global_wall_s']
    def surface(self,p,V,T,eq):
        tri=V[T];relative=self.projection._closest_triangle_points_origin(tri-np.asarray(p))
        points=relative+p;dist=np.sum(relative*relative,axis=1);i=int(np.argmin(dist));point=points[i];a,b,c=tri[i]
        weights=np.linalg.lstsq(np.c_[(b-a),(c-a)],point-a,rcond=None)[0];bary=np.r_[1-weights.sum(),weights]
        assert np.min(bary)>=-1e-10 and np.max(abs(bary@tri[i]-point))<1e-10
        n=eq[i,:3]/np.linalg.norm(eq[i,:3]);assert np.max((V-point)@n)<1e-10
        return dict(triangle_id=i,triangle_vertex_ids=T[i],triangle_vertices_local_m=tri[i],barycentric=bary,
            point_local_m=point,outward_unit_normal_local=n,projection_distance_m=math.sqrt(float(dist[i])),
            triangle_area_m2=.5*float(np.linalg.norm(np.cross(b-a,c-a))),on_edge_or_vertex=bool(np.min(bary)<1e-8),
            scope='Actual filled ideal-convex triangle point, not centroid/functional pad or pressure patch truth')
    def configure(self,entry):
        self.entry=entry;source=read(checked(self.cfg['inputs'][entry['source_input']]))
        anchor_bundle=read(checked(self.cfg['inputs'][entry['anchor_input']]))
        self.source=source;self.bundle=anchor_bundle;q=np.array(source['native_probe']['raw']['qpos27'])
        assert q.shape==(27,) and source['candidate']['assembly_gate'] and source['final_geometry']['geometry_gate_pass']
        self.q0=q[:20].copy();self.rest=q[12:20].copy();R0=self.Rotation.from_quat(q[23:27][[1,2,3,0]]).as_matrix();self.t0=self.Rp.T@(q[20:23]-self.pp)
        self.rv0=self.Rotation.from_matrix(self.Rp.T@R0).as_rotvec();self.p0=q[20:23].copy();self.R0=R0
        self.hand=[];self.phone=[]
        for i,row in enumerate(anchor_bundle['anchors']):
            hh=row['surfaces']['hand'];pp=row['surfaces']['phone'];gid=PADS[i]
            self.hand.append(self.surface(np.array(hh['projected_point_geom_local_m']),self.geom_local[gid],self.triangles[gid],self.equations[gid]))
            self.phone.append(self.surface(np.array(pp['projected_point_phone_body_m']),self.phoneV,self.phoneT,self.phoneEq))
            assert self.hand[-1]['projection_distance_m']<1e-10 and self.phone[-1]['projection_distance_m']<1e-10
        # The same local material location is projected to the actual third-tip mesh; equal topology is not assumed.
        self.hand.append(self.surface(self.hand[1]['point_local_m'],self.geom_local[41],self.triangles[41],self.equations[41]))
        self.Alocal=np.array([v['point_local_m'] for v in self.hand]);self.nlocal=np.array([v['outward_unit_normal_local'] for v in self.hand])
        x=np.r_[self.q0[:12],self.t0,self.rv0];base=self.fk(x,register=False)
        if entry['third_target_rule']=='source_middle_anchor_nearest_phone':target=self.R0.T@(base['A'][2]-self.p0)
        elif entry['third_target_rule']=='index_material_toward_COM_30mm':
            target=self.phone[1]['point_local_m'].copy();target[2]+=np.clip(float(self.m.body_ipos[self.E.oid,2])-target[2],-.03,.03)
        else:raise ValueError('Unknown named third target')
        self.phone.append(self.surface(target,self.phoneV,self.phoneT,self.phoneEq))
        self.Blocal=np.array([v['point_local_m'] for v in self.phone]);self.nPlocal=np.array([v['outward_unit_normal_local'] for v in self.phone])
        # Only mutable evaluator observation/design fields change, never the compiled model or old source.
        self.E.pads=PADS.copy();self.E.anchors=self.Alocal.copy();self.E.anchor_targets=self.Blocal.copy()
        self.x0=x;self.lo=np.r_[self.m.jnt_range[:12,0],self.t0-.035,self.rv0-math.pi/6]
        self.hi=np.r_[self.m.jnt_range[:12,1],self.t0+.035,self.rv0+math.pi/6]
        self.full_cache={};self.best_full=None;self.best_fast=None;self.cheap_entry=0;self.full_entry=0
        self.trace=(self.out/(entry['name']+'_evaluations.jsonl')).open('x')
        return x
    def fk(self,x,register=True):
        if time.perf_counter()>=self.global_deadline:raise BudgetStop('Global wall budget')
        if register and (time.perf_counter()>=self.phase_deadline or self.cheap_entry>=self.cheap_max):raise BudgetStop('Phase cheap-FK budget')
        x=np.asarray(x,dtype=float);assert x.shape==(18,) and np.isfinite(x).all()
        q=np.r_[x[:12],self.rest];assert np.all(q>=self.m.jnt_range[:20,0]) and np.all(q<=self.m.jnt_range[:20,1])
        d=self.d;R=self.Rp@self.Rotation.from_rotvec(x[15:18]).as_matrix();p=self.pp+self.Rp@x[12:15]
        d.qpos[:]=self.E.initial['qpos'];d.qpos[:20]=q;d.qpos[20:23]=p;d.qpos[23:27]=self.Rotation.from_matrix(R).as_quat()[[3,0,1,2]]
        d.qvel[:]=0;d.ctrl[:]=0;d.qfrc_applied[:]=0;d.xfrc_applied[:]=0
        self.mj.mj_kinematics(self.m,d);self.mj.mj_comPos(self.m,d)
        self.counts['cheap_FK']+=1;self.counts['cheap_comPos']+=1
        if register:self.cheap_entry+=1
        assert np.array_equal(d.xpos[self.E.pid],self.pp) and np.array_equal(d.xmat[self.E.pid].reshape(3,3),self.Rp)
        GR=np.array([d.geom_xmat[g].reshape(3,3) for g in PADS]);GP=np.array([d.geom_xpos[g] for g in PADS])
        A=np.einsum('nij,nj->ni',GR,self.Alocal)+GP;nh=np.einsum('nij,nj->ni',GR,self.nlocal)
        return dict(x18=x.copy(),q20=q,qpos27=d.qpos.copy(),R=R,p=p,COM=d.xipos[self.E.oid].copy(),A=A,nH=nh,GR=GR,GP=GP)
    def residual(self,x):
        k=self.fk(x);B=self.Blocal@k['R'].T+k['p'];nP=self.nPlocal@k['R'].T
        closure=B-k['A']+.000025*k['nH']
        v=np.r_[closure.ravel()/.001,.25*(k['nH']+nP).ravel(),.02*(x[:12]-self.q0[:12])/.2,
            .03*(x[12:15]-self.t0)/.01,.03*(x[15:18]-self.rv0)/.25]
        objective=float(v@v)
        if self.best_fast is None or objective<self.best_fast[0]:self.best_fast=(objective,np.asarray(x).copy())
        return v
    def full(self,x):
        key=np.asarray(x,dtype=float).tobytes()
        if key in self.full_cache:return self.full_cache[key]
        if self.full_entry>=self.cfg['budget']['full_queries_per_entry']:raise BudgetStop('Full-state query budget')
        k=self.fk(x);s=self.E.evaluate(k['q20'],x[12:15],x[15:18],exact_all=False)
        self.full_entry+=1;self.counts['full_geometry_evaluations']+=1
        assert np.max(abs(s['qpos27']-k['qpos27']))<1e-13
        vals=[];labels=[];unknown=s['unknown'].copy()
        assert len(s['phone_geometry_rows'])==22 and len(s['self_geometry_rows'])==189
        for row in s['phone_geometry_rows']:
            if row['geom_id'] in PADS:
                assert row['status']=='EXACT_NUMERICAL_IDEAL_CONVEX' # all3 E.pads force exact even when far
                vals.extend([row['lower_bound_m']+.0001,.0002-row['upper_bound_m']]);labels.extend([row['name']+'_pad_lower',row['name']+'_pad_upper'])
            else:vals.append(row['lower_bound_m'] if row['status']!='UNKNOWN' else -1.);labels.append(row['name']+'_nonpad')
        for row in s['self_geometry_rows']:vals.append(row['lower_bound_m'] if row['status']!='UNKNOWN' else -1.);labels.append('/'.join(row['names'])+'_self')
        assert len(vals)==214
        translation=float(np.linalg.norm(k['p']-self.p0));angle=float(self.Rotation.from_matrix(k['R']@self.R0.T).magnitude())
        vals.extend([.035-translation,math.pi/6-angle]);labels.extend(['functional_phone_translation_trust_m','functional_phone_rotation_trust_rad'])
        values=np.array(vals);passed=not unknown and bool(np.all(values>=0));objective=float(np.sum(self.residual(x)**2))
        result=dict(k=k,state=s,values=values,labels=labels,unknown=unknown,passed=passed,objective=objective,
            translation_from_source_m=translation,angle_from_source_rad=angle)
        rank=(max(0.,-float(values.min()))+(1. if unknown else 0.),objective)
        if self.best_full is None or rank<self.best_full[0]:self.best_full=(rank,np.asarray(x).copy())
        self.trace.write(json.dumps(clean(dict(stage=self.phase,x18=x,objective=objective,worst_constraint=float(values.min()),unknown=unknown,geometry_passed=passed,counts=self.counts)))+'\n');self.trace.flush()
        self.full_cache[key]=result
        if len(self.full_cache)>8:self.full_cache.pop(next(iter(self.full_cache)))
        if len(self.E.query_cache)>512:self.E.query_cache=dict(list(self.E.query_cache.items())[-128:]);self.counts['query_cache_prunes']+=1
        return result
    def final_candidate(self,x,entry_id):
        # A fresh full geometry evaluation after search; no cache-as-final qualification.
        self.full_cache.pop(np.asarray(x,dtype=float).tobytes(),None);f=self.full(x);k=f['k'];s=f['state'];bindings=[]
        bygeom={v['geom_id']:v for v in s['phone_geometry_rows']}
        for j,gid in enumerate(PADS):
            hpoint=k['A'][j];bodypoint=k['R'].T@(hpoint-k['p']);nearest=self.surface(bodypoint,self.phoneV,self.phoneT,self.phoneEq)
            query=bygeom[gid]['query_phone_frame'];features=self.E._world_features(query,k['R'],k['p']);force=-np.asarray(features['normal_world'])
            hand=self.hand[j];phone=self.phone[j]
            bindings.append(dict(hand_geom_id=gid,hand_body_id=int(self.m.geom_bodyid[gid]),hand_anchor_world_m=hpoint,
                hand_outward_unit_normal_world=k['nH'][j],force_on_object_unit_normal_world=force,
                geom_local_hand_anchor_m=hand['point_local_m'],geom_local_force_on_object_normal=k['GR'][j].T@force,
                hand_surface=hand,phone_surface=phone,phone_body_material_point_m=phone['point_local_m'],phone_body_material_outward_normal=phone['outward_unit_normal_local'],
                phone_material_world_m=k['R']@phone['point_local_m']+k['p'],phone_nearest_to_hand_anchor_surface=nearest,
                query_gap_m=bygeom[gid]['exact_gap_m'],independent_query_features_world=features,
                scope='Real filled hand material anchor for a subsequent PD proposal, not a certified native force point or finite patch; force direction is minus hand-exit query normal, local version explicitly F_on_object.'))
        a=k['A'];cross=np.cross(a[1]-a[0],a[2]-a[0]);area=.5*float(np.linalg.norm(cross));plane=None
        if area>0:
            normal=cross/(2*area);distance=float((k['COM']-a[0])@normal);projection=k['COM']-distance*normal
            uv=np.linalg.lstsq(np.c_[a[1]-a[0],a[2]-a[0]],projection-a[0],rcond=None)[0]
            plane=dict(COM_signed_plane_distance_m=distance,COM_plane_projection_world_m=projection,projection_barycentric=np.r_[1-uv.sum(),uv])
        anchors=self.bundle['anchors'];material=[]
        for j,row in enumerate(anchors):
            ego=np.array(row['original_ego_material_target_phone_body_m']);pbody=k['R'].T@(a[j]-k['p'])
            material.append(dict(finger=row['finger'],original_ego_material_target_body_m=ego,actual_hand_anchor_phone_body_m=pbody,
                anchor_to_original_ego_target_distance_m=float(np.linalg.norm(pbody-ego)),
                fixed_material_world_shift_from_source_m=float(np.linalg.norm((k['R']-self.R0)@ego+k['p']-self.p0))))
        return dict(schema='functional_tripod_geometric_reference_candidate:1',candidate_id=entry_id,
            source_entry=self.entry,source_record=self.cfg['inputs'][self.entry['source_input']],qpos27_F64=k['qpos27'],qvel26_F64=np.zeros(26),q20_F64=k['q20'],
            x18=k['x18'],fixed_qrest8_F64=self.rest,designated_geom_ids=PADS,ctrl20=None,control_qualified=False,
            geometry=dict(passed=f['passed'],phone_pair_count=22,self_pair_count=189,table_complete=True,unknown=f['unknown'],
                pad_window_m=[-.0001,.0002],nonpad_minimum_gap_m=0.,self_minimum_gap_m=0.,constraint_values=f['values'],constraint_labels=f['labels'],
                source_full_state=s,legacy_static_support_available_not_used=s['support_operands']['available'],
                scope='Own214 geometry constraints use all3 designated IDs regardless of old helper support len==2. AABB certificates allowed only for separated nondesignated/self pairs.'),
            pad_bindings=bindings,source_material_deviation=material,
            functional_phone_pose_deviation=dict(translation_m=f['translation_from_source_m'],angle_rad=f['angle_from_source_rad'],source_p_world_m=self.p0,source_R_world=self.R0,
                scope='35mm/30degree algorithmic search trust; not measured ego uncertainty/error bounds.'),
            lever_descriptors=dict(hand_anchor_to_COM_world3x3_m=a-k['COM'],three_anchor_triangle_area_m2=area,COM_plane=plane,
                scope='Pure material-anchor geometry, not actual native contact patch or wrench feasibility.'),
            static_or_dynamic_holding_qualified=False,ideal_LP_as_necessary_gate=False,source_functional_pad_truth='UNKNOWN')

def preflight(path):
    cfg=read(path);assert cfg['schema']=='functional_tripod_geometry_protocol:1' and type(cfg['CPU_static_run_authorized']) is bool
    assert cfg['integration_or_training_authorized'] is False and cfg['designated_geom_ids']==PADS
    for item in [*cfg['inputs'].values(),*cfg['source_pins'].values()]:checked(item)
    assert cfg['source_pins']['solver']==pin(__file__) and len(cfg['entries'])==4
    assert cfg['budget']==dict(global_wall_s=360,phaseA_wall_s=15,phaseB_wall_s=60,final_wall_s=8,cheap_FK_per_phase=1200,
        full_queries_per_entry=250,final_full_query_reserve=2,max_total_heavy_queries=20000,heavy_queries_per_entry=4500)
    out=Path(cfg['output']);assert out.is_absolute() and out.parent==H and not out.exists()
    return cfg,out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',action='store_true');a=ap.parse_args()
    cfg,out=preflight(a.protocol)
    if not a.run:print(json.dumps(dict(status='SOURCE_METADATA_ONLY_NO_MODEL',entries=4,variables=18,steps=0)));return
    assert cfg['CPU_static_run_authorized'] is True
    from scipy.optimize import least_squares,minimize
    out.mkdir();write(out/'protocol.json',cfg);started=time.perf_counter();E=None;results=[];phase='construct'
    try:
        E=Tripod(cfg,out)
        for ei,entry in enumerate(cfg['entries']):
            E.counts['entries_attempted']+=1;phase=entry['name'];x=E.configure(entry);stages={}
            E.phase='A';E.phase_deadline=min(E.global_deadline,time.perf_counter()+15);E.cheap_max=E.cheap_entry+1200
            try:
                z=least_squares(E.residual,x,bounds=(E.lo,E.hi),max_nfev=80,x_scale=np.r_[np.full(12,.2),np.full(3,.01),np.full(3,.2)],ftol=1e-8,xtol=1e-8,gtol=1e-8)
                x=z.x;stages['A']=dict(status=int(z.status),nfev=int(z.nfev),cost=float(z.cost),success=bool(z.success))
            except (BudgetStop,E.G.StaticBudgetExceeded) as err:
                x=E.best_fast[1] if E.best_fast else x;stages['A']=dict(budget_stop=repr(err),selection='Best recorded cheap objective; includes finite-difference proposal calls, no physical qualification')
            E.phase='B';E.phase_deadline=min(E.global_deadline,time.perf_counter()+60);E.cheap_max=E.cheap_entry+1200
            E.E.set_budget(deadline_monotonic=E.phase_deadline,max_additional_heavy_queries=4500)
            try:
                fun=lambda z:float(np.sum(E.residual(z)**2))
                cons=lambda z:E.full(z)['values']/np.r_[np.full(214,.001),.01,.1]
                z=minimize(fun,x,method='SLSQP',bounds=list(zip(E.lo,E.hi)),constraints=[dict(type='ineq',fun=cons)],
                    options=dict(maxiter=20,ftol=1e-8,eps=1e-6,disp=False))
                stages['B']=dict(status=int(z.status),message=str(z.message),nit=int(z.nit),success=bool(z.success))
            except (BudgetStop,E.G.StaticBudgetExceeded) as err:stages['B']=dict(budget_stop=repr(err))
            x=E.best_full[1] if E.best_full else x
            E.phase='FINAL';E.phase_deadline=min(E.global_deadline,time.perf_counter()+8);E.cheap_max=E.cheap_entry+100
            E.E.set_budget(deadline_monotonic=E.phase_deadline,max_additional_heavy_queries=500)
            # Reserve a fresh full query even if B consumed its250-search limit.
            E.full_entry=max(0,E.full_entry-2)
            candidate=E.final_candidate(x,ei);candidate['stages']=stages;candidate['counts']=dict(E.counts)
            cp=out/(entry['name']+'_candidate.json');write(cp,candidate);npz=out/(entry['name']+'_state.npz')
            with npz.open('xb') as f:np.savez_compressed(f,qpos27_F64=candidate['qpos27_F64'],qvel26_F64=candidate['qvel26_F64'])
            results.append(dict(candidate=pin(cp),state_npz=pin(npz),geometry_pass=candidate['geometry']['passed'],candidate_id=ei))
            E.counts['entries_returned']+=1;E.trace.close();print(json.dumps(dict(entry=entry['name'],geometry_pass=candidate['geometry']['passed'],source_pose_deviation=clean(candidate['functional_phone_pose_deviation']),counts=E.counts)),flush=True)
        assert E.E._parameter_digest()==E.model_initial
        write(out/'report.json',dict(schema='functional_tripod_geometric_reference_result:1',status='COMPLETED_BOUNDED_FOUR_ENTRY_GEOMETRY_SEARCH',
            protocol=pin(a.protocol),candidates=results,counts=E.counts,static_geometry_counts=E.E.counts,
            model_audit=dict(passed=True,model_digest=E.model_initial,physical_parameters_changed=False),elapsed_s=time.perf_counter()-started,
            qualified_geometry_candidates=sum(v['geometry_pass'] for v in results),holding_or_ego_retarget_qualified=False,
            scope='New functional3tip hypothesis; all4entries retained; no dynamic, force LP, training or new native forward. Physical/control qualification belongs to a separate preregistered diagnostic.'))
    except BaseException as error:
        write(out/'failure.json',dict(phase=phase,error=repr(error),completed_entries=results,counts=None if E is None else E.counts,
            static_counts=None if E is None else E.E.counts,automatic_retry=False));raise
if __name__=='__main__':main()
