"""Fixed302-slot matched material-pyramid CPU screen; no simulation imports.

Execution is gated by a separately frozen protocol. Default only checks saved
metadata/arrays; SciPy is imported only for an explicit authorized CPU --run.
"""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,os,sys,time
import numpy as np
STARTED=time.perf_counter()
ROOT=Path(__file__).resolve().parents[1]
WRITER=Path('D:/workspace/supermimic-rough-prefix-learning-study/vendor/runtime_io.py')
WRITER_SHA='159c0d11e50017a4b5fab511e6594efb290955c5ee4a543f67374b7ed19675d1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):p=Path(p).resolve(strict=True);return {'path':str(p),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def writer():
    if sha(WRITER)!=WRITER_SHA:raise ValueError('Qualified atomic IO changed')
    sp=importlib.util.spec_from_file_location('_matched_material_cpu_atomic159c',WRITER);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def pointer(value,path):
    if not isinstance(path,str) or not path.startswith('/'):raise ValueError('Explicit contract JSON pointer required')
    for name in path[1:].split('/'):
        name=name.replace('~1','/').replace('~0','~');value=value[int(name)] if isinstance(value,list) else value[name]
    return value

def contact_basis(inward):
    n=np.asarray(inward,float)
    if n.shape!=(3,) or not np.isfinite(n).all() or abs(np.linalg.norm(n)-1)>1e-8:raise ValueError('Known unit inward normal required')
    n=n/np.linalg.norm(n);s=1. if n[2]>=0 else -1.;a=-1/(s+n[2]);b=n[0]*n[1]*a
    t1=np.array([1+s*n[0]**2*a,s*b,-s*n[0]]);t1/=np.linalg.norm(t1)
    t2=np.array([b,s+n[1]**2*a,-n[1]]);t2/=np.linalg.norm(t2)
    return np.column_stack((n,t1,t2))

def wrench_matrix(points_body,outward_body,com_body,radius,local_columns):
    """Normal-force coefficients; torsion rays retain normalF and p cross F."""
    p=np.asarray(points_body,float)-np.asarray(com_body,float)
    n=-np.asarray(outward_body,float);local=np.asarray(local_columns,float)
    if p.ndim!=2 or p.shape[1]!=3 or p.shape!=n.shape or not np.isfinite(p).all() or not np.isfinite(radius) or radius<=0:raise ValueError('Complete finite geometry required')
    if local.shape!=(6,6) or not np.isfinite(local).all() or not np.array_equal(local[0],np.ones(6)):raise ValueError('Six native-style columns must have exactly1 normal force')
    columns=[]
    for point,normal in zip(p,n):
        B=contact_basis(normal);force=B@local[:3];direct=B@local[3:]
        torque=np.cross(np.broadcast_to(point,(6,3)),force.T).T+direct
        columns.append(np.vstack((force,torque/radius)))
    return np.concatenate(columns,axis=1)

def preflight(path):
    p=read(path)
    if p['schema']!='full151_matched_actual_material_cone_CPU_protocol:1' or p['condition_order']!=['H123','H12345'] or p['slot_order']!='frame_major:frame*2+condition_index' or p['max_LP_calls']!=302:raise ValueError('Fixed paired302schedule required')
    if p['soft_worker_seconds']!=30 or p['hard_process_seconds']!=45 or p['automatic_retry'] is not False or p['GPU_used'] is not False:raise ValueError('Fixed CPU budget/no retry required')
    if p['solver_options']!={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9} or p['equality_residual_gate_N']!=1e-7:raise ValueError('Numerical contract changed')
    for value in p['source_pins'].values():
        if not Path(value['path']).is_absolute() or sha(value['path'])!=value['sha256']:raise ValueError('Missing/changed frozen file')
    for key in ('cues','old_H123_targets','demand_matrices','material_contract'):
        value=p[key]
        if not any(Path(v['path']).resolve()==Path(value['path']).resolve() and v['sha256']==value['sha256'] for v in p['source_pins'].values()):raise ValueError('Unpinned input alias '+key)
        if sha(value['path'])!=value['sha256']:raise ValueError('Input alias changed '+key)
    for value in (pin(__file__),pin(WRITER)):
        if not any(Path(x['path']).resolve()==Path(value['path']).resolve() and x['sha256']==value['sha256'] for x in p['source_pins'].values()):raise ValueError('Worker/atomic IO not frozen')
    cue=load(p['cues']['path']);old=load(p['old_H123_targets']['path']);demand=load(p['demand_matrices']['path']);material=read(p['material_contract']['path']);m=p['material_binding']
    if type(pointer(material,m['condim_pointer'])) is not int or pointer(material,m['condim_pointer'])!=4:raise ValueError('Frozen actual-material condim4 required')
    local=np.asarray(pointer(material,m['local_template_pointer']),float)
    if m['local_template_storage']=='ray_rows_by_contact_components':local=local.T
    elif m['local_template_storage']!='contact_components_by_ray_columns':raise ValueError('Template orientation must be explicit')
    if m['local_component_order']!=['force_normal','force_tangent1','force_tangent2','torque_normal','torque_tangent1','torque_tangent2']:raise ValueError('Native contact6 ordering changed')
    if local.shape!=(6,6) or not np.isfinite(local).all() or not np.array_equal(local[0],np.ones(6)) or np.any(local[4:]):raise ValueError('Normal-force-one condim4 six-column pyramid required')
    if not all(np.count_nonzero(local[1:,j])==1 for j in range(6)) or sorted(np.argmax(np.abs(local[1:]),axis=0).tolist())!=[0,0,1,1,2,2]:raise ValueError('No independent torsion or eightedge/cartesianproduct rays')
    for row in (1,2,3):
        vals=local[row,local[row]!=0]
        if len(vals)!=2 or vals[0]!=-vals[1]:raise ValueError('Each sliding/torsion component needs its signed native pair')
    if cue['finger_ids'].tolist()!=[1,2,3,4,5] or cue['landmark_indices'].tolist()!=[3,7,11,15,19]:raise ValueError('Fixed five sourcecue membership required')
    if cue['object_surface_closest_body_m'].shape!=(151,5,3) or cue['object_surface_outward_normal_body'].shape!=(151,5,3) or cue['geometry_valid_mask'].shape!=(151,5):raise ValueError('Fivecue151 ABI mismatch')
    if cue['geometry_valid_mask'].dtype!=np.bool_ or not np.array_equal(cue['time_s'],old['time_s']) or not np.array_equal(cue['mapped_object_pose_w'],old['mapped_object_pose_w']):raise ValueError('Source time/pose lineage differs')
    com=np.asarray(old['object_com_offset_body_m'],float);radius=float(old['object_nominal_enclosing_radius_m'])
    if not np.array_equal(cue['object_surface_closest_body_m'][:,:3]-com,old['point_com_body_m']) or not np.array_equal(-cue['object_surface_outward_normal_body'][:,:3],old['inward_normal_body']):raise ValueError('Matched H123 must use exactly the same first3 sourcecues')
    W=demand['required_wrench_N_equivalent']
    if W.shape!=(151,6) or not np.isfinite(W).all() or not np.array_equal(W[:,:3],demand['required_force_body_N']) or not np.array_equal(W[:,3:]*radius,demand['required_moment_body_Nm']):
        # A divide-then-multiply is not bitwise invertible. Check its roundoff,
        # while preserving the original demand matrix as the exact LP RHS.
        if W.shape!=(151,6) or not np.isfinite(W).all() or not np.array_equal(W[:,:3],demand['required_force_body_N']) or np.max(np.abs(W[:,3:]*radius-demand['required_moment_body_Nm']))>1e-12:raise ValueError('Frozen demand binding differs')
    return p,cue,local,W,com,radius,{'status':'CPU_METADATA_ARRAY_PREFLIGHT_ONLY','source_pins':len(p['source_pins']),'frames':151,'max_LP_calls':302,'condim':4,'ray_columns_per_slot':6,'normalforce_coefficients':True,'Torch_Warp_Env_imported':False,'LP_calls':0}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--output',type=Path);ap.add_argument('--run',action='store_true');a=ap.parse_args();protocol=a.protocol.resolve(strict=True)
    p,cue,local,W,com,radius,proof=preflight(protocol)
    if not a.run:print(json.dumps(proof,indent=2));return
    if p['CPU_run_authorized'] is not True:raise ValueError('Explicit Root CPU authorization required')
    out=Path(p['output']).resolve()
    if a.output is None or a.output.resolve()!=out or out.exists() or not out.is_relative_to((ROOT/'outputs').resolve()):raise ValueError('Fresh owned exact output required')
    out.mkdir(parents=True,exist_ok=False);io=writer();write=io.write_json
    runtime={'pid':os.getpid(),'parent_pid':os.getppid(),'started_utc':utc(),'plan_sha256':sha(protocol),'source_sha256':sha(__file__),'status':'initializing_CPU','GPU_used':False,'automatic_retry':False}
    write(out/'runtime.json',runtime);write(out/'config.json',p);write(out/'CPU_preflight.json',proof);(out/'execution_source.py').write_bytes(Path(__file__).read_bytes())
    from scipy.optimize import linprog
    deadline=STARTED+30;conditions=[('H123',3),('H12345',5)];rows=[];matrices={};lam={};completed=np.zeros((151,2),bool);lp_calls=0
    for name,K in conditions:
        matrices[name]=np.full((151,6,6*K),np.nan);lam[name]=np.full((151,6*K),np.nan)
    for k in range(151):
        for ci,(name,K) in enumerate(conditions):
            available=np.flatnonzero(cue['geometry_valid_mask'][k,:K]).tolist()
            rows.append({'rowid':k*2+ci,'frame':k,'time_s':float(cue['time_s'][k]),'condition':name,'available_finger_ids':[int(cue['finger_ids'][v]) for v in available],'status':'NOT_RUN','solver_status':None,'lambda_sum_normal_N':None,'lambda_min_N':None,'residual_inf_N_equivalent':None,'force_residual_inf_N':None,'torque_residual_inf_Nm':None})
            if len(available)==K:matrices[name][k]=wrench_matrix(cue['object_surface_closest_body_m'][k,:K],cue['object_surface_outward_normal_body'][k,:K],com,radius,local)
    def preserve():
        write(out/'frame_results.json',rows)
        dest=out/'matrices_and_witnesses.npz';tmp=out/('arrays_'+str(os.getpid())+'.tmp')
        with tmp.open('wb') as stream:np.savez_compressed(stream,A_H123_N_equivalent=matrices['H123'],A_H12345_N_equivalent=matrices['H12345'],lambda_H123_normal_N=lam['H123'],lambda_H12345_normal_N=lam['H12345'],required_wrench_N_equivalent=W,local_template_contact6x6=local,completed=completed,geometry_valid_mask=cue['geometry_valid_mask'])
        io.replace_with_bounded_retry(tmp,dest)
    nesting=[]
    for k in range(151):
        if cue['geometry_valid_mask'][k].all():
            same=np.array_equal(matrices['H123'][k],matrices['H12345'][k,:,:18]);nesting.append(bool(same))
            if not same:raise ValueError('H123 must be exact column subset of H12345')
    terminal='FAILED_ENGINEERING_PARTIAL';error=None
    try:
        preserve()
        for row in rows:
            if time.perf_counter()>=deadline:terminal='SOFT_BUDGET_PARTIAL';break
            k=row['frame'];ci=0 if row['condition']=='H123' else 1;name,K=conditions[ci]
            if len(row['available_finger_ids'])!=K:
                row['status']='UNKNOWN_MISSING_GEOMETRY';row['reason']='Full set cone not constructed; no partialsubset infeasibility claim';completed[k,ci]=True
            else:
                lp_calls+=1;runtime.update(status='running_CPU',rowid=row['rowid'],LP_calls=lp_calls);write(out/'runtime.json',runtime)
                result=linprog(np.ones(K*6),A_eq=matrices[name][k],b_eq=W[k],bounds=(0,None),method='highs',options=p['solver_options'])
                row.update(solver_status=int(result.status),solver_message=str(result.message),status='SOLVER_UNKNOWN')
                if result.status==2:row['status']='INFEASIBLE_DECLARED_FULL_SET'
                elif result.status==0 and result.success and result.x is not None and np.isfinite(result.x).all():
                    value=np.asarray(result.x,float);lam[name][k]=value;res=matrices[name][k]@value-W[k]
                    row.update(lambda_sum_normal_N=float(value.sum()),lambda_min_N=float(value.min()),residual_inf_N_equivalent=float(np.abs(res).max()),force_residual_inf_N=float(np.abs(res[:3]).max()),torque_residual_inf_Nm=float(np.abs(res[3:]*radius).max()))
                    if value.min()>=0 and row['residual_inf_N_equivalent']<=1e-7:row['status']='FEASIBLE_CONDITIONAL_WITNESS'
                completed[k,ci]=True
            preserve()
        else:terminal='COMPLETED_MATCHED_CONE_CPU_SCREEN'
    except BaseException as exc:error=repr(exc);raise
    finally:
        preserve();counts={name:{status:sum(row['condition']==name and row['status']==status for row in rows) for status in sorted({row['status'] for row in rows})} for name,_ in conditions}
        report={'schema':'full151_matched_actual_material_cone_CPU_report:1','status':terminal,'protocol':pin(protocol),'source':pin(__file__),'source_pins':p['source_pins'],'LP_calls':lp_calls,'max_LP_calls':302,'accounted_slots':int(completed.sum()),'condition_counts':counts,'H123_exact_subset_H12345_complete_geometry_frames':len(nesting),'all_available_nesting_exact':all(nesting),'error':error,'wall_seconds':time.perf_counter()-STARTED,'physical_reference_qualified':False,'GPU_Env_physics_WM_optimizer':False,'scope':'Same151demand and5sourcecue hypotheses; H123 uses first3 exact. Contactframe is hypotheticalFrisvad, not measuredcontact. Nativecondim4-style6columns/slot each carries normalF plus its lever moment and signed sliding/torsion. Coefficients are normal forces; no vector norm normalization, separate torsion generator, upperforcecap, muscan or retry. Infeasible is solver diagnosis for the complete declared set, not general physical impossibility; missinggeometry UNKNOWN and no partialsubset negative conclusion.'}
        write(out/'report.json',report);runtime.update(status=terminal,finished_utc=utc(),LP_calls=lp_calls,completed_slots=int(completed.sum()),wall_seconds=report['wall_seconds']);write(out/'runtime.json',runtime)
    print(json.dumps({'status':terminal,'LP_calls':lp_calls,'counts':counts,'report':pin(out/'report.json')},indent=2));raise SystemExit(0 if terminal=='COMPLETED_MATCHED_CONE_CPU_SCREEN' else 1)

if __name__=='__main__':main()
