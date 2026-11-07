"""Saved COM/velocity/load review and MJB load in the research interpreter.

This does not rerun decomposition, contacts or drop integration. Penetration
maxima are execution-time measurements, not independently reconstructible from
the compact drop trace. Source snapshots retain those executed loops.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin


def rotations(q):
    w,x,y,z=q.T;R=np.empty((len(q),3,3))
    R[:,0,0]=w*w+x*x-y*y-z*z;R[:,1,1]=w*w-x*x+y*y-z*z;R[:,2,2]=w*w-x*x-y*y+z*z
    R[:,0,1]=2*(x*y-w*z);R[:,1,0]=2*(x*y+w*z);R[:,0,2]=2*(x*z+w*y);R[:,2,0]=2*(x*z-w*y);R[:,1,2]=2*(y*z-w*x);R[:,2,1]=2*(y*z+w*x)
    return R


def run(args):
    import mujoco
    final=json.loads(args.report.read_text());meta=json.loads(args.mesh_report.read_text());prior=meta['physical_prior'];out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or final['passed']!=final['total'] or not final['qualified_numerical_asset_prototype']:raise ValueError('Qualified numerical prototype and fresh review')
    checks={};details=[]
    for drop in final['drops']:
        pin=drop['trace']
        if file_pin(pin['path'])!=pin:raise ValueError('Changed trace')
        with np.load(pin['path'],allow_pickle=False) as z:a=z['trace'].copy()
        N=len(a);dt=drop['dt_s'];q=a[:,2:9];v=a[:,9:15];R=rotations(q[:,3:7]);lever=R@np.array(prior['COM_model_m']);COM=q[:,:3]+lever
        omega=(R@v[:,3:6,None])[:,:,0];velocity=v[:,:3]+np.cross(omega,lever);tail=np.arange(N)>=round(1.5/dt)
        error=max(float(abs(COM-a[:,15:18]).max()),float(abs(velocity-a[:,18:21]).max()),float(abs(omega-a[:,21:24]).max()))
        metrics=drop['metrics'];checks[str(dt)+'_COM_velocity_frames']=error<1e-12
        checks[str(dt)+'_integer_clock_and_raw_quaternion']=N==round(2/dt) and np.array_equal(a[:,0],np.arange(1,N+1)) and np.max(abs(a[:,1]-np.arange(1,N+1)*dt))<1e-10 and np.max(abs(np.sum(q[:,3:7]**2,axis=1)-1))<1e-12
        checks[str(dt)+'_settle_and_load']=abs(float(np.sqrt(np.mean(np.sum(velocity[tail]**2,axis=1))))-metrics['tail_vCOM_RMS_m_s'])<1e-12 and abs(float(a[tail,24].mean())-prior['mass_kg']*9.81)<.05
        details.append(dict(dt_s=dt,rows=N,maximum_frame_reconstruction_error=error))
    p=[np.array(d['metrics']['final_COM_world_m']) for d in final['drops']];q=[np.array(d['final_quaternion_wxyz']) for d in final['drops']]
    distance=float(np.linalg.norm(p[0]-p[1]));angle=float(2*np.arccos(np.clip(abs((q[0]/np.linalg.norm(q[0]))@(q[1]/np.linalg.norm(q[1]))),0,1)))
    checks['drop_dt_endpoint_consistency']=distance<.002 and angle<np.deg2rad(2)
    if file_pin(final['asset_mjb']['path'])!=final['asset_mjb']:raise ValueError('Changed MJB')
    model=mujoco.MjModel.from_binary_path(final['asset_mjb']['path']);body=model.body('hot3d_mug9').id
    checks['research_environment_binary_load']=(model.nq,model.nv,model.nu)==(7,6,0) and abs(model.body_mass[body]-prior['mass_kg'])<1e-15 and np.max(abs(model.body_ipos[body]-prior['COM_model_m']))<1e-15
    checks['visual_and_aid_excluded']=model.geom_contype[model.geom('mug9_visual').id]==0 and model.geom_conaffinity[model.geom('mug9_visual').id]==0 and model.neq==model.nmocap==model.nplugin==0 and not np.any(model.body_gravcomp)
    checks['no_neural_execution']='torch' not in sys.modules
    if not all(checks.values()):raise ValueError('Saved asset review failed')
    out.mkdir(parents=True);report=dict(schema='supermimic_research.mug_asset_saved_review:1',asset_report=file_pin(args.report),mesh_report=file_pin(args.mesh_report),auditor=file_pin(__file__),
        checks={k:bool(v) for k,v in checks.items()},passed=sum(bool(v) for v in checks.values()),total=len(checks),traces=details,
        dt_pair=dict(final_COM_difference_m=distance,final_orientation_difference_rad=angle,quaternion_normalization='Descriptor only; no state repair'),
        interpreter=sys.executable,native_version=mujoco.__version__,counts=dict(binary_model_reads=1,MjData=0,forward=0,integration_steps=0,NN_FW=0,GPU_calls=0),
        scope='Independent saved POST COM/velocity reconstruction, integer clock/unit-quaternion checks, tail load/settle and endpoint consistency; MJB loaded in research interpreter. Does not independently recompute native penetration, contact forces or all geometry distances.')
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(dict(report=file_pin(out/'report.json'),passed=report['passed'],total=report['total'],dt_pair=report['dt_pair'])))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for flag in ('report','mesh-report','output'):p.add_argument('--'+flag,type=Path,required=True)
    run(p.parse_args())
