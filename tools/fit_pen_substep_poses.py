"""Fit only registered basis/sham TRAIN rows; evaluate the full TEST policy after freezing weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation,Slerp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.controlled_pen_records import native_state55
from supermimic_research.mechanics.transition_records import state_rotation
from supermimic_research.mechanics.pen_substep_pose import SCHEMA, NATIVE_DT, cv_qpos, features, SubstepPose, load_substep_pose
from supermimic_research.mechanics.pen_load_forecast import LoadForecast
from supermimic_research.mechanics.pen_imagined_load import pen_load_step
from supermimic_research.strict_json import write_json

SEED=7411
STEPS=400
READ=lambda p:json.loads(verify_file_pin(p).read_bytes())


def inputs_and_labels(row):
    with np.load(verify_file_pin(row['packet'])) as f:x=f['x'].copy();pd=f['pd'].copy()
    with np.load(verify_file_pin(row['trace'])) as f:pre=f['pre_qpos'].copy();vel=f['pre_qvel'][0].copy();w=f['initial_warmstart'].copy()
    if not np.isfinite(x).all() or not np.isfinite(pre).all() or pre.shape!=(8,27) or x.shape!=(102,):
        raise ValueError('Unexpected native pose record')
    s=native_state55(pre[0],vel,np.arange(7,27),np.arange(6,26),0,0)
    np.testing.assert_array_equal(s,x[:55]);np.testing.assert_array_equal(x[81:101],pd);np.testing.assert_array_equal(x[55:81],w)
    cv=cv_qpos(torch.from_numpy(s).double()).numpy()
    actual=Rotation.from_quat(pre[1:,[4,5,6,3]])
    nominal=Rotation.from_quat(cv[1:,[4,5,6,3]])
    correction=np.column_stack((pre[1:,7:]-cv[1:,7:],pre[1:,:3]-cv[1:,:3],(actual*nominal.inv()).as_rotvec()))
    return x,correction,pre


def training_rows(roles):
    data=[];pairs=[];counts={};sources=set()
    for role in ('TRAIN_A','TRAIN_B'):
        group=roles[role];index={}
        for row in group['rows']:
            if row['name'].startswith('mixed_'):continue
            if not (row['valid'] and not row['terminal']):continue
            source=row['source']['sha256'];sources.add(source)
            key=row['phase'],row['name']
            if key in index:raise ValueError('Duplicate source-action row')
            index[key]=len(data)
            x,y,_=inputs_and_labels(row)
            data.append((x,y,source))
        counts[role]=len(index)
        for phase in sorted({p for p,_ in index}):
            for j in group['selected']:
                a=index.get((phase,f'j{j}_minus'));b=index.get((phase,f'j{j}_plus'))
                if a is None or b is None:continue
                np.testing.assert_array_equal(data[a][0][:81],data[b][0][:81])
                assert data[a][2]==data[b][2]
                length=np.linalg.norm(data[a][0][81:101]-data[b][0][81:101])
                if length>1e-6:pairs.append((a,b,length))
    return np.array([r[0] for r in data]),np.array([r[1] for r in data]),pairs,counts,len(sources)


def fit(raw,y,pairs,out):
    data=torch.tensor(raw,dtype=torch.float32)
    x=features(data[:,:55],data[:,81:101],data[:,55:81]);y=torch.tensor(y,dtype=torch.float32)
    minus=torch.tensor([p[0] for p in pairs]);plus=torch.tensor([p[1] for p in pairs]);length=torch.tensor([p[2] for p in pairs],dtype=x.dtype)[:,None,None]
    derivative=(y[plus]-y[minus])/length
    # Separate physical units and time scales, fitted exclusively on TRAIN contrasts.
    dscale=derivative.square().mean(0).sqrt().clamp_min(1e-7)
    result={}
    for name,weight in [('absolute',0.),('paired',1.)]:
        torch.manual_seed(SEED);net=SubstepPose();net.normalize_from_train(x,y);opt=torch.optim.Adam(net.parameters(),lr=.001)
        rng=torch.Generator().manual_seed(SEED+1);hist=hashlib.sha256()
        for _ in range(STEPS):
            ai=torch.randint(len(x),(96,),generator=rng);pi=torch.randint(len(minus),(48,),generator=rng)
            for ids in (ai,pi):hist.update(ids.numpy().tobytes())
            pa,pm,pp=net(torch.cat((x[ai],x[minus[pi]],x[plus[pi]]))).split((96,48,48))
            absolute=((pa-y[ai])/net.out_scale).square().mean()
            response=(((pp-pm)/length[pi]-derivative[pi])/dscale).square().mean()
            opt.zero_grad();(absolute+weight*response).backward();opt.step()
        if not torch.isfinite(absolute+response):raise ValueError('Nonfinite fit')
        path=out/f'{name}.pt';torch.save(dict(schema=SCHEMA,state_dict=net.state_dict()),path)
        result[name]=dict(checkpoint=file_pin(path),update_count=STEPS,sampling_hash=hist.hexdigest(),response_weight=weight)
    assert result['absolute']['sampling_hash']==result['paired']['sampling_hash']
    return result


def load_interpolation(raw,root):
    load_pin=READ(file_pin(root.parent/'pen_load_forecast_v1/report.json'))['outcomes']['state']['model']
    w=torch.load(verify_file_pin(load_pin),map_location='cpu',weights_only=True)
    net=LoadForecast(torch.zeros(2,len(w['center'])),torch.zeros(2,len(w['out_center'])))
    net.load_state_dict(w);net=net.double().eval().requires_grad_(False)
    folder=root/'native/TEST'
    m=mujoco.MjModel.from_binary_path(str(folder/'model.mjb'));b=m.body('object/cube').id
    with np.load(folder/'physical_context.npz') as f:
        mass=float(f['body_mass'][0,b]);Q=Rotation.from_quat(f['body_iquat'][0,b,[1,2,3,0]]).as_matrix();I=Q@np.diag(f['body_inertia'][0,b])@Q.T;g=f['gravity'][0]
    s=torch.tensor(raw[:,:55],dtype=torch.float64);u=torch.tensor(raw[:,81:101],dtype=torch.float64)
    nxt=pen_load_step(s,u,None,net,mass,I,g)[0].numpy();result=[]
    fractions=np.arange(8)/8
    for a,b in zip(s.numpy(),nxt,strict=True):
        q=np.empty((8,27));q[:,:3]=a[40:43]+fractions[:,None]*(b[40:43]-a[40:43])
        q[:,7:]=a[:20]+fractions[:,None]*(b[:20]-a[:20])
        rots=Rotation.from_matrix(np.stack([state_rotation(a),state_rotation(b)]))
        q[:,3:7]=Slerp([0,1],rots)(fractions).as_quat()[:,[3,0,1,2]]
        result.append(q)
    return np.array(result),load_pin


def rot(q):
    return Rotation.from_quat(q[..., [4,5,6,3]].reshape(-1,4))


def pose_errors(q,true):
    delta=q-true;N=len(q)
    angle=np.rad2deg((rot(q)*rot(true).inv()).magnitude()).reshape(N,8)
    e={'com_mm':1000*np.linalg.norm(delta[:,:,:3],axis=-1),'angle_deg':angle,'joint_rad':np.sqrt(np.mean(delta[:,:,7:]**2,axis=-1))}
    return {k:dict(t1_to_t7_rms=float(np.sqrt(np.mean(x[:,1:]**2))),by_tick_rms=np.sqrt(np.mean(x**2,axis=0)).tolist()) for k,x in e.items()}


def contrast(q,truth,rows,selected):
    by_key={(a['phase'],a['name']):i for i,a in enumerate(rows) if a['valid'] and not a['terminal']}
    result={}
    for kind,names in [('basis',[f'j{j}' for j in selected]),('mixed',['mixed'])]:
        obs=[];pred=[]
        for phase in sorted({p for p,n in by_key}):
            for name in names:
                inds=[by_key.get((phase,name+s)) for s in ('_minus','_plus')]
                if None in inds:continue
                a,b=inds
                def change(z):
                    za,zb=z[a,1:],z[b,1:]
                    return np.c_[zb[:,:3]-za[:,:3],(rot(zb)*rot(za).inv()).as_rotvec(),zb[:,7:]-za[:,7:]]
                obs.append(change(truth));pred.append(change(q))
        O,P=np.array(obs),np.array(pred)
        stats={}
        for k,u in [('com',slice(0,3)),('rotation',slice(3,6)),('hand',slice(6,None))]:
            v=O[:,:,u];z=P[:,:,u];den=np.linalg.norm(v)
            stats[k]=dict(error_over_zero=float(np.linalg.norm(z-v)/den),amp_ratio=float(np.linalg.norm(z)/den),positive_dot=int(((v*z).sum(axis=(1,2))>0).sum()),n=len(obs))
        result[kind]=stats
    return result


def main(root,out):
    root=root.resolve();out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    source=file_pin(root/'report.json');master=READ(source)
    write_json(out/'intent.json',dict(source=source,seed=SEED,updates=STEPS,batch=96,pair_batch=48,lr=.001,hidden=[96,96],train=['TRAIN_A','TRAIN_B'],admission='valid, nonterminal, basis and sham. Secants require positive transmitted PD length >1e-6',
              test='All TEST rows predicted only after both fits finish. Evaluation restricted to valid nonterminal rows. Previously inspected single-reference holdout.',
              baseline='8 pre-tick poses at 0..17.5 ms. Position-CV and world-omega-CV reference; predict 7 displacement residuals with common net. Both fits also use abs endpoints in every step and same sampling order. Fixed final checkpoint; no TEST tuning.',
              implementation=[file_pin(Path(__file__)),file_pin(ROOT/'src/supermimic_research/mechanics/pen_substep_pose.py')]))
    raw,y,pairs,counts,n=training_rows(master['roles']);models=fit(raw,y,pairs,out)
    rows=master['roles']['TEST']['rows'];data=[inputs_and_labels(r) for r in rows];test=np.array([t[0] for t in data])
    valid=np.array([r['valid'] and not r['terminal'] for r in rows]);truth=np.array([t[2] for t in data])
    with torch.no_grad():
        s=torch.tensor(test[:,:55],dtype=torch.float64);w=torch.tensor(test[:,55:81],dtype=torch.float64);u=torch.tensor(test[:,81:101],dtype=torch.float64)
        variants={'constant_velocity':cv_qpos(s).numpy()}
        variants['load_interpolation'],load_pin=load_interpolation(test,root)
        for key,model in models.items():
            net=load_substep_pose(torch.load(verify_file_pin(model['checkpoint']),map_location='cpu',weights_only=True),dtype=torch.float64)
            variants[key]=net.predict_qpos(s,w,u).numpy()
    for value in variants.values():
        assert value.shape==truth.shape and np.isfinite(value).all()
        np.testing.assert_allclose(value[:,0,:3],truth[:,0,:3],atol=1e-9)
        np.testing.assert_allclose(value[:,0,7:],truth[:,0,7:],atol=1e-9)
        assert np.max((rot(value[:,0])*rot(truth[:,0]).inv()).magnitude())<1e-6
    p=out/'predictions.npz'
    np.savez_compressed(p,phase=[r['phase'] for r in rows],name=[r['name'] for r in rows],valid=valid,**variants)
    results={k:dict(absolute=pose_errors(v[valid],truth[valid]),contrasts=contrast(v,truth,rows,master['roles']['TEST']['selected'])) for k,v in variants.items()}
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),models=models,load_baseline=load_pin,training_counts=counts,training_sources=n,training_pairs=len(pairs),test_rows=len(rows),valid_rows=int(valid.sum()),predictions=file_pin(p),results=results))
    print({k:{c:round(m['t1_to_t7_rms'],6) for c,m in r['absolute'].items()} for k,r in results.items()})
    print({k:[round(m['error_over_zero'],3) for m in r['contrasts']['mixed'].values()] for k,r in results.items()})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);args=p.parse_args();main(args.root,args.out)
