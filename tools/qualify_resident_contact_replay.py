"""GPU metric graph versus independent NumPy on archived native raw contacts.

Uploads saved inputs into a minimal native-layout buffer. No simulator step,
training, source checkpoint restoration or throughput claim is made here.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import mujoco
import numpy as np
import torch
import warp as wp
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.resident_contact_metrics import ResidentContactMetrics,vec5


FIELDS={'worldid':int,'geom':wp.vec2i,'dim':int,'dist':float,'pos':wp.vec3,'frame':wp.mat33,'friction':vec5,'efc_address':int}


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def canonical_joints(m):
    return np.array([m.joint(f'robot/right_finger{i}_joint{j}').id for i in range(1,6) for j in range(1,5)],dtype=np.int32)


def oracle(m,body,jids,bounds,packet):
    """Independent double arithmetic, no resident kernels/label helper imports."""
    c=packet['contact'];W=len(packet['qpos']);geometry=np.zeros((W,3));fn=np.zeros((W,6));other=np.zeros(W);wrench=np.zeros((W,4,6));counts=np.zeros(W,np.int32)
    q=packet['qpos'][:,m.jnt_qposadr[jids]].astype(np.float64)
    lo=bounds[:,jids,0].astype(np.float64);hi=bounds[:,jids,1].astype(np.float64)
    geometry[:,2]=np.maximum(np.maximum(lo-q,q-hi),0).max(axis=-1)
    decoded=np.zeros((len(c['dim']),6));legacy=np.zeros((W,6))
    for i,(g,h) in enumerate(c['geom']):
        w=int(c['worldid'][i]);g,h=int(g),int(h);counts[w]+=1
        gn=m.geom(g).name or '';hn=m.geom(h).name or '';og=m.geom_bodyid[g]==body;oh=m.geom_bodyid[h]==body
        if og!=oh:geometry[w,0]=max(geometry[w,0],-float(c['dist'][i]))
        if gn.startswith('robot/') and hn.startswith('robot/'):geometry[w,1]=max(geometry[w,1],-float(c['dist'][i]))
        dim=int(c['dim'][i]);addr=c['efc_address'][i]
        if addr[0]>=0:
            width=1 if dim==1 else (2*(dim-1) if packet['cone']==0 else dim)
            ids=addr[:width]
            if len(ids)!=width or np.any(ids<0) or np.any(ids>=c['nefc'][w]):raise ValueError('Invalid active raw EFC input')
            values=c['efc_force'][w,ids].astype(np.float64)
            if packet['cone']==0 and dim>1:
                decoded[i,0]=sum(values)
                for k in range(dim-1):decoded[i,k+1]=(values[2*k]-values[2*k+1])*float(c['friction'][i,k])
            else:decoded[i,:dim]=values
        if og==oh:continue
        sign=-1. if og else 1.;name=hn if og else gn;normal=max(0.,decoded[i,0]);robot=name.startswith('robot/')
        if robot:
            digit=next((j-1 for j in range(1,6) if name.startswith(f'robot/right_finger{j}_')),5);fn[w,digit]+=normal
        else:other[w]+=normal
        axes=c['frame'][i].astype(np.float64);F=sign*(axes.T@decoded[i,:3]);direct=sign*(axes.T@decoded[i,3:])
        arm=c['pos'][i].astype(np.float64)-c['COM_efc_phase_world'][w].astype(np.float64)
        world=np.r_[F,direct+np.cross(arm,F)];wrench[w,0]+=world
        if robot:wrench[w,1]+=world
        # Preserve the archived NumPy reader's float32 arm subtraction separately.
        old_arm=c['pos'][i]-c['COM_efc_phase_world'][w]
        legacy[w]+=np.r_[F,direct+np.cross(old_arm,F)]
    for w in range(W):
        R=c['R_efc_phase_world'][w].astype(np.float64)
        for pair in range(2):
            wrench[w,pair+2,:3]=R.T@wrench[w,pair,:3];wrench[w,pair+2,3:]=R.T@wrench[w,pair,3:]
    return dict(geometry=geometry,Fn6=fn,nonhand_N=other,wrench=wrench,contacts=counts,errors=np.zeros(W,np.int32),global_error=np.zeros(1,np.int32)),decoded,legacy


class Buffer:
    def __init__(self,m,body,bounds,packets,device,*,capacity=None):
        self.m=m;self.body=body;self.W=len(packets[0]['qpos']);self.B=m.nbody
        minimum=max(len(p['contact']['dim']) for p in packets)+3
        cap=minimum if capacity is None else int(capacity)
        if cap<minimum:raise ValueError('Capacity includes active records and inactive guards')
        nj=max(p['contact']['efc_force'].shape[1] for p in packets)
        def a(x,t):return wp.array(x,dtype=t,device=device)
        def z(shape,t):return wp.zeros(shape,dtype=t,device=device)
        contact=SimpleNamespace(worldid=z(cap,int),geom=z(cap,wp.vec2i),dim=z(cap,int),dist=z(cap,float),pos=z(cap,wp.vec3),frame=z(cap,wp.mat33),friction=z(cap,vec5),efc_address=z((cap,10),int))
        self.data=SimpleNamespace(nacon=z(1,int),naconmax=cap,njmax=nj,contact=contact,nefc=z(self.W,int),efc=SimpleNamespace(force=z((self.W,nj),float)),
            xmat=z((self.W,self.B),wp.mat33),xipos=z((self.W,self.B),wp.vec3),qpos=z((self.W,m.nq),float),time=z(self.W,float),overflow=z(self.W,int))
        self.dm=SimpleNamespace(opt=SimpleNamespace(cone=packets[0]['cone']),jnt_range=a(bounds,wp.vec2));self.load(packets[0])
        self.metrics=ResidentContactMetrics(m,self.dm,self.data,body,canonical_joints(m))

    def load(self,p):
        d=self.data;c=p['contact'];n=len(c['dim']);cap=d.naconmax
        for name in FIELDS:
            target=getattr(d.contact,name);shape=target.numpy().shape
            v=np.full(shape,999999 if name in ['worldid','geom','dim','efc_address'] else np.nan,dtype=np.int32 if name in ['worldid','geom','dim','efc_address'] else np.float32)
            source=c[name];v[:n,...] = -1 if name=='efc_address' else 0
            if name=='efc_address':v[:n,:source.shape[1]]=source
            else:v[:n]=source
            target.assign(v)
        force=np.full(d.efc.force.shape,np.nan,np.float32)
        for w in range(self.W):force[w,:c['nefc'][w]]=c['efc_force'][w,:c['nefc'][w]]
        R=np.tile(np.eye(3,dtype=np.float32),(self.W,self.B,1,1));R[:,self.body]=c['R_efc_phase_world']
        C=np.zeros((self.W,self.B,3),np.float32);C[:,self.body]=c['COM_efc_phase_world']
        for array,value in [(d.nacon,[n]),(d.nefc,c['nefc']),(d.efc.force,force),(d.qpos,p['qpos']),(d.xmat,R),(d.xipos,C),(d.time,p['time'])]:array.assign(np.asarray(value,dtype=np.int32 if array.dtype==wp.int32 else np.float32))

    def fingerprint(self):
        arrays=[getattr(self.data.contact,k) for k in FIELDS]+[self.data.nacon,self.data.nefc,self.data.efc.force,self.data.qpos,self.data.xmat,self.data.xipos,self.data.time,self.data.overflow,self.dm.jnt_range,self.metrics.object_tag,self.metrics.digit,self.metrics.joint_ids,self.metrics.qids]
        h=hashlib.sha256()
        for a in arrays:h.update(a.numpy().tobytes())
        return h.hexdigest()


def empty_window(p):
    W=len(p['qpos']);return dict(geometry_max=np.zeros((W,3)),Fn6_sum=np.zeros((W,6)),nonhand_max_sum=np.zeros((W,2)),wrench_sum=np.zeros((W,8,6)),
        counts=np.zeros((W,2),np.int32),errors=np.zeros(W,np.int32),global_error=np.zeros(1,np.int32),
        source_R=p['contact']['R_efc_phase_world'].copy(),source_C=p['contact']['COM_efc_phase_world'].copy(),clock=np.repeat(p['time'][:,None].astype(np.float32).astype(np.float64),2,axis=1))


def reset_selected(v,p,ids):
    new=empty_window(p)
    for k in v:
        if k!='global_error':v[k][ids]=new[k][ids]


def fold(v,r,p):
    v['geometry_max']=np.maximum(v['geometry_max'],r['geometry']);v['Fn6_sum']+=r['Fn6']
    v['nonhand_max_sum'][:,0]=np.maximum(v['nonhand_max_sum'][:,0],r['nonhand_N']);v['nonhand_max_sum'][:,1]+=r['nonhand_N']
    v['wrench_sum'][:,:4]+=r['wrench'];v['counts'][:,0]+=1;v['counts'][:,1]+=r['contacts'];v['clock'][:,1]=p['time'].astype(np.float32)
    for w in range(len(p['qpos'])):
        R=v['source_R'][w].astype(np.float64);arm=p['contact']['COM_efc_phase_world'][w].astype(np.float64)-v['source_C'][w].astype(np.float64)
        for pair in range(2):
            F=r['wrench'][w,pair,:3];M=r['wrench'][w,pair,3:]
            v['wrench_sum'][w,pair+4]+=np.r_[R.T@F,R.T@M]
            v['wrench_sum'][w,pair+6]+=np.r_[R.T@F,R.T@(M+np.cross(arm,F))]


def evaluation_packets(case):
    m=mujoco.MjModel.from_binary_path(str(verify_file_pin(case['model'])));body=m.body('object/cube').id
    trace=dict(np.load(verify_file_pin(case['trace'])));raw=dict(np.load(verify_file_pin(case['raw_contacts'])));d=mujoco.MjData(m);packets=[]
    d.mocap_pos[:]=trace['initial_mocap_pos'].reshape(d.mocap_pos.shape);d.mocap_quat[:]=trace['initial_mocap_quat'].reshape(d.mocap_quat.shape)
    for i,q in enumerate(trace['pre_qpos']):
        d.qpos[:]=q;mujoco.mj_kinematics(m,d)
        take=raw['tick']==i;c={k:raw[k][take] for k in FIELDS if k!='worldid'};c['worldid']=np.zeros(take.sum(),np.int32)
        low,high=map(int,raw['efc_offsets'][i:i+2]);c['nefc']=np.array([high-low],np.int32);c['efc_force']=raw['efc_force'][None,low:high]
        c['COM_efc_phase_world']=d.xipos[body][None].astype(np.float32);c['R_efc_phase_world']=d.xmat[body].reshape(1,3,3).astype(np.float32);c['local_force6']=raw['local_force6'][take]
        packets.append(dict(contact=c,qpos=trace['post_qpos'][i:i+1].astype(np.float32),time=trace['post_GPU_time'][i].reshape(1).astype(np.float32),cone=case['cone']))
    return m,body,m.jnt_range.astype(np.float32)[None],packets


def run_case(name,m,body,bounds,packets,cfg,out):
    b=Buffer(m,body,bounds,packets,cfg['device']);metrics=b.metrics;checks={};maximum={};records={};hashes=[];window=None
    def compare(label,a,e):
        delta=float(np.max(abs(a.astype(np.float64)-e))) if a.size else 0.;maximum[label]=max(maximum.get(label,0.),delta)
        checks[label]=checks.get(label,True) and bool(np.allclose(a,e,atol=cfg['absolute_tolerance'],rtol=cfg['relative_tolerance']))
    for tick,p in enumerate(packets):
        b.load(p)
        if tick%cfg['window_samples']==0:metrics.reset_window();window=empty_window(p)
        if len(p['qpos'])>1 and tick==cfg['partial_reset_tick']:
            ids=np.array(cfg['partial_reset_worlds']);metrics.reset_window(torch.tensor(ids,device=cfg['device'],dtype=torch.int64));reset_selected(window,p,ids)
        before=b.fingerprint();metrics.sample();result=metrics.numpy();after=b.fingerprint();hashes.append(before)
        checks['inputs_byte_unchanged']=checks.get('inputs_byte_unchanged',True) and before==after
        ref,local,legacy=oracle(m,body,canonical_joints(m),bounds,p);fold(window,ref,p)
        compare('raw_local_force',local,p['contact']['local_force6'])
        for key in ref:compare('instant_'+key,result['instant'][key],ref[key])
        for key in window:compare('window_'+key,result['window'][key],window[key])
        for group,values in result.items():
            for key,value in values.items():records.setdefault(group+'_'+key,[]).append(value)
        if 'wrench_WORLD_COM' in p['contact']:
            compare('archived_legacy_WORLD',legacy,p['contact']['wrench_WORLD_COM'])
            error=float(abs(ref['wrench'][:,0]-legacy).max());maximum['new_vs_legacy_rounding']=max(maximum.get('new_vs_legacy_rounding',0.),error)
            checks['new_vs_legacy_rounding']=checks.get('new_vs_legacy_rounding',True) and error<=cfg['legacy_torque_tolerance']
            compare('archived_Fn6',ref['Fn6'],p['contact']['Fn6']);compare('archived_nonhand',ref['nonhand_N'],p['contact']['nonhand_N'])
    target=out/(name+'.npz');np.savez_compressed(target,**{k:np.stack(v) for k,v in records.items()},input_SHA256=np.array(hashes))
    return dict(name=name,worlds=b.W,packets=len(packets),world_samples=b.W*len(packets),active_contact_rows=sum(len(p['contact']['dim']) for p in packets),
        checks=checks,max_abs_errors=maximum,record=file_pin(target),metric_metadata=metrics.metadata(),captured_graph=metrics.graph is not None,
        partial_reset_worlds=cfg['partial_reset_worlds'] if b.W>1 else [],native_integrations=0,optimizer_steps=0)


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh replay output under runs')
    out.mkdir();sources=sorted((ROOT/'src/supermimic_research').rglob('*.py'))+[Path(__file__).resolve(),cp]
    receipt=freeze_sources(sources,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    lp=(cp.parent/cfg['evaluation_ledger']).resolve();ledger=json.loads(lp.read_text());report=json.loads(verify_file_pin(ledger['roots']['evaluation']).read_text())
    plp=(cp.parent/cfg['portable_ledger']).resolve();pl=json.loads(plp.read_text());producer=json.loads(verify_file_pin(pl['roots'][cfg['portable_case']]).read_text())
    state=torch.load(verify_file_pin(producer['checkpoint']),map_location='cpu',weights_only=False)
    continuation=torch.load(verify_file_pin(producer['continuation']),map_location='cpu',weights_only=False)
    cases=[]
    for c in report['cases']:
        m,body,bounds,packets=evaluation_packets(c);cases.append(run_case(c['name'],m,body,bounds,packets,cfg,out))
    # Same topology is verified against the producer's actual native arrays;
    # bounds come from that producer rather than a later host model's defaults.
    arrays=state['native']['model']['arrays']
    for key,expected in [('geom_bodyid',m.geom_bodyid),('jnt_qposadr',m.jnt_qposadr),('jnt_bodyid',m.jnt_bodyid)]:
        if not np.array_equal(np.asarray(arrays[key]).reshape(-1),expected):raise ValueError('Archived model topology mismatch '+key)
    packets=[dict(contact=s['contact'],qpos=s['qpos'],time=s['time'],cone=state['native']['model']['scalars']['opt.cone']) for s in continuation['substeps']]
    cases.append(run_case('W32_ARCHIVED',m,body,arrays['jnt_range'],packets,cfg,out))
    verify_source_snapshot(receipt,live=True)
    checks={c['name']+'_'+k:v for c in cases for k,v in c['checks'].items()};checks['all_CUDA_graphs']=all(c['captured_graph'] for c in cases)
    inputs=[file_pin(lp),file_pin(plp),ledger['roots']['evaluation'],pl['roots'][cfg['portable_case']],producer['checkpoint'],producer['continuation']]
    for c in report['cases']:inputs.extend([c[k] for k in ['model','trace','raw_contacts']])
    engines=[file_pin(Path(mod.__file__)) for mod in [wp,mujoco,torch]]
    result=dict(schema='resident_contact_saved_raw_replay:1',config=file_pin(cp),source_receipt=file_pin(out/'source_receipt.json'),inputs=inputs,engine_entries=engines,
        cases=cases,checks=checks,passed=sum(checks.values()),total=len(checks),packets=sum(c['packets'] for c in cases),world_samples=sum(c['world_samples'] for c in cases),
        native_integrations=0,optimizer_steps=0,CPU_kinematics_queries=sum(c['native_integrations'] for c in report['cases']),scope=cfg['scope']+' EFC geometry uses pre-step FK while joint excursion uses saved post-step qpos; no portable-state or W32-physics repeatability qualification.')
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],packets=result['packets'],world_samples=result['world_samples'])))
    if not all(checks.values()):raise ValueError('Preserve metric replay discrepancies')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
