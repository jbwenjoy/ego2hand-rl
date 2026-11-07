"""One bounded analytic/saved tensor check; no native simulator or NN."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
import argparse, hashlib, importlib.util, json, math, sys
import numpy as np
import torch

H=Path(__file__).resolve().parent;P=H.parent
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def checked(p):
    assert pin(p['path'])==p,'Source/input pin changed';return Path(p['path'])
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def close(a,b,tol=1e-12):assert np.max(np.abs(np.asarray(a)-np.asarray(b)))<=tol,(a,b)
def fixtures(G):
    D=torch.cat((torch.eye(3,dtype=torch.float64),-torch.eye(3,dtype=torch.float64)))
    signs=torch.tensor([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]],dtype=torch.float64)
    P0=signs;rows=[];calls=0
    def gap(V,dirs=D):
        nonlocal calls;calls+=1;return G.vertex_plane_gaps(P0,V,dirs).max()
    separated=signs*.5+torch.tensor([1.8,0.,0.],dtype=torch.float64)
    close(gap(separated).item(),.3)
    Q=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],dtype=torch.float64)
    calls+=1;close(G.vertex_plane_gaps(P0@Q.T,separated@Q.T,D@Q.T).max().item(),.3)
    rows.append(dict(name='separated_boxes_and_common_rigid_rotation',expected_m=.3))
    overlap=signs*.5+torch.tensor([1.3,0.,0.],dtype=torch.float64)
    close(gap(overlap).item(),-.2);limited=gap(overlap,D[1:3]).item();assert limited<-.2
    rows.append(dict(name='penetrating_boxes_finite_directions_more_negative',full_axes_m=-.2,limited_m=limited))
    corner=signs+torch.tensor([3.,3.,0.],dtype=torch.float64)
    close(gap(corner).item(),1.);diag=torch.cat((D,torch.tensor([[1.,1.,0.]],dtype=torch.float64)))
    close(gap(corner,diag).item(),math.sqrt(2))
    rows.append(dict(name='corner_separation_is_lower_bound_not_exact_SDF',axes_m=1.,with_diagonal_m=math.sqrt(2)))
    center=torch.tensor([1.5,0.,0.],dtype=torch.float64,requires_grad=True)
    segment=center+torch.tensor([[0.,0.,-.4],[0.,0.,.4]],dtype=torch.float64)
    calls+=1;caps=G.capsule_plane_gaps(P0,segment,.2,D).max();close(caps.item(),.3)
    grad=torch.autograd.grad(caps,center)[0];close(grad.numpy(),[1.,0.,0.])
    rows.append(dict(name='capsule_analytic_support_and_translation_AD',gap_m=.3,gradient=grad.tolist()))
    return rows,calls

def run(protocol):
    p=read(protocol)
    assert p['schema']=='phone_directional_gap_bounded_tensor_contract:1' and p['pure_tensor_geometry_authorized'] is True and p['native_NN_training_physics_authorized'] is False
    for item in [*p['sources'].values(),*p['inputs'].values()]:checked(item)
    O=Path(p['output']);O.mkdir(exist_ok=False)
    G=module('bounded_directional_gap',p['sources']['provider']['path'])
    cat=read(p['inputs']['catalog']['path']);provider=G.DirectionalGapProvider(p['inputs']['catalog']['path'])
    torch.set_num_threads(4);rng=torch.get_rng_state().clone()
    analytic,support_calls=fixtures(G)
    codec_path=P/'phone_transition_dataset_v1/transition_codec_v1.py'
    assert pin(codec_path)['sha256']=='1bf33a7fd27ae51d38a71c1f7e095dbe3047c4828ee6e65c199117bf536f4cac'
    codec=module('frozen_saved_phone_codec',codec_path)
    meta=provider.provider.meta;ipos=np.array(meta['phone_body_ipos']);arrays={};summaries=[];FD0=None
    # Direct saved-coordinate check, independent of query values being loose.
    with np.load(p['inputs']['reference_compiled_geometry']['path'],allow_pickle=False) as z:
        pb=provider.provider.geometry_by_id[71]
        raw=provider.provider.geometry_arrays[pb['vertices_key']].numpy().astype(np.float64)
        phone_body=raw@np.array(meta['geom_R'][71]).T+np.array(meta['geom_pos'][71])
        close(phone_body,z['phone_body_vertices'])
        for gid in provider.provider.pair_geom_ids:
            b=provider.provider.geometry_by_id[gid]
            if b['kind']=='mesh':close(provider.provider.geometry_arrays[b['vertices_key']].numpy(),z[f'geom{gid}_vertices'])
    for j,key in enumerate(['state_exact','state_source1mm']):
        c=read(p['inputs'][key]['path']);assert c['geometry']['passed'] is True and c['compression_target_pass'] is True
        x=codec.actual_state(c['qpos27_F64'],c['qvel26_F64'],ipos);arrays[key+'_state58_F64']=x
        exact={r['geom_id']:r for r in c['geometry']['source_full_state']['phone_geometry_rows']}
        results={}
        for dt in [torch.float64,torch.float32]:
            t=torch.tensor(x,dtype=dt,requires_grad=(j==0 and dt==torch.float64))
            f=provider.features(t);G.validate_features(f);assert f['directional_gap_known'].all()
            label='F64' if dt==torch.float64 else 'F32';value=f['directional_gap_m'].detach().numpy()
            arrays[key+'_directional_gap_'+label]=value
            arrays[key+'_guard_'+label]=f['numeric_heuristic_guard_m'].detach().numpy()
            arrays[key+'_argmax_direction_BODY3_'+label]=f['maximizing_plane_direction_BODY3'].detach().numpy()
            rows=[]
            for i,gid in enumerate(f['pair_geom_ids']):
                row=exact[gid];ex=row.get('exact_gap_m');known=row['status']=='EXACT_NUMERICAL_IDEAL_CONVEX' and ex is not None
                delta=float(value[i]-ex) if known else None
                if known:assert delta<=p['reference_comparison']['F64_arithmetic_allowance_m' if dt==torch.float64 else 'F32_diagnostic_allowance_m'],(key,label,gid,delta)
                rows.append(dict(geom_id=gid,name=f['pair_names'][i],descriptor_m=float(value[i]),source_query_status=row['status'],source_exact_gap_m=ex if known else None,descriptor_minus_source_exact_m=delta,guard_m=float(f['numeric_heuristic_guard_m'][i].detach()),exact_gap_known=False))
            results[label]=rows
            if j==0 and dt==torch.float64:
                i=f['pair_geom_ids'].index(13);AD=torch.autograd.grad(f['directional_gap_m'][i],t)[0].detach().numpy();FD0=(x,i,AD)
                arrays['exact_descriptor_geom13_AD_state58_F64']=AD
        summaries.append(dict(state=key,source=p['inputs'][key],pairs22=results))
    x,i,AD=FD0;h=p['AD']['FD_h_rad'];fdvals=[]
    for sign in [1.,-1.]:
        xx=x.copy();xx[0]+=sign*h;ff=provider.features(torch.tensor(xx,dtype=torch.float64));fdvals.append(float(ff['directional_gap_m'][i]))
    fd=(fdvals[0]-fdvals[1])/(2*h);assert np.isfinite(AD).all() and math.isfinite(fd)
    derivative=dict(pair_geom_id=13,q_index=0,FD_h_rad=h,AD_per_rad=float(AD[0]),central_FD_per_rad=fd,absolute_difference=float(abs(fd-AD[0])),scope='Derivative of finite-direction descriptor only, may be piecewise and does not validate physical action/force derivatives')
    assert provider.counts['descriptor_calls']==provider.counts['pure_Torch_FK_calls']==6
    assert torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
    assert not any(k in sys.modules for k in ['mujoco','mujoco_warp','warp'])
    with (O/'descriptor_arrays.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    write(O/'pair_comparison44.json',summaries)
    checks={k:True for k in ['four_closed_form_support_fixtures','compiled_geom_once_and_BODY_COM_frame','all22_pairs_two_named_saved_states_kept','F64_directional_values_not_above_saved_exact_with_numeric_allowance','FP32_diagnostic_scope_and_separate_masks','pure_tensor_descriptor_AD_and_FD_finite','six_Torch_FK_no_native_NN_physics','Torch_rng_unchanged_CPU_only']}
    report=dict(schema='phone_directional_gap_bounded_tensor_result:1',status='PASS_BOUNDED_DIRECTIONAL_DESCRIPTORS_NOT_EXACT_GAP_OR_CONTACT_TRUTH',protocol=pin(protocol),catalog=p['inputs']['catalog'],sources=p['sources'],inputs=p['inputs'],checks=checks,passed=len(checks),total=len(checks),direction_count=cat['direction_count'],analytic_fixtures=analytic,descriptor_derivative=derivative,
        counts=dict(provider.counts,analytic_fixture_cases=4,analytic_support_evaluations=support_calls,pure_geometry_AD_calls=2,new_native_model_loads=0,new_native_FK=0,new_native_forward=0,new_physics_steps=0,new_NN_calls=0,training_updates=0,GPU_calls=0),
        numerical_comparison_scope=p['reference_comparison'],exact_gap_qualified=False,formal_floating_certificate=False,full_H_zero_contact_certificate=False,
        outputs={f.name:pin(f) for f in O.iterdir()},scope='Current ideal compiled convex supports only. Saved exact bounds are source-backed comparison evidence; no fresh exact query. Finite directional descriptor can be loose/more negative, maximizing plane is not unique closest/contact normal. No causal/Markov/force/dynamics certificate.')
    write(O/'report.json',report);return pin(O/'report.json')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,default=H/'protocol_v1.json');ap.add_argument('--run',action='store_true');a=ap.parse_args()
    p=read(a.protocol)
    if not a.run:
        for item in [*p['sources'].values(),*p['inputs'].values()]:checked(item)
        print(json.dumps(dict(status='METADATA_ONLY_NO_TENSOR_FK',pairs=22,states=2,protocol=pin(a.protocol))));return
    assert not (H/'failure_v1.json').exists() and not Path(p['output']).exists(),'Fresh bounded once check'
    try:print(json.dumps(run(a.protocol)))
    except BaseException as e:
        write(H/'failure_v1.json',dict(error=repr(e),source=pin(Path(__file__)),automatic_retry=False,new_native_calls=0));raise
if __name__=='__main__':main()
