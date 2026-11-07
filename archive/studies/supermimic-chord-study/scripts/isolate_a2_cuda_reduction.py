"""Diagnostic-only amax/saved-index comparison; production code is untouched."""
from pathlib import Path
import importlib.util
import inspect
import json
import os
import time

ROOT=Path(__file__).resolve().parents[1]
DEX=Path('D:/workspace/dex-supermimic-research')
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(DEX/'.cache/torchinductor/a2_geometry_probe')
os.environ['TRITON_CACHE_DIR']=str(DEX/'.cache/triton/controlled_pen')
import numpy as np
import torch
import torch._functorch.config as aot_config

def main():
    started=time.perf_counter()
    output=ROOT/'outputs/a2_cuda_reduction_isolation_v1';output.mkdir(exist_ok=False)
    spec=importlib.util.spec_from_file_location('train',ROOT/'scripts/train_controlled_pen.py')
    train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train)
    frozen=json.loads((ROOT/'outputs/a2_hook_final_cpu_20261102/source/contact_objective/frozen_inputs.json').read_text())
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    torch._dynamo.config.suppress_errors=False
    module=train.build_contact_objective(frozen,'cuda:0')
    from chord_study.contact_geometry import weighted_wrench_support
    from dex_supermimic.rl.dex_supermimic.pen import unflatten_pen_memory
    source=inspect.getsource(weighted_wrench_support)
    original='return (support*weights[..., None]).amax(-2).clamp_min(0)'
    replacement='return (support*weights[..., None]).max(dim=-2).values.clamp_min(0)'
    assert source.count(original)==1
    altered=source.replace(original,replacement).replace('def weighted_wrench_support(', 'def saved_index_support(',1)
    namespace=dict(weighted_wrench_support.__globals__);exec(altered,namespace)
    saved_index=namespace['saved_index_support']
    (output/'diagnostic_saved_index_function.py').write_text(altered,encoding='utf-8')
    repro=np.load(ROOT/'outputs/a2_inductor_gradient_repro_v2/state_memory_values_gradients.npz')
    x=torch.tensor(repro['state58'],device='cuda:0');packed=torch.tensor(repro['packed_pen_memory'],device='cuda:0')
    with torch.no_grad():g=module.geometry_proxy(x,module.bandwidth_m,return_geometry=True)
    base=[g[key].detach() for key in ('projected_contact_com_o_m','inward_normal_o','soft_contact_weight')]
    basis=module.geometry_proxy.basis;radius=module.geometry_proxy.radius
    result={'status':'running','production_source_changed':False,'diagnostic_difference':original+' -> '+replacement,
            'expected_hashes':frozen['expected_hashes'],'backward_pass_autocast':'off','measurements':[]}
    arrays={key:value.cpu().numpy() for key,value in zip(('projected','normal','weights'),base)}
    def save():
        train.write_json(output/'result.json',result)
        np.savez_compressed(output/'reproduction_arrays.npz',**arrays)
    def support_case(name,function,compiled):
        tick=time.perf_counter();inputs=[v.clone().requires_grad_() for v in base]
        fn=torch.compile(function,backend='inductor',fullgraph=True,dynamic=False) if compiled else function
        values=fn(*inputs,basis,radius)
        gradients=torch.autograd.grad(values.sum(),inputs)
        torch.cuda.synchronize()
        arrays[name+'_support']=values.detach().cpu().numpy()
        record={'name':name,'seconds':time.perf_counter()-tick,'nonfinite_gradient_counts':[]}
        for i,gradient in enumerate(gradients):
            record['nonfinite_gradient_counts'].append(int((~torch.isfinite(gradient)).sum()))
            arrays[name+'_gradient_'+str(i)]=gradient.cpu().numpy()
        result['measurements'].append(record);save();print(json.dumps(record),flush=True)
    with aot_config.patch(backward_pass_autocast='off'):
        support_case('eager_amax',weighted_wrench_support,False)
        support_case('inductor_amax',weighted_wrench_support,True)
        support_case('inductor_saved_index_max',saved_index,True)
        compiled=torch.compile(module,backend='inductor',fullgraph=True,dynamic=False)
        state=x.clone().requires_grad_();tick=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16):values,terms=compiled(state,unflatten_pen_memory(packed))
        gradient=torch.autograd.grad(values.sum(),state)[0];torch.cuda.synchronize()
        arrays['full_off_bonus']=values.detach().cpu().numpy();arrays['full_off_gradient']=gradient.cpu().numpy()
        record={'name':'full_module_outerAMP_backward_autocast_off','seconds':time.perf_counter()-tick,
                'gradient_nonfinite_count':int((~torch.isfinite(gradient)).sum()),
                'bonus_max_abs_vs_eager':float(np.max(np.abs(arrays['full_off_bonus']-repro['eager_bonus'])))}
        result['measurements'].append(record);save();print(json.dumps(record),flush=True)
    result['support_forward_max_abs_amax_vs_saved_index']=float(np.max(np.abs(arrays['inductor_amax_support']-arrays['inductor_saved_index_max_support'])))
    result['total_wall_seconds']=time.perf_counter()-started;result['status']='diagnostic_completed'
    save();print(json.dumps({'output':str(output),'wall_seconds':result['total_wall_seconds']}),flush=True)

if __name__=='__main__':main()
