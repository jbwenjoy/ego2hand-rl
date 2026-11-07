"""Whole-graph counterfactual in a probe process; no production file changes."""
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
    started=time.perf_counter();output=ROOT/'outputs/a2_whole_max_variant_v1';output.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch._dynamo.config.suppress_errors=False
    spec=importlib.util.spec_from_file_location('train',ROOT/'scripts/train_controlled_pen.py')
    train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train)
    frozen=json.loads((ROOT/'outputs/a2_hook_final_cpu_20261102/source/contact_objective/frozen_inputs.json').read_text())
    module=train.build_contact_objective(frozen,'cuda:0')
    from dex_supermimic.rl.dex_supermimic.pen import unflatten_pen_memory
    import chord_study.contact_geometry as cg
    original=cg.weighted_wrench_support
    source=inspect.getsource(original)
    assert source.count('.amax(-2)')==1
    modified=source.replace('.amax(-2)', '.max(dim=-2).values')
    namespace=dict(original.__globals__);exec(modified,namespace)
    cg.weighted_wrench_support=namespace['weighted_wrench_support']
    (output/'temporary_variant.py').write_text(modified,encoding='utf-8')
    raw=np.load(ROOT/'outputs/a2_inductor_gradient_repro_v2/state_memory_values_gradients.npz')
    base=torch.tensor(raw['state58'],device='cuda:0');memory=unflatten_pen_memory(torch.tensor(raw['packed_pen_memory'],device='cuda:0'))
    result={'production_files_changed':False,'only_probe_difference':'weighted support point amax -> saved-index max; ties choose first winner instead of mean',
            'expected_hashes':frozen['expected_hashes'],'backward_pass_autocast':'off','status':'running'}
    arrays={}
    try:
        with aot_config.patch(backward_pass_autocast='off'):
            compiled=torch.compile(module,backend='inductor',fullgraph=True,dynamic=False)
            for name,forward in [('eager_saved_max',module),('inductor_saved_max',compiled)]:
                x=base.clone().requires_grad_();tick=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.bfloat16):bonus,terms=forward(x,memory)
                gradient=torch.autograd.grad(bonus.sum(),x)[0];torch.cuda.synchronize()
                arrays[name+'_bonus']=bonus.detach().cpu().numpy();arrays[name+'_gradient']=gradient.cpu().numpy()
                result[name]={'seconds':time.perf_counter()-tick,'gradient_nonfinite_count':int((~torch.isfinite(gradient)).sum()),
                    'gradient_nonzero_finite_max':float(gradient[torch.isfinite(gradient)].abs().max()),
                    'bonus_max_abs_vs_original_eager':float(np.max(np.abs(arrays[name+'_bonus']-raw['eager_bonus'])))}
                print(json.dumps({name:result[name]}),flush=True)
        old=np.load(ROOT/'outputs/a2_cuda_reduction_isolation_v1/reproduction_arrays.npz')
        result['compiled_original_vs_savedmax_value_max_abs']=float(np.max(np.abs(arrays['inductor_saved_max_bonus']-old['full_off_bonus'])))
        result['status']='counterfactual_completed'
    finally:
        cg.weighted_wrench_support=original
        result['total_wall_seconds']=time.perf_counter()-started
        train.write_json(output/'result.json',result);np.savez_compressed(output/'values_gradients.npz',**arrays)
        print(json.dumps({'output':str(output),'seconds':result['total_wall_seconds']}),flush=True)

if __name__=='__main__':main()
