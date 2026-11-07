"""Save an unchanged frozen-prior eager/Inductor gradient failure reproducer."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import time

ROOT=Path(__file__).resolve().parents[1]
DEX=Path('D:/workspace/dex-supermimic-research')
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(DEX/'.cache/torchinductor/a2_geometry_probe')
os.environ['TRITON_CACHE_DIR']=str(DEX/'.cache/triton/controlled_pen')
import numpy as np
import torch
from dex_supermimic.rl.dex_supermimic.pen import unflatten_pen_memory

def main():
    started=time.perf_counter()
    output=ROOT/'outputs/a2_inductor_gradient_repro_v1'
    output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch._dynamo.config.suppress_errors=False
    spec=importlib.util.spec_from_file_location('controlled_pen_train',ROOT/'scripts/train_controlled_pen.py')
    train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train)
    frozen=json.loads((ROOT/'outputs/a2_hook_final_cpu_20261102/source/contact_objective/frozen_inputs.json').read_text())
    module=train.build_contact_objective(frozen,'cuda:0')
    compiled=torch.compile(module,backend='inductor',fullgraph=True,dynamic=False)
    trace=np.load(ROOT/'outputs/a1l_paired_v2_clip007_2000iter_seed0_restart1/compiled/replay_control_trace.npz')
    state=torch.tensor(trace['x1'][0],device='cuda:0').repeat(32,1)
    packed=torch.tensor(trace['m'][0],device='cuda:0').repeat(32,1)
    memory=unflatten_pen_memory(packed)
    arrays={'state58':state.cpu().numpy(),'packed_pen_memory':packed.cpu().numpy()}
    result={'expected_hashes':frozen['expected_hashes'],'input_source':'A1L restart1 replay_control_trace x1/m t=0, original4 rows repeated32 times',
            'script_sha256':train.digest(Path(__file__)),'device':torch.cuda.get_device_name(),'measurements':[]}
    for name,forward,outer_amp in [('eager',module,False),('compiled_outer_AMP',compiled,True),('compiled_no_outer_AMP',compiled,False)]:
        x=state.detach().clone().requires_grad_()
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=outer_amp):
            bonus,terms=forward(x,memory)
        record={'name':name,'bonus_dtype':str(bonus.dtype),'terms':{}}
        arrays[name+'_bonus']=bonus.detach().cpu().numpy()
        for label,quantity in [('bonus',bonus),('support',terms['geometry_intent_support_match']),('penetration',terms['geometry_intent_penetration_prior'])]:
            gradient=torch.autograd.grad(quantity.sum(),x,retain_graph=True)[0]
            mask=~torch.isfinite(gradient)
            arrays[name+'_'+label+'_gradient']=gradient.detach().cpu().numpy()
            record['terms'][label]={'nan_count':int(torch.isnan(gradient).sum()),'inf_count':int(torch.isinf(gradient).sum()),
                'nonfinite_state_components':mask.sum(0).cpu().tolist(),'nonfinite_original_rows':[int(i) for i in range(4) if mask[i].any()],
                'finite_gradient_max_abs':float(gradient[~mask].abs().max()) if (~mask).any() else None,
                'value':quantity[:4].detach().cpu().tolist()}
        torch.cuda.synchronize()
        result['measurements'].append(record)
        np.savez_compressed(output/'state_memory_values_gradients.npz',**arrays)
        train.write_json(output/'result.json',result)
        print(json.dumps(record),flush=True)
    points_palm=module.geometry_proxy.hand_points_palm(state[:4,:20])
    rotation=state[:4,43:52].reshape(4,3,3)
    points_o=torch.einsum('...ji,...pj->...pi',rotation,points_palm-state[:4,None,40:43])
    sdf,normal=module.geometry_proxy.pen_sdf_normal(points_o)
    result['four_state_sampled_geometry']={'min_abs_sdf_m':sdf.abs().amin(-1).cpu().tolist(),
        'min_sdf_m':sdf.amin(-1).cpu().tolist(),'normal_norm_min':torch.linalg.vector_norm(normal,dim=-1).amin(-1).cpu().tolist()}
    arrays['points_o_first4']=points_o.detach().cpu().numpy();arrays['sdf_first4']=sdf.cpu().numpy()
    np.savez_compressed(output/'state_memory_values_gradients.npz',**arrays)
    result['total_wall_seconds']=time.perf_counter()-started
    result['status']='failure_reproduced_without_source_changes'
    train.write_json(output/'result.json',result)
    print(json.dumps({'output':str(output),'wall_seconds':result['total_wall_seconds']}),flush=True)

if __name__=='__main__':main()
