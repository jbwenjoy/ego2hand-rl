"""Resume the compound response WM with explicit frozen state and CUDA batches."""
import torch
from .compound_response import CompoundResponseWM
from .world_model import restore_rng,capture_rng


def restore_response(saved,device):
    config=saved['config'];model=CompoundResponseWM(saved['scales'],width=config['width']).to(device)
    model.load_state_dict(saved['model'],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'],fused=torch.device(device).type=='cuda')
    optimizer.load_state_dict(saved['optimizer'])
    sampler=torch.Generator(device=device);sampler.set_state(saved['sampler'])
    restore_rng(saved['rng'],device=device)
    return model,optimizer,sampler


def response_update(model,optimizer,sampler,x,y,config,*,capture_gradient=False):
    indices=torch.randint(len(x),(config['batch_size'],),generator=sampler,device=x.device)
    optimizer.zero_grad(set_to_none=True);loss=model.loss(x[indices],y[indices]);loss.backward()
    before={k:p.grad.detach().clone() for k,p in model.named_parameters()} if capture_gradient else None
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip_norm'],error_if_nonfinite=False)
    gradient=dict(pre_clip=before,post_clip={k:p.grad.detach().clone() for k,p in model.named_parameters()},clip_norm=norm.detach().clone()) if capture_gradient else None
    optimizer.step()
    return loss.detach(),indices,gradient


def response_payload(model,optimizer,sampler,parent,*,update,device):
    return dict(schema=parent['schema'],update=update,planned_updates=parent['planned_updates'],config=parent['config'],scales=parent['scales'],
        model=model.state_dict(),optimizer=optimizer.state_dict(),sampler=sampler.get_state(),rng=capture_rng(device=device),
        normalizer_fit_role=parent['normalizer_fit_role'],dataset_pins=parent['dataset_pins'],packed=parent['packed'],
        model_input_schema=parent['model_input_schema'],model_output_schema=parent['model_output_schema'],device=str(device),torch_version=torch.__version__)
