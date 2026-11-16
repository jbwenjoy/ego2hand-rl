"""Finite-amplitude response supervision with separate point/pair RNG streams."""
import copy
import torch

from .impulse_stage import update_impulse
from .pen_stage import restore_arm


def finite_contrast_loss(plus,minus,target,epsilon,scale,*,include_fn=False):
    predicted=(plus-minus)/(2*epsilon[:,None])
    error=((predicted-target)/scale).square()
    existing=error[:,40:52].mean()+error[:,78:84].mean()
    return existing+error[:,84:90].mean() if include_fn else existing


@torch.no_grad()
def response_components(arm,cache):
    model=arm['model'];pred=model.prediction(torch.cat([cache['x_plus'],cache['x_minus']]));plus,minus=pred.chunk(2)
    error=(((plus-minus)/(2*cache['epsilon'][:,None])-cache['finite_contrast'])/cache['response_scale']).square()
    terms={k:float(error[:,sl].mean()) for k,sl in dict(object12=slice(40,52),wrench6=slice(78,84),Fn6=slice(84,90)).items()}
    terms['total_used']=terms['object12']+terms['wrench6']+(terms['Fn6'] if 'Fn6' in arm['response_contract']['blocks'] else 0.)
    terms['weighted_total_used']=arm['contrast_weight']*terms['total_used']
    return terms


def update_contrast(arm,cache,cfg,*,capture_gradient=False):
    if 'pair_contract' in arm and arm['pair_contract']!=cache.get('pair_contract'):raise ValueError('Response pair/cache identity mismatch before sampling')
    if 'response_contract' in arm and arm['response_contract']!=cache.get('response_contract'):raise ValueError('Response units/cache identity mismatch before sampling')
    if arm['contrast_weight']==0:
        values,point_ids,gradient=update_impulse(arm,cache,cfg,capture_gradient=capture_gradient)
        return torch.cat([values,values.new_zeros(1)]),point_ids,None,gradient
    model,opt=arm['model'],arm['optimizer'];device=cache['x'].device
    point_ids=torch.randint(len(cache['x']),(cfg['batch_size'],),generator=arm['sampler'],device=device)
    pair_ids=torch.randint(len(cache['x_plus']),(cfg['pair_batch_size'],),generator=arm['pair_sampler'],device=device)
    opt.zero_grad(set_to_none=True)
    point_loss,terms=model.supervised_loss(cache['x'][point_ids],cache['y'][point_ids],cache[arm['mode']][point_ids],auxiliary_weight=cfg['auxiliary_weight'])
    inputs=torch.cat([cache['x_plus'][pair_ids],cache['x_minus'][pair_ids]])
    predictions=model.prediction(inputs);plus,minus=predictions.chunk(2)
    response=finite_contrast_loss(plus,minus,cache['finite_contrast'][pair_ids],cache['epsilon'][pair_ids],cache['response_scale'] if 'response_contract' in arm else model.y_scale,include_fn='Fn6' in arm.get('response_contract',{}).get('blocks',[]))
    loss=point_loss+arm['contrast_weight']*response;loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'],error_if_nonfinite=False)
    gradient=dict(post_clip={k:p.grad.detach().clone() for k,p in model.named_parameters()},clip_norm=norm.detach().clone()) if capture_gradient else None
    opt.step()
    return torch.cat([loss.detach()[None],terms,response.detach()[None]]),point_ids,pair_ids,gradient


def restore_contrast_arm(saved,device):
    contract=saved.get('pair_contract')
    if saved['config'].get('study_schema') in [3,4,5]:
        if not isinstance(contract,dict) or set(contract)!={'rank','rows','index_sha256','point_bank','pair_bank','subset_bank'} or contract['rank'] not in [0,4,20] or type(contract['rows']) is not int or contract['rows']<0 or (contract['rank']==0)!=(contract['rows']==0):raise ValueError('Explicit serialized study3 pair-subset identity required')
    elif contract is not None:raise ValueError('Historical arm cannot silently acquire study3 pair identity')
    units=saved.get('response_contract')
    if saved['config'].get('study_schema')==4:
        if not isinstance(units,dict) or set(units)!={'schema','mode','weight','scale_sha256','scale_report'} or units['schema']!='physical_response_units:1' or units['mode'] not in ['POINT_Y_SCALE','TRAIN_DERIVATIVE_RMS'] or units['weight']!=saved['contrast_weight']:raise ValueError('Explicit serialized study4 response units required')
    elif saved['config'].get('study_schema')==5:
        if not isinstance(units,dict) or set(units)!={'schema','mode','weight','scale_sha256','scale_report','blocks'} or units['schema']!='physical_response_units:2' or units['mode']!='TRAIN_DERIVATIVE_RMS' or units['weight']!=saved['contrast_weight'] or units['blocks'] not in [['object12','wrench6'],['object12','wrench6','Fn6']]:raise ValueError('Explicit serialized study5 response blocks required')
    elif units is not None:raise ValueError('Historical arm cannot silently acquire study4 response units')
    arm=restore_arm(saved,device)
    arm['pair_sampler']=torch.Generator(device=device);arm['pair_sampler'].set_state(saved['pair_sampler'])
    arm['contrast_weight']=saved['contrast_weight']
    if contract is not None:arm['pair_contract']=copy.deepcopy(contract)
    if units is not None:arm['response_contract']=copy.deepcopy(units)
    return arm
