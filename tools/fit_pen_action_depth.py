"""Fit and evaluate a paired state-only / state-action depth predictor at fixed budget."""
import argparse
import copy
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.nn import functional as F

from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.learning.world_model import atomic_checkpoint, raw_tree_equal
from supermimic_research.mechanics.pen_action_depth import inputs, ActionDepthHead
from supermimic_research.strict_json import write_json


UPDATES = 800
FIRST = 40
SEED = 8211
ARMS = ('state_only', 'state_action')
GATES = np.array([2., .2])


def read_npz(pin):
    with np.load(verify_file_pin(pin), allow_pickle=False) as z:
        return dict(z)


def tensors(data, device):
    components = [torch.as_tensor(data[k], device=device, dtype=torch.float32) for k in
                  ('state55', 'warm26', 'boundary_mm', 'pd20')]
    x = inputs(*components)
    y = torch.as_tensor(data['y_mm'], device=device, dtype=torch.float32)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all() or (y < 0).any():
        raise ValueError('Depth training requires finite inputs and nonnegative targets')
    return x,y


def grouped_indices(data, mask):
    groups = []
    for identifier in np.unique(data['state_id'][mask]):
        rows = np.flatnonzero(mask & (data['state_id']==identifier))
        rows = rows[np.argsort(data['action_id'][rows])]
        np.testing.assert_array_equal(data['action_id'][rows], np.arange(len(rows)))
        for k in ('state55','warm26','boundary_mm'):
            np.testing.assert_array_equal(data[k][rows],np.broadcast_to(data[k][rows[0]],data[k][rows].shape))
        groups.append(rows)
    return np.stack(groups)


def train_step(net, optim, x, z, indices):
    predicted = net(x[indices])
    target = z[indices]
    # All commands in one group share an exact initial state. Comparing with
    # the executed command explicitly supervises the within-state response.
    weight = 1 + (target > np.log(2.)).to(target.dtype)
    fit = (F.smooth_l1_loss(predicted,target,beta=.1,reduction='none')*weight).mean()
    response = F.smooth_l1_loss(predicted-predicted[:,:1],target-target[:,:1],beta=.1)
    loss = fit + .5*response
    optim.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True)
    optim.step()
    return loss.detach()


def metrics(actual, pred):
    unsafe, called = actual>GATES, pred>GATES
    return dict(n=len(pred), positives=unsafe.sum(0).tolist(),
        true_positive=(unsafe & called).sum(0).tolist(), false_positive=(~unsafe & called).sum(0).tolist(),
        false_negative=(unsafe & ~called).sum(0).tolist(), true_negative=(~unsafe & ~called).sum(0).tolist(),
        mae_mm=np.abs(actual-pred).mean(0).tolist())


def summarize(data, mask, predicted):
    merged = {'current_depth':data['boundary_mm'][:,-2:], **predicted}
    return {name:metrics(data['y_mm'][mask], pred[mask]) for name,pred in merged.items()}


def main(source, out, device):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    out.mkdir(parents=True,exist_ok=False)
    fit_pin, test_pin = [file_pin(source/name) for name in ('fit.npz','test.npz')]
    fit = read_npz(fit_pin)
    labels = fit['split'].astype(str)
    train, valid = labels=='train', labels=='val'
    if not train.any() or not valid.any() or not (train|valid).all():
        raise ValueError('Expected train and validation groups only in fit pack')
    if set(fit['state_id'][train]) & set(fit['state_id'][valid]):
        raise ValueError('Source leakage between fit and validation')
    np.testing.assert_array_equal(fit['y_mm'],fit['cpu_mm'])
    groups = grouped_indices(fit,train)
    write_json(out/'intent.json',dict(dataset_source=file_pin(source/'report.json'),fit=fit_pin,test=test_pin,
               seed=SEED, matched_updates=UPDATES, groups_per_update=12,actions_per_group=groups.shape[1],
               train_groups=len(groups), validation_groups=len(np.unique(fit['state_id'][valid])), device=str(device),
               widths=[109,96,96,2],learning_rate=.001,first_checkpoint=FIRST,
               code=[file_pin(__file__),file_pin(Path(__file__).resolve().parents[1]/'src/supermimic_research/mechanics/pen_action_depth.py')],
               primary='Last checkpoint in both arms. Physical thresholds 2.0 mm pen / 0.2 mm self. Compare Warp-labelled unseen trajectories and report numeric repeats separately. No evaluation-time tuning.'))
    x,y = tensors(fit,device)
    groups_t = torch.as_tensor(groups,device=device)
    ids = groups_t.flatten()
    mean = x[ids].mean(0)
    scale = x[ids].std(0,unbiased=False).clamp_min(.001)
    def make(arm):
        model = ActionDepthHead(arm=='state_action').to(device)
        opt = torch.optim.Adam(model.parameters(), lr=.001)
        return model,opt
    models = {}
    opts = {}
    for arm in ARMS:
        torch.manual_seed(SEED)
        models[arm],opts[arm] = make(arm)
        models[arm].center.copy_(mean)
        models[arm].scale.copy_(scale)
    assert raw_tree_equal(models[ARMS[0]].state_dict(),models[ARMS[1]].state_dict())
    schedule = np.random.default_rng(SEED).integers(0,len(groups),size=(UPDATES,12))
    schedule_t = torch.as_tensor(schedule,device=device)
    z = torch.log1p(y/models[ARMS[0]].gates)
    start = time.perf_counter()
    history = []
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(1,UPDATES+1):
        rows = groups_t[schedule_t[step-1]]
        losses = {name:train_step(models[name],opts[name],x,z,rows) for name in ARMS}
        if step in (1,FIRST,200,400,UPDATES):
            history.append(dict(step=step,loss={a:float(v) for a,v in losses.items()}))
        if step == FIRST:
            first = out/'first.pt'
            bundle = {a:dict(model=models[a].state_dict(),optimizer=opts[a].state_dict()) for a in ARMS}
            atomic_checkpoint(first,dict(step=step,paired=bundle,schedule=torch.from_numpy(schedule),dataset=fit_pin))
            loaded = torch.load(first,map_location=device,weights_only=True)
            restart = {a:make(a) for a in ARMS}
            assert raw_tree_equal(bundle,loaded['paired'])
            for a,(net,opt) in restart.items():
                net.load_state_dict(loaded['paired'][a]['model'])
                opt.load_state_dict(loaded['paired'][a]['optimizer'])
                # Compare a full next update, including Adam moments, on copies
                # of the same checkpoint. Preserve the uninterrupted cursor.
                direct, direct_opt = make(a)
                direct.load_state_dict(models[a].state_dict())
                direct_opt.load_state_dict(copy.deepcopy(opts[a].state_dict()))
                next_rows = groups_t[schedule_t[step]]
                train_step(direct,direct_opt,x,z,next_rows)
                train_step(net,opt,x,z,next_rows)
                assert raw_tree_equal(direct.state_dict(),net.state_dict()) and raw_tree_equal(direct_opt.state_dict(),opt.state_dict())
            elapsed = time.perf_counter()-start
            write_json(out/'first_health.json',dict(step=step,checkpoint=file_pin(first),
                model_and_adam_next_update_exact=True, loss=history[-1]['loss'],seconds=elapsed,
                remaining_fit_s_at_checkpoint=elapsed*(UPDATES-step)/step))
    if device.type=='cuda':
        torch.cuda.synchronize(device)
    fit_seconds = time.perf_counter()-start
    frozen = {}
    estimates = {}
    with torch.inference_mode():
        for arm,net in models.items():
            path = out/(arm+'.pt')
            atomic_checkpoint(path,dict(schema='pen_action_depth:1',arm=arm,updates=UPDATES,conditioned=net.action_conditioned,
                                        state_dict=net.state_dict(),fit=fit_pin))
            frozen[arm]=file_pin(path)
            estimates[arm]=net.predict_mm(x).cpu().numpy()
    freeze = out/'frozen.json'
    write_json(freeze,dict(models=frozen,executed_updates_per_arm=UPDATES,fit_seconds=fit_seconds,
        gpu_peak_allocated_B=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        history=history,validation=summarize(fit,valid,estimates)))
    # The locked native test suite is first decoded after both final weights are frozen.
    test = read_npz(test_pin)
    assert (test['split'].astype(str)=='test').all()
    test_x,_ = tensors(test,device)
    estimates = {'cpu_dynamics':test['cpu_mm']}
    for name,pin in frozen.items():
        bundle = torch.load(verify_file_pin(pin),map_location=device,weights_only=True)
        net = ActionDepthHead(bundle['conditioned']).to(device)
        net.load_state_dict(bundle['state_dict'])
        with torch.inference_mode():
            estimates[name] = net.predict_mm(test_x).cpu().numpy()
            assert np.isfinite(estimates[name]).all()
    output = out/'native_predictions.npz'
    np.savez_compressed(output,**estimates,actual_mm=test['y_mm'],phase=test['phase'],arm=test['arm'],
                        clip=test['clip'],rep=test['rep'])
    split_scores = {}
    for clip in np.unique(test['clip']):
        for rep in np.unique(test['rep'][test['clip']==clip]):
            key=f'{clip}_rep{rep}'
            split_scores[key]=summarize(test,(test['clip']==clip)&(test['rep']==rep),estimates)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),frozen=file_pin(freeze),
        first_checkpoint_health=file_pin(out/'first_health.json'),predictions=file_pin(output),native=split_scores))
    print('Frozen after',round(fit_seconds,2),'s; native test evaluated',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args = parser.parse_args()
    main(args.source.resolve(),args.out.resolve(),torch.device(args.device))
