"""Measure one-step and short open-loop model errors on separate native simulator worlds."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.spatial.transform import Rotation

from supermimic_research.artifacts import file_pin
from supermimic_research.learning.flash_imagination import Dynamics, NativeSteps, fit_scales, model_update, predictions, sequence_model_update
from supermimic_research.strict_json import write_json


def check(path, output, updates, fit_horizon):
    start = perf_counter()
    with np.load(path, allow_pickle=False) as z:
        data = NativeSteps(*(jnp.asarray(z[k]) for k in (
            "obs","critic_context","action","next_obs","next_critic_context","objective_terms","terminated","truncated")))
        weights, dt = z["weights"], float(z["dt"])
        voc = z["applied_voc"] if "applied_voc" in z.files else z["applied_voc_scale"]
        no_assist = bool(np.all(voc == 0))
    receipt = json.loads(path.with_name("report.json").read_bytes())
    blocks = {b["name"]: slice(b["start"],b["end"]) for b in receipt["policy_schema"]["observation"]["blocks"]}
    t, worlds = data.action.shape[:2]
    if worlds < 4 or t < 4:
        raise ValueError("At least four worlds and four contiguous controls required for this split")
    evaluation = np.arange(worlds) % 4 == 0
    train_ids, eval_ids = np.flatnonzero(~evaluation), np.flatnonzero(evaluation)
    def grouped(ids):
        return jax.tree.map(lambda x: x[:, ids].reshape((-1, *x.shape[2:])), data)
    train = grouped(train_ids)
    scales = fit_scales(train)
    module = Dynamics(train.targets().shape[-1])
    params = module.init(jax.random.key(51), train.state()[:1], train.action[:1])["params"]
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(3e-4))
    optim = optimizer.init(params)
    def update(params, opt_state, indices):
        batch = jax.tree.map(lambda x: x[indices], train)
        return model_update(module, optimizer, params, opt_state, scales, batch)
    def unrolled_update(params, opt_state, indices):
        windows = indices[None,:] + jnp.arange(fit_horizon)[:,None]*len(train_ids)
        sequence = jax.tree.map(lambda x: x[windows],train)
        return sequence_model_update(module,optimizer,params,opt_state,scales,sequence)
    if fit_horizon != 1:
        update = unrolled_update
    update = jax.jit(update)
    predict = jax.jit(lambda p,x,u: predictions(module,p,scales,x,u))
    rng = np.random.default_rng(51)
    for _ in range(updates):
        ids = rng.integers(0, (t-fit_horizon+1)*len(train_ids), 128, dtype=np.int32)
        params, optim, training_loss = update(params, optim, ids)
    all_metrics = {}
    for steps in (1,3):
        n = t - steps + 1
        def flat(a, offset):
            v = a[offset:offset+n,eval_ids]
            return v.reshape((-1,*a.shape[2:]))
        initial = jnp.concatenate((flat(data.observation,0),flat(data.context,0)),axis=-1)
        x = initial
        valid = np.ones(len(initial), dtype=bool)
        for offset in range(steps):
            x, terms, _ = predict(params, x, flat(data.action,offset))
            if offset < steps - 1:
                valid &= ~np.asarray(flat(data.terminated | data.truncated,offset), dtype=bool)
        target = np.column_stack((flat(data.next_observation,steps-1),flat(data.next_context,steps-1)))
        if not valid.any():
            all_metrics[str(steps)] = {"world_windows": 0}
            continue
        width = train.observation.shape[1] - 3
        # Geometry extents are static; score only the native policy observation channels.
        target = target[valid,:width]
        scale = np.asarray(scales.state_scale)[:width]
        error = (np.asarray(x)[valid,:width] - target) / scale
        fixed = (np.asarray(initial)[valid,:width] - target) / scale
        truth_reward = np.asarray(flat(data.terms, steps-1))[valid] @ weights * dt
        reward = np.asarray(terms)[valid] @ weights * dt
        pred, baseline = np.asarray(x)[valid,:width], np.asarray(initial)[valid,:width]
        pos, quat = blocks["object_position_e_m"], blocks["object_orientation_e_wxyz"]
        def object_metrics(values):
            distance = np.linalg.norm(values[:,pos] - target[:,pos],axis=1)
            delta = Rotation.from_quat(values[:,quat],scalar_first=True).inv()*Rotation.from_quat(target[:,quat],scalar_first=True)
            return dict(position_rms_mm=float(1000*np.sqrt(np.mean(distance**2))),
                        rotation_rms_deg=float(np.rad2deg(np.sqrt(np.mean(delta.magnitude()**2)))),
                        quaternion_norm_rmse=float(np.sqrt(np.mean((np.linalg.norm(values[:,quat],axis=1)-1)**2))))
        all_metrics[str(steps)] = dict(world_windows=int(valid.sum()),
            object_model=object_metrics(pred), object_static=object_metrics(baseline),
            scaled_observation_rmse=float(np.sqrt(np.mean(error**2))),
            frozen_state_rmse=float(np.sqrt(np.mean(fixed**2))),
            reward_rmse=float(np.sqrt(np.mean((reward-truth_reward)**2))),
            reward_train_mean_rmse=float(np.sqrt(np.mean((np.asarray(train.terms.mean(0))@weights*dt-truth_reward)**2))))
    write_json(output,dict(data=file_pin(path), implementation=file_pin(__file__), backend=jax.default_backend(),
                           updates=updates, fit_horizon=fit_horizon, train_world_ids=train_ids.tolist(), validation_world_ids=eval_ids.tolist(),
                           label_applied_voc_verified_zero=bool(no_assist), metrics=all_metrics,
                           last_train_loss=float(training_loss[0]),wall_s=perf_counter()-start))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data",type=Path)
    p.add_argument("result",type=Path)
    p.add_argument("--updates",type=int,default=240)
    p.add_argument("--fit-horizon",type=int,choices=(1,3),default=1)
    args=p.parse_args()
    check(args.data,args.result,args.updates,args.fit_horizon)
