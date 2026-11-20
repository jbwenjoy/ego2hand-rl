"""Measure one-control penetration forecasts against new-clip Warp traces."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.mechanics.controlled_pen_records import native_state55
from supermimic_research.mechanics.pen_substep_pose import cv_qpos, load_substep_pose
from supermimic_research.strict_json import write_json


ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS_MM = np.array([2., .2])


def read(pin):
    return json.loads(verify_file_pin(pin).read_bytes())


def sources():
    runs = ROOT/'runs'
    specs = [
        ('clip_003_handoff', runs/'pen_prefix_digit_handoff_v1/report.json', None),
        ('clip_003_balanced', runs/'pen_start_prefix_balanced_w1_v1/report.json', ('base',)),
        ('clip_002', runs/'pen_dev_init_compare_v1/refit002/report.json', None),
    ]
    result = []
    for split, path, keep in specs:
        pin = file_pin(path)
        j = read(pin)
        if 'arms' in j:
            rows = [(arm, c) for arm, a in j['arms'].items() if keep is None or arm in keep
                    for c in a['cases']]
        else:
            rows = [(c['arm'], c) for c in j['rows']]
        for arm, c in rows:
            result.append(dict(split=split, arm=arm, rep=c['rep'], report=pin, case=c['case']))
    return result


def penetration_mm(data, pen_geoms, hand_geoms):
    pairs = data.contact.geom
    depths = np.maximum(-np.asarray(data.contact.dist), 0.)
    pen = pen_geoms[pairs[:, 0]] != pen_geoms[pairs[:, 1]]
    hand = hand_geoms[pairs].all(axis=1)
    return 1000*np.array([depths[pen].max(initial=0.), depths[hand].max(initial=0.)])


def geometry_peaks(model, data, paths, masks):
    peaks = np.zeros((len(paths), 2))
    for i, path in enumerate(paths):
        for qpos in path:
            data.qpos[:] = qpos
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            mujoco.mj_collision(model, data)
            np.maximum(peaks[i], penetration_mm(data, *masks), out=peaks[i])
    return peaks


def simulate(model, starts, velocities, warm, pd, mount, masks):
    data = mujoco.MjData(model)
    peaks = np.zeros((len(starts), 2))
    for i in range(len(starts)):
        mujoco.mj_resetData(model, data)
        data.qpos[:], data.qvel[:], data.qacc_warmstart[:] = starts[i], velocities[i], warm[i]
        data.mocap_pos[:], data.mocap_quat[:] = mount
        data.time = .02*i
        data.ctrl[:] = pd[i]
        for _ in range(8):
            mujoco.mj_step(model, data)
            np.maximum(peaks[i], penetration_mm(data, *masks), out=peaks[i])
    return peaks


def evaluate_case(model, nets, case, masks, anchors):
    folder = Path(case['case']['path']).parent
    exported = folder/'cpu_export/report.json'
    if exported.exists():
        receipt = json.loads(exported.read_bytes())
        assert receipt['model']['sha256'] == anchors['model']['sha256']
        phys = verify_file_pin(receipt['physical_context'])
    else:
        phys = folder/'physical_context.npz'
        assert file_pin(folder/'model.mjb')['sha256'] == anchors['legacy']['sha256']
    assert file_pin(phys)['sha256'] == anchors['physics']['sha256']
    with np.load(verify_file_pin(read(case['case'])['trace'])) as trace:
        qpaths = trace['pre_qpos'].reshape(-1,8,27)
        starts = qpaths[:,0]
        v = trace['pre_qvel'][::8]
        raw_pd = trace['post_ctrl'].reshape(-1,8,20)
        pd = raw_pd[:,0]
        np.testing.assert_array_equal(raw_pd, np.broadcast_to(pd[:,None], raw_pd.shape))
        warm = np.vstack([np.zeros(model.nv), trace['post_qacc_warmstart'][7:-1:8]])
        mount = []
        for k in ('post_mocap_pos','post_mocap_quat'):
            array = trace[k]
            np.testing.assert_array_equal(array, np.broadcast_to(array[0], array.shape))
            mount.append(array[0])
        actual = 1000*np.column_stack([trace[k].reshape(-1,8).max(axis=1) for k in
            ('pre_penetration_m', 'pre_self_penetration_m')])
        assert not trace['post_xfrc_applied'].any() and not trace['post_qfrc_applied'].any()
    data = mujoco.MjData(model)
    data.mocap_pos[:], data.mocap_quat[:] = mount
    tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64)
    states = tensor([native_state55(q, vel, np.arange(7,27), np.arange(6,26), 0, 0)
                     for q, vel in zip(starts,v,strict=True)])
    with torch.inference_mode():
        paths = {'constant_velocity':cv_qpos(states).numpy()}
        for name, net in nets.items():
            paths[name] = net.predict_qpos(states,tensor(warm),tensor(pd)).numpy()
    for pred in paths.values():
        np.testing.assert_allclose(pred[:,0,:3], starts[:,:3], atol=2e-7, rtol=0)
        assert np.isfinite(pred).all()
    predictions = {name:geometry_peaks(model, data, path, masks) for name, path in paths.items()}
    predictions['cpu_dynamics'] = simulate(model, starts, v, warm, pd, mount, masks)
    reconstruction = geometry_peaks(model, data, qpaths, masks)
    return dict(**case, warp=actual.tolist(), reconstruction=reconstruction.tolist(),
                estimates={m:arr.tolist() for m,arr in predictions.items()})


def scores(rows):
    gold = np.concatenate([r['warp'] for r in rows])
    reconstruction = np.concatenate([r['reconstruction'] for r in rows])
    positive = gold > THRESHOLDS_MM
    result = {}
    for name in rows[0]['estimates']:
        pred = np.concatenate([r['estimates'][name] for r in rows])
        called = pred > THRESHOLDS_MM
        result[name] = dict(true_positive=(called & positive).sum(0).tolist(),
            false_positive=(called & ~positive).sum(0).tolist(),
            false_negative=(~called & positive).sum(0).tolist(),
            true_negative=(~called & ~positive).sum(0).tolist(),
            mean_abs_error_mm=abs(pred-gold).mean(0).tolist())
    return dict(controls=len(gold), cases=len(rows), positive=positive.sum(0).tolist(),
                measured_pose_geometry_max_error_mm=abs(reconstruction-gold).max(0).tolist(),
                measured_pose_geometry_disagreements=np.sum((reconstruction>THRESHOLDS_MM)!=positive,axis=0).tolist(),
                estimates=result)


def main(out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=False)
    fit = file_pin(ROOT/'runs/pen_substep_pose_fit_v1/report.json')
    model_pins = {name:m['checkpoint'] for name,m in read(fit)['models'].items() if name in ('absolute','paired')}
    nets = {name:load_substep_pose(torch.load(verify_file_pin(pin),map_location='cpu',weights_only=True))
            for name,pin in model_pins.items()}
    catalog = json.loads((ROOT/'runs/pen_refit_catalog_v2/refit.json').read_bytes())
    source = read(file_pin(ROOT/'runs/pen_scaled_train_v1/eval_02781/repeat_0/report.json'))
    anchors = dict(model=catalog['model'],legacy=catalog['source'],physics=source['physical_context'])
    selected = sources()
    write_json(out/'intent.json',dict(cases=selected,anchors=anchors,models=model_pins,fit=fit,code=file_pin(__file__),
        metrics=['object penetration','hand self penetration'],threshold_mm=THRESHOLDS_MM.tolist(),
        scope='Every executed control including the first offending one. Fresh forecast from measured action-time state, previous native warmstart (zero at initialization), and transmitted target. Old clip001 models frozen. Repetition zero is primary; other repetitions measure numerical effects. Saved substeps are used only to score and check CPU geometry.'))
    model = mujoco.MjModel.from_binary_path(str(verify_file_pin(anchors['model'])))
    body = model.body('object/cube').id
    np.testing.assert_array_equal(model.jnt_qposadr[model.actuator_trnid[:,0]], np.arange(7,27))
    np.testing.assert_array_equal(model.body_ipos[body], np.zeros(3))
    assert model.na == model.neq == 0 and np.isclose(model.opt.timestep,.0025)
    masks = (model.geom_bodyid==body,np.array([model.geom(i).name.startswith('robot/') for i in range(model.ngeom)]))
    rows = [evaluate_case(model,nets,c,masks,anchors) for c in selected]
    file = out/'forecasts.json'
    write_json(file,rows)
    groups = {}
    for rep in sorted({r['rep'] for r in rows}):
        groups[str(rep)] = {split:scores([r for r in rows if r['split']==split and r['rep']==rep])
                            for split in sorted({r['split'] for r in rows})}
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),records=file_pin(file),by_rep_and_split=groups))
    print(json.dumps(groups['0'],indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    main(parser.parse_args().out.resolve())
