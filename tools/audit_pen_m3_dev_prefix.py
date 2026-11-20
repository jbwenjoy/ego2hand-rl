"""Frozen M3 versus constant velocity on observed PD windows of two development clips."""
import argparse
from pathlib import Path
import json

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.mechanics.controlled_pen_records import native_state55
from supermimic_research.mechanics.pen_contact_groups import normal_group
from supermimic_research.mechanics.pen_imagined_load import pen_load_step
from supermimic_research.mechanics.pen_load_forecast import LoadForecast
from supermimic_research.mechanics.pen_task_contracts import pose_components
from supermimic_research.mechanics.transition_records import state_rotation
from supermimic_research.strict_json import write_json


ROOT = Path(__file__).resolve().parents[1] / 'runs'
HORIZONS = (1, 2, 4, 8, 16)
TRACE_KEYS = ('pre_qpos','pre_qvel','post_qpos','post_qvel','post_ctrl','post_qfrc_constraint',
              'pre_object_nonhand_normal_N','pre_hand_normal_N','post_qfrc_applied','post_xfrc_applied')


def read(pin):
    return json.loads(verify_file_pin(pin).read_bytes())


def sources():
    sources = {}
    folders = ('pen_dev_init_compare_v1/refit002','pen_start_prefix_w1_v1','pen_start_prefix_balanced_w1_v1')
    for idx, folder in enumerate(folders):
        pin = file_pin(ROOT/folder/'report.json')
        master = read(pin)
        sources[folder] = pin
        if idx == 0:
            rows = master['rows']
        else:
            rows = [dict(arm=arm, **e) for arm,g in master['arms'].items() for e in g['cases']]
        for case in rows:
            row = read(case['case'])
            path = verify_file_pin(row['trace'])
            direct = idx == 0
            group = f"002/{row['arm']}" if direct else f"003/{('original','balanced_kl')[idx-1]}/{row['arm']}"
            model = path.parent/('model.mjb' if direct else 'cpu_export/model.mjb')
            phys = path.parent/('physical_context.npz' if direct else 'cpu_export/device_parameters/physical_context.npz')
            with np.load(path, allow_pickle=False) as z:
                arrays = {k:z[k] for k in TRACE_KEYS}
            ok = np.array([c['ok'] for c in row['checks']])
            if row['stop'] == 'TASK_FAILURE':
                ok[-1] = False
            yield group,row['rep'],case['case'],row['stop'],arrays,ok,model,phys,None

    folder = ROOT/'pen_gpu_refit_preloads_v1'
    master = read(file_pin(folder/'report.json'))
    model = mujoco.MjModel.from_binary_path(str(folder/'model.mjb'))
    body = model.body('object/cube').id
    for row in master['rows']:
        if row['clip'] != 'clip_002':
            continue
        audits = read(row['report'])['audits']
        parts, groups = [], []
        for audit in audits:
            with np.load(verify_file_pin(audit['trace']), allow_pickle=False) as z:
                part = {k:z[k] for k in TRACE_KEYS if k in z}
                norms = np.zeros((8,6))
                nonhand = np.zeros(8)
                for tick, g1, g2, *rest in z['raw_contact_tick_plus23']:
                    g1,g2 = int(g1),int(g2)
                    whose = model.geom_bodyid[[g1,g2]] == body
                    if whose.sum() != 1:
                        continue
                    digit = normal_group(model.geom(g2 if whose[0] else g1).name)
                    normal = max(0., rest[15])  # original row[18], EFC-decoded normal
                    if digit < 0:
                        nonhand[int(tick)] += normal
                    else:
                        norms[int(tick),digit] += normal
                part.update(pre_hand_normal_N=norms.sum(1),pre_object_nonhand_normal_N=nonhand)
                parts.append(part)
                groups.append(norms.mean(0))
        arrays = {k:np.concatenate([part[k] for part in parts]) for k in TRACE_KEYS}
        yield (f"002/fixed_hold/{row['arm']}", row['rep'],row['report'],'FIXED_TEN_CONTROLS',arrays,
               np.array([a['valid'] for a in audits]),folder/'model.mjb',folder/'physical_context.npz',np.array(groups))


def load_model(fit):
    weights = torch.load(verify_file_pin(fit['model']), weights_only=True, map_location='cpu')
    if len(weights['center']) != 75 or len(weights['out_center']) != 52:
        raise ValueError('Expected the state-only M3 load/hand checkpoint')
    net = LoadForecast(torch.zeros(2,75),torch.zeros(2,52))
    net.load_state_dict(weights)
    return net.double().eval().requires_grad_(False)


def pose(state):
    quat = Rotation.from_matrix(state_rotation(state)).as_quat()[:,[3,0,1,2]]
    return np.c_[state[:,40:43],quat]


def endpoint_errors(predicted, observed):
    good = np.isfinite(predicted).all(1)
    mm = np.full(len(good), np.nan)
    deg = np.full((len(good),3), np.nan)
    if good.any():
        mm[good] = np.linalg.norm(predicted[good,:3] - observed[good,:3], axis=1)*1000
        angles = pose_components(predicted[good,3:], observed[good,3:])
        deg[good] = np.rad2deg(np.stack([angles[k] for k in (
            'full_angle_rad','shaft_angle_rad','reference_axis_twist_rad')],axis=1))
    return np.column_stack((mm,deg))


def get_trajectory(z, model_path, physical_path, controls):
    model = mujoco.MjModel.from_binary_path(str(model_path))
    body = model.body('object/cube').id
    jid = model.body_jntadr[body]
    qa, da = model.jnt_qposadr[jid], model.jnt_dofadr[jid]
    names = [f'robot/right_finger{digit}_joint{joint}' for digit in range(1,6) for joint in range(1,5)]
    joints = np.array([model.joint(name).id for name in names])
    aids = np.array([model.actuator(name+'_actuator').id for name in names])
    np.testing.assert_array_equal(joints, model.actuator_trnid[aids,0])
    with np.load(physical_path, allow_pickle=False) as p:
        np.testing.assert_array_equal(p['body_ipos'][0,body], [0,0,0])
        ori = Rotation.from_quat(np.roll(p['body_iquat'][0,body],-1)).as_matrix()
        physics = (float(p['body_mass'][0,body]), (ori*p['body_inertia'][0,body])@ori.T,p['gravity'][0])
    if model.nv!=26 or model.nq!=27 or model.neq!=0 or np.any(z['post_qfrc_applied']) or np.any(z['post_xfrc_applied']):
        raise ValueError('Expected a no-aid free pen')
    if len(z['pre_qpos']) != 8*controls:
        raise ValueError('Expected eight native ticks per control')
    np.testing.assert_array_equal(z['post_qpos'][:-1],z['pre_qpos'][1:])
    np.testing.assert_array_equal(z['post_qvel'][:-1],z['pre_qvel'][1:])
    pd = z['post_ctrl'][:,aids].reshape(controls,8,20)
    np.testing.assert_array_equal(pd, np.broadcast_to(pd[:,:1],pd.shape))
    pos = np.concatenate([z['pre_qpos'][::8], z['post_qpos'][-1:]])
    vel = np.concatenate([z['pre_qvel'][::8], z['post_qvel'][-1:]])
    states = np.array([native_state55(q,v,model.jnt_qposadr[joints],model.jnt_dofadr[joints],qa,da)
                       for q,v in zip(pos,vel)])
    # The pen free-body COM coincides with the joint origin. Rotate native local
    # free-joint torque back to world at each solve, then average in source axes.
    local_rot = Rotation.from_quat(z['pre_qpos'][:,qa+3:qa+7][:,[1,2,3,0]])
    gf = z['post_qfrc_constraint'][:,da:da+6]
    force_world = gf[:,:3].reshape(controls,8,3).mean(1)
    torque_world = local_rot.apply(gf[:,3:]).reshape(controls,8,3).mean(1)
    src = local_rot[::8].inv()
    truth = np.c_[src.apply(force_world),src.apply(torque_world),z['pre_hand_normal_N'].reshape(controls,8).mean(1)]
    hand_only = z['pre_object_nonhand_normal_N'].reshape(controls,8).max(1) == 0.
    return states,pd[:,0],physics,truth,hand_only


@torch.no_grad()
def predict(states,pd,physics,ok,net):
    count = len(pd)
    tensor = lambda x:torch.as_tensor(x,dtype=torch.float64,device='cpu')
    estimate = tensor(states[:-1])
    actual_pose = pose(states)
    horizons = {}
    for h in range(1,min(max(HORIZONS), count)+1):
        n = count-h+1
        estimate,_,load = pen_load_step(estimate[:n],tensor(pd[h-1:]),None,net,*physics)
        if h == 1:
            predicted_load = load.numpy()
        if h not in HORIZONS:
            continue
        predicted = np.full((n,7),np.nan)
        finite = torch.isfinite(estimate).all(-1).numpy()
        if finite.any():
            predicted[finite] = pose(estimate.numpy()[finite])
        source = states[:n]
        cv_rot = (Rotation.from_rotvec(.02*h*source[:,52:55]) * Rotation.from_matrix(state_rotation(source)))
        cv = np.c_[source[:,40:43] + .02*h*source[:,49:52],cv_rot.as_quat()[:,[3,0,1,2]]]
        truth = actual_pose[h:]
        good = np.lib.stride_tricks.sliding_window_view(ok,h).all(-1)
        horizons[h] = dict(native_pose=truth,m3_pose=predicted,cv_pose=cv,valid=good,
                           m3_error=endpoint_errors(predicted,truth),cv_error=endpoint_errors(cv,truth))
    return horizons,predicted_load


def rms(arr):
    if len(arr) == 0:
        return None
    return np.sqrt(np.mean(np.square(arr),axis=0)).tolist()


def summarize_group(cases):
    rows = {}
    for h in HORIZONS:
        paths = [case['h'][h] for case in cases if h in case['h']]
        scopes = {}
        for name in ('observed','valid_only','from_reset'):
            masks = [np.ones(len(p['valid']),bool) if name=='observed' else
                     p['valid'] if name=='valid_only' else np.arange(len(p['valid']))==0 for p in paths]
            out = dict(trajectories=sum(bool(m.any()) for m in masks),windows=sum(int(m.sum()) for m in masks))
            for model in ('m3','cv'):
                errs = [p[model+'_error'][m] for p,m in zip(paths,masks)]
                e = np.concatenate(errs) if errs else np.empty((0,4))
                finite = np.isfinite(e[:,:3]).all(1)
                out[model] = dict(finite_windows=int(finite.sum()),rms_COM_mm_SO3_deg_shaft_deg=rms(e[finite,:3]),
                                  rms_twist_deg=rms(e[np.isfinite(e[:,3]),3]))
            scopes[name] = out
        rows[str(h)] = scopes
    delta = np.concatenate([c['load_diff'] for c in cases])
    out = dict(controls=len(delta),force_vector_RMSE_N=rms(np.linalg.norm(delta[:,:3],axis=1)),
               torque_vector_RMSE_Nm=rms(np.linalg.norm(delta[:,3:6],axis=1)),
               hand_total_normal_RMSE_N=rms(delta[:,6]))
    if all(c['digit_diff'] is not None for c in cases):
        out['per_group_normal_RMSE_N'] = rms(np.concatenate([c['digit_diff'] for c in cases]))
    return dict(paths=len(cases),observed_controls=[len(c['ok']) for c in cases],horizons=rows,H1_hand_only_load=out)


def run(out):
    torch.set_num_threads(2)
    root_reports = ['pen_dev_init_compare_v1/refit002','pen_start_prefix_w1_v1','pen_start_prefix_balanced_w1_v1',
                    'pen_gpu_refit_preloads_v1','pen_load_forecast_v1']
    pins = {key:file_pin(ROOT/key/'report.json') for key in root_reports}
    fit = read(pins['pen_load_forecast_v1'])['outcomes']['state']
    net = load_model(fit)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'intent.json',dict(sources=pins,script=file_pin(__file__),model=fit['model'],horizons=HORIZONS,
        contract='All overlapping observed fixed-PD windows, source state measured once per window. '
                 'M3 and constant world linear/angular velocity get identical starts. No tuning, native stepping or GPU. '
                 'Terminal observed controls included in observed; valid_only excludes any failed control. '
                 'from_reset includes a terminal endpoint if actually observed.'))
    arrays, manifest, groups = {},[],{}
    for nr,(name,rep,pin,stop,z,ok,model_path,phys_path,labels) in enumerate(sources()):
        s,pd,ph,truth,hand_only = get_trajectory(z,model_path,phys_path,len(ok))
        h,load = predict(s,pd,ph,ok,net)
        diff = np.c_[load[:,:6],load[:,6:].sum(1)] - truth
        digit_diff = None if labels is None else load[hand_only,6:] - labels[hand_only]
        key = f'p{nr:02}'
        arrays.update({f'{key}_h{H}_{field}':a for H,rec in h.items() for field,a in rec.items()})
        arrays.update({f'{key}_m3_load12':load,f'{key}_true_wrench6_total_normal':truth,
                       f'{key}_hand_only':hand_only,f'{key}_valid':ok})
        if labels is not None:
            arrays[f'{key}_true_normal6'] = labels
        groups.setdefault(name,[]).append(dict(h=h,load_diff=diff[hand_only],digit_diff=digit_diff,ok=ok))
        manifest.append(dict(key=key,group=name,rep=rep,source=pin,stop=stop,observed_controls=len(ok),
                             geometry_task_valid_controls=sum(ok),model=file_pin(model_path),physics=file_pin(phys_path)))
    archive = out/'windows.npz'
    np.savez_compressed(archive,**arrays)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),windows=file_pin(archive),trajectories=manifest,
         groups={key:summarize_group(val) for key,val in groups.items()},actual_native_steps=0,optimizer_updates=0,
         train=dict(reference='single controlled clip_001',train_controls=fit['train_controls'],val_controls=fit['val_controls'],
                    data='Four TRAIN policies A-D, VAL separate, frozen stored normalization and weights.'),
         interpretation='Fixed realized PD, not regenerated Actor actions. Native trace endpoints compare simulated '
                        'motion, not reference motion. Horizons are controls of 20 ms. RMS is window-weighted, '
                        'overlapping rows and numerical repeats are dependent. The optional fixed-hold cohort '
                        'has no moving reference or Actor; valid_only there uses archived physical gates. '
                        'Hand-only force/torque uses saved generalized constraint forces at zero COM offset and '
                        'zero nonhand load. Per-digit normals exist only for fixed-hold raw contact rows. '
                        'Pose arrays contain world COM followed by body quaternion wxyz; load axes are source BODY.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'pen_m3_dev_prefix_v1')
    run(p.parse_args().out)
