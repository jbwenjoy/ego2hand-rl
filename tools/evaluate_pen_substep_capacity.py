"""Recompute contact capacities from held-out substep pose predictions on CPU."""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from check_pen_capacity_geometry_oracle import geometry
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.mechanics.pen_wrench_capacity import balanced_directions
from supermimic_research.strict_json import write_json

METHODS = ('constant_velocity', 'load_interpolation', 'absolute', 'paired')


def hand_pen_pair(model, body, g, h):
    on = model.geom_bodyid[[g,h]] == body
    if on.sum() != 1:
        return None
    pen, hand = (g,h) if on[0] else (h,g)
    return (pen,hand) if (model.geom(hand).name or '').startswith('robot/') else None


def measured_signature(model, body, raw):
    ticks = [[] for _ in range(8)]
    for row in raw:
        if row[5] > 0:
            continue
        tick,g,h = map(int,row[:3])
        pair = hand_pen_pair(model,body,g,h)
        if pair is not None:
            ticks[tick].append(pair)
    return tuple(tuple(sorted(t)) for t in ticks)


def geometry_sequence(model,data,body,directions,q):
    if q.shape != (8,model.nq) or not np.isfinite(q).all():
        raise ValueError('Eight finite native-qpos states expected')
    values,signature = [],[]
    for state in q:
        values.append(geometry(model,data,state,body,directions))
        pairs = []
        for c in data.contact:
            if c.dist <= 0:
                pair = hand_pen_pair(model,body,*map(int,c.geom))
                if pair is not None:
                    pairs.append(pair)
        signature.append(tuple(sorted(pairs)))
    return np.asarray(values), tuple(signature)


def rms_summary(pairs):
    if not pairs:
        return dict(n=0,relative_error=None,amplitude=None,positive_dot=None)
    actual = np.stack([p['native_delta'] for p in pairs])
    estimate = np.stack([p['pred_delta'] for p in pairs])
    denom = np.linalg.norm(actual)
    dots = np.sum(actual*estimate,axis=1)
    return dict(n=len(pairs),native_energy=float(np.square(actual).sum()),
                relative_error=float(np.linalg.norm(actual-estimate)/denom) if denom else None,
                amplitude=float(np.linalg.norm(estimate)/denom) if denom else None,
                positive_dot=int((dots>0).sum()))


def main(native_run,pose_run,old_run,topology_run,out):
    root_report = file_pin(native_run/'report.json')
    source = json.loads(verify_file_pin(root_report).read_bytes())
    old = json.loads((old_run/'report.json').read_bytes())
    topo = json.loads((topology_run/'report.json').read_bytes())
    if topo['source'] != root_report or json.loads(verify_file_pin(old['intent']).read_bytes())['source'] != root_report:
        raise ValueError('Historical controls must have identical native data')
    topo_counts = {(r['phase'],r['kind']):r['differs_ticks'] for r in topo['rows'] if r['role']=='TEST'}
    old_rows = {(r['phase'],r['name']):r for r in old['rows']}
    fit = json.loads((pose_run/'report.json').read_bytes())
    with np.load(verify_file_pin(fit['predictions']),allow_pickle=False) as f:
        ids = list(zip(map(int,f['phase']),map(str,f['name'])))
        valid = f['valid'].copy()
        preds = {m:f[m].astype(float) for m in METHODS}
    declared = source['roles']['TEST']['rows']
    if len(set(ids))!=len(ids) or ids!=[(r['phase'],r['name']) for r in declared]:
        raise ValueError('TEST source order/coverage mismatch')
    out.mkdir(exist_ok=False,parents=True)
    write_json(out/'intent.json',dict(native=root_report,model=file_pin(pose_run/'report.json'),
        historic=file_pin(old_run/'report.json'),topology=file_pin(topology_run/'report.json'),
        scope='All TEST rows for prediction; valid, nonterminal, nonzero effective PD pairs for contrasts. Native sorted geom-pair multisets at all eight ticks define the topology strata. Oracle-only ablations replace one body with measured substep poses.'))
    folder=native_run/'native/TEST'
    model=mujoco.MjModel.from_binary_path(str(folder/'model.mjb'))
    data=mujoco.MjData(model)
    body=model.body('object/cube').id
    qa=model.jnt_qposadr[model.body_jntadr[body]]
    qs=model.jnt_qposadr[model.actuator_trnid[:,0]]
    assert len(qs)==20 and not set(qs).intersection(range(qa,qa+7))
    directions=balanced_directions()
    modes=METHODS+('true_substeps','paired_hand_true_object','true_hand_paired_object')
    caps={m:[] for m in modes}; signatures={m:[] for m in modes}
    native=[];native_sets=[];pd=[];native_ok=[]
    for i,r in enumerate(declared):
        with np.load(verify_file_pin(r['trace'])) as f:
            truth=f['pre_qpos'].copy();top=measured_signature(model,body,f['raw_contact_tick_plus23'])
            data.mocap_pos[:]=f['post_mocap_pos'][0];data.mocap_quat[:]=f['post_mocap_quat'][0]
        with np.load(verify_file_pin(r['packet'])) as f:
            native.append(f['capacity'].copy());pd.append(f['pd'].copy())
        native_sets.append(top);native_ok.append(bool(r['valid'] and not r['terminal']))
        hand=truth.copy();hand[:,qs]=preds['paired'][i][:,qs]
        obj=truth.copy();obj[:,qa:qa+7]=preds['paired'][i][:,qa:qa+7]
        poses=dict(true_substeps=truth,paired_hand_true_object=hand,true_hand_paired_object=obj)
        poses.update({m:preds[m][i] for m in METHODS})
        for name,q in poses.items():
            # Quaternion sign has no effect on the pose or collision geometry.
            np.testing.assert_allclose(q[0,qs],truth[0,qs],rtol=0,atol=1e-6)
            np.testing.assert_allclose(q[0,qa:qa+3],truth[0,qa:qa+3],rtol=0,atol=1e-6)
            np.testing.assert_allclose(np.abs(q[0,qa+3:qa+7]@truth[0,qa+3:qa+7]),1,atol=1e-6)
            values,sig=geometry_sequence(model,data,body,directions,q)
            caps[name].append(values.mean(0));signatures[name].append(sig)
        np.testing.assert_allclose(caps['true_substeps'][-1],old_rows[ids[i]]['measured_substeps'],atol=1e-8)
        np.testing.assert_array_equal(native[-1],old_rows[ids[i]]['native'])
    native_ok=np.array(native_ok);native=np.stack(native);lookup={k:i for i,k in enumerate(ids)}
    if not np.array_equal(valid,native_ok):
        raise ValueError('Prediction validity flags differ from raw native validity and termination')
    summaries={};rows=[];omissions=[]
    for name in modes:
        estimates=np.stack(caps[name])
        summary=dict(abs_capacity_MAE=float(np.mean(abs(estimates[native_ok]-native[native_ok]))))
        for kind,actions in [('mixed',['mixed']),('basis',[f'j{j}' for j in source['roles']['TEST']['selected']])]:
            pairs=[]
            for phase in sorted({p for p,_ in ids}):
                for action in actions:
                    a,b=[lookup[(phase,action+s)] for s in ('_minus','_plus')]
                    if not(native_ok[a] and native_ok[b]) or np.linalg.norm(pd[b]-pd[a])<=1e-7:
                        if name==modes[0]:omissions.append((phase,action))
                        continue
                    changed=sum(u!=v for u,v in zip(native_sets[a],native_sets[b]))
                    assert changed==topo_counts[(phase,action)]
                    predicted=sum(u!=v for u,v in zip(signatures[name][a],signatures[name][b]))
                    n= native[b]-native[a];v=estimates[b]-estimates[a]
                    pairs.append(dict(native_delta=n,pred_delta=v,changed=changed,predicted=predicted))
                    rows.append(dict(method=name,kind=kind,phase=phase,action=action,changed=changed,predicted=predicted,
                                     native_l2=float(np.linalg.norm(n)),prediction_l2=float(np.linalg.norm(v)),error_l2=float(np.linalg.norm(v-n))))
            summary[kind]=dict(all=rms_summary(pairs),unchanged=rms_summary([p for p in pairs if not p['changed']]),
                               changed=rms_summary([p for p in pairs if p['changed']]),
                               changed_classification={tag:sum((p['changed']>0)==truth and (p['predicted']>0)==est for p in pairs)
                                                       for tag,truth,est in [('TP',True,True),('FP',False,True),('FN',True,False),('TN',False,False)]})
        summaries[name]=summary
    caps_path=out/'capacities.npz'
    np.savez_compressed(caps_path,phase=np.array([p for p,_ in ids]),name=np.array([n for _,n in ids]),native=native,valid=native_ok,**{k:np.stack(v) for k,v in caps.items()})
    old_interp=np.stack([old_rows[key]['learned_endpoint'] for key in ids])
    historic_gap=np.abs(caps['load_interpolation']-old_interp)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),predictions=fit['predictions'],model=file_pin(folder/'model.mjb'),
        historic_load_interpolation=dict(max_abs=float(historic_gap.max()),mean_abs=float(historic_gap.mean()),mixed=old['summary']['learned_endpoint']['mixed']),
        estimates=file_pin(caps_path),available_rows=len(ids),used_rows=int(native_ok.sum()),excluded_pairs=omissions,summary=summaries,pairs=rows,
        historic_cpu_mixed=old['summary']['cpu_openloop_substeps']['mixed'],historic_cpu_basis=old['summary']['cpu_openloop_substeps']['basis']))
    for key,s in summaries.items():
        print(key,s['abs_capacity_MAE'],[(kind,s[kind]['all']['relative_error'],s[kind]['all']['positive_dot'],s[kind]['changed']['relative_error']) for kind in ['mixed','basis']])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['native_run','pose_run','old_run','topology_run','out']:parser.add_argument(name,type=Path)
    args=parser.parse_args();main(args.native_run,args.pose_run,args.old_run,args.topology_run,args.out)
