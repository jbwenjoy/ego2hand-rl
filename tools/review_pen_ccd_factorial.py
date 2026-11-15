"""Independent array/model/output audit of native-only CCD numerical cells."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import verify_source_snapshot
from supermimic_research.runtime.controlled_evaluation import write


def main(a):
    r=json.loads(a.report.read_text());branch=json.loads(verify_file_pin(r['branch']).read_text())
    if a.output.exists() or {(c['iterations'],c['requested_tolerance']) for c in r['cells']}!={(i,t) for i in [50,100] for t in [1e-6,1e-7]} or len(r['cells'])!=4:raise ValueError('Fresh complete2x2 independent review')
    for key in ['source_receipt','installed_source_receipt']:verify_source_snapshot(json.loads(verify_file_pin(r[key]).read_text()))
    original=dict(np.load(verify_file_pin(r['original_model_arrays'])));old_scalars=json.loads(verify_file_pin(r['original_model_scalars']).read_text());cells=[]
    for c in r['cells']:
        arrays=dict(np.load(verify_file_pin(c['fixed_model_arrays'])));scalars=json.loads(verify_file_pin(c['fixed_model_scalars']).read_text())
        changes=sorted(k for k in original if original[k].tobytes()!=arrays[k].tobytes());scalar_changes=sorted(k for k in old_scalars if old_scalars[k]!=scalars[k])
        checks=dict(all_model_array_keys=set(arrays)==set(original),all_model_scalar_keys=set(scalars)==set(old_scalars),
            only_declared_array_changed=set(changes)<=set(['opt.ccd_tolerance']),only_declared_scalar_changed=set(scalar_changes)<=set(['opt.ccd_iterations']),
            declared_tolerance=np.array_equal(arrays['opt.ccd_tolerance'],np.full_like(original['opt.ccd_tolerance'],c['requested_tolerance'])),
            declared_iterations=scalars['opt.ccd_iterations']==c['iterations'],reported_changes_exact=changes==c['changed_arrays'] and scalar_changes==c['changed_scalars'],
            collector_model_unchanged=bool(c['model_unchanged_after_queries']),exact20_cases=len(c['records'])==20 and len({(e['pose'],e['repeat']) for e in c['records']})==20)
        outputs=[]
        for e in c['records']:
            z=np.load(verify_file_pin(e['trace']));mask=np.all(np.sort(z['geom'],axis=1)==[1,29],axis=1)
            baseline=c['iterations']==50 and c['requested_tolerance']==1e-6;old=next(q for q in branch['records'] if q['pose']==e['pose'] and q['repeat']==e['repeat'])
            x=dict(raw_contact_count=len(z['geom'])==e['nacon'],raw_pair_count=int(mask.sum())==e['pair_contacts'],
                raw_pair_distances=np.array_equal(z['dist'][mask].astype(float),np.array(e['pair_distances_m'])),finite_geometry=all(np.isfinite(z[k]).all() for k in z.files),
                baseline_original_exact=(not baseline) or (int(mask.sum())==old['pair_contacts'] and np.array_equal(np.sort(z['dist'][mask]),np.sort(np.array(old['pair_distances_m'],np.float32)))),
                exact_restore_record=bool(e['checks']['exact_public_restore']),capacity_record=bool(e['checks']['capacity']),overflow_record=bool(e['checks']['no_overflow']))
            outputs.append(dict(pose=e['pose'],repeat=e['repeat'],checks=x,pair_contacts=int(mask.sum()),distances_m=z['dist'][mask].astype(float).tolist()))
        summaries=[]
        for name in [p['name'] for p in branch['poses']]:
            selected=[o for o in outputs if o['pose']==name]
            summaries.append(dict(pose=name,contact_presence=[o['pair_contacts']>0 for o in selected],
                exact_fixed_pose_repeat=all(o['pair_contacts']==selected[0]['pair_contacts'] and o['distances_m']==selected[0]['distances_m'] for o in selected),
                distances_m=[o['distances_m'] for o in selected]))
        cells.append(dict(iterations=c['iterations'],tolerance=c['observed_tolerance'],checks=checks,outputs=outputs,summaries=summaries))
    passed=sum(sum(c['checks'].values())+sum(sum(e['checks'].values()) for e in c['outputs']) for c in cells)
    total=sum(len(c['checks'])+sum(len(e['checks']) for e in c['outputs']) for c in cells)
    a.output.mkdir();result=dict(schema='native_ccd_factorial_saved_review:1',run=file_pin(a.report),source=file_pin(__file__),cells=cells,passed=passed,total=total,
        native_integrations=0,neural_updates=0,scope='Independent saved model field diffs, numerical values, raw contact outcomes, baseline and exact per-pose repeat checks. Local sensitivity on two unique pose images, not global detector repair, valid force/gradient/controller/ego result. Original reconstructed Data static limits persist.')
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),contracts=[passed,total],cells=[dict(iterations=c['iterations'],tolerance=c['tolerance'],presence={p['pose']:sum(p['contact_presence']) for p in c['summaries']}) for c in cells])))
    if passed!=total:raise ValueError('Retain factorial review failure')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
