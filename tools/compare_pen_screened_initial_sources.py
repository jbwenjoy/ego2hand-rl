"""All original/sham first-two-repeat sources, without native replay."""
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.shared_snapshot_transport import load_shared
from supermimic_research.learning.world_model import raw_tree_equal
from supermimic_research.strict_json import write_json


def read(p):return json.loads(Path(p).read_text())


def main():
    vp=ROOT/'runs/pen_native_screened_review_v3/indices/completed_36.json';index=read(vp);out=ROOT/'runs/pen_first_two_repeat_initial_comparison_v1'
    if out.exists() or len(index['reviewed_cases'])!=36:raise ValueError('Fresh fixed first-two-repeat source comparison required')
    lookup={r['name']:r for r in index['reviewed_cases']};rows=[];checks={}
    for teacher in ['RESIDUAL','SUPPORT']:
        for controller in ['UNMODIFIED_ACTOR','NATIVE_SHAM']:
            inputs=[];states=[];traces=[]
            for repeat in [0,1]:
                slot=lookup[f'R{repeat}_{teacher}_{controller}'];review=read(verify_file_pin(slot['report']));case=read(verify_file_pin(review['producer_case']));node=case['nodes'][0];packet=read(verify_file_pin(node['BASE']));states.append(load_shared(packet['source']));traces.append(dict(np.load(verify_file_pin(packet['trace']))));inputs.append(dict(review=slot['report'],case=review['producer_case'],first_BASE=node['BASE'],source=packet['source'],trace=packet['trace']))
            native={};differences={}
            for group in ['model','data']:
                x,y=[s['native'][group] for s in states];native[group]=raw_tree_equal(x,y);delta=[]
                for key,a in x['arrays'].items():
                    b=y['arrays'][key]
                    if a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes()!=b.tobytes():
                        delta.append(dict(field=key,dtype_equal=a.dtype==b.dtype,shape_equal=a.shape==b.shape,max_abs=float(np.max(abs(a.astype(np.float64)-b.astype(np.float64)))) if a.shape==b.shape and a.size else None))
                differences[group]=delta
            a,b=traces
            checks[teacher+'_'+controller+'_first_source_epochs']=all(np.array_equal(s['native']['data']['arrays']['qpos'][0],z['pre_qpos'][0]) and np.array_equal(s['native']['data']['arrays']['qvel'][0],z['pre_qvel'][0]) for s,z in zip(states,traces))
            rows.append(dict(teacher=teacher,controller=controller,inputs=inputs,native_equal=native,native_array_differences=differences,python_equal=raw_tree_equal(states[0]['python'],states[1]['python']),rng_equal=raw_tree_equal(states[0]['rng'],states[1]['rng']),
                first_sampled_RAW_equal=np.array_equal(a['actions20'],b['actions20']),first_actual_PD_equal=np.array_equal(a['post_ctrl'],b['post_ctrl']),first_pre_qpos_equal=np.array_equal(a['pre_qpos'][0],b['pre_qpos'][0]),first_pre_qvel_equal=np.array_equal(a['pre_qvel'][0],b['pre_qvel'][0]),
                first_control_max_abs_qpos=float(abs(a['post_qpos']-b['post_qpos']).max()),first_control_max_abs_qvel=float(abs(a['post_qvel']-b['post_qvel']).max())))
    checks['all4_comparisons']=len(rows)==4
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__)],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt)
    verify_source_snapshot(receipt,live=True);write_json(out/'report.json',dict(schema='pen_first_two_repeat_initial_comparison:1',index=file_pin(vp),source_receipt=file_pin(out/'source_receipt.json'),rows=rows,checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,scope='All original/sham teachers across first two numerical repeats only. Native/python/RNG differences reported,not required equal. Source equality is not future determinism or a solver-root-cause proof.'))
    print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[sum(checks.values()),len(checks)],native=0)),flush=True)
    if not all(checks.values()):raise ValueError('Initial source/trace epoch discrepancy')


if __name__=='__main__':main()
