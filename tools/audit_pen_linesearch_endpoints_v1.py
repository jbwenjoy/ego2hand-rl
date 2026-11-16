"""All saved endpoint epochs against exact installed source, no native work."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.mechanics.pen_ray_epoch_audit_v1 import source_facts,endpoint_metrics
from supermimic_research.strict_json import write_json


def read(pin): return json.loads(verify_file_pin(pin).read_text())


def main():
    out=ROOT/'runs/pen_linesearch_endpoint_audit_v1'
    if out.exists(): raise ValueError('Fresh offline endpoint audit required')
    regpin=file_pin(ROOT/'runs/pen_termination_comparator_registration_v1/registration.json'); reg=read(regpin)
    installed=reg['inputs']['installed_solver']; source_path=verify_file_pin(installed); text=source_path.read_text(); facts=source_facts(text)
    tests=ROOT/'runs/pen_ray_epoch_audit_v1_tests.log'
    if not all(facts['checks'].values()) or 'Ran 2 tests' not in tests.read_text(encoding='utf-8-sig') or '\nOK\n' not in tests.read_text(encoding='utf-8-sig'):
        raise ValueError('Exact source/epoch regression tests required')
    inputs=[regpin,installed,file_pin(tests)]; rows=[]; peaks=[]; checks=dict(facts['checks'])
    for replica in ['A','B']:
        pp=file_pin(ROOT/f'runs/pen_termination_comparator_v1/{replica}/report.json'); rp=file_pin(ROOT/f'runs/pen_termination_comparator_review_v2/{replica}/report.json'); p,v=read(pp),read(rp); inputs += [pp,rp]
        if v['producer']!=pp or v['passed']!=v['total'] or not all(v['checks'].values()): raise ValueError('Complete independently reviewed producer required')
        for slot in p['rows']:
            if slot['declaration']['scope']!='PACKET_PLUS_SOLVE_EXIT': continue
            q=read(slot['report']); inputs += [slot['report'],q['solve_exit'],q['balance'],q['epoch_review']]
            o=dict(np.load(verify_file_pin(q['solve_exit']))); b=dict(np.load(verify_file_pin(q['balance']))); e=read(q['epoch_review'])
            peak=int(np.linalg.norm(b['original_total'],axis=1).argmax())
            if peak!=e['peak_tick']: raise ValueError('Original peak identity changed')
            for tick in range(8):
                row=dict(replica=replica,algorithm=q['declaration']['algorithm'],repeat=q['declaration']['repeat'],row_index=q['declaration']['row_index'],
                    original_name=q['declaration']['original_name'],tick=tick,query_impulse_valid=q['new_impulse_valid'],
                    original_impulse_Ns=float(np.linalg.norm(b['original_total'][tick])),**endpoint_metrics(o,tick))
                rows.append(row)
                if not q['new_impulse_valid'] and tick==peak: peaks.append(row)
    candidate=[r for r in peaks if r['algorithm']=='DROP_IMPROVEMENT_ONLY']; original=[r for r in peaks if r['algorithm']=='INSTALLED_ORIGINAL_OR']
    checks.update(all960_ticks=len(rows)==960,all35_failed_peaks=len(peaks)==35 and len(candidate)==12 and len(original)==23,
        all_failed_post_rebuild_flags=all(r['post_update_state_changed']==1 and r['post_update_quad_changed']==0 and not r['post_update_search_unchanged'] and r['post_update_grad_scale']==1 for r in peaks),
        all_failed_prior_step_zero=all(r['alpha_from_last_linesearch']==r['improvement_from_last_linesearch']==0 and r['step_noise_flag_from_last_linesearch'] for r in peaks),
        all_candidate_failed_cap=all(r['iterations']==50 for r in candidate),
        all_failed_stored_next_search_negative=all(r['post_update_G_dot_stored_search']<0 for r in peaks))
    # Numerical outcomes are observed facts for this sample, never a gate change.
    for pin in inputs: verify_file_pin(pin)
    out.mkdir(); receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'tests/test_pen_ray_epoch_audit_v1.py'],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()); write_json(out/'source_receipt.json',receipt)
    shutil.copyfile(source_path,out/'installed_solver.py')
    if file_pin(out/'installed_solver.py')['sha256']!=installed['sha256']: raise ValueError('Installed source copy differs')
    with (out/'all960_endpoint_ticks.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result=dict(schema='pen_linesearch_endpoint_source_audit:1',inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),
        installed_copy=file_pin(out/'installed_solver.py'),source_facts=facts,checks=checks,passed=sum(checks.values()),total=len(checks),failed_peaks=peaks,
        csv=file_pin(out/'all960_endpoint_ticks.csv'),actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,
        conclusions=dict(exhausted_flag_means='abs(last alpha)<noise_floor;not proof LS_ITERATIONS exhausted',
            mixed_endpoint_epochs='alpha/improvement/noise flag describe last line search;grad/search/decrement/change flags are later post-update values',
            rebuild='All35 failed peaks match the forced-rebuild path,not proof that the prior ray inputs matched the saved post-update search',
            shifted_cost='Quadratic pyramidal/equality path already evaluates shifted costs;friction branch also contains absolute cost differences',
            missing=['same-iteration pre-line-search gradient/scale/search/Ma/smooth/mv/qacc',
                'actual row inputs ne/nf/nefc/D/frictionloss/Jaref and computed jv',
                'GPU p0 row and gauss reductions,gtol,noise_floor,initial Newton alpha/point/acceptance',
                'each actual bracket iteration points/alphas,swap/stop flags,accepted alpha/improvement',
                'paired post-gradient-update values with exact outer iteration index']),
        scope='Offline full saved endpoint population and pinned source only. Stored next-search descent does not identify the ray that produced alpha0;noise flag is not a loop-cap flag. No precision/bracketing cause,engine fix,backend/Actor/ego or speed admission.')
    verify_source_snapshot(receipt,live=True); write_json(out/'report.json',result)
    print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[sum(checks.values()),len(checks)],ticks=len(rows),failed_peaks=len(peaks),native=0)))
    if not all(checks.values()): raise ValueError('Endpoint source/outcome discrepancy retained')


if __name__=='__main__': main()
