"""Pure-CPU schedule and replay-channel contracts; no native physics claim."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rough_prefix.io import pin
from rough_prefix.physical_fd import branch_schedule,baseline_errors
from rough_prefix.physical_probe_consumer_math import slots


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    checks={};rows=branch_schedule();expected=slots()
    checks['producer_consumer_slot_identity_all560']=len(rows)==560 and all(all(r[k]==v for k,v in e.items()) for r,e in zip(rows,expected))
    checks['all_unexecuted_rows_explicitly_unmeasured']=all(r['status']=='NOT_RUN' and r['actualu'] is None and r['r_k'] is None and r['r_kplus1'] is None and r['restorequalified'] is False for r in rows)
    x=np.zeros(58,np.float32)
    checks['identical_baseline_passes']=baseline_errors(x,x,7.,7.)['passed']
    for name,index,limit in [('q',0,2e-5),('qd',20,2e-4),('p',40,2e-5),('R',43,2e-5),('v',52,2e-4),('omega',55,2e-4)]:
        bad=x.copy();bad[index]=2*limit;v=baseline_errors(bad,x,7.,7.)
        checks[name+'_channel_in_58_layout_rejects']=v['passed'] is False and v['errors'][name]>limit and all(v['errors'][k]==0 for k in v['errors'] if k!=name)
    checks['reward_replay_gate_is_separate']=baseline_errors(x,x,7.+4e-5,7.)['passed'] is False
    bad=x.copy();bad[40]=np.nan
    checks['nonfinite_output_does_not_pass']=baseline_errors(bad,x,7.,7.)['passed'] is False
    checks['default_module_import_has_no_frameworks']=not any(k in sys.modules for k in ('torch','warp','mujoco','mujoco_warp'))
    report=dict(schema='current104_physical_worker_CPU_contract:1',checks=checks,passed=all(checks.values()),
        source=pin(ROOT/'src/rough_prefix/physical_fd.py'),consumer_math=pin(ROOT/'src/rough_prefix/physical_probe_consumer_math.py'),
        scope='Schedule ABI and individual replay-channel boundary failures only. Does not construct native Env or qualify restore/cadence/physical gradients.',
        Env_created=False,physics_steps=0,WM_forwards=0,GPU_used=False)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],checks=len(checks),failed=[k for k,v in checks.items() if not v])))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
