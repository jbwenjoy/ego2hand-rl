"""Attach eight-tick capacity labels to independent exact-source action pairs."""
import json,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import collect_guarded_finger_bank as bank
from supermimic_research.mechanics.pen_capacity_trace import packet_capacity
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json


def main(out):
    out.mkdir(exist_ok=False)
    write_json(out/'intent.json',dict(source=file_pin(__file__),trajectory_driver=file_pin(Path(bank.__file__)),
        phases=bank.base.PHASES,split='Same TRAIN_A/B basis-only training and TEST actor held out of fitting. All mixed probes withheld; previous experiments on same actors disclosed.',
        design='Two sham, sixteen signed single-joint RAW and two mixed directions from each native source; verify actual PD endpoints and normalize by actual contrast. Geometry labels are means of eight moving-BODY capacity vectors.'))
    b=bank.base;original=b.decode_saved_case;captured=[]
    def label(case,z,model,physical):
        checked=original(case,z,model,physical)
        c=packet_capacity(z,model,case['model_layout'],physical['geom_friction'])
        if c['maximum_containment_excess']>2e-5 or c['positive_gap_skipped_max_load_N']>1e-5:raise ValueError('Containment failed')
        path=out/f'trace_{len(captured):04}.npz';np.savez_compressed(path,**z)
        captured.append((file_pin(path),c['capacity'],c['mask'],z['post_ctrl'][0],z['pre_action_phase'][0]))
        return checked
    with patch.object(b,'decode_saved_case',label):bank.main(out/'native')
    native=json.loads((out/'native/report.json').read_bytes());roles={};index=0
    for role in ('TRAIN_A','TRAIN_B','TEST'):
        arm=json.loads(verify_file_pin(native['arms'][role]).read_bytes())
        group=out/role;group.mkdir();rows=[]
        with np.load(out/'native'/role/'physical_context.npz') as ph:assert ph['geom_friction'].shape[0]==1
        for row in arm['rows']:
            trace,capacity,mask,ctrl,phase=captured[index];index+=1
            assert row['phase']==phase
            with np.load(verify_file_pin(row['packet'])) as p:
                assert np.array_equal(p['pd'],p['x'][0,81:101])
                # All 20 actuators are the transmission-resolved joint order of the pen.
                np.testing.assert_array_equal(p['pd'],ctrl)
                dst=group/(Path(row['packet']['path']).stem+'.npz')
                np.savez_compressed(dst,x=p['x'][0],pd=p['pd'],capacity=capacity.mean(0),ticks=capacity,masks=mask)
            rows.append(dict(phase=int(phase),name=row['name'],valid=row['valid'],terminal=row['terminal'],source=row['source'],native_packet=row['packet'],trace=trace,packet=file_pin(dst)))
        roles[role]=dict(selected=arm['selected'],rows=rows)
    assert index==len(captured)==600 and native['native']==10200
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),native_report=file_pin(out/'native/report.json'),roles=roles,native_ticks=native['native']))
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
