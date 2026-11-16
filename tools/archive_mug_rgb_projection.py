"""Independent saved-image/mesh/pixel accounting and immutable RGB audit roots."""
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from archive_pen_source_ccd_evidence import pins


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    out=ROOT/'provenance/mug_rgb_projection_v1';analysis=ROOT/'runs/mug_rgb_projection_archive_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh RGB archive required')
    rp=ROOT/'runs/mug_rgb_projection_v1/report.json';r=read(rp);reg_path=verify_file_pin(r['registration']);reg=read(reg_path);root=rp.parent
    prior=ROOT/'provenance/pen_backend_preparation_v1/ledger.json';checks=dict(root11=r['passed']==r['total']==11 and all(r['checks'].values()),
        exact61_frames=[row['frame'] for row in r['rows']]==list(range(47,108)),fixed_displays=r['display_frames']==[47,77,107],zero_work=(r['actual_native_integrations'],r['optimizer_steps'],r['GPU_kernels'])==(0,0,0),
        registration_inputs=reg['inputs']==r['inputs'])
    geometry=np.load(verify_file_pin(r['projection_geometry']))
    for prefix,vk,fk in [('source','source_vertices','source_faces'),('compiled','compiled_BODY_vertices','compiled_faces')]:
        v,f=geometry[vk],geometry[fk];edges=v[f]-v[f][:,[1,2,0]];checks[prefix+'_max5mm_edges']=float(np.linalg.norm(edges,axis=-1).max())<=.005+1e-12
    for row in r['rows']:
        masks=np.load(verify_file_pin(row['masks']));shape=tuple(masks['shape']);n=int(np.prod(shape));m={k:np.unpackbits(masks[k])[:n].reshape(shape).astype(bool) for k in ['source','compound','provided_amodal','provided_modal']}
        for name,a,b in [('source_vs_provided_amodal','provided_amodal','source'),('source_vs_provided_modal','provided_modal','source'),('compound_vs_source','source','compound'),('compound_vs_provided_amodal','provided_amodal','compound')]:
            x,y=m[a],m[b];reported=row[name];intersection=int((x&y).sum());union=int((x|y).sum());actual=dict(reference_pixels=int(x.sum()),candidate_pixels=int(y.sum()),intersection_pixels=intersection,missed_pixels=int((x&~y).sum()),extra_pixels=int((y&~x).sum()),IoU=intersection/union if union else 1.)
            checks[f'frame{row["frame"]}_{name}_pixels']=all(reported[k]==v for k,v in actual.items())
    receipts=read(verify_file_pin(r['member_receipts']));members={p['member']:p for p in receipts['records']}
    for frame in r['display_frames']:
        pin=file_pin(root/f'RGB_{frame:06d}.jpg');expected=members[f'{frame:06d}.image_214-1.jpg'];checks[f'RGB{frame}_source_bytes']=all(pin[k]==expected[k] for k in ['bytes','sha256'])
    log=ROOT/'runs/mug_rgb_projection_tests484_v1.log';txt=log.read_text();checks['tests484']='Ran 484 tests in 10.462s' in txt and '\nOK\n' in txt
    training=ROOT/'runs/pen_backend_long_training_v1/source_receipt.json';response_reg=ROOT/'runs/pen_backend_response_registration_v1/registration.json'
    for name,path in [('training',training),('response',verify_file_pin(read(response_reg)['source_receipt']))]:verify_source_snapshot(read(path),live=True);checks[name+'_source_bytes_unchanged']=True
    for item in [r,reg]:verify_source_snapshot(read(verify_file_pin(item['source_receipt'])),live=True)
    if not all(checks.values()):raise ValueError('Independent RGB archive failed: '+repr([k for k,v in checks.items() if not v]))
    docs=[ROOT/'docs'/name for name in ['mug-rgb-projection-protocol-v1.md','mug-rgb-projection-results-2026-11-16.md','research-plan-v1.29.md']]
    scripts=[Path(__file__),ROOT/'tools/audit_mug_rgb_projection.py'];analysis.mkdir();receipt=freeze_sources(docs+scripts,analysis/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(analysis/'source_receipt.json',receipt)
    todo=[file_pin(p) for p in [rp,reg_path,prior,log,training,response_reg,analysis/'source_receipt.json']+docs+scripts]+[file_pin(root/f'RGB_{f:06d}.jpg') for f in r['display_frames']]
    allowed={root,reg_path.parent,analysis,Path(r['inputs']['upstream']['path']).parent};verified={}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        path=verify_file_pin(pin);verified[key]=pin
        if path.suffix=='.json' and any(path.is_relative_to(p) for p in allowed):todo.extend(pins(read(path)))
    out.mkdir();write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'summary.json',dict(checks=checks,passed=sum(checks.values()),total=len(checks),report=file_pin(rp),aggregate=r['aggregate'],source_frames=61,display_frames=r['display_frames'],
        contact_sheet=r['contact_sheet'],visual_review='Inspected original47 and fixed47/77/107 sheet;coarse cup/handle correspondence with visible edge/occlusion differences.No manual exact ground truth or contact labels.',
        actual_native_integrations=0,optimizer_steps=0,GPU_kernels=0,scope=r['scope']))
    write(out/'ledger.json',dict(schema='real_ego_RGB_projection_ledger:1',prior_evidence=file_pin(prior),report=file_pin(rp),registration=file_pin(reg_path),summary=file_pin(out/'summary.json'),
        source_receipts=[r['source_receipt'],reg['source_receipt'],file_pin(analysis/'source_receipt.json')],documents=[file_pin(p) for p in docs],verified_manifest=file_pin(analysis/'verified_pins.json'),verified_pins=len(verified),
        actual_native_integrations=0,optimizer_steps=0,GPU_kernels=0))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),checks=[sum(checks.values()),len(checks)],verified_pins=len(verified))),flush=True)


if __name__=='__main__':main()
