"""CPU-only preparation of all five retained impulse failures and controls."""
import argparse
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import mujoco
import numpy as np
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.shared_snapshot_transport import load_shared
from supermimic_research.mechanics.warp_packet_decode import decode_saved_case
from supermimic_research.mechanics.pen_solver_tick_balance import mass_product
from supermimic_research.strict_json import write_json


def read(p):return json.loads(Path(p).read_text())


def main(a):
    started=time.perf_counter();cp=a.config.resolve();cfg=tomllib.loads(cp.read_text())
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['partition_report','study_config','protocol','output','required_closed_producer','required_closed_review']}
    if (cfg['schema'],cfg['mode'],cfg['replicas'],cfg['repeats'],cfg['scopes'],cfg['controls_per_condition'],cfg['native_ticks_per_control'],cfg['future_native_cap'],cfg['optimizer_budget'],cfg['normalizer_fit_budget'])!=(1,'PREPARE_SAME_SOLVE_EXIT_DIAGNOSTIC',['A','B'],3,['PACKET_BASELINE','PACKET_PLUS_SOLVE_EXIT'],1,8,960,0,0):raise ValueError('Fixed diagnostic preparation required')
    out=paths['output']
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh preparation output required')
    prior=read(paths['partition_report']);base=read(verify_file_pin(prior['prior_failed_diagnostic']))
    if prior['passed']!=prior['total'] or not all(prior['checks'].values()):raise ValueError('Completed corrected partition required')
    verify_source_snapshot(read(verify_file_pin(prior['source_receipt'])),live=True)
    if [p['failure'] for p in cfg['pairs']]!=base['failed_impulse_rows']:raise ValueError('All five failures in original order required')
    life=read(verify_file_pin(base['inputs']['lifecycle']));node=read(verify_file_pin(base['inputs']['node']));cases=node['branches']+life['rows'];case_reports=[]
    for pin in base['inputs']['first3_cases']:
        case_reports.append(pin);r=read(verify_file_pin(pin))
        for n in r['nodes']:
            for b in ['BASE','ALT']:
                if n[b] is not None:cases.append(read(verify_file_pin(n[b])))
    lookup={c['name']:c for c in cases};old={r['name']:r for r in base['rows']};rows=[];checks={};snapshots={}
    with patch.object(mujoco,'mj_step',side_effect=AssertionError('CPU-only preparation')),patch.object(mujoco,'mj_forward',side_effect=AssertionError('No forward dynamics in preparation')):
        for pi,pair in enumerate(cfg['pairs']):
            for role in ['failure','control']:
                c=lookup[pair[role]];name=c['name'];key=json.dumps(c['source'],sort_keys=True)
                if key not in snapshots:snapshots[key]=load_shared(c['source'])
                s=snapshots[key];m=s['native']['model'];d=s['native']['data'];ma=m['arrays'];da=d['arrays'];nv=m['scalars']['nv'];adr=c['model_layout']['object_dofadr']
                model=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['model'])));physical=dict(np.load(verify_file_pin(c['physical_context'])));z=dict(np.load(verify_file_pin(c['trace'])));metrics,_=decode_saved_case(c,z,model,physical)
                product=mass_product(da['M'],da['qacc'],ma['M_mulm_rowadr'],ma['M_mulm_col'],ma['M_mulm_madr'])
                # Reconstruct M with basis vectors independently of cached Ma.
                dense=mass_product(np.repeat(da['M'],nv,axis=0),np.eye(nv),ma['M_mulm_rowadr'],ma['M_mulm_col'],ma['M_mulm_madr']).T
                reference=np.einsum('ij,tj->ti',dense,da['qacc'].astype(np.float64))
                local=dict(source_native4=s['schema']=='controlled_pen_owned_state:4' and s['worlds']==1,
                    original_mask=metrics['checks']['linear_impulse']==(role=='control')==old[name]['original_impulse_pass'],
                    original_decoder=metrics==c['physical_audit']['decoder_metrics'],
                    source_window_not_pending=not s['python']['scalars'][(('attr','_window_pending'),)],
                    expected_solver=m['scalars']['opt.solver']==2 and m['scalars']['opt.cone']==0 and m['scalars']['opt.integrator']==3 and m['scalars']['opt.enableflags']==0,
                    original8_ticks=z['post_qpos'].shape[0]==8 and z['actions20'].shape==(1,20),
                    mass_product=bool(np.allclose(product,reference,rtol=0,atol=64*np.finfo(np.float64).eps*(1+abs(reference)).max())),
                    symmetric_mass=bool(np.allclose(dense,dense.T,rtol=0,atol=1e-12)),
                    mass_chart=bool(np.allclose(dense[adr:adr+3,adr:adr+3],c['physics']['mass']*np.eye(3),rtol=0,atol=1e-12)),
                    actual_dt=bool(np.all(ma['opt.timestep']==np.float32(.0025))))
                checks.update({name+'_'+k:bool(v) for k,v in local.items()})
                rows.append(dict(pair=pi,role=role,comparison_role=pair['control_role'],name=name,case=c,
                    original_metrics=metrics,checks=local,nv=nv,M_shape=list(da['M'].shape),M_dtype=str(da['M'].dtype),
                    solver_factorization_is_sparse=m['scalars']['is_sparse'],
                    source_cached_Ma_difference_max=float(abs(product-da['efc.Ma']).max()),
                    source_epoch_scope='SOURCE_SNAPSHOT_ONLY_NOT_ORIGINAL_FAILING_TICK',
                    model_options={k:m['scalars'][k] for k in ['opt.solver','opt.cone','opt.integrator','opt.iterations','opt.graph_conditional','opt.enableflags','opt.disableflags']},
                    tolerance=ma['opt.tolerance'].tolist(),meaninertia=ma['stat.meaninertia'].tolist()))
    checks.update(all10=len(rows)==10,all5_original_failures=sum(not r['original_metrics']['checks']['linear_impulse'] for r in rows)==5,native_budget=10*len(cfg['replicas'])*cfg['repeats']*len(cfg['scopes'])*8==cfg['future_native_cap'])
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,paths['protocol']],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt)
    write_json(out/'report.json',dict(schema='pen_solver_tick_preparation:1',config=file_pin(cp),protocol=file_pin(paths['protocol']),prior=file_pin(paths['partition_report']),source_receipt=file_pin(out/'source_receipt.json'),original_case_reports=case_reports,rows=rows,checks=checks,passed=sum(checks.values()),total=len(checks),future_prerequisite_paths={k:str(paths[k]) for k in ['required_closed_producer','required_closed_review']},future_native_cap=cfg['future_native_cap'],actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,wall_s=time.perf_counter()-started,scope='CPU source/decoder/layout preparation only. No failing-tick private context captured;no replay/instrumentation/solver fix/Actor or ego admission. Full54 is separate;future worker must reject until both final prerequisites are successful.'))
    verify_source_snapshot(receipt,live=True);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[sum(checks.values()),len(checks)],native=0)),flush=True)
    if not all(checks.values()):raise ValueError('Retain preparation discrepancy '+repr([k for k,v in checks.items() if not v]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);main(p.parse_args())
