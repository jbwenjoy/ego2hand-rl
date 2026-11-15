"""Live read-only native hook qualification, distinct from checkpoint schemas."""
import argparse
import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import warp as wp
from qualify_resident_contact_replay import oracle,empty_window,reset_selected,fold,canonical_joints,FIELDS
from supermimic_research.artifacts import file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.resident_contact_metrics import ResidentContactMetrics
from supermimic_research.runtime.pen_native_policy import NativePenPolicyAdapter
from supermimic_research.runtime.gpu_model_snapshot import model_leaves
from supermimic_research.runtime.pen_portable_state import engine_sources
from supermimic_research.mechanics.portable_contact_records import contacts


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def host_state(root):
    arrays,scalars=model_leaves(root)
    return dict(arrays={k:v.numpy().copy() for k,v in arrays.items()},scalars=scalars)


def exact_state(a,b):
    return a['scalars']==b['scalars'] and a['arrays'].keys()==b['arrays'].keys() and all(a['arrays'][k].tobytes()==b['arrays'][k].tobytes() for k in a['arrays'])


def fingerprint(arrays):
    h=hashlib.sha256()
    for a in arrays:h.update(a.numpy().tobytes())
    return h.hexdigest()


def source_packet(adapter):
    s=adapter.env.sim;d=s.wp_data
    return dict(qpos=d.qpos.numpy().copy(),time=d.time.numpy().copy(),cone=int(s.wp_model.opt.cone),
        contact=dict(R_efc_phase_world=d.xmat.numpy()[:,adapter.features.body].copy(),COM_efc_phase_world=d.xipos.numpy()[:,adapter.features.body].copy()))


def qualify_worlds(adapter,cfg,out,original_step):
    from mjlab.sim.sim import Simulation
    sim=adapter.env.sim;m=sim.mj_model;d=sim.wp_data;W=adapter.num_envs;jids=canonical_joints(m);body=adapter.features.body
    metric=ResidentContactMetrics(m,sim.wp_model,d,body,jids);bounds=sim.wp_model.jnt_range.numpy().copy()
    checks={};maximum={};records=[];sources={};sample=0;window=None;terminal_resets=[]
    input_arrays=[getattr(d.contact,k) for k in FIELDS]+[d.nacon,d.nefc,d.efc.force,d.qpos,d.xmat,d.xipos,d.time,d.overflow,sim.wp_model.jnt_range,
        metric.object_tag,metric.digit,metric.joint_ids,metric.qids]
    model_before=host_state(sim.wp_model)
    # An alternate caller stream supplies CUDA IDs; there is no host ID readback
    # inside reset_window. Invalid IDs must leave every world untouched.
    alternate=torch.cuda.Stream();bad_checks=[]
    metric.sample();initial=metric.numpy()['window']
    for values in ([0,0],[W],[2**32]):
        with torch.cuda.stream(alternate):metric.reset_window(torch.tensor(values,device='cuda:0',dtype=torch.int64))
        result=metric.numpy()['window'];bad_checks.append(np.array_equal(initial['counts'],result['counts']) and np.array_equal(initial['wrench_sum'],result['wrench_sum']) and bool(result['global_error'][0]&1024))
    metric.reset_window();checks['GPU_reset_invalid_no_partial_write']=all(bad_checks);checks['full_reset_clears_error']=not metric.numpy()['window']['global_error'].any()
    checks['metric_setup_no_physics']=adapter.physics_integrations==0
    reset_ids=torch.tensor([0] if W==1 else [0,7,31],device='cuda:0',dtype=torch.int64)
    def compare(label,a,e):
        delta=float(np.max(abs(a.astype(np.float64)-e))) if a.size else 0.;maximum[label]=max(maximum.get(label,0.),delta)
        checks[label]=checks.get(label,True) and bool(np.allclose(a,e,atol=cfg['absolute_tolerance'],rtol=cfg['relative_tolerance']))
    def step(current):
        nonlocal sample,window
        if current is not sim:return original_step(current)
        save=sample in cfg['source_capture_samples']
        before_physics=host_state(d) if save else None
        original_step(current)
        packet=dict(qpos=d.qpos.numpy().copy(),qvel=d.qvel.numpy().copy(),ctrl=d.ctrl.numpy().copy(),time=d.time.numpy().copy(),cone=int(sim.wp_model.opt.cone),
            contact=contacts(sim.wp_model,d,m,body))
        pre_metric=host_state(d) if save else None
        before=fingerprint(input_arrays)
        if sample==cfg['partial_window_reset_before_sample']:
            with torch.cuda.stream(alternate):metric.reset_window(reset_ids)
            reset_selected(window,packet,np.array([0] if W==1 else [0,7,31]))
        metric.sample();actual=metric.numpy();after=fingerprint(input_arrays)
        checks['metric_input_bytes_unchanged']=checks.get('metric_input_bytes_unchanged',True) and before==after
        expected,local,legacy=oracle(m,body,jids,bounds,packet);fold(window,expected,packet)
        compare('raw_local_force',local,packet['contact']['local_force6'])
        compare('raw_legacy_WORLD',legacy,packet['contact']['wrench_WORLD_COM'])
        for key in expected:compare('instant_'+key,actual['instant'][key],expected[key])
        for key in window:compare('window_'+key,actual['window'][key],window[key])
        if save:
            post_metric=host_state(d);checks['complete_Data_bytes_unchanged_'+str(sample)]=exact_state(pre_metric,post_metric)
            sources[sample]=dict(before_physics=before_physics,after_physics_before_metrics=pre_metric,after_metrics=post_metric,
                source_scope='New live diagnostic at zero-based native sample; not the historical W32 failure source or a numerical replay experiment.')
        records.append(dict(packet=packet,metrics=actual,input_SHA256=before))
        sample+=1
    Simulation.step=step
    try:
        for k in range(cfg['controls']):
            if k==cfg['reset_before_control']:adapter.reset(reset_ids)
            metric.reset_window();window=empty_window(source_packet(adapter))
            w=torch.arange(W,device=adapter.device)[:,None];j=torch.arange(20,device=adapter.device)[None,:]
            action=.05*torch.sin(.2*k+.13*w+.03*j)
            _,_,terminated,truncated,_=adapter.step(action)
            ids=(terminated|truncated).nonzero().flatten()
            if len(ids):
                terminal_resets.append(dict(after_control=k,worlds=ids.cpu().tolist()))
                adapter.reset(ids)
            checks['exact_decimation']=checks.get('exact_decimation',True) and sample==(k+1)*cfg['decimation']
    finally:Simulation.step=original_step
    checks['observed_integrations']=sample*W==adapter.physics_integrations==cfg['controls']*cfg['decimation']*W
    checks['native_model_bytes_unchanged']=exact_state(model_before,host_state(sim.wp_model))
    checks['every_sample_recorded']=len(records)==cfg['controls']*cfg['decimation']
    checks['not_attached_to_checkpoint_schema']=not hasattr(adapter,'resident_contact_metrics')
    target=out/f'W{W}_records.pt';torch.save(records,target)
    full=out/f'W{W}_full_sources.pt';torch.save(dict(model=model_before,data_sources=sources),full)
    return dict(worlds=W,checks=checks,passed=sum(checks.values()),total=len(checks),max_abs_errors=maximum,records=file_pin(target),full_sources=file_pin(full),
        native_integrations=sample*W,diagnostic_metric_samples=metric.setup_metric_samples+1,optimizer_steps=0,metric_metadata=metric.metadata(),
        model_array_fields=len(model_before['arrays']),data_array_fields=len(next(iter(sources.values()))['before_physics']['arrays']),
        no_automatic_reset=True,terminal_resets=terminal_resets,explicit_reset_before_control=cfg['reset_before_control'],reset_worlds=reset_ids.cpu().tolist())


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or cfg['controls']!=16 or cfg['decimation']!=8:raise ValueError('Fresh declared live metric qualification')
    out.mkdir();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})
    paths += [Path(__file__).resolve(),ROOT/'tools/qualify_resident_contact_replay.py',cp,(cp.parent/cfg['adapter_config']).resolve()]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    from mjlab.sim.sim import Simulation
    original=Simulation.step;cases=[];observed=0
    def count_step(sim):
        nonlocal observed
        original(sim);observed+=sim.num_envs
    Simulation.step=count_step
    try:
        with torch.jit.optimized_execution(False):
            for W in cfg['worlds']:
                adapter=NativePenPolicyAdapter((cp.parent/cfg['adapter_config']).resolve(),W)
                try:cases.append(qualify_worlds(adapter,cfg,out,count_step))
                finally:adapter.close()
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),observed_native_integrations=observed,optimizer_steps=0))
        raise
    finally:Simulation.step=original
    verify_source_snapshot(receipt,live=True);checks={f"W{c['worlds']}_"+k:v for c in cases for k,v in c['checks'].items()}
    checks['actual_native_budget']=observed==sum(c['native_integrations'] for c in cases)==4224
    result=dict(schema='resident_contact_live_native_hook:1',config=file_pin(cp),source_receipt=file_pin(out/'source_receipt.json'),engine_sources=engine_sources(),
        cases=cases,checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=observed,optimizer_steps=0,
        portable_schema_change=False,role='DIAGNOSTIC_NEVER_FIT',scope=cfg['scope']+' Metric source captures are complete native Data/model fields for this run, not a portable resume or a matched historical divergence diagnosis.')
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],native_integrations=observed)))
    if not all(checks.values()):raise ValueError('Preserve live metric discrepancy before training integration')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
