"""Reopen reviewed selected controls in the unchanged102/90 response schema."""
import json
import numpy as np
import mujoco
import torch
from ..artifacts import file_pin,verify_file_pin
from ..learning.pen_diagnostics import decode_pen_torch
from .controlled_pen_records import pack_case,native_state55
from .warp_packet_decode import decode_saved_case


def read(pin):return json.loads(verify_file_pin(pin).read_text())


def prepare(index):
    if len(index['reviewed_cases'])!=18 or index['complete']:raise ValueError('Exact inspected first18 partial cohort required')
    x=[];y=[];heads=[];valid=[];ids=[];phases=[];targets=[];meta=[];inputs={};models={};physical={};checks=[]
    for ci,slot in enumerate(index['reviewed_cases']):
        review=read(slot['report']);case=read(review['producer_case']);inputs[slot['report']['path']]=slot['report'];inputs[review['producer_case']['path']]=review['producer_case']
        if review['passed']!=review['total'] or not all(review['checks'].values()):raise ValueError('Complete input-capsule reviewed case required')
        previous=None
        for node,rnode in zip(case['nodes'],review['nodes']):
            pin=node[node['selection']['selected']];c=read(pin);inputs[pin['path']]=pin;key=c['model']['path']
            if key not in models:models[key]=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['model'])));physical[key]=dict(np.load(verify_file_pin(c['physical_context'])));inputs[key]=c['model'];inputs[c['physical_context']['path']]=c['physical_context']
            z=dict(np.load(verify_file_pin(c['trace'])));inputs[c['trace']['path']]=c['trace'];metrics,d=decode_saved_case(c,z,models[key],physical[key]);p=pack_case(models[key],c,z,d,np.asarray(c['qids']),np.asarray(c['dofs']),np.asarray(c['aids']),z['initial_warmstart'],c['physics'])
            qa=c['model_layout']['object_qposadr'];da=c['model_layout']['object_dofadr'];t=native_state55(z['post_qpos'][-1],z['post_qvel'][-1],c['qids'],c['dofs'],qa,da)
            decoded=decode_pen_torch(torch.tensor(p['x'][0,:55]),torch.tensor(p['y'][0])).numpy();tol=128*np.finfo(np.float64).eps*(1+abs(t));row_checks=dict(one_response=p['x'].shape==(1,102) and p['y'].shape==(1,90),decoder_receipt=metrics==c['physical_audit']['decoder_metrics'],original_packet_mask=bool(p['valid'][0])==c['physical_audit']['packet_quality'],original_selected_mask=c['physical_audit']['valid']==rnode['selected_physical_valid'],target_chart=bool(np.all(abs(decoded-t)<=tol)),actual_warm=bool(np.array_equal(p['y'][0,52:78],z['post_qacc_warmstart'][-1])))
            if previous is not None:row_checks['warm_chain']=np.array_equal(previous,p['x'][0,55:81])
            if not all(row_checks.values()):raise ValueError('Selected dataset discrepancy '+c['name']+' '+repr(row_checks))
            x.append(p['x'][0]);y.append(p['y'][0]);heads.append(p['heads'][0]);targets.append(t);valid.append(c['physical_audit']['valid']);ids.append(ci);phases.append(node['phase']);previous=p['y'][0,52:78];checks.append(row_checks)
            meta.append(dict(case_id=ci,case=case['name'],phase=node['phase'],selected_query=pin,physical_valid=c['physical_audit']['valid'],source_task_terminated=c['terminated'],source_reference_complete=c['truncated']))
    for pin in inputs.values():verify_file_pin(pin)
    data=dict(x=np.asarray(x),y=np.asarray(y),heads=np.asarray(heads),valid=np.asarray(valid,bool),case_id=np.asarray(ids,np.int64),control_index=np.asarray(phases,np.int64),next55=np.asarray(targets))
    if not all(np.isfinite(data[k]).all() for k in ['x','y','heads','next55']):raise ValueError('Finite original recorded matrices required')
    return data,meta,list(inputs.values()),checks
