"""Assemble only independently reviewed selected native poses,without replay."""
import json
import numpy as np
from ..artifacts import verify_file_pin


def selected_poses(case,review):
    if (review['passed']!=review['total'] or not all(review['checks'].values()) or review['summary']['name']!=case['name'] or len(review['nodes'])!=len(case['nodes'])):raise ValueError('Exact independently reviewed selected case required')
    post=[];mocap_pos=[];mocap_quat=[];selected=[];initial=None;model=None;layout=None;previous=None
    for node,r in zip(case['nodes'],review['nodes']):
        choice=node['selection']['selected']
        if choice!=r['selected'] or node['phase']!=r['phase']:raise ValueError('Selection changed after review')
        pin=node[choice]
        if pin is None:raise ValueError('Selected query missing')
        packet=json.loads(verify_file_pin(pin).read_text());z=dict(np.load(verify_file_pin(packet['trace'])))
        if packet['phase']!=node['phase'] or len(z['post_qpos'])!=8:raise ValueError('Selected query phase/length differs')
        if model is None:model=packet['model'];layout=packet['model_layout'];initial=z['pre_qpos'][0].copy()
        if packet['model']!=model or packet['model_layout']!=layout:raise ValueError('Selected physical model changed')
        if previous is not None and not np.array_equal(previous,z['pre_qpos'][0]):raise ValueError('Selected pose chain has a reset/discontinuity')
        previous=z['post_qpos'][-1].copy();post.append(z['post_qpos']);mocap_pos.append(z['post_mocap_pos']);mocap_quat.append(z['post_mocap_quat']);selected.append(pin)
    q=np.concatenate(post);mp=np.concatenate(mocap_pos);mq=np.concatenate(mocap_quat)
    if q.shape!=(8*len(case['nodes']),27) or not all(np.isfinite(v).all() for v in [q,mp,mq]):raise ValueError('Finite original Hand2/pen selected native poses required')
    return dict(initial_qpos=initial,post_qpos=q,post_mocap_pos=mp,post_mocap_quat=mq,model=model,model_layout=layout,selected_queries=selected,duration_s=.0025*len(q),dt_s=.0025)
