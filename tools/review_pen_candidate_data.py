"""Independent saved packet/chart/mask/split/TRAIN-scaler readback for every row."""
import argparse
import json
import sys
import tomllib
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.mechanics.warp_packet_decode import decode_saved_case
from supermimic_research.mechanics.pen_replay_fidelity import compare_response
from supermimic_research.learning.pen_response import fit_pen_scales
from supermimic_research.runtime.controlled_evaluation import write


def load(pin):return dict(np.load(verify_file_pin(pin)))


def main(a):
    if a.output.exists():raise ValueError('Fresh independent readback')
    p=json.loads(a.prepared.read_text());r=json.loads(verify_file_pin(p['parents']).read_text());cp=verify_file_pin(p['config']);cfg=tomllib.loads(cp.read_text())
    gates=tomllib.loads((cp.parent/cfg['gates']).read_text());data={role:load(pin) for role,pin in p['outputs'].items()}
    roles={k:role for role in ['TRAIN','CHECK'] for k in r['response_parent_split'][role]}
    a.output.mkdir();receipt=freeze_sources([Path(__file__),ROOT/'src/supermimic_research/mechanics/warp_packet_decode.py',
        ROOT/'src/supermimic_research/mechanics/pen_replay_fidelity.py',ROOT/'src/supermimic_research/learning/pen_response.py'],a.output/'source_snapshot',ROOT,'independent-readback-source')
    write(a.output/'source_receipt.json',receipt);checks=[];parents=[];cache={};max_errors={'x':0.,'y':0.,'heads':0.}
    for parent in p['parents_detail']:
        pid,role,k=parent['parent_id'],parent['role'],parent['k'];key=(parent['mode'],parent['dt'])
        c=next(c for c in r['cases'] if (c['name'],c['dt'])==key)
        if key not in cache:
            m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])));physical=load(c['GPU_physical_context'])
            jids=np.array([m.joint(f'robot/right_finger{i}_joint{j}').id for i in range(1,6) for j in range(1,5)])
            qids,dofs=m.jnt_qposadr[jids],m.jnt_dofadr[jids];aids=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in jids]);cache[key]=(m,physical,qids,dofs,aids)
        m,physical,qids,dofs,aids=cache[key];body,qa,da=(c['model_layout'][key] for key in ['object_body','object_qposadr','object_dofadr']);source=load(parent['source'])
        src_q,src_v=source['qpos'][0],source['qvel'][0];R0=Rotation.from_quat(src_q[qa+3:qa+7][[1,2,3,0]]).as_matrix()
        s=np.r_[src_q[qids],src_v[dofs],src_q[qa:qa+3],R0[:,0],R0[:,1],src_v[da:da+3],R0@src_v[da+3:da+6]]
        source_inside=Path(parent['source']['path']).is_relative_to(Path(p['parents']['path']).parent)
        z=data[role];global_index={(int(z['repeat'][i]),int(z['candidate'][i])):i for i in np.flatnonzero(z['parent_id']==pid)}
        local=[row for row in p['rows'] if row['parent_id']==pid];packet_cache={};wrenches={};Fn_cache={};base_valid={};graphs={}
        for row in local:
            rep,j=row['repeat'],row['candidate'];packet=load(row['packet'])
            if rep not in graphs:graphs[rep]=load(row['prediction'])
            d=graphs[rep];metrics,decoded=decode_saved_case(c,packet,m,physical);W=decoded['world_COM_wrench'];Fn=np.zeros((len(W),6));nonhand=np.zeros(len(W))
            for raw in packet['raw_contact_tick_plus23']:
                tick,g,h=int(raw[0]),int(raw[1]),int(raw[2]);aobj,bobj=m.geom_bodyid[g]==body,m.geom_bodyid[h]==body
                if aobj==bobj:continue
                name=m.geom(h if aobj else g).name or ''
                if name.startswith('robot/'):
                    group=5
                    for digit in range(1,6):
                        if name.startswith('robot/right_finger'+str(digit)+'_'):group=digit-1;break
                    Fn[tick,group]+=max(0.,float(raw[18]))
                else:nonhand[tick]+=max(0.,float(raw[18]))
            q,v=packet['post_qpos'][-1],packet['post_qvel'][-1];R1=Rotation.from_quat(q[qa+3:qa+7][[1,2,3,0]]).as_matrix()
            t=np.r_[q[qids],v[dofs],q[qa:qa+3],R1[:,0],R1[:,1],v[da:da+3],R1@v[da+3:da+6]]
            rotation=Rotation.from_matrix(R1@R0.T).as_rotvec();mean=W.mean(0)
            x=np.r_[s,source['qacc_warmstart'][0],packet['post_ctrl'][0,aids],c['dt']]
            y=np.r_[t[:43]-s[:43],rotation,t[49:]-s[49:],packet['post_qacc_warmstart'][-1],R0.T@mean[:3],R0.T@mean[3:],Fn.mean(0)]
            h=y.copy();mass=p['physics']['mass'];I=np.array(p['physics']['inertia_BODY']);gravity=np.array(p['physics']['gravity'])
            h[46:49]=mass*(t[49:52]-s[49:52])-.02*(mean[:3]+mass*gravity)
            h[49:52]=(R1@I@R1.T@t[52:55]-R0@I@R0.T@s[52:55])-.02*mean[3:]
            predicted={key:d[field][j].astype(float) for key,field in [('post_qpos','Q'),('post_qvel','V'),('post_ctrl','U'),('post_GPU_time','T')]}
            _,fidelity=compare_response(predicted,packet,d['W'][j].astype(float),W,qids,dofs,qa,da,gates)
            own=dict(source_qv_exact=np.array_equal(packet['pre_qpos'][0],source['qpos'][0]) and np.array_equal(packet['pre_qvel'][0],source['qvel'][0]),
                PD_exact=np.array_equal(packet['post_ctrl'][:,aids],np.broadcast_to(d['PD_targets'][j],packet['post_ctrl'][:,aids].shape)),no_aid=not np.any(packet['post_qfrc_applied']) and not np.any(packet['post_xfrc_applied']),
                capacity=not np.any(packet['overflow']),finite_targets=bool(np.isfinite(np.r_[x,y,h]).all()),no_nonhand_contact=bool(np.all(nonhand<=1e-9)),
                Fn_graph_agreement=bool(np.max(abs(Fn-d['Fn'][j]))<=gates['EFC_force_N']),rotation_chart=np.linalg.norm(rotation)<np.pi-.001,
                actual_next_warm=np.array_equal(y[52:78],packet['post_qacc_warmstart'][-1]))
            index=global_index[(rep,j)];errors={name:float(abs(actual-expected).max()) for name,actual,expected in [('x',z['x'][index],x),('y',z['y'][index],y),('heads',z['heads'][index],h)]}
            for name,value in errors.items():max_errors[name]=max(max_errors[name],value)
            base=bool(all(metrics['checks'].values()) and all(fidelity.values()) and all(own.values()));base_valid[(rep,j)]=base
            row_checks=dict(source_role=source_inside and role==roles[k] and k not in [0,4],target_chart=max(errors.values())<1e-12,
                recorded_checks={key:bool(value) for key,value in own.items()}==row['checks'],recorded_EFC={key:bool(value) for key,value in metrics['checks'].items()}==row['EFC_checks'],
                recorded_fidelity=fidelity==row['fidelity'],base_mask=base==row['base_valid'],no_graph_overflow=not np.any(d['bad'][j]),
                parent_keys=int(z['phase'][index])==k and row['parent_key']==parent['key'])
            checks.append(dict(parent_id=pid,repeat=rep,candidate=j,checks={key:bool(val) for key,val in row_checks.items()}));packet_cache[(rep,j)]=packet;wrenches[(rep,j)]=W;Fn_cache[(rep,j)]=Fn
        group_checks=[]
        for j in range(57):
            stable=True
            for rep in [1,2]:
                _,fidelity=compare_response(packet_cache[(rep,j)],packet_cache[(0,j)],wrenches[(rep,j)],wrenches[(0,j)],qids,dofs,qa,da,gates)
                stable &= all(fidelity.values()) and float(abs(Fn_cache[(rep,j)]-Fn_cache[(0,j)]).max())<=gates['EFC_force_N']
            good=all(base_valid[(rep,j)] for rep in range(3)) and stable
            group_checks.append(all(bool(z['valid'][global_index[(rep,j)]])==good for rep in range(3)))
        parents.append(dict(parent_id=pid,role=role,rows=len(local),shared_action_masks_exact=all(group_checks),all57x3=set(global_index)=={(rep,j) for rep in range(3) for j in range(57)}))
        print(json.dumps(dict(parent=pid,rows=len(local),max_errors=max_errors)),flush=True)
    z=data['TRAIN'];scales,head_scales=fit_pen_scales(z['x'][z['valid']],z['y'][z['valid']],z['heads'][z['valid']]);saved_scales=load(p['scales']);saved_heads=load(p['head_scales'])
    final_checks=dict(TRAIN_only_scales=all(scales[k].tobytes()==saved_scales[k].tobytes() for k in scales),TRAIN_only_head_scales=all(head_scales[k].tobytes()==saved_heads[k].tobytes() for k in head_scales),
        all_parent_masks=all(e['shared_action_masks_exact'] and e['all57x3'] for e in parents),role_parents_disjoint=not bool(set(data['TRAIN']['parent_id'])&set(data['CHECK']['parent_id'])),
        all_rows_checked=len(checks)==sum(len(z['x']) for z in data.values()))
    passed=sum(sum(e['checks'].values()) for e in checks)+sum(final_checks.values());total=sum(len(e['checks']) for e in checks)+len(final_checks)
    verify_source_snapshot(receipt,live=True)
    result=dict(schema='pen_candidate_response_readback:1',prepared=file_pin(a.prepared),source_receipt=file_pin(a.output/'source_receipt.json'),source=file_pin(__file__),
        checks=final_checks,row_checks=checks,parents=parents,max_target_errors=max_errors,passed=passed,total=total,native_integrations=0,neural_updates=0,
        scope='Every stored independent packet/chart/base and shared action mask, exact parent-phase roles and valid-TRAIN-only scales rechecked. Masks retain failed rows; no expert, unseen trajectory, physical gradient, real-ego or acceleration claim.')
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),checks=[passed,total],max_errors=max_errors)))
    if passed!=total:raise ValueError('Preserve failed readback; no fitting')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
