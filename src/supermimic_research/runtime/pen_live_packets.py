"""Complete current-source packets for native-screened evaluation and lifecycle QA."""
import time
import tomllib
import mujoco
import numpy as np
import torch
from unittest.mock import patch
from . import owned_pen_state as native
from .shared_snapshot_transport import SharedImmutableTransport,load_shared
from .pen_owned_probe_recorder import OwnedProbeRecorder,save_physics,source_frame_gap
from ..artifacts import file_pin,verify_file_pin
from ..strict_json import write_json
from ..learning.world_model import raw_tree_equal
from ..learning.pen_live_decision import from_owned_snapshot
from ..mechanics.probe_row_review import review_recorded_row
from ..mechanics.pen_reward_chain import native_components


class LivePenPackets:
    """External observer/storage;never inserted in owned runtime inventory."""
    def __init__(self,runtime,receipt,counter,folder,role):
        self.runtime=runtime;self.receipt=receipt;self.counter=counter;self.folder=folder;self.recorder=OwnedProbeRecorder(runtime);self.restores=[];self.captures=[]
        self.store=SharedImmutableTransport(folder/'shared_immutable.pt.gz');physics=save_physics(runtime,folder);model=runtime.env.sim.mj_model;f=runtime.features;body=f.body;qa=f.qa;da=int(model.jnt_dofadr[model.body_jntadr[body]])
        self.template=dict(dt=.0025,qids=f.qids.cpu().numpy().tolist(),dofs=f.dofs.cpu().numpy().tolist(),aids=runtime.action._position_ctrl_ids.cpu().numpy().tolist(),physics=physics,model=file_pin(folder/'model.mjb'),physical_context=file_pin(folder/'physical_context.npz'),
            model_layout=dict(object_body=body,object_qposadr=qa,object_dofadr=da,COM_BODY=model.body_ipos[body].tolist()),role=role)
        self.model=mujoco.MjModel.from_binary_path(str(verify_file_pin(self.template['model'])));self.physical=dict(np.load(verify_file_pin(self.template['physical_context'])));self.reward_cfg=tomllib.loads(runtime.config_path.read_text())['reward']

    def capture(self,folder,name):
        t=time.perf_counter();state=native.capture(self.runtime,self.receipt);record=self.store.save(folder/(name+'.pt.gz'),state);decoded=load_shared(record)
        if not raw_tree_equal(state,decoded):raise ValueError('Live compressed snapshot readback differs')
        write_json(folder/(name+'.transport.json'),record);self.captures.append(dict(path=str(folder/name),wall_s=time.perf_counter()-t,transport=record));return decoded,record

    def restore(self,record,label):
        t=time.perf_counter();before=self.counter.metadata();state=load_shared(record);audit=native.restore(self.runtime,state);after=native.capture(self.runtime,self.receipt)
        equal={k:raw_tree_equal(state[k],after[k]) for k in ['native','python','rng']};item=dict(label=label,audit=audit,complete_state_equal=equal,external_counter_unchanged=before==self.counter.metadata(),wall_s=time.perf_counter()-t);self.restores.append(item)
        if not audit['all_array_bytes_equal'] or not audit['all_scalar_values_equal'] or not all(equal.values()) or not item['external_counter_unchanged']:raise ValueError('Complete restore readback differs')
        return state

    def query(self,raw,folder,name,state,record):
        folder.mkdir(parents=True);t=time.perf_counter();runtime=self.runtime;phase=int(runtime.motion.time_steps[0]);raw=np.asarray(raw)
        if raw.shape!=(20,) or raw.dtype!=np.float32 or not np.isfinite(raw).all():raise ValueError('Original finite float32 sampled RAW20 required')
        decision=from_owned_snapshot(state,phase,self.template,raw);self.recorder.begin(raw);before=self.counter.native
        _,reward,term,trunc,info=runtime.step(torch.as_tensor(raw,device=runtime.device)[None]);z=self.recorder.finish();w={k:v.cpu().numpy().copy()[0] for k,v in runtime.contact_views.items()}
        np.savez_compressed(folder/'trace.npz',**z);np.savez_compressed(folder/'window.npz',**w);write_json(self.folder/'work.json',dict(stage=name,**self.counter.metadata()));end,endpin=self.capture(folder,'end')
        q=state['native']['data']['arrays'];qa=self.template['model_layout']['object_qposadr'];body=self.template['model_layout']['object_body'];gap=source_frame_gap(q['qpos'][0],qa,q['xmat'][0,body],q['xipos'][0,body]);view=dict(native=state,phase=phase,source_qpos_R=gap['qpos_R'],source_cached_R=q['xmat'][0,body]);actual=z['post_ctrl'][0,self.template['aids']]
        quality=dict(eight_ticks=self.counter.native-before==8,actual_PD=bool(abs(actual-decision['actual_PD20']).max()<=2e-7),no_aid=not bool(np.any(z['post_qfrc_applied']) or np.any(z['post_xfrc_applied'])))
        case=dict(self.template,name=name,available=True,qualified=all(quality.values()),quality=quality,actual_PD20=actual.tolist(),trace=file_pin(folder/'trace.npz'),window=file_pin(folder/'window.npz'),source=record,end=endpin,phase=phase,
            source_reward=float(reward[0]),terminated=bool(term[0]),truncated=bool(trunc[0]),native_integrations=self.counter.native-before,sampled_RAW20=raw.tolist(),applied_RAW20=decision['applied_RAW20'].tolist(),RAW_clip_count=decision['clipped_components'],
            termination_reasons={k:bool(v[0]) for k,v in info['termination_reasons'].items()})
        with patch.object(mujoco,'mj_step',side_effect=AssertionError('Read-only packet audit')),patch.object(mujoco,'mj_forward',side_effect=AssertionError('Read-only packet audit')):
            packet,audit=review_recorded_row(case,z,w,view,self.model,self.physical,dict(role=case['role'],object_penetration_m=.002,self_penetration_m=.0002,joint_excursion_rad=.001));component,formula,errors=native_components(case,z,w,decision['controller_context'],self.reward_cfg,runtime.objective_cfg)
        case.update(physical_audit=audit,reward_checks=formula,reward_errors=errors,source_reward_equal=abs(float(component[-1])-case['source_reward'])<=2e-6,packed_input_bytes=np.array_equal(packet['x'].astype(np.float32),decision['input102']),selected_path_internal_counter=runtime.owned_native_integrations,wall_s=time.perf_counter()-t)
        write_json(folder/'report.json',case)
        if not case['packed_input_bytes'] or not all(formula.values()) or not case['source_reward_equal']:raise ValueError('Input/controller/reward contract discrepancy retained')
        return case,end
