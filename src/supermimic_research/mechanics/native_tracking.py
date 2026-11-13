"""Reference-conditioned native receding-horizon baseline, not learned control."""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .transition_records import state_rotation


class TrackingResidual:
    def __init__(self,reference,grasp,joint_range,config,task):
        self.reference=reference;self.grasp=np.asarray(grasp);self.joint_range=np.asarray(joint_range);self.config=config;self.task=task
    def goal(self,k):
        r=self.reference;i=0 if self.task=='HOLD' else int(k)
        return r['COM_world_m'][i],r['T_world_from_BODY'][i,:3,:3],np.zeros(3) if self.task=='HOLD' else r['COM_velocity_world_m_s'][i],np.zeros(3) if self.task=='HOLD' else r['angular_velocity_world_rad_s'][i]
    def residual(self,prediction,k,action,previous):
        c=self.config;rows=[]
        for h,(state,contact) in enumerate(zip(prediction['states55'],prediction['contacts'])):
            p,R,v,w=self.goal(k+h+1);angle=Rotation.from_matrix(state_rotation(state)@R.T).as_rotvec()
            excursion=max(0.,float(np.max(self.joint_range[:,0]-state[:20])),float(np.max(state[:20]-self.joint_range[:,1])))
            rows.extend([(state[40:43]-p)/c['COM_scale_m'],angle/c['angle_scale_rad'],(state[49:52]-v)/c['velocity_scale_m_s'],(state[52:55]-w)/c['omega_scale_rad_s'],
                np.sqrt(c['joint_weight'])*(state[:20]-self.grasp)/.1,
                np.sqrt(c['contact_weight'])*np.maximum(.1-contact['normal_by_finger_N'][:3],0)/.1,
                np.array([np.sqrt(c['other_force_weight'])*max(contact['other_normal_N']-.1,0)/.1,contact['table_normal_N'],
                    max(contact['max_native_penetration_m']-.002,0)/.002,max(contact['max_self_penetration_m']-.0002,0)/.0002,excursion/.001])])
        rows.extend([np.sqrt(c['action_weight'])*action,np.sqrt(c['smooth_weight'])*(action-previous)])
        value=np.concatenate(rows)
        if not np.isfinite(value).all():raise ValueError('Nonfinite native optimization residual')
        return value


class NativeTrackingPlanner:
    def __init__(self,oracle,residual,config):
        self.oracle=oracle;self.objective=residual;self.config=config;self.previous=np.zeros(20);self.decisions=0
    def state_dict(self):return dict(previous=self.previous.copy(),decisions=self.decisions)
    def restore(self,state):self.previous=np.array(state['previous'],float).copy();self.decisions=int(state['decisions'])
    def choose(self,snapshot):
        c=self.config;active=np.array(c['active_joints']);horizon=min(c['horizon'],100-snapshot['k']);previous=self.previous.copy();records=[];start=time.perf_counter()
        def evaluate(action,label):
            prediction=self.oracle.rollout(snapshot,action,horizon);r=self.objective.residual(prediction,snapshot['k'],action,previous);cost=float(r@r)
            records.append(dict(label=label,action=action.copy(),residual=r.copy(),cost=cost));return cost,r,prediction
        zero=np.zeros(20);cz,rz,pz=evaluate(zero,'zero_incumbent');cp,rp,pp=evaluate(previous,'previous_incumbent')
        action,r,prediction,cost=(zero,rz,pz,cz) if cz<=cp else (previous.copy(),rp,pp,cp);incumbent_cost=cost;J=[]
        for joint in active:
            low,high=action.copy(),action.copy();low[joint]=max(-1.,action[joint]-c['FD_action']);high[joint]=min(1.,action[joint]+c['FD_action'])
            _,a,_=evaluate(low,f'FD_minus_{joint}');_,b,_=evaluate(high,f'FD_plus_{joint}');J.append((b-a)/(high[joint]-low[joint]))
        J=np.array(J).T;delta=-np.linalg.solve(J.T@J+c['ridge']*np.eye(len(active)),J.T@r);delta=np.clip(delta,-c['max_action_step'],c['max_action_step'])
        for factor in c['line_search']:
            candidate=action.copy();candidate[active]=np.clip(action[active]+factor*delta,-1,1);cc,rr,pr=evaluate(candidate,f'line_{factor}')
            if cc<cost:cost,r,prediction=cc,rr,pr;best=candidate.copy()
        selected=best if cost<incumbent_cost else action
        if np.any(abs(selected)>1) or np.any(selected[12:]!=0):raise ValueError('Declared bounded first-three-finger action mask')
        self.previous=selected.copy();self.decisions+=1
        return selected,dict(k=snapshot['k'],horizon=horizon,incumbent_cost=incumbent_cost,selected_cost=cost,wall_s=time.perf_counter()-start,oracle_calls=len(records),
            proposal_actions=np.array([p['action'] for p in records]),proposal_costs=np.array([p['cost'] for p in records]),proposal_residuals=np.array([p['residual'] for p in records]),
            selected_first_native_state=prediction['native_states'][0].copy(),selected_first55=prediction['states55'][0].copy(),
            prediction_scope='Privileged full-state native horizon, constant raw action with actual EMA updates. Sampled endpoint contact costs; execution validates every tick.')
