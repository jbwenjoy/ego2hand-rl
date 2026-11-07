"""Small exact selected-pair pose engagement; qualification remains separate."""
import numpy as np


def engage_pairs(model,qpos,audit,pairs,target_m,*,maximum_iterations=8):
    import mujoco
    data=mujoco.MjData(model);q=np.asarray(qpos,float).copy();log=[];queries=0
    lower=np.maximum(model.jnt_range[:20,0],model.actuator_ctrlrange[:,0]);upper=np.minimum(model.jnt_range[:20,1],model.actuator_ctrlrange[:,1])
    if len(pairs)!=3 or not -.000075<=target_m<0:raise ValueError('Three explicit groups and declared small engagement depth')
    def gap(proposal,pair):
        nonlocal queries
        if np.any(proposal[:20]<lower) or np.any(proposal[:20]>upper):raise ValueError('Raw legal pose and command, never clamp')
        data.qpos[:]=proposal;mujoco.mj_kinematics(model,data);queries+=1
        answer=audit.query(audit.shape(data,pair[0]),audit.shape(data,pair[1]),force_exact=True)
        if answer['gap_m'] is None:raise ValueError('Unknown selected-pair gap')
        return answer['gap_m']
    for digit,pair in enumerate(pairs):
        for iteration in range(maximum_iterations):
            current=gap(q,pair);error=current-target_m
            if abs(error)<=5e-6:break
            gradient=np.zeros(4);indices=np.arange(digit*4,digit*4+4)
            for j,k in enumerate(indices):
                step=min(1e-5,(q[k]-lower[k])/2,(upper[k]-q[k])/2)
                if step<1e-10:continue
                plus=q.copy();minus=q.copy();plus[k]+=step;minus[k]-=step
                gradient[j]=(gap(plus,pair)-gap(minus,pair))/(2*step)
            if np.linalg.norm(gradient)<1e-10:break
            delta=-error*gradient/(gradient@gradient)
            if abs(delta).max()>.02:delta*=.02/abs(delta).max()
            accepted=False
            for alpha in [1.,.5,.25]:
                proposal=q.copy();proposal[indices]+=alpha*delta
                if np.any(proposal[:20]<lower) or np.any(proposal[:20]>upper):continue
                after=gap(proposal,pair);accepted=abs(after-target_m)<abs(error)
                log.append(dict(digit=digit+1,iteration=iteration,pair=pair,before_m=current,after_m=after,
                    q20=proposal[:20].tolist(),accepted=bool(accepted)))
                if accepted:q=proposal;break
            if not accepted:break
    if not np.array_equal(q[12:],qpos[12:]):raise ValueError('Only three declared fingers may engage')
    return q,dict(queries=queries,proposals=log,final_selected_gaps_m=[gap(q,pair) for pair in pairs])
