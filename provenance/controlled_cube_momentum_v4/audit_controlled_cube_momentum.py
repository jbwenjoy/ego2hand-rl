"""Short original-GPU replay isolating contact/generalized force/integration."""
from pathlib import Path
import argparse,json,random,sys,tomllib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import ArtifactRegistry,file_pin,verify_file_pin
from supermimic_research.runtime.controlled_assets import bind_controlled_assets
from supermimic_research.mechanics.warp_contact_labels import read_warp_contact_labels


def main(args):
    import torch,warp as wp,mujoco,mujoco_warp as mjwarp
    from mjlab.envs import ManagerBasedRlEnv
    from dex_supermimic.tasks.dex_reorient.config.task import make_env_cfg
    if args.output.exists():raise ValueError('Fresh audit output')
    r=json.loads(args.report.read_text(encoding='utf8'));c=next(c for c in r['cases'] if c['task']=='hand2-reorient' and c['controller']=='ZERO');z=np.load(verify_file_pin(c['trace']))
    cp=verify_file_pin(r['config']);config=tomllib.loads(cp.read_text());registry=ArtifactRegistry(cp.parent/config['artifact_manifest'])
    torch.set_num_threads(4);wp.set_device('cuda:0');random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    cfg=make_env_cfg(1,50,play=True,task='hand2-reorient');cfg.scene.num_envs=1;cfg.seed=config['seed'];cfg.auto_reset=False
    cfg.commands['reorient_command'].min_goal_interval=3.;cfg.commands['reorient_command'].debug_vis=False
    bind_controlled_assets(cfg,'hand2-reorient',registry)
    args.output.mkdir();(args.output/'intent.json').write_text(json.dumps(dict(source=file_pin(__file__),run=file_pin(args.report),ticks=12,neural_updates=0),indent=2)+'\n')
    env=ManagerBasedRlEnv(cfg,device='cuda:0');original=env.sim.step
    try:
        env.reset();m=env.sim.mj_model;d=mujoco.MjData(m);mjwarp.get_data_into(d,m,env.sim.wp_data)
        native=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,native,mujoco.mjtState.mjSTATE_INTEGRATION)
        if not np.array_equal(native,z['initial_native_state']):raise ValueError('Exact original initial state required')
        body=c['model_layout']['object_body'];da=c['model_layout']['object_dofadr'];mass=float(env.sim.wp_model.body_mass.numpy()[0,body]);dt=c['dt'];rows=[];contact_debug=[]
        def step():
            before=env.sim.data.qvel[0].cpu().numpy().copy();original();mjwarp.get_data_into(d,m,env.sim.wp_data)
            F=np.zeros(3)
            for i in range(d.ncon):
                con=d.contact[i];g1,g2=map(int,con.geom);b1,b2=m.geom_bodyid[[g1,g2]]
                if (b1==body)==(b2==body):continue
                force=np.empty(6);mujoco.mj_contactForce(m,d,i,force);F+=(1. if b2==body else -1.)*con.frame.reshape(3,3).T@force[:3]
            if np.any(d.qfrc_applied) or np.any(d.xfrc_applied):raise ValueError('No force aid')
            labels=read_warp_contact_labels(env.sim.wp_model,env.sim.wp_data)
            for i in range(d.ncon):
                cf=np.empty(6);mujoco.mj_contactForce(m,d,i,cf)
                if np.max(abs(cf-labels['local_force6'][i]))>1e-6:
                    contact_debug.append(dict(tick=len(rows),geoms=[m.geom(int(g)).name for g in labels['geom'][i]],
                        GPU_efc_addresses=labels['efc_address'][i].tolist(),host_efc_address=int(d.contact[i].efc_address),
                        nefc=labels['nefc'],dimension=int(labels['dim'][i]),invalid=bool(labels['invalid_constraint_index'][i]),
                        host_force6=cf.tolist(),direct_force6=labels['local_force6'][i].tolist(),
                        host_adhesion=float(d.contact[i].adhesion),
                        normal_difference_plus_host_adhesion=float(cf[0]-labels['local_force6'][i,0]+d.contact[i].adhesion),
                        GPU_friction=labels['friction'][i].tolist(),host_friction=d.contact[i].friction.tolist()))
            direct=np.zeros(3);inactive=0
            for i,(g1,g2) in enumerate(labels['geom']):
                b1,b2=m.geom_bodyid[[g1,g2]]
                if (b1==body)==(b2==body):continue
                direct+=(1. if b2==body else -1.)*labels['frame'][i].reshape(3,3).T@labels['local_force6'][i,:3]
                inactive+=int(labels['efc_address'][i,0]<0)
            rows.append(dict(before=before,after=d.qvel.copy(),qacc=d.qacc.copy(),constraint=d.qfrc_constraint.copy(),bias=d.qfrc_bias.copy(),passive=d.qfrc_passive.copy(),actuator=d.qfrc_actuator.copy(),contact_F=F,direct_F=direct,inactive_object_contacts=inactive,post_qpos=d.qpos.copy()))
        env.sim.step=step
        for _ in range(6):
            _,_,terminated,truncated,_=env.step(torch.zeros((1,20),device='cuda:0'))
            if bool(terminated[0] or truncated[0]):raise ValueError('Unexpected early native termination')
        a={key:np.array([row[key] for row in rows]) for key in rows[0]};dv=a['after'][:,da:da+3]-a['before'][:,da:da+3]
        constraint=a['constraint'][:,da:da+3];total=constraint-a['bias'][:,da:da+3]+a['passive'][:,da:da+3]+a['actuator'][:,da:da+3]
        errors=dict(contact_vs_generalized_force_N=a['contact_F']-constraint,
            direct_contact_vs_generalized_force_N=a['direct_F']-constraint,
            direct_contact_momentum_residual_N_s=mass*dv-dt*(a['direct_F']+mass*m.opt.gravity),
            force_balance_N=mass*a['qacc'][:,da:da+3]-total,
            integrator_velocity_residual_m_s=dv-dt*a['qacc'][:,da:da+3],
            generalized_momentum_residual_N_s=mass*dv-dt*total,
            contact_only_momentum_residual_N_s=mass*dv-dt*(a['contact_F']+mass*m.opt.gravity))
        np.savez_compressed(args.output/'trace.npz',**a,**errors)
        replay=np.array_equal(a['post_qpos'],z['post_qpos'][:12]) and np.array_equal(a['after'],z['post_qvel'][:12])
        result=dict(schema='controlled_cube_momentum_receipt:1',source=file_pin(__file__),original=file_pin(args.report),trace=file_pin(args.output/'trace.npz'),
            exact_initial=True,exact_twelve_tick_state_replay=replay,ticks=len(rows),neural_updates=0,
            maximum_norms={key:float(np.linalg.norm(value,axis=1).max()) for key,value in errors.items()},
            mass_kg=mass,dt=dt,integrator=int(m.opt.integrator),cone=int(m.opt.cone),
            inactive_object_contact_counts=a['inactive_object_contacts'].tolist(),
            differing_contact_decodes=contact_debug,
            decoder_source=file_pin(ROOT/'src/supermimic_research/mechanics/warp_contact_labels.py'),
            scope='Original GPU solver, ZERO, first120ms. Separate copied constraint force, contact force, qacc and actual dv; no altered plant or force aid.')
        (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    finally:
        env.sim.step=original;env.close()


if __name__=='__main__':
    import torch
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    with torch.jit.optimized_execution(False):main(p.parse_args())
