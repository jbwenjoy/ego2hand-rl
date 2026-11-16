import unittest
from types import SimpleNamespace
import numpy as np
from supermimic_research.runtime.pen_solve_exit_observer_v2 import expected_layout,SolveExitGraphObserverV2
from supermimic_research.mechanics.pen_solve_exit_epoch_v2 import physical_view,review
from supermimic_research.runtime.pen_solve_exit_version2 import worker_bindings
from supermimic_research.runtime import pen_solve_exit_observer as observer_module
from supermimic_research.mechanics import pen_solve_exit_epoch as epoch_module


class ActualNewtonContextLayout(unittest.TestCase):
    def test_actual_CPU_allocation_factory_matches_padded_and_empty_buffers(self):
        import warp as wp
        from mujoco_warp._src import solver
        m=SimpleNamespace(nv=26,nv_pad=28,opt=SimpleNamespace(solver=2));d=SimpleNamespace(nworld=1,njmax=64)
        with wp.ScopedDevice('cpu'):ctx=solver._create_solver_context(m,d)
        for key,shape in expected_layout(26,28,2,1).items():self.assertEqual(getattr(ctx,key).shape,shape)
        self.assertEqual(ctx.grad.dtype,wp.float32);np.testing.assert_array_equal(ctx.grad.numpy()[0,26:],0)

    def test_empty_native_Mgrad_is_preserved_without_alias_rejection(self):
        import warp as wp
        from mujoco_warp._src import solver
        with wp.ScopedDevice('cpu'):
            ctx=solver._create_solver_context(SimpleNamespace(nv=26,nv_pad=28,opt=SimpleNamespace(solver=2)),SimpleNamespace(nworld=1,njmax=64))
            obs=object.__new__(SolveExitGraphObserverV2);obs.wp=wp;obs.context_owners=[];obs.data=SimpleNamespace(qacc=wp.zeros((1,26),dtype=float));obs.executions=wp.zeros(1,dtype=int)
            obs.buffers={'data.qacc':wp.empty_like(obs.data.qacc),'context.grad':wp.empty_like(ctx.grad),'context.Mgrad':wp.empty_like(ctx.Mgrad)}
            before=ctx.grad.numpy().copy();obs._emit(ctx)
        self.assertEqual(obs.count(),1);self.assertEqual(obs.buffers['context.Mgrad'].numpy().shape,(1,0));np.testing.assert_array_equal(ctx.grad.numpy(),before);np.testing.assert_array_equal(obs.buffers['context.grad'].numpy(),before)

    def example(self):
        ticks,nv=8,26;acc=np.zeros((ticks,nv));acc[:,0]=1.;v0=np.arange(ticks)[:,None]*.0025*acc;v1=v0+.0025*acc;q0=np.zeros((ticks,27));q0[:,3]=1.;q0[:,0]=.0025**2*np.arange(ticks)*(np.arange(ticks)+1)/2;q1=q0.copy();q1[:,0]+=.0025*v1[:,0];zero=np.zeros_like(acc);contact=.02*acc
        packet=dict(pre_qpos=q0,post_qpos=q1,pre_qvel=v0,post_qvel=v1,post_qacc=acc,post_qfrc_constraint=contact,post_qfrc_applied=zero,post_xfrc_applied=np.zeros((ticks,1,6)),post_GPU_time=(np.arange(ticks)+1)*.0025)
        fields=dict(qpos=q0,qvel=v0,time=np.arange(ticks)*.0025,qacc=acc,M=np.tile((.03*np.eye(nv))[None],(ticks,1,1)),**{'efc.Ma':.03*acc},qfrc_smooth=zero,qfrc_constraint=contact,qfrc_applied=zero,xfrc_applied=np.zeros((ticks,1,6)),solver_niter=np.ones(ticks))
        grad=np.zeros((ticks,28));grad[:,0]=.002;context=dict(grad=grad,Mgrad=np.empty((ticks,0)),grad_scale=np.full(ticks,5.),grad_dot=np.ones(ticks),newton_decrement=np.ones(ticks),improvement=np.ones(ticks));copies={'data.'+k:v[:,None] for k,v in fields.items()};copies.update({'context.'+k:v[:,None] for k,v in context.items()})
        model={'M_mulm_rowadr':None,'M_mulm_col':None,'M_mulm_madr':None,'stat.meaninertia':[1.],'opt.tolerance':[1e-12]};case=dict(dt=.0025,physics=dict(mass=.03,gravity=[0,0,0]),model_layout=dict(object_dofadr=0));decoded=dict(world_COM_wrench=np.c_[contact[:,:3],np.zeros((ticks,3))],linear_impulse_residual_N_s=np.tile([.000025,0,0],(ticks,1)))
        return packet,copies,model,case,decoded,[0,0,0],50

    def test_physical_force_uses26_while_raw28_is_unchanged(self):
        args=self.example();before=args[1]['context.grad'].copy();r,a=review(*args)
        self.assertEqual(r['passed'],r['total']);self.assertEqual(a['scaled_gradient_N'].shape,(8,26));np.testing.assert_allclose(a['actual_scaled_gradient'][:,0],.000025);np.testing.assert_array_equal(args[1]['context.grad'],before);self.assertEqual(r['context_layout']['Mgrad_shape'],[8,1,0])

    def test_bad_padding_or_fabricated_Mgrad_rejects(self):
        a=self.example()[1];a['context.grad'][0,0,27]=1.
        with self.assertRaises(ValueError):physical_view(a,26)
        a=self.example()[1];a['context.Mgrad']=np.zeros((8,1,26))
        with self.assertRaises(ValueError):physical_view(a,26)

    def test_explicit_bindings_restore_on_failure(self):
        old_class,old_review=observer_module.SolveExitGraphObserver,epoch_module.review
        with self.assertRaises(RuntimeError):
            with worker_bindings() as audit:
                self.assertIs(observer_module.SolveExitGraphObserver,SolveExitGraphObserverV2)
                raise RuntimeError('controlled scope failure')
        self.assertIs(observer_module.SolveExitGraphObserver,old_class);self.assertIs(epoch_module.review,old_review);self.assertTrue(audit['bindings_restored'])


if __name__=='__main__':unittest.main()
