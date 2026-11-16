"""Copy a completed iteration to its external trace slot; no native writes."""
import warp as wp

POST_VECTOR_FIELDS=['raw_grad','search','qacc','Ma','qfrc_constraint']
POST_FLOAT_FIELDS=['grad_scale','grad_dot','newton_decrement','alpha','improvement','time']
POST_INT_FIELDS=['valid','solver_niter','state_changed_count','quad_changed_count','done','ls_exhausted','search_unchanged']


@wp.kernel
def copy_post_iteration(
    niter:wp.array[int],grad:wp.array2d[float],search:wp.array2d[float],qacc:wp.array2d[float],ma:wp.array2d[float],constraint:wp.array2d[float],
    scale:wp.array[float],grad_dot:wp.array[float],decrement:wp.array[float],alpha:wp.array[float],improvement:wp.array[float],time:wp.array[float],
    state:wp.array[int],quad:wp.array[int],done:wp.array[bool],exhausted:wp.array[bool],unchanged:wp.array[bool],
    vectors:wp.array4d[float],floats:wp.array3d[float],integers:wp.array3d[int],errors:wp.array[int],
):
    world,dof=wp.tid(); i=niter[world]-1
    if i<0 or i>=vectors.shape[0]:
        if dof==0: wp.atomic_add(errors,0,1)
        return
    vectors[i,world,dof,0]=grad[world,dof]
    vectors[i,world,dof,1]=search[world,dof]
    vectors[i,world,dof,2]=qacc[world,dof]
    vectors[i,world,dof,3]=ma[world,dof]
    vectors[i,world,dof,4]=constraint[world,dof]
    if dof==0:
        floats[i,world,0]=scale[world]
        floats[i,world,1]=grad_dot[world]
        floats[i,world,2]=decrement[world]
        floats[i,world,3]=alpha[world]
        floats[i,world,4]=improvement[world]
        floats[i,world,5]=time[world]
        integers[i,world,0]=1
        integers[i,world,1]=niter[world]
        integers[i,world,2]=state[world]
        integers[i,world,3]=quad[world]
        integers[i,world,4]=int(done[world])
        integers[i,world,5]=int(exhausted[world])
        integers[i,world,6]=int(unchanged[world])
