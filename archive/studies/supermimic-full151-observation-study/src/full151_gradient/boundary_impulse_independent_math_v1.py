"""Independent old152 four-substep impulse labels; scalar sums/raw basis.

No producer, model, geometry or backend imports. Raw native R9 and raw recorded
palm transpose are preserved; the 3x3 solve uses partial-pivot elimination.
"""
import math
import numpy as np

PHASES=(0,49,99,149);JOINTS=(0,4,8,12)


def require(v,message):
    if not v:raise ValueError(message)


def finite(a,shape,name,dtype=None):
    a=np.asarray(a);require(a.shape==shape and np.isfinite(a).all() and (dtype is None or a.dtype==np.dtype(dtype)),'Finite typed '+name);return a


def expected_restored_schedule():
    rows=[]
    for pi,path in enumerate(('A0','A2')):
        for phase_index,k in enumerate(PHASES):
            s=4*pi+phase_index
            for off in range(19):
                bid=19*s+off;g=226*pi+150+19*phase_index+off;base=off<3
                j=None if base else (off-3)%8//2;ei=None if base else (off-3)//8
                rows.append({'branch_id':bid,'snapshot_id':s,'path':path,'k':k,'offset':off,'group_id':g,'mode':'baseline' if base else 'direction',
                  'joint_index':None if base else JOINTS[j],'j':j,'epsilon_index':ei,'sign':0 if base else (-1,1)[(off-3)%2],
                  'substep_source_rows':[4*g+t for t in range(4)],'post20_GPU_row':1808+g,'terminal_observation':k+1})
    return rows


def require_complete_dataset(*masks):
    require(bool(masks),'Explicit full population masks')
    for v in masks:
        a=np.asarray(v);require(a.dtype==np.bool_ and a.size>0 and a.all(),'Unknown required evidence blocks GLOBAL enrollment, no selected subset')


def source_witness(valid_contact_by_substep_source):
    a=np.asarray(valid_contact_by_substep_source);require(a.dtype==np.bool_ and a.shape==(4,2),'Four-step two-source constraint witness')
    return a.any(axis=0)


def raw_transpose_vector(P,v):
    P=finite(P,(3,3),'recorded palm FP32',np.float32);v=finite(v,(3,),'world vector')
    return np.asarray([math.fsum(float(P[k,a])*float(v[k]) for k in range(3)) for a in range(3)],np.float64)


def matvec(A,v):
    A=finite(A,(3,3),'F64 matrix');v=finite(v,(3,),'vector')
    return np.asarray([math.fsum(float(A[i,j])*float(v[j]) for j in range(3)) for i in range(3)],np.float64)


def congruence_native_R(A,I):
    A=finite(A,(3,3),'raw native R9');I=finite(I,(3,3),'body inertia')
    return np.asarray([[math.fsum(float(A[r,i])*float(I[i,j])*float(A[c,j]) for i in range(3) for j in range(3)) for c in range(3)] for r in range(3)],np.float64)


def solve3_partial_pivot(A,b):
    a=finite(A,(3,3),'raw endpoint inertia').astype('f8').tolist();rhs=finite(b,(3,),'angular momentum RHS').astype('f8').tolist()
    for c in range(3):
        p=max(range(c,3),key=lambda r:abs(a[r][c]));require(a[p][c]!=0,'Singular native-R9 inertia blocks whole enrollment')
        if p!=c:a[c],a[p]=a[p],a[c];rhs[c],rhs[p]=rhs[p],rhs[c]
        for r in range(c+1,3):
            f=a[r][c]/a[c][c];a[r][c]=0.
            for j in range(c+1,3):a[r][j]=math.fsum((a[r][j],-f*a[c][j]))
            rhs[r]=math.fsum((rhs[r],-f*rhs[c]))
    x=[0.,0.,0.]
    for r in range(2,-1,-1):x[r]=math.fsum((rhs[r],-math.fsum(a[r][j]*x[j] for j in range(r+1,3))))/a[r][r]
    require(all(math.isfinite(v) for v in x),'Finite solve required, no pseudo-inverse or repair')
    return np.asarray(x,np.float64)


def determinant3(A):
    a=finite(A,(3,3),'raw R9')
    return math.fsum((float(a[0,0])*float(a[1,1])*float(a[2,2]),float(a[0,1])*float(a[1,2])*float(a[2,0]),float(a[0,2])*float(a[1,0])*float(a[2,1]),
      -float(a[0,2])*float(a[1,1])*float(a[2,0]),-float(a[0,1])*float(a[1,0])*float(a[2,2]),-float(a[0,0])*float(a[1,2])*float(a[2,1])))


def raw_orthogonality(A):
    A=finite(A,(3,3),'raw R9')
    return np.asarray([[math.fsum(float(A[k,i])*float(A[k,j]) for k in range(3))-(1. if i==j else 0.) for j in range(3)] for i in range(3)],np.float64)


def four_substep_labels(pre58,post58,P,dt4,F_world4x2x3,tau_world4x2x3,mass,gravity_world,I_body):
    pre=finite(pre58,(58,),'native PRE58',np.float32).astype('f8');post=finite(post58,(58,),'native target58',np.float32).astype('f8')
    P=finite(P,(3,3),'original palm basis',np.float32);h=finite(dt4,(4,),'effective WP FP32 substeps',np.float32).astype('f8');require((h>0).all(),'Positive source dt')
    F=finite(F_world4x2x3,(4,2,3),'saved source forces',np.float64);T=finite(tau_world4x2x3,(4,2,3),'saved source COM torques',np.float64)
    require(type(mass) is float and math.isfinite(mass) and mass>0,'Actual positive mass');g=finite(gravity_world,(3,),'source-effective world gravity',np.float64);Ib=finite(I_body,(3,3),'bound normalized-iquat inertia',np.float64)
    Jw=np.asarray([[math.fsum(float(h[t])*float(F[t,c,j]) for t in range(4)) for j in range(3)] for c in range(2)],np.float64)
    Kw=np.asarray([[math.fsum(float(h[t])*float(T[t,c,j]) for t in range(4)) for j in range(3)] for c in range(2)],np.float64)
    Jp=np.asarray([raw_transpose_vector(P,v) for v in Jw]);Kp=np.asarray([raw_transpose_vector(P,v) for v in Kw]);gp=raw_transpose_vector(P,g);deltaT=math.fsum(float(x) for x in h)
    rv=np.asarray([math.fsum((float(post[52+j]),-float(pre[52+j]),-math.fsum((float(Jp[0,j]),float(Jp[1,j]),mass*float(gp[j])*deltaT))/mass)) for j in range(3)])
    R0,R1=pre[43:52].reshape(3,3),post[43:52].reshape(3,3);I0,I1=congruence_native_R(R0,Ib),congruence_native_R(R1,Ib)
    initial_L=matvec(I0,pre[55:58]);rhs=np.asarray([math.fsum((float(initial_L[j]),float(Kp[0,j]),float(Kp[1,j]))) for j in range(3)])
    solved=solve3_partial_pivot(I1,rhs);rw=np.asarray([math.fsum((float(post[55+j]),-float(solved[j]))) for j in range(3)])
    solve_res=matvec(I1,solved)-rhs
    return {'J_world_Ns_F64':Jw,'K_world_Nms_F64':Kw,'J_palm_Ns_F64':Jp,'K_palm_Nms_F64':Kp,'gravity_palm_m_s2_F64':gp,
      'DeltaT_s_F64':deltaT,'I_p0_F64':I0,'I_p1_F64':I1,'angular_RHS_F64':rhs,'angular_solved_omega_F64':solved,'angular_solve_residual_F64':solve_res,
      'r_v_saved_m_s_F64':rv,'r_w_saved_rad_s_F64':rw,'raw_R0_det_F64':determinant3(R0),'raw_R1_det_F64':determinant3(R1),
      'raw_R0_orthogonality_residual_F64':raw_orthogonality(R0),'raw_R1_orthogonality_residual_F64':raw_orthogonality(R1),
      'scope':'Observer defects retained, not forced zero or physical qualification; raw palm transpose and native R9 preserved.'}
