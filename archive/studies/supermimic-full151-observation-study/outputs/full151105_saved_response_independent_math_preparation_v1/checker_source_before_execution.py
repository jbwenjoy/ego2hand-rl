"""Closed-form NumPy fixtures; no actual saved-response decomposition."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'src/full151_gradient/saved_action_response_independent_math_v1.py'
OUT=ROOT/'outputs/full151105_saved_response_independent_math_preparation_v1'


def pin(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    OUT.mkdir(exist_ok=False);pins={'math':pin(PATH),'checker':pin(Path(__file__))}
    for name,p in [('math',PATH),('checker',Path(__file__))]:
        ast.parse(p.read_bytes());(OUT/(name+'_source_before_execution.py')).write_bytes(p.read_bytes())
    (OUT/'predata_freeze.json').write_text(json.dumps({'source_pins':pins,'actual_saved_response_arrays_read':False},indent=2))
    spec=importlib.util.spec_from_file_location('independent_response_math',PATH);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    snapshot=np.zeros((8,58),dtype='f4');snapshot[:,43:52]=np.eye(3,dtype='f4').reshape(9)
    pre=np.broadcast_to(snapshot[:,None],(8,19,58)).copy()
    def controls(center=0.):
        u0=np.full((8,20),center,dtype='f4');u=np.broadcast_to(u0[:,None],(8,19,20)).copy()
        for s in range(8):
            for ei,eps in enumerate(m.EPS):
                for ji,j in enumerate(m.JOINTS):
                    off=3+ei*8+ji*2;unit=np.eye(20,dtype='f4')[j]
                    u[s,off]=u0[s]-np.float32(eps)*unit;u[s,off+1]=u0[s]+np.float32(eps)*unit
        return u,u0
    def archived(u,u0):
        a={'plus':np.empty((8,4,2,20),dtype='f4'),'minus':np.empty((8,4,2,20),dtype='f4'),
           'd_eff':np.empty((8,4,2,20),dtype='f8'),'center_offset':np.empty((8,4,2,20),dtype='f8'),'clipped':np.zeros((8,4,2),dtype=bool)}
        for s in range(8):
            for ei,eps in enumerate(m.EPS):
                for ji,j in enumerate(m.JOINTS):
                    off=3+ei*8+ji*2;minus,plus=u[s,off],u[s,off+1];unit=np.eye(20,dtype='f4')[j]
                    a['minus'][s,ji,ei]=minus;a['plus'][s,ji,ei]=plus
                    a['d_eff'][s,ji,ei]=(plus.astype('f8')-minus.astype('f8'))/(2*eps)
                    a['center_offset'][s,ji,ei]=(plus.astype('f8')+minus.astype('f8'))/2-u0[s].astype('f8')
                    a['clipped'][s,ji,ei]=not(np.array_equal(plus,u0[s]+np.float32(eps)*unit) and np.array_equal(minus,u0[s]-np.float32(eps)*unit))
        return a
    def affine(u):
        # Powers of two: multiplication of exact saved F32 steps is exact.
        z=u.sum(-1,dtype='f4')
        return np.broadcast_to((2*z)[...,None],(8,19,58)).copy(),np.broadcast_to((4*z)[...,None],(8,19,58)).copy()
    u,u0=controls();y,p=affine(u);r=m.decompose(pre,y,p,u,snapshot,u0,archived(u,u0));a=r['arrays']
    checks={'all128_64_closedform_finite_response_retained':a['physical_response128_F64'].shape==(128,58) and a['physical_secant64_F64'].shape==(64,58),
        'closedform_affine_secants_exact_two_four_residual_two':np.all(a['physical_secant64_F64']==2) and np.all(a['predicted_secant64_F64']==4) and np.all(a['secant_residual64_F64']==2),
        'closedform_bias_identity_closure_zero':np.all(a['bias_identity_closure128_F64']==0),
        'legacy_dimensionless_not_rad_halfstep':not np.array_equal(a['legacy_d_eff64_dimensionless'],a['actual_halfstep64_rad'])}
    yy=np.zeros((8,19,58),dtype='f4');pp=np.full_like(yy,3)
    yy[:,:3]=np.array([-1,0,1],dtype='f4')[None,:,None];pp[:,:3]=np.array([2,3,4],dtype='f4')[None,:,None]
    noise=m.decompose(pre,yy,pp,u,snapshot,u0,archived(u,u0))
    checks['baseline_means_bias_and_population_variance_closedform']=np.all(noise['arrays']['baseline_bias_F64']==3) and np.all(noise['baseline_noise']['physical']['population_variance_F64']==2/3) and np.all(noise['arrays']['response_residual128_F64']==0)
    rounded,one=controls(1.);rr=m.decompose(pre,yy,pp,rounded,snapshot,one,archived(rounded,one))
    checks['F32_rounding_delta_not_2eps_does_not_imply_clipping']=not rr['arrays']['legacy_clipped64'].any() and rr['arrays']['actual_denominator64_rad'][0]!=2*m.EPS[0]
    zero=u.copy();zero[0,3:5]=u0[0];zy,zp=affine(zero);zr=m.decompose(pre,zy,zp,zero,snapshot,u0,archived(zero,u0))
    checks['zero_denominator_preserves_pair_NaNs_mask_no_fill']=len(zr['pair_identities'])==64 and not zr['arrays']['secant_defined64'][0] and np.isnan(zr['arrays']['physical_secant64_F64'][0]).all() and zr['summaries']['physical_secant']['undefined_rows']==1
    clipped=u.copy();clipped[0,4]=u0[0];cy,cp=affine(clipped);cr=m.decompose(pre,cy,cp,clipped,snapshot,u0,archived(clipped,u0))
    checks['asymmetric_clipped_actualdenom_not_nominal_central']=cr['arrays']['hplus64_rad'][0]==0 and cr['arrays']['hminus64_rad'][0]>0 and cr['arrays']['legacy_clipped64'][0] and cr['arrays']['midpoint_offset64_rad'][0,0]<0 and cr['arrays']['physical_secant64_F64'][0,0]==2
    wrong=u.copy();wrong[0,3,19]=np.nextafter(np.float32(0),np.float32(1));w=m.decompose(pre,y,p,wrong,snapshot,u0,archived(wrong,u0))
    checks['nonjoint_ulp_blocks_whole_interpretation_no_subset']=not w['fixed_visible_state_interpretation_allowed'] and 'arrays' not in w
    zeros=pre.copy();zeros[0,5,40]=np.float32(-0.);zg=m.visible_state_guard(zeros,snapshot)
    checks['other18_numeric_zero_guard_retains_signedzero_bits']=zg['same_visible58_all_passed'] and not zg['all58_raw_bit_equal_mask'][0,5,40]
    zeros=pre.copy();zeros[0,5,0]=np.float32(-0.);qg=m.decompose(zeros,y,p,u,snapshot,u0,archived(u,u0))
    checks['q_signedzero_bit_change_blocks_interpretation']=not qg['fixed_visible_state_interpretation_allowed']
    reverse=u.copy();reverse[0,[3,4]]=reverse[0,[4,3]];rv=m.decompose(pre,y,p,reverse,snapshot,u0,archived(reverse,u0))
    checks['negative_source_order_blocks_no_relabel']=not rv['fixed_visible_state_interpretation_allowed']
    checks['no_actual_decomposition_or_backend_model_import']=not any(k in sys.modules for k in ('torch','warp','mujoco','scipy','dex_supermimic'))
    checks={k:bool(v) for k,v in checks.items()}
    report={'schema':'full151105_saved_response_independent_math_preparation:1','source_pins':pins,'passed':sum(checks.values()),'total':len(checks),'checks':checks,
        'synthetic_only':True,'actual_saved_response_decompositions':0,'new_model_FW_AD_MJB_Env_physics_GPU':0,'scope':'No actual response value or qualification inferred from closedform toy arrays.'}
    (OUT/'report.json').write_text(json.dumps(report,indent=2));ready={'source_pins':pins,'cpu_report':pin(OUT/'report.json'),'cpu_checks':{'passed':report['passed'],'total':report['total']}}
    (OUT/'READY.json').write_text(json.dumps(ready,indent=2));print(json.dumps({'READY':pin(OUT/'READY.json'),**ready},indent=2))
    return 0 if all(checks.values()) else 2


if __name__=='__main__':raise SystemExit(main())
