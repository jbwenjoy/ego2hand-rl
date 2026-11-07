"""Pure installed MuJoCo algebra, not MjData/contacts/physics or material data."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def pin(p):
    p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def main():
    out=ROOT/'outputs/native_pyramidal4_algebra_CPU_v1'
    if out.exists():raise FileExistsError('Freeze once')
    out.mkdir(parents=True);(out/'source_before.py').write_bytes(Path(__file__).read_bytes())
    import mujoco
    calls={}
    for name in ('MjData','mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_collision','mj_geomDistance','mj_contactForce','mj_kinematics','mj_comPos'):
        def deny(*args,_name=name,**kwargs):calls[_name]=calls.get(_name,0)+1;raise AssertionError('Forbidden native physics/query '+_name)
        setattr(mujoco,name,deny)
    # Deliberately unequal arbitrary fixture coefficients; these are not
    # measured or chosen simulation material values.
    mu=np.array([.31,.47,.006],np.float64)
    expected=np.zeros((6,4));expected[:,0]=1
    for i in range(6):expected[i,1+i//2]=mu[i//2]*(1 if i%2==0 else -1)
    decoded=[]
    for coefficient in np.eye(6):
        force=np.zeros(4);mujoco.mju_decodePyramid(force,coefficient,mu);decoded.append(force)
    decoded=np.asarray(decoded);lam=np.arange(1,7,dtype=float)/10
    mixed=np.zeros(4);mujoco.mju_decodePyramid(mixed,lam,mu)
    checks={'six_columns_match_native_decoder_exact':np.array_equal(decoded,expected),
        'all_columns_unit_normal_force':np.array_equal(decoded[:,0],np.ones(6)),
        'torsion_columns_keep_normal_force':np.array_equal(decoded[4:,:3],[[1,0,0],[1,0,0]]),
        'torsion_is_signed_mu2_not_unit_or_radius_scaled':np.array_equal(decoded[4:,3],[mu[2],-mu[2]]),
        'mixed_sum_normal_coefficient':mixed[0]==lam.sum(),
        'mixed_linear_combination':np.allclose(mixed,lam@expected,rtol=0,atol=1e-15),
        'native_pyramid_L1_friction_budget':np.sum(np.abs(mixed[1:])/mu)<=mixed[0]+1e-15,
        'no_native_physics_or_contact_queries':not calls,
        'no_device_or_model_runtime':not any(k.split('.')[0] in ('torch','warp','mujoco_warp','mjlab','dex_supermimic') for k in sys.modules)}
    np.savez_compressed(out/'decoder_columns.npz',fixture_mu=mu,expected_columns=expected,native_decoded_columns=decoded,mixed_lambda=lam,mixed_contact_force=mixed)
    report={'schema':'native_pyramidal4_algebra_CPU_contract:1','status':'PASS_ALGEBRA_ONLY' if all(checks.values()) else 'FAIL_RETAINED','checks':{k:bool(v) for k,v in checks.items()},'passed':sum(checks.values()),'total':len(checks),'source':pin(__file__),'mujoco_version':mujoco.__version__,'mujoco_native_function_binary':pin(mujoco._functions.__file__),'pure_mju_decodePyramid_calls':7,'native_model_or_data_instances':0,'forbidden_physics_or_contact_query_calls':calls,'fixture_mu':mu.tolist(),'material_value_claim':False,'physical_reference_qualified':False,'scope':'Installed condim4 decoder verifies six normal-force-normalized rays and coupled torsion, using arbitrary unequal fixture mu. No actual model material claim, contact/frame observation, simulation or LP. Normal force is sum(lambda), not Euclidean ray normalization; torsion stays coupled to normal force.'}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'passed':int(report['passed']),'total':report['total'],'report':pin(out/'report.json')}))
    if not all(checks.values()):raise SystemExit(1)
if __name__=='__main__':main()
