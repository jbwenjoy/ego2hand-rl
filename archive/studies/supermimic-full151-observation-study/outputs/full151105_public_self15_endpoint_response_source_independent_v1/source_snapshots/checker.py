"""Exactly one closed-form synthetic old152/64 finite-response fixture."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from full151_gradient import public_self15_endpoint_response105_v1 as c

def main():
    out=c.PREP;c.require(not out.exists(),'Fresh proof only');out.mkdir()
    sources={k:c.pin(v) for k,v in c.OWN.items()};c.write(out/'source_freeze.json',{'source_pins':sources,'before_fixture':True,'actual_saved_array_reads':0})
    import numpy as np
    ids=[];meta=[];groups=[];x=np.zeros((152,58),np.float32);u=np.zeros((152,20),np.float32);y=np.zeros((152,58),np.float32)
    for bid in range(152):
        s,off=divmod(bid,19);pi,si=divmod(s,4);base=off<3;j=None if base else ((off-3)%8)//2;ei=None if base else (off-3)//8;group=226*pi+150+19*si+off;mode='baseline' if base else 'direction'
        ids.append({'row_id':bid,'snapshot_id':s,'path':('A0','A2')[pi],'mode':mode,'k':c.PHASES[si],'branch_id':bid,'group_id':group,'physical_rows':list(range(4*group,4*group+4)),'terminal_observation':c.PHASES[si]+1})
        meta.append({'row_id':bid,'snapshot_id':s,'path':('A0','A2')[pi],'type':mode,'k':c.PHASES[si],'group_id':group,'j':j,'epsilon_index':ei,'epsilon_rad':None if base else c.EPS[ei],
            'sign':0 if base else (-1,1)[(off-3)%2],'baseline_repeat':off if base else None,'joint_index':None if base else c.JOINTS[j],'native_actuator_id':None if base else c.JOINTS[j]+1,'native_joint_id':None if base else c.JOINTS[j]+2})
        groups.append(group);x[bid,0]=bid
        if not base:u[bid,c.JOINTS[j]]=np.float32(meta[-1]['sign']*c.EPS[ei])
        y[bid,40]=np.float32(2)*u[bid].sum(dtype=np.float32)
    pred=y.copy();pred[:,40]=2*y[:,40];predictions={role:pred.copy() for role in c.ROLES};g=np.asarray(groups,np.int64);sub=4*g[:,None]+np.arange(4,dtype=np.int64)
    a,rows,stats,baseline=c.analyze(x,u,y,predictions,ids,meta,g,sub,np)
    checks={'single_nonzero_closedform_all64_actual_response':bool(np.array_equal(a[c.ROLES[0]+'__actual_secant58_F64'][:,40],np.full(64,4.)) and np.array_equal(a['physical_actual_secant58_F64'][:,40],np.full(64,2.))),
        'all64_and13_cohorts_no_selection':len(rows)==64 and all(len(s)==13 for s in stats.values()) and len(baseline)==8,
        'actual_pre_differences_retained':bool(np.all(a['actual_delta_x58_F64'][:,0]==1)),
        'nominal_uses_requested_eps_separate':bool(np.array_equal(a['physical_nominal_secant58_F64'][:,40],2*a['actual_coordinate_denominator_rad_F64']/a['nominal_2epsilon_rad_F64']))}
    # Boundary in the same fixture: one encoded zero-du pair remains, with NaN
    # secants/mask. No replacement epsilon, row deletion, or extra fixture run.
    pl=rows[0]['plus_branch_id'];mi=rows[0]['minus_branch_id'];u[pl]=u[mi]
    az,rz,sz,bz=c.analyze(x,u,y,predictions,ids,meta,g,sub,np)
    checks['zero_denominator_retains_row_with_NaN']=bool(len(rz)==64 and not az['actual_defined_mask64'][0] and np.isnan(az['physical_actual_secant58_F64'][0]).all() and sz[c.ROLES[0]]['ALL']['actual_undefined_pairs']==1)
    checks['no_model_framework_or_actual_saved_arrays']=not any(k in sys.modules for k in ('torch','mujoco','warp'))
    c.require(all(checks.values()),'Single synthetic fixture failed')
    r={'schema':'full151105_public_self15_endpoint_response_CPU_contract:1','status':'PASS_SINGLE_SYNTHETIC_FINITE_ENDPOINT_RESPONSE_NO_NN','checks':checks,'passed':len(checks),'total':len(checks),
        'source_pins':sources,'synthetic_fixture_count':1,'actual_saved_array_reads':0,'model_FW_AD_physics_calls':0,'scope':'Synthetic152 only, no actual old152 alignment or physical eligibility claim'}
    c.write(out/'report.json',r);ready={'schema':'full151105_public_self15_endpoint_response_CPU_READY:1','source_pins':sources,'cpu_report':c.pin(out/'report.json'),
        'cpu_checks':{'passed':r['passed'],'total':r['total']},'api':{'entry':'full151_gradient.public_self15_endpoint_response105_v1.run_analysis','default':'metadata_preflight / declaration'},'input_aliases':sorted(c.INPUTS)}
    c.write(out/'READY.json',ready);print({'status':r['status'],'passed':r['passed'],'total':r['total'],'READY':c.pin(out/'READY.json')})

if __name__=='__main__':main()
