"""SOURCE DRAFT: focused synthetic contracts; no real checkpoint/data load.

Do not execute before the scientific contract and source scope are locked.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from full151_gradient import public_self15_fresh_gradient105_v1 as c

def main():
    c.require(c.CONTRACT_SHA is not None and c.pin(c.CONTRACT)['sha256']==c.CONTRACT_SHA,'Contract must be frozen before single synthetic execution')
    c.require(not c.PREP.exists(),'Fresh proof only');c.PREP.mkdir();sources={k:c.pin(v) for k,v in c.OWN.items()}
    c.c._write_json(c.PREP/'source_freeze.json',{'source_pins':sources,'source_contract':c.pin(c.CONTRACT),'before_synthetic':True})
    import numpy as np
    import torch
    torch.set_num_threads(4)
    from full151_gradient.serial_probe105_calibration_v1 import controlled_object_score_FP32
    # Case 1: full synthetic EPS-major metadata, 64 pairs and one clipped chord.
    x8=np.zeros((8,58),np.float32);x8[:,40]=.01;x8[:,43:52]=np.eye(3,dtype=np.float32).reshape(1,9)
    u8=np.zeros((8,20),np.float32);u=np.zeros((152,20),np.float32);rows=[];snaps=[]
    minus=np.zeros((8,4,2,20),np.float32);plus=minus.copy()
    for s in range(8):
        snaps.append({'snapshot_id':s,'path':('A0','A2')[s//4],'k':c.PHASES[s%4],'baseline3_qualified':True})
        for off in range(19):
            base=off<3;ei=None if base else (off-3)//8;j=None if base else ((off-3)%8)//2;sign=0 if base else (-1,1)[(off-3)%2];bid=19*s+off
            if not base:
                u[bid,c.tg.JOINTS[j]]=np.float32(sign*c.tg.EPS[ei])
                if s==j==ei==0 and sign==1:u[bid,c.tg.JOINTS[j]]=0  # Preserve clipped/asymmetric chord.
                (plus if sign==1 else minus)[s,j,ei]=u[bid]
            rows.append({'row_id':bid,'snapshot_id':s,'path':('A0','A2')[s//4],'k':c.PHASES[s%4],'type':'baseline' if base else 'direction',
                'baseline_repeat':off if base else None,'j':j,'epsilon_index':ei,'epsilon_rad':None if base else c.tg.EPS[ei],
                'sign':sign,'joint_index':None if base else c.tg.JOINTS[j],'status':'COMPLETED','restorequalified':True,'actualu':u[bid].tolist()})
    effective=(plus.astype(np.float64)-minus.astype(np.float64))/(2*np.asarray(c.tg.EPS,np.float64)[None,None,:,None])
    center=(plus.astype(np.float64)+minus.astype(np.float64))*.5-u8[:,None,None]
    clipped=np.zeros((8,4,2),bool);clipped[0,0,0]=True
    arrays={'snapshot_x':x8,'snapshot_u':u8,'branch_x_next':np.repeat(x8,19,axis=0),'branch_u':u,'direction__centers':u8.copy(),
        'direction__minus':minus,'direction__plus':plus,'direction__d_eff':effective,'direction__center_offset':center,'direction__clipped':clipped,
        'completed':np.ones(152,bool),'snapshot_completed':np.ones(8,bool),'restore_qualified':np.ones(152,bool),'restore_inventory_bitexact':np.ones(152,bool)}
    data,identities,pairs=c.cf.prepare_test(arrays,rows,snaps,np)
    physical=[]
    for pair in pairs:
        s=pair['snapshot_id'];j=pair['j'];ei=pair['epsilon_index'];mi=pair['minus_branch_id'];pl=pair['plus_branch_id'];du=u[pl].astype(np.float64)-u[mi].astype(np.float64);den=du[pair['joint_index']]
        physical.append({**pair,'actual_delta_u20_rad_F64':du.tolist(),'actual_coordinate_denominator_rad':float(den),'actual_coordinate_defined':True,
            'physical_actual_coordinate_secant':4.,'physical_legacy_nominal_epsilon_slope':float(4*den/(2*c.tg.EPS[ei])),
            'legacy_d_eff_dimensionless':effective[s,j,ei].tolist()})
    projected=c.tg.projected_rows(np.full((8,20),3.,np.float32),physical,np)
    case1=(len(identities)==152 and len(pairs)==64 and np.array_equal(data['x152_FP32'],np.repeat(x8,19,axis=0))
        and all(r['WM_AD_actual_delta_projected_per_coordinate']==3. for r in projected)
        and abs(projected[0]['WM_AD_legacy_effective_direction']-1.5)<=8*np.finfo(np.float32).eps
        and abs(projected[0]['physical_legacy_nominal_epsilon_slope']-2.)<=8*np.finfo(np.float32).eps
        and bool(clipped[0,0,0]))

    # Case 2: exact call routes and object-only analytic gradient on Tiny models.
    # These one-parameter fixtures are NOT actual native14/final16 qualification.
    class Tiny(torch.nn.Module):
        def __init__(self,aux=False):super().__init__();self.register_parameter('p',torch.nn.Parameter(torch.zeros(()),requires_grad=False));self.aux=aux
        def forward(self,x,u,dynamics):
            pred=torch.cat((x[:,:40],x[:,40:41]+2*u[:,:1],x[:,41:]),-1);logits=x.new_zeros((len(x),15))
            return (pred,logits,x.new_zeros((len(x),2,6)),x.new_zeros((len(x),2))) if self.aux else (pred,logits)
    before=c.c._rng(torch,np);counts={key:0 for key in c.BUDGET};x=torch.from_numpy(data['x152_FP32']);uv=torch.from_numpy(data['actual_u152_FP32']);xi=torch.zeros(46)
    dynamics=torch.cat((x,xi[None].expand(152,-1),x.new_zeros((152,174))),-1);xt=torch.from_numpy(x8);gp=torch.zeros((8,3));gR=torch.eye(3).expand(8,3,3).clone()
    analytical=-8*xt[:,40]/(.05**2)*torch.exp(-(xt[:,40]/.05)**2)-32*xt[:,40]/(.04**2)*torch.exp(-(xt[:,40]/.04)**2)
    gradients=[]
    for n,role in enumerate(c.ROLES):
        model=Tiny(n>0)
        with torch.no_grad():c.call_model(model,x,uv,dynamics,role,counts,torch,'endpoint')
        with torch.enable_grad():
            uc=torch.zeros((8,20),requires_grad=True);d8=torch.cat((xt,torch.zeros((8,220))),-1)
            pred,logits=c.call_model(model,xt,uc,d8,role,counts,torch,'center');terms,score=controlled_object_score_FP32(pred,gp,gR)
            gradient,=torch.autograd.grad(score.sum(),uc);counts['uVJP_calls']+=1
        c.require(model.p.grad is None and not model.p.requires_grad,'No fixture parameter gradient');gradients.append(gradient)
    case2=(counts['native_FW_attempted']==counts['native_FW_returned']==6 and counts['endpoint_predictions']==456 and counts['center_predictions']==24
        and counts['aux_head_FW']==4 and counts['uVJP_calls']==3 and all(bool(torch.allclose(g[:,0],analytical,rtol=8*torch.finfo(torch.float32).eps,atol=1e-4))
        and bool((g[:,1:]==0).all()) for g in gradients))

    # Case 3: cached15 is separate, ties negative; raw RNG values unchanged.
    stats,_=c.cached15_statistics(np.zeros((152,15),np.float32),np.tile(np.asarray([1]+[0]*14,np.float32),(152,1)),identities,np)
    after=c.c._rng(torch,np);rng=c.ev.parity.raw_rng_equality(before,after,np,torch)
    case3=(len(stats)==13 and stats['ALL']['TP']==0 and stats['ALL']['FN']==152 and stats['ALL']['tie_zero_negative']==2280
        and len(stats['ALL']['perfeature15'])==15 and stats['ALL']['positive_precision'] is None and stats['ALL']['perfeature15'][1]['positive_recall'] is None
        and abs(stats['ALL']['BCE_mean_elements_F64']-np.log(2.))<=4*np.finfo(np.float64).eps and all(rng.values()) and not torch.cuda.is_initialized())
    checks={'synthetic_fresh_design_pair_and_clip_conventions':bool(case1),'six_routes_three_object_only_uVJP_analytic':bool(case2),'cached15_separate_and_raw_RNG_unchanged':bool(case3)}
    c.require(all(checks.values()),'Focused synthetic contract failed')
    report={'schema':'full151105_public_self15_fresh_gradient_CPU_contract:1','status':'PASS_SYNTHETIC_ONLY_FRESH_DESIGN_OBJECT_SCORE_AND_BUDGET',
        'checks':checks,'passed':len(checks),'total':len(checks),'source_pins':sources,'source_contract':c.pin(c.CONTRACT),
        'actual_component_loads_fresh_NPZ_physics':0,'synthetic_only':True,'actual_native14_final16_qualification':False,
        'synthetic_counts':counts,'raw_RNG_comparison':rng,'scope':'Synthetic NumPy metadata and three Tiny models only; no real checkpoint, fresh dataset, native constructor or physical qualification'}
    c.c._write_json(c.PREP/'report.json',report)
    c.c._write_json(c.PREP/'READY.json',{'schema':'full151105_public_self15_fresh_gradient_CPU_READY:1','source_pins':sources,'source_contract':c.pin(c.CONTRACT),
        'cpu_report':c.pin(c.PREP/'report.json'),'cpu_checks':{'passed':report['passed'],'total':report['total']},'api':{'entry':'full151_gradient.public_self15_fresh_gradient105_v1.run_analysis','default':'metadata_preflight / declaration'}})
    print({'status':report['status'],'passed':report['passed'],'total':report['total'],'READY':c.pin(c.PREP/'READY.json')})

if __name__=='__main__':main()
