"""Focused synthetic contact bookkeeping contracts; never classify saved samples."""
import copy,json,os,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_gradient import contact_regime608105_v1 as c

def main():
    out=c.PREP;c.require(not out.exists(),'fresh CPU preparation');out.mkdir()
    pins={k:c.pin(v) for k,v in c.OWN.items()};c.write(out,'source_freeze_before_tests.json',pins)
    for k,v in pins.items():(out/(k+'_source.py')).write_bytes(Path(v['path']).read_bytes())
    checks={}
    def test(name,fn):
        fn();checks[name]=True
    def eq(v):c.require(v,'synthetic contract')
    def reject(fn):
        try:fn()
        except (ValueError,KeyError):return
        raise AssertionError('Expected rejection')
    catalog={'native_ngeom':4,'native_object_body_id':3,'native_geom_catalog':[
      {'id':0,'name':'robot/right_palm_collision','type':6,'native_body_id':1,'contype':1,'conaffinity':1},
      {'id':1,'name':'object','type':6,'native_body_id':3,'contype':1,'conaffinity':1},
      {'id':2,'name':'robot/right_finger1_link1_col','type':6,'native_body_id':2,'contype':1,'conaffinity':1},
      {'id':3,'name':'unknown_other','type':0,'native_body_id':0,'contype':1,'conaffinity':1}]}
    sets=c.catalog_sets(catalog)
    contact={'geom':[0,1],'efc_address':0,'dist':-.004,'pos':[0.,0.,0.],'frame':np.eye(3).tolist(),'force6':[2.,0.,0.,0.,0.,0.]}
    raw={'geom':[0,1],'type':3,'worldid':0,'efc_address':[0,1,-1]}
    def row(cc=None,rr=None,ne=2):return c.inspect_contact(0,cc or contact,rr or raw,ne,sets)
    test('type3_constraint_sensor_padding_valid',lambda:eq(row()['secondary_positive_force'] and row()['constraint_type_bit'] and row()['sensor_type_bit']))
    def ineligible():
        for patch in ({'type':2},{'worldid':1},{'efc_address':[-1,-1]},{'efc_address':[2,-1]}):
            r=row(rr={**raw,**patch});eq(r['generated_candidate'] and r['category']=='hand_object' and not r['force_evidence_eligible'])
    test('force_ineligible_candidates_retained',ineligible)
    test('host_address_not_raw_slot_index',lambda:eq(row(cc={**contact,'efc_address':1000})['force_evidence_eligible']))
    test('exact_Fn_threshold_separate_from_geometry',lambda:eq(not row(cc={**contact,'force6':[1e-6,0,0,0,0,0]})['secondary_positive_force'] and row()['observed_candidate_depth_m']==.004))
    def cats():
        for pair,expected,flag in [([0,2],'hand_self',None),([1,3],'other','object_nonhand'),([0,3],'other','hand_other'),([1,1],'other','both_object')]:
            r=row(cc={**contact,'geom':pair},rr={**raw,'geom':pair});eq(r['category']==expected and (flag is None or r[flag]))
    test('native_categories_not_static_environment',cats)
    test('empty_unknown_clearance_and_summary_generator',lambda:eq(c.summary([])['all']=={'generated_candidate_count':0,'minimum_generated_distance_m':None,'observed_candidate_depth_m':0.,'generated_candidate_absent':True,'secondary_positive_force_count':0,'global_clearance':'UNKNOWN'} and c.summary(r for r in [row()])['hand_object']['generated_candidate_count']==1))
    def parser():
        s={'nacon':0,'nefc':0,'contacts':[],'raw_wp_contact':{k:[] for k in raw},'raw_efc_force':[],'raw_efc_type':[]};eq(c.sample_contacts(s,sets)==[])
        reject(lambda:c.sample_contacts({**s,'nacon':1},sets))
        reject(lambda:row(rr={**raw,'geom':[1,0]}))
        reject(lambda:row(rr={**raw,'efc_address':[]}))
        reject(lambda:c.integer(True,'count',0))
    test('counts_original_index_shape_and_bool_rejections',parser)
    def encoding():
        a=np.zeros(58,'f4');b=a.copy();b[40]=-0.;r=c.encoded(a,b);eq(r['p_all_numeric_zero'] and not r['p_all_bit_equal'])
        reject(lambda:c.encoded(a.astype('f8'),b))
    test('FP32_numeric_vs_bit_encoding_separate',encoding)
    test('four_substep_profiles_all_categories',lambda:eq(c._profile([{'substep':i,'categories':c.summary([row()])} for i in range(1,5)])['hand_object']['generated_count_by_substep']==[1,1,1,1]))
    report={'schema':'full151105_contact_regime608_CPU_contract:1','status':'PASS_FOCUSED_SYNTHETIC_CONTACT_BOOKKEEPING_NO_ACTUAL608','checks':checks,'passed':len(checks),'total':len(checks),'source_pins':pins,
      'actual_saved_contact_regime_runs':0,'actual_physical_samples_read':0,'actual_NPZ_values_decoded':0,'new_model_FW_AD_MJB_FK_Env_GPU_physics':0}
    c.write(out,'report.json',report)
    ready={'schema':'full151105_contact_regime608_CPU_READY:1','source_pins':pins,'cpu_report':c.pin(out/'report.json'),'cpu_checks':{'passed':len(checks),'total':len(checks)},'protocol':c.pin(c.PREREG),'api':{'entry':str(c.OWN['CLI']),'module':str(c.OWN['consumer'])},'actual608_execution_authorized':False}
    c.write(out,'READY.json',ready);print(json.dumps({'status':report['status'],'passed':len(checks),'total':len(checks),'READY':c.pin(out/'READY.json')}))

if __name__=='__main__':main()
