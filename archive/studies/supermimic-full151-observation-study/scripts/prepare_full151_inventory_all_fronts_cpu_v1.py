"""Create isolated schema2 fronts; preserve all executed schema1 sources."""
from pathlib import Path
import ast,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def once(s,a,b):
    assert s.count(a)==1,a
    return s.replace(a,b,1)
def save(name,s):
    ast.parse(s);p=ROOT/'scripts'/name
    with p.open('x',encoding='utf-8',newline='') as f:f.write(s)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def main():
    entry=(ROOT/'scripts/execute_full151_gradient_v1.py').read_text(encoding='utf-8')
    entry=once(entry,"SCHEMA='full151105_same_warp_physical_probe_run:1'","SCHEMA='full151105_same_warp_physical_probe_run:2'")
    entry=once(entry,"'planned_episode_substeps':1808}}","'planned_episode_substeps':1808},\n       'inventory_discovery_all':{'donor_intervals':300,'snapshots':8,'baseline_branches':0,'perturbation_branches':0,'planned_episode_substeps':1200}}")
    entry=entry.replace('scripts/run_full151_gradient_v1.py','scripts/run_full151_gradient_v2.py').replace('full151_gradient/serial_probe105_v1.py','full151_gradient/serial_probe105_v2.py').replace('full151_gradient.serial_probe105_v1','full151_gradient.serial_probe105_v2')
    entry=entry.replace("protocol['schema']!='full151105_physical_probe_execution_protocol:1'","protocol['schema']!='full151105_physical_probe_execution_protocol:2'")
    old="""    if c['probe_mode']=='serial_full':
        if c.get('restore_smoke_health') is None:raise ValueError('Actual positive native restore smoke required before full152')
        smoke=positive(c['restore_smoke_health'])
        if smoke.get('native_restore_qualified') is not True or smoke.get('physical_gradient_qualified') is not False:raise ValueError('Separate actual restore engineering proof required; no gradient claim')
"""
    new="""    if c['probe_mode']=='inventory_discovery_all' and c['foreign_inventory_review'] is not None:raise ValueError('Discovery-all captures fresh donors only, with no restore/foreign authorization')
    if c['probe_mode']=='serial_full':
        if c.get('restore_smoke_health') is None or c.get('inventory_discovery_health') is None:raise ValueError('Actual empirical phase0 and complete8 inventory saved reviews required')
        smoke=positive(c['restore_smoke_health']);discovery=positive(c['inventory_discovery_health'])
        if smoke.get('empirical_phase0_baseline_passed') is not True or smoke.get('physical_gradient_qualified') is not False or smoke.get('whole_hidden_state_qualified') is not False:raise ValueError('Typed empirical baseline scope required, never a hidden-state certificate')
        if discovery.get('probe_mode')!='inventory_discovery_all' or discovery.get('completed_inventory_snapshots')!=8 or discovery.get('episode_step_returned')!=1200 or discovery.get('restored_branches')!=0 or discovery.get('physical_gradient_qualified') is not False:raise ValueError('Complete actual8snapshot zero-restore donor inventory proof required')
        for evidence in (smoke,discovery):
            if not same(evidence['input_bundle'],c['input_bundle']):raise ValueError('Same actual native input bundle required')
"""
    entry=once(entry,old,new)
    entry=once(entry,'    validate_config(c,bundle)\n',"""    validate_config(c,bundle)
    from full151_gradient.serial_probe105_v2 import execution_plan
    plan=execution_plan(c['probe_mode']);expected=MODES[c['probe_mode']]
    if plan['episode_substeps']!=expected['planned_episode_substeps'] or plan['snapshot_count']!=expected['snapshots'] or plan['branch_count']!=expected['baseline_branches']+expected['perturbation_branches']:raise ValueError('Independent front/worker scheduling contract differs')
""")
    entry=entry.replace("('COMPLETED_ENGINEERING_RESTORE_SMOKE','COMPLETED_PHYSICAL_PROBE','INVENTORY_REVIEW_REQUIRED')","('COMPLETED_ENGINEERING_RESTORE_SMOKE','COMPLETED_PHYSICAL_PROBE','COMPLETED_INVENTORY_DISCOVERY','INVENTORY_REVIEW_REQUIRED')")
    entry=entry.replace("physics_steps=result.get('physics_steps','SEE_PROBE_COUNTERS')","episode_steps_returned=result.get('episode_step_returned','SEE_PROBE_COUNTERS'),step_count_scope='Episode API returns; constructor/capture counters separate'")
    sup=(ROOT/'scripts/run_full151_gradient_v1.py').read_text(encoding='utf-8')
    sup=once(sup,"ENTRY=ROOT/'scripts/execute_full151_gradient_v1.py'","ENTRY=ROOT/'scripts/execute_full151_gradient_v2.py'")
    builder=(ROOT/'scripts/build_full151_gradient_plan_cpu_v3.py').read_text(encoding='utf-8')
    builder=builder.replace("choices=['engineering_restore_smoke','serial_full']","choices=['engineering_restore_smoke','serial_full','inventory_discovery_all']")
    builder=builder.replace("ap.add_argument('--authorize-gpu',action='store_true')","ap.add_argument('--inventory-discovery-health',type=Path);ap.add_argument('--authorize-gpu',action='store_true')")
    builder=builder.replace("{a.tag}-v1.json","{a.tag}-v2.json").replace("full151105_gradient_{a.tag}_v1","full151105_gradient_{a.tag}_v2")
    builder=builder.replace('scripts/execute_full151_gradient_v1.py','scripts/execute_full151_gradient_v2.py').replace('scripts/run_full151_gradient_v1.py','scripts/run_full151_gradient_v2.py').replace('full151_gradient/serial_probe105_v1.py','full151_gradient/serial_probe105_v2.py').replace('from execute_full151_gradient_v1 import','from execute_full151_gradient_v2 import')
    builder=builder.replace('full151105_serial_probe_preparation_CPU_v2/','full151105_serial_probe_v2_preparation_CPU_v1/')
    builder=builder.replace("'schema':'full151105_physical_probe_execution_protocol:1'","'schema':'full151105_physical_probe_execution_protocol:2'")
    builder=once(builder,"    if smoke is not None:source['restore_smoke_health']=smoke\n","    if smoke is not None:source['restore_smoke_health']=smoke\n    discovery=None if a.inventory_discovery_health is None else pin(a.inventory_discovery_health)\n    if discovery is not None:source['inventory_discovery_health']=discovery\n")
    builder=builder.replace("'restore_smoke_health':smoke,'readiness'","'restore_smoke_health':smoke,'inventory_discovery_health':discovery,'readiness'")
    pins=[save('execute_full151_gradient_v2.py',entry),save('run_full151_gradient_v2.py',sup),save('build_full151_gradient_plan_cpu_v4.py',builder)]
    print(json.dumps({'created':pins,'AST':'PASS','production_executed':False}))
if __name__=='__main__':main()
