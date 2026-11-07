"""Author description of every saved snapshot; no independent qualification."""
import collections,csv,hashlib,json
from pathlib import Path

ROOT=Path('D:/workspace/supermimic-full151-observation-study')
RUN=ROOT/'outputs/full151105_saved_contact_regime608_CPU_v1'
OUT=ROOT/'outputs/full151105_saved_contact_regime608_author_description_v1'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def pin(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def save(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def main():
    if OUT.exists():raise ValueError('Fresh author description only')
    report=read(RUN/'report.json')
    if pin(RUN/'report.json')['sha256']!='fa62879353cebdbbeee8032fe1f3fed601da76b6b9907725698020790c181932':raise ValueError('actual report pin')
    samples=read(RUN/'sample608.json');branches=read(RUN/'branch152.json');snaps=read(RUN/'snapshot8.json');pairs=read(RUN/'response_pairs64.json')
    contacts=[json.loads(line) for line in (RUN/'contact_long.jsonl').open(encoding='utf-8')]
    catalog=read(RUN/'geometry_catalog_source.json');bodies={b['id']:b['name'] for b in catalog['native_body_catalog']}
    if (len(samples),len(contacts),len(branches),len(snaps),len(pairs))!=(608,2831,152,8,64):raise ValueError('All saved rows retained')
    rows=[];detail=[]
    for snapshot in snaps:
        s=snapshot['snapshot_id'];ss=[x for x in samples if x['snapshot_id']==s];cs=[x for x in contacts if x['snapshot_id']==s]
        bs=branches[19*s:19*s+19];pert=[b['response_descriptor'] for b in bs[3:]];ps=[p for p in pairs if p['snapshot_id']==s]
        row={'snapshot_id':s,'path':snapshot['path'],'k':snapshot['k'],'branch_count':len(bs),'PRE_sample_count':len(ss),
          'all_generated_contact_rows':len(cs),'HO_generated':sum(c['category']=='hand_object' for c in cs),
          'HO_secondary_positive':sum(c['category']=='hand_object' and c['secondary_positive_force'] for c in cs),
          'HH_generated':sum(c['category']=='hand_self' for c in cs),
          'HH_secondary_positive':sum(c['category']=='hand_self' and c['secondary_positive_force'] for c in cs),
          'other_generated':sum(c['category']=='other' for c in cs),
          'object_nonhand_generated':sum(c['object_nonhand'] for c in cs),
          'object_nonhand_secondary_positive':sum(c['object_nonhand'] and c['secondary_positive_force'] for c in cs),
          'hand_other_generated':sum(c['hand_other'] for c in cs),
          'samples_with_generated_HO':sum(x['categories']['hand_object']['generated_candidate_count']>0 for x in ss),
          'samples_with_secondary_positive_HO':sum(x['categories']['hand_object']['secondary_positive_force_count']>0 for x in ss),
          'perturbation_endpoints':len(pert),
          'physical_centered_p_and_R_encoded_flat_endpoints':sum(d['physical_p_centered_all_numeric_zero'] and d['physical_R_centered_all_numeric_zero'] for d in pert),
          'model_centered_p_and_R_encoded_flat_endpoints':sum(d['model_p_centered_all_numeric_zero'] and d['model_R_centered_all_numeric_zero'] for d in pert),
          'full20_control_unchanged_endpoints':sum(d['zero_control_change_all20_numeric'] for d in pert),
          'epsilon_pairs':len(ps),'physical_pair_p_and_R_numeric_flat':sum(d['physical_encoding']['p_all_numeric_zero'] and d['physical_encoding']['R_all_numeric_zero'] for d in ps),
          'pair_all20_control_delta_zero':sum(d['zero_actual_all20_delta'] for d in ps)}
        rows.append(row)
        groups=collections.defaultdict(list)
        for c in cs:groups[(c['category'],tuple(c['ordered_geom_pair']))].append(c)
        names=[]
        for (category,geom),values in sorted(groups.items()):
            first=values[0]
            names.append({'category':category,'ordered_native_geom_ids':list(geom),'native_geom_names':first['native_geom_names'],
              'native_body_ids':first['native_body_ids'],'native_body_names':[bodies[i] for i in first['native_body_ids']],
              'generated_rows':len(values),'secondary_positive_rows':sum(x['secondary_positive_force'] for x in values),
              'source_helper_active_rows':sum(x['source_helper_active_by_original_formula'] for x in values),
              'force_evidence_eligible_rows':sum(x['force_evidence_eligible'] for x in values)})
        detail.append({**row,'all_native_pair_names':names,'all_four_substep_profiles':snapshot['category_four_substep_profile_summed_over_19_branches'],
          'all_branch_ids':snapshot['branch_ids'],'all_pair_ids':snapshot['related_pair_ids']})
    OUT.mkdir();save('source_freeze_before_description.json',pin(Path(__file__)))
    save('all8_snapshot_table.json',detail)
    with (OUT/'all8_snapshot_table.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    table=['| Path | k | PRE | Generated | HO / secondary+ | HH | object–nonhand / secondary+ | p/R-flat perturbations | unchanged u endpoints | flat p/R pairs |',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:table.append(f"| {r['path']} | {r['k']} | {r['PRE_sample_count']} | {r['all_generated_contact_rows']} | {r['HO_generated']} / {r['HO_secondary_positive']} | {r['HH_generated']} | {r['object_nonhand_generated']} / {r['object_nonhand_secondary_positive']} | {r['physical_centered_p_and_R_encoded_flat_endpoints']}/16 | {r['full20_control_unchanged_endpoints']}/16 | {r['physical_pair_p_and_R_numeric_flat']}/8 |")
    md='''# Saved contact regime: author description

This is an author summary of all 608 branch PRE samples, 2,831 original contact records, 152 restored intervals, eight snapshots and 64 epsilon pairs. It is not an independent audit. Every snapshot appears below; no flat-response, force-positive or old stable-group selection was applied.

'''+ '\n'.join(table)+'''

The table counts repeated generated contact records, not unique persistent contacts or independent observations. Each snapshot contains 19 separate four-substep intervals. “Secondary+” means the original decoded normal force exceeds 1e-6 N and the saved constraint-type bit, world, raw-slot and host-address guards pass. It does not replace the primary generated-geometry count.

A2 k49 has no generated hand–object or object–nonhand pair across its 76 PRE samples. Its 171 generated records are hand–hand. A2 k99 and k149 also have no generated hand–object records, but each has 608 object–nonhand records, all secondary-positive under the saved evidence rule, plus 152 hand–hand records. Thus the late p/R encoding-flat observations cannot all be described as contact-free object motion.

At k99 and k149, the original ordered native pairs are (0,2), `terrain` ↔ `object/A_bottom`, and (0,3), `terrain` ↔ `object/A_top`; their body names are `terrain` ↔ `object/cube`. Each pair contributes 304 records per snapshot. These are exact saved names, not a newly certified static-environment or support-contact model. The same snapshots contain hand pairs (53,66), `robot/right_finger4_link3_col` ↔ `robot/right_finger5_link3_col`, and (14,27), `robot/right_finger1_link3_col` ↔ `robot/right_finger2_link3_col`. At k49 the hand pairs are (55,70), `robot/right_finger4_link4_col` ↔ `robot/right_finger5_tip_sensor_col`, and (18,29), `robot/right_finger1_tip_sensor_col` ↔ `robot/right_finger2_link4_col`. Full native body names and all other snapshots’ pairs are retained in the JSON table.

All 16 perturbed endpoints at each of A2 k49, k99 and k149 have zero saved physical p/R centered response; all eight plus/minus p/R pairs are numerically flat. Only two perturbed endpoints per snapshot have unchanged full20 control. The other 42 of the 48 late endpoints therefore have a nonzero recorded control change despite flat object p/R encoding. This is a finite, FP32-encoded, one-interval observation for the tested coordinates and amplitudes. Hand-state responses, finite resolution, contact constraints and hidden state remain separate issues; this table does not identify a unique cause or establish a zero physical Jacobian.

Absence of generated candidates is not global clearance. Force6 values were decoded by the original producer; contact dimension is missing for a complete independent pyramid decode. Stored world force and COM moment were only copied by the producer and are not recomputed here. Host addresses and raw WP slots remain different address spaces. No model, state converter, FK, MJB, world wrench calculation, AD, new physics or training is executed by this description.
'''
    (OUT/'description.md').write_text(md,encoding='utf-8')
    inputs={name:pin(RUN/name) for name in ('report.json','sample608.json','contact_long.jsonl','branch152.json','snapshot8.json','response_pairs64.json','geometry_catalog_source.json','response_branch152.json')}
    save('report.json',{'schema':'full151105_contact_regime608_author_saved_description:1','status':'COMPLETED_AUTHOR_ALL8_SNAPSHOT_DESCRIPTION_NOT_INDEPENDENT_AUDIT',
      'producer_report':inputs['report.json'],'input_pins':inputs,'author_source':pin(Path(__file__)),'counts':report['counts'],
      'new_model_state_MJB_FK_AD_world_wrench_physics_training':0,'physical_gradient_qualified':False,'whole_hidden_state_qualified':False,
      'global_clearance_qualified':False,'causal_explanation_qualified':False})
    save('output_pins.json',{x.name:pin(x) for x in OUT.iterdir() if x.is_file()})
    print(json.dumps({'status':'COMPLETED_AUTHOR_ALL8_SNAPSHOT_DESCRIPTION','report':pin(OUT/'report.json'),'description':pin(OUT/'description.md')}))

if __name__=='__main__':main()
