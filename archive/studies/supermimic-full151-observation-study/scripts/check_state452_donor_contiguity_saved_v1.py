"""Saved-only donor endpoint/prestate byte check; constructs no H8 windows."""
from pathlib import Path
import hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'outputs/full151105_state_prediction452_CPU_v1'
OUTPUT=ROOT/'outputs/full151105_H8_donor_source_feasibility_v1'


def pin(path):
    return {'path':str(path),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def same_pin(a,b):
    return a['sha256']==b['sha256'] and Path(a['path']).samefile(Path(b['path']))


def main():
    assert not OUTPUT.exists()
    source=pin(Path(__file__).resolve())
    manifest=json.loads((INPUT/'output_pins.json').read_text())
    relevant={k:manifest[k] for k in ('matched_input_arrays.npz','group_identities452.json')}
    for value in relevant.values():assert same_pin(pin(Path(value['path'])),value)
    rows=json.loads((INPUT/'group_identities452.json').read_text())
    with np.load(INPUT/'matched_input_arrays.npz',allow_pickle=False) as z:
        pre=z['pre_x452_FP32'].copy();post=z['target_x452_FP32'].copy();u=z['actual_u452_FP32'].copy();u4=z['all_substep_u452x4x20_FP32'].copy()
    assert pre.shape==post.shape==(452,58) and pre.dtype==post.dtype==np.float32
    assert np.isfinite(pre).all() and np.isfinite(post).all()
    assert u.shape==(452,20) and u4.shape==(452,4,20) and u.dtype==u4.dtype==np.float32
    assert np.isfinite(u).all() and np.isfinite(u4).all()
    lookup={};counts={}
    for index,row in enumerate(rows):
        counts[row['mode']]=counts.get(row['mode'],0)+1
        if row['mode']=='donor':
            key=(row['path'],row['k']);assert key not in lookup and row['branch_id'] is None;lookup[key]=index
    assert counts=={'donor':300,'baseline':24,'direction':128}
    links=[];control_rows=[]
    for path in ('A0','A2'):
        assert {k for p,k in lookup if p==path}==set(range(150))
        for k in range(150):
            index=lookup[path,k]
            same=all(np.array_equal(u[index].view(np.uint32),u4[index,s].view(np.uint32)) for s in range(4))
            control_rows.append({'path':path,'k':k,'group_id':index,'four_actual_absolute_controls_bitexact':same})
        for k in range(149):
            a,b=lookup[path,k],lookup[path,k+1]
            bitmask=post[a].view(np.uint32)==pre[b].view(np.uint32)
            features=np.flatnonzero(~bitmask)
            links.append({'path':path,'post_k':k,'next_pre_k':k+1,'post_group_id':a,'next_pre_group_id':b,
              'all58_bitexact':bool(bitmask.all()),'mismatched_features':features.tolist(),
              'mismatches':[{'feature':int(j),'post_bits':int(post[a,j].view(np.uint32)),'next_pre_bits':int(pre[b,j].view(np.uint32)),
                            'signed_difference_F64':float(post[a,j])-float(pre[b,j])} for j in features]})
    for value in relevant.values():assert same_pin(pin(Path(value['path'])),value)
    assert same_pin(pin(Path(__file__).resolve()),source)
    passed=all(x['all58_bitexact'] for x in links)
    result={'schema':'full151105_H8_donor_saved_contiguity_source_feasibility:1',
      'status':'PASS_SAVED_DONOR_CONTIGUITY_ONLY' if passed else 'DONOR_CONTIGUITY_FAILURE_RETAINED',
      'source':source,'input_pins':relevant,'donor_paths':2,'donor_intervals':300,'unique_adjacent_links':298,
      'components_per_link':58,'compared_component_bits':298*58,'all_links_all58_bitexact':passed,
      'failed_link_count':sum(not x['all58_bitexact'] for x in links),'all_donor_four_controls_bitexact':all(x['four_actual_absolute_controls_bitexact'] for x in control_rows),
      'links':links,'control_rows':control_rows,
      'H8_metadata_only':{'declared_start_range_inclusive':[0,142],'starts_per_path':143,'total_overlapping_windows_if_separately_authorized':286,
         'last_window_controls':'142..149','last_window_final_observation':150,'restored_branch_intervals_excluded_by_availability':152,
         'actual_window_arrays_constructed':0,'future_window_filtering':'No error-based selection or skipping; any continuity break is reported, not repaired.'},
      'execution':{'saved_NPZ_files_read':1,'models_or_WM_or_Actor_loads':0,'forward_calls':0,'AD_calls':0,'state_conversions':0,'window_arrays_constructed':0,'Env_physics_GPU':0,
                   'Torch_imported':'torch' in sys.modules,'Warp_imported':'warp' in sys.modules},
      'scope':'Source feasibility plus actual saved byte-contiguity only; not an H8 prediction, hidden-state certificate or policy/imagination/gradient replay.'}
    OUTPUT.mkdir()
    with (OUTPUT/'contiguity.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('links','control_rows')},indent=2))


if __name__=='__main__':main()
