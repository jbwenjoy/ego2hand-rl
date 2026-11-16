"""Declared PD coverage and finite response diagnostics; no fitting or selection."""
import numpy as np
from .pen_native_response import vector_comparison,repeated_contrast_spread

BLOCKS=dict(object12=slice(40,52),wrench6=slice(78,84),Fn6=slice(84,90))


def spectrum(matrix,columns):
    a=np.asarray(matrix,dtype=np.float64).reshape(-1,columns)
    if not np.isfinite(a).all():raise ValueError('Finite matrix required')
    if not len(a):return dict(rows=0,columns=columns,numerical_rank=0,singular_values=[],condition_nonzero=None,rank_scope='Finite matrix numerical rank,not a differentiability/robustness claim.')
    s=np.linalg.svd(a,compute_uv=False);rank=int(np.linalg.matrix_rank(a));positive=s[:rank]
    return dict(rows=len(a),columns=columns,numerical_rank=rank,singular_values=s.tolist(),condition_nonzero=float(positive[0]/positive[-1]) if rank else None,
        rank_scope='Finite matrix numerical rank,not a differentiability/robustness claim.')


def review_coverage(data,metadata,groups,directions,source_names,scale,*,relative_error=.25,cosine=.9):
    x,y,mask=np.asarray(data['x']),np.asarray(data['y']),np.asarray(data['valid']);d=np.asarray(directions,dtype=np.float64);scale=np.asarray(scale,dtype=np.float64)
    if x.shape!=(len(metadata),102) or y.shape!=(len(metadata),90) or mask.shape!=(len(metadata),) or mask.dtype!=np.bool_ or scale.shape!=(90,) or np.any(scale<=0) or not all(np.isfinite(v).all() for v in [x,y,d,scale]):raise ValueError('Aligned finite source/labels/masks and fixed90-unit scales required')
    if d.shape!=(20,20) or not np.allclose(d@d.T,np.eye(20),atol=2e-15,rtol=0) or (relative_error,cosine)!=(.25,.9):raise ValueError('Original QR20 and predeclared .25/.9 amplitude screen required')
    if len(set(source_names))!=len(source_names) or any(m['source_name'] not in source_names or m['row_index']!=i or m['valid']!=bool(mask[i]) for i,m in enumerate(metadata)):raise ValueError('Exact row/source/mask identities required')
    expected=[(s,j,e) for s in source_names for j in range(20) for e in [.002,.005]]
    identities=[(g['source_name'],g['direction'],g['epsilon']) for g in groups]
    if identities!=expected:raise ValueError('All declared source/direction/amplitude groups,including empty,required')
    paired={};audit=[];seen_rows=set()
    for gi,g in enumerate(groups):
        plus=np.asarray(g['rows']['1'],dtype=np.int64);minus=np.asarray(g['rows']['-1'],dtype=np.int64);ids=np.r_[plus,minus]
        if plus.ndim!=1 or minus.ndim!=1 or len(set(ids.tolist()))!=len(ids) or np.any(ids<0) or np.any(ids>=len(metadata)):raise ValueError('Distinct in-range signed repeats required')
        for sign,selected in [(1,plus),(-1,minus)]:
            repeats=[]
            for i in selected:
                m=metadata[i];p=m['profile']
                if m['source_name']!=g['source_name'] or (p['direction'],p['epsilon'],p['sign'])!=(g['direction'],g['epsilon'],sign) or i in seen_rows:raise ValueError('Pair row relabeled or reused')
                repeats.append(p['repeat']);seen_rows.add(int(i))
            if len(set(repeats))!=len(repeats) or any(r not in [0,1] for r in repeats):raise ValueError('Two actual distinct repeats,not duplicated rows')
        complete=plus.shape==(2,) and minus.shape==(2,)
        if g['complete']!=complete or g['qualified']!=bool(complete and mask[ids].all()):raise ValueError('Group cannot override physical masks/completeness')
        row=dict(group_index=gi,source=g['source_name'],direction=g['direction'],epsilon=g['epsilon'],complete=complete,qualified=g['qualified'])
        if complete:
            xp,xn=x[plus[0]],x[minus[0]];same=np.array_equal(x[plus],np.tile(xp,(2,1))) and np.array_equal(x[minus],np.tile(xn,(2,1))) and np.array_equal(xp[:81],xn[:81]) and xp[-1]==xn[-1]
            error=float(abs(xp[81:101]-xn[81:101]-2*g['epsilon']*d[g['direction']]).max());row.update(identical_source_warm_dt=bool(same),actual_PD_error_rad=error,actual_PD_matches_declared=bool(error<=4e-7))
            # Unqualified groups remain visible with their exact defects. A
            # qualified group must also satisfy every source/actual-PD check.
            if g['qualified'] and (not same or error>4e-7):raise ValueError('Qualified response has mismatched actual source/PD')
            paired[identities[gi]]=dict(group=gi,qualified=g['qualified'],actual=(xp[81:101]-xn[81:101])/(2*g['epsilon']),fd=(y[plus].mean(0)-y[minus].mean(0))/(2*g['epsilon']),
                spread={b:repeated_contrast_spread(y[plus,sl],y[minus,sl],scale[sl]) for b,sl in BLOCKS.items()})
        audit.append(row)
    expected_rows={i for i,m in enumerate(metadata) if m['profile']['direction']>=0}
    if seen_rows!=expected_rows:raise ValueError('All available signed rows must appear exactly once')
    ranks=[];stability=[]
    for source in source_names:
        sets=[]
        for e in [.002,.005]:
            ids=[j for j in range(20) if (source,j,e) in paired and paired[(source,j,e)]['qualified']];sets.append(set(ids))
            ranks.append(dict(source=source,epsilon=e,eligible_directions=ids,declared_input=spectrum(d[ids],20),actual_PD_input=spectrum([paired[(source,j,e)]['actual'] for j in ids],20),
                normalized_finite_output={b:spectrum([paired[(source,j,e)]['fd'][sl]/scale[sl] for j in ids],sl.stop-sl.start) for b,sl in BLOCKS.items()}))
        both=sorted(sets[0]&sets[1]);ranks.append(dict(source=source,epsilon='BOTH',eligible_directions=both,declared_input=spectrum(d[both],20)))
        for j in range(20):
            small,large=paired.get((source,j,.002)),paired.get((source,j,.005))
            if small is None or large is None:continue
            blocks={}
            for b,sl in BLOCKS.items():
                comparison=vector_comparison(small['fd'][sl]/scale[sl],large['fd'][sl]/scale[sl])
                passed=small['qualified'] and large['qualified'] and comparison['relative_L2_error']<=relative_error and comparison['cosine'] is not None and comparison['cosine']>=cosine and small['spread'][b]['contrast_exceeds_observed_spread'] and large['spread'][b]['contrast_exceeds_observed_spread']
                blocks[b]=dict(amplitude=comparison,small_spread=small['spread'][b],large_spread=large['spread'][b],physical_screen=bool(passed))
            stability.append(dict(source=source,direction=j,blocks=blocks))
    return dict(group_audit=audit,source_spectra=ranks,finite_amplitude_screens=stability,
        physical_screen_direction_counts={b:sum(r['blocks'][b]['physical_screen'] for r in stability) for b in BLOCKS},
        eligible_pair_groups={f'rank{n}':[i for i,g in enumerate(groups) if g['qualified'] and g['direction']<n] for n in [4,20]},
        scope='Input direction coverage and finite output spectra are distinct. Screen is descriptive only;no TRAIN pair filtering by response size/stability,or inference of infinitesimal derivatives.')
