"""Fixed complete ten-condition replay schedule; no simulator imports."""


def schedule(rows, replica):
    if replica not in ['A','B'] or len(rows)!=10 or len({r['name'] for r in rows})!=10:
        raise ValueError('Two replicas and ten distinct declared conditions required')
    if {(r['pair'],r['role']) for r in rows}!={(i,k) for i in range(5) for k in ['failure','control']}:
        raise ValueError('All five failure/comparison pairs required')
    ordinal=['A','B'].index(replica);result=[]
    for teacher in ['RESIDUAL','SUPPORT']:
        ids=[i for i,r in enumerate(rows) if ('SUPPORT' if r['name'].startswith('SUPPORT_') else 'RESIDUAL')==teacher]
        if len(ids)!=(8 if teacher=='RESIDUAL' else 2):raise ValueError('Original teacher allocation required')
        for repeat in range(3):
            offset=(repeat+ordinal)%len(ids);order=ids[offset:]+ids[:offset]
            scopes=['PACKET_BASELINE','PACKET_PLUS_SOLVE_EXIT']
            if (repeat+ordinal)%2:scopes.reverse()
            for ri in order:
                for scope in scopes:
                    result.append(dict(replica=replica,teacher=teacher,repeat=repeat,row_index=ri,
                        original_name=rows[ri]['name'],scope=scope,
                        name=f'{replica}_{teacher}_R{repeat}_C{ri:02}_{scope}'))
    return result
