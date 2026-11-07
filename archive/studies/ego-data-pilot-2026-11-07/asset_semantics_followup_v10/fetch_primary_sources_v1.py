"""Small official text/metadata snapshots; never downloads models or CAD."""
from pathlib import Path
import datetime,hashlib,json,urllib.request

HERE=Path(__file__).resolve().parent
UP='https://api.github.com/repos/wuji-technology/wuji-description'
PINNED='829c9aa27cb12c3a49d5b8d63c9869dd8f656702'


def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'saved-asset-source-audit','Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(req,timeout=40) as r:return r.read(),dict(r.headers)


def main():
    dest=HERE/'official_sources'
    dest.mkdir(exist_ok=False)
    manifest=[]
    def save(name,url):
        b,h=get(url);f=dest/name;f.write_bytes(b)
        manifest.append({'path':str(f.resolve()),'url':url,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),
            'HTTP_date':h.get('Date'),'last_modified':h.get('Last-Modified'),'etag':h.get('ETag')})
        return b
    current=json.loads(save('current_commit.json',UP+'/commits/main'))
    pinned=json.loads(save('pinned_commit.json',UP+'/commits/'+PINNED))
    head=current['sha']
    for label,sha in [('current',head),('pinned',PINNED)]:
        tree=json.loads(save(label+'_tree.json',UP+'/git/trees/'+sha+'?recursive=1'))
        for name,path in [('README.md','README.md'),('CHANGELOG.md','CHANGELOG.md'),
                          ('right.xml','hand2/hand2_2/body/mjcf/right.xml')]:
            save(label+'_'+name,'https://raw.githubusercontent.com/wuji-technology/wuji-description/'+sha+'/'+path)
    for name,url in [('docs_index.txt','https://docs.wuji.tech/llms.txt'),
        ('hardware_overview.md','https://docs.wuji.tech/docs/zh/wuji-hand/latest/overview.md'),
        ('version_compatibility.md','https://docs.wuji.tech/docs/zh/wuji-hand/latest/version-compatibility.md'),
        ('SDK_fingertip.md','https://docs.wuji.tech/docs/zh/wuji-sdk/latest/wuji-hand-2-tactile.md')]:
        save(name,url)
    result={'schema':'official_hand2_asset_semantics_source_snapshots:1',
        'retrieved_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repo':'wuji-technology/wuji-description',
        'current_commit':head,'current_commit_date':current['commit']['committer']['date'],
        'pinned_commit':PINNED,'pinned_commit_date':pinned['commit']['committer']['date'],
        'files':manifest,'total_downloaded_bytes':sum(r['bytes'] for r in manifest),
        'model_or_CAD_downloads':0,'new_model_physics_calls':0}
    (HERE/'source_manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('current_commit','current_commit_date','pinned_commit_date','total_downloaded_bytes')}))


if __name__=='__main__':main()
