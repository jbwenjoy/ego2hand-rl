"""Read-only source completion after an absent current CHANGELOG URL, no retry."""
from pathlib import Path
import datetime,hashlib,json,urllib.request,urllib.error

HERE=Path(__file__).resolve().parent
D=HERE/'official_sources'
PIN='829c9aa27cb12c3a49d5b8d63c9869dd8f656702'
HEAD=json.loads((D/'current_commit.json').read_text(encoding='utf-8'))['sha']
manifest=[];errors=[{'url':f'https://raw.githubusercontent.com/wuji-technology/wuji-description/{HEAD}/CHANGELOG.md','status':404,'scope':'Initial source fetch stopped; current tree confirms file absent. No scientific execution.'}]

def save(name,url):
    f=D/name
    if f.exists():raise FileExistsError(f)
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'asset-semantics-source-audit','Accept':'text/plain, application/json'})
        with urllib.request.urlopen(req,timeout=40) as r:b=r.read();h=dict(r.headers)
        f.write_bytes(b)
        manifest.append({'path':str(f.resolve()),'url':url,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'HTTP_date':h.get('Date'),'last_modified':h.get('Last-Modified')})
        return b
    except urllib.error.HTTPError as e:
        errors.append({'url':url,'status':e.code});return None

for name,url in [
    ('pinned_tree.json',f'https://api.github.com/repos/wuji-technology/wuji-description/git/trees/{PIN}?recursive=1'),
    ('pinned_README.md',f'https://raw.githubusercontent.com/wuji-technology/wuji-description/{PIN}/README.md'),
    ('pinned_right.xml',f'https://raw.githubusercontent.com/wuji-technology/wuji-description/{PIN}/hand2/hand2_2/body/mjcf/right.xml'),
    ('current_right.xml',f'https://raw.githubusercontent.com/wuji-technology/wuji-description/{HEAD}/hand2/hand2_2/body/mjcf/right.xml'),
    ('current_release.json','https://api.github.com/repos/wuji-technology/wuji-description/releases/tags/v2026.10.26'),
    ('pinned_release.json','https://api.github.com/repos/wuji-technology/wuji-description/releases/tags/v2026.10.12'),
    ('hardware_overview.md','https://docs.wuji.tech/docs/zh/wuji-hand/latest/overview.md'),
    ('model_integration.md','https://docs.wuji.tech/docs/zh/wuji-hand/latest/model-integration.md'),
    ('version_compatibility.md','https://docs.wuji.tech/docs/zh/wuji-hand/latest/version-compatibility.md'),
    ('SDK_fingertip.md','https://docs.wuji.tech/docs/zh/wuji-sdk/latest/wuji-hand-2-tactile.md'),
    ('sdk_current_commit.json','https://api.github.com/repos/wuji-technology/wuji-sdk/commits/main')]:save(name,url)
sdk=json.loads((D/'sdk_current_commit.json').read_text(encoding='utf-8'))
save('SDK_fingertip_example.py',f"https://raw.githubusercontent.com/wuji-technology/wuji-sdk/{sdk['sha']}/examples/python/wuji_hand_2/3.fingertip_typed.py")
for name,path in [('current_commit.json','commits/main'),('pinned_commit.json','commits/'+PIN),('current_tree.json','git/trees/'+HEAD+'?recursive=1')]:
    f=D/name;b=f.read_bytes();manifest.append({'path':str(f.resolve()),'url':'https://api.github.com/repos/wuji-technology/wuji-description/'+path,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
f=D/'current_README.md';b=f.read_bytes();manifest.append({'path':str(f.resolve()),'url':f'https://raw.githubusercontent.com/wuji-technology/wuji-description/{HEAD}/README.md','bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
out={'schema':'official_hand2_asset_semantics_source_snapshots:1','retrieved_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'current_commit':HEAD,'pinned_commit':PIN,'SDK_commit':sdk['sha'],'files':manifest,'errors':errors,'total_source_downloaded_bytes':sum(x['bytes'] for x in manifest),'model_or_CAD_downloads':0,'new_model_physics_calls':0}
dest=HERE/'source_manifest.json'
if dest.exists():raise FileExistsError(dest)
dest.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'current_commit':HEAD,'SDK_commit':sdk['sha'],'files':len(manifest),'source_bytes':out['total_source_downloaded_bytes'],'errors':errors}))
