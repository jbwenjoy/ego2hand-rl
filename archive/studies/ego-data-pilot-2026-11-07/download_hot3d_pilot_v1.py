"""Bounded public dataset download; no SDK/model/imported remote code."""
import datetime,hashlib,json,pathlib,urllib.request
ROOT=pathlib.Path(__file__).resolve().parent
REPO='bop-benchmark/hot3d'
REV=json.loads((ROOT/'sources/hot3d_repository_metadata.json').read_text())['sha']
IDS=('001852','001856','001884')
MAX_BYTES=300_000_000
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def pin(p):return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)}
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def url(path):return f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{path}'
def small(name,u,limit=2_000_000):
    with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'ego-data-pilot/1.0'}),timeout=30) as r:b=r.read(limit+1)
    if len(b)>limit:raise ValueError('Metadata size limit')
    p=ROOT/'sources'/name
    with p.open('xb') as f:f.write(b)
    return {'requested_url':u,**pin(p)}
def main():
    tree=json.loads((ROOT/'sources/hot3d_train_aria_tree_first100.json').read_text());lookup={x['path']:x for x in tree};items=[lookup[f'train_aria/clip-{i}.tar'] for i in IDS]
    if sum(x['size'] for x in items)>MAX_BYTES:raise ValueError('Download budget exceeded')
    write(ROOT/'download_plan_v1.json',{'schema':'hot3d_small_ego_download_plan:1','repo':REPO,'revision':REV,'clip_ids':list(IDS),'files':items,'total_expected_bytes':sum(x['size'] for x in items),'maximum_bytes':MAX_BYTES,'selection':'Named RGB train clips selected before inspection; compatibility pilot, no independent trajectory/participant claim.','download_script':pin(pathlib.Path(__file__))})
    sources=[small('hot3d_clips_README.md','https://raw.githubusercontent.com/facebookresearch/hot3d/main/hot3d/clips/README.md'),small('hot3d_dataset_license_agreement.pdf',url('hot3d_dataset_license_agreement.pdf')),small('hot3d_clip_splits.json',url('clip_splits.json'))]
    rows=[];dest=ROOT/'hot3d'/'train_aria';dest.mkdir(parents=True,exist_ok=True)
    try:
        for item in items:
            name=pathlib.PurePosixPath(item['path']).name;p=dest/name;temp=p.with_suffix('.tar.partial')
            if p.exists() or temp.exists():raise ValueError('Fresh file required; preserve existing download')
            print(json.dumps({'status':'DOWNLOADING','file':item['path'],'expected_bytes':item['size']}),flush=True)
            request=urllib.request.Request(url(item['path']),headers={'User-Agent':'ego-data-pilot/1.0'})
            done=0;reported=0;h=hashlib.sha256()
            with urllib.request.urlopen(request,timeout=30) as response,temp.open('xb') as f:
                while True:
                    b=response.read(1024*1024)
                    if not b:break
                    done+=len(b)
                    if done>item['size']:raise ValueError('Unexpected larger content')
                    f.write(b);h.update(b)
                    if done-reported>=16*1024*1024:print(json.dumps({'file':name,'received_bytes':done}),flush=True);reported=done
            if done!=item['size'] or h.hexdigest()!=item['lfs']['oid']:raise ValueError('LFS original size/SHA256 mismatch')
            temp.rename(p);rows.append({'requested_url':url(item['path']),'expected_LFS_SHA256':item['lfs']['oid'],**pin(p)})
            print(json.dumps({'status':'VERIFIED','file':name,'bytes':done,'sha256':h.hexdigest()}),flush=True)
        write(ROOT/'download_manifest_v1.json',{'schema':'hot3d_small_ego_download_manifest:1','created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repo':REPO,'revision':REV,'clips':rows,'metadata_and_license':sources,'total_clip_bytes':sum(x['bytes'] for x in rows),'models_training_SDK_execution':0,'raw_data_only_not_retargeting_or_dynamics_qualification':True})
        print(json.dumps({'status':'COMPLETE_THREE_VERIFIED_RAW_EGO_CLIPS','total_clip_bytes':sum(x['bytes'] for x in rows)}),flush=True)
    except BaseException as e:
        write(ROOT/'download_failure_v1.json',{'type':type(e).__name__,'message':str(e),'verified_files':rows,'metadata':sources});raise
if __name__=='__main__':main()
