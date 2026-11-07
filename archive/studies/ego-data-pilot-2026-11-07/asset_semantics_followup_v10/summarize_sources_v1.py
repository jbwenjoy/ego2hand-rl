"""Describe immutable source identity and metadata; no asset/model loading."""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
D=HERE/'official_sources'
LOCAL=Path(r'D:/workspace/dex-supermimic-research/packages/wuji-description')
BODY=LOCAL/'src/wuji_description/resources/hand2/hand2_2/body'

def pin(f):
    f=Path(f).resolve();b=f.read_bytes()
    return {'path':str(f),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def read(f):return json.loads(Path(f).read_text(encoding='utf-8'))

def main():
    dest=HERE/'report.json'
    if dest.exists():raise FileExistsError(dest)
    manifest=read(HERE/'source_manifest.json')
    current=read(D/'current_commit.json');pinned=read(D/'pinned_commit.json');sdk=read(D/'sdk_current_commit.json')
    trees={label:{x['path']:x for x in read(D/(label+'_tree.json'))['tree']} for label in ('pinned','current')}
    mesh=[]
    for finger in ('thumb','index_finger','middle_finger','ring_finger','pinky'):
        name='r_'+finger+'_tip_sensor_frame.STL';path='hand2/hand2_2/body/meshes/right/'+name
        f=BODY/'meshes/right'/name;b=f.read_bytes();localblob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        mesh.append({'file':name,'local_pin':pin(f),'local_git_blob_sha1':localblob,
            'pinned_git_blob_sha1':trees['pinned'][path]['sha'],'current_git_blob_sha1':trees['current'][path]['sha'],
            'all_three_identical':localblob==trees['pinned'][path]['sha']==trees['current'][path]['sha']})
    local=BODY/'mjcf/right.xml';a=(D/'pinned_right.xml').read_bytes();b=(D/'current_right.xml').read_bytes()
    xml=ET.fromstring(a);sensor=xml.find('sensor')
    pad_geoms=[]
    for body in xml.findall('.//body'):
        if body.get('name','').endswith('tip_sensor_frame'):
            pad_geoms.append({'body':body.get('name'),'geoms':[dict(g.attrib) for g in body.findall('geom')]})
    sources={Path(r['path']).name:{'url':r['url'],'pin':{'path':r['path'],'bytes':r['bytes'],'sha256':r['sha256']}} for r in manifest['files']}
    def version(r):return {'sha':r['sha'],'commit_date':r['commit']['committer']['date'],'message':r['commit']['message'].splitlines()[0]}
    result={'schema':'official_Hand2_functional_pad_semantics_followup:10','role':'source_information_only_not_asset_or_physics_qualification',
        'source':pin(__file__),'retrieved_UTC':manifest['retrieved_UTC'],'source_manifest':pin(HERE/'source_manifest.json'),
        'upstream_repo':'https://github.com/wuji-technology/wuji-description','local_wrapper_package_version':'0.1.0 (local resource packaging, not upstream release)',
        'local_provenance_README':pin(LOCAL/'README.md'),'pinned_version':version(pinned),'current_version':version(current),'SDK_version':version(sdk),
        'mesh_identity_all5':mesh,
        'right_MJCF_comparison':{'official_pinned':pin(D/'pinned_right.xml'),'official_current':pin(D/'current_right.xml'),
            'local':pin(local),'official_pinned_current_raw_bytes_equal':a==b,'local_raw_bytes_equal_pinned':local.read_bytes()==a,
            'local_newline_normalized_text_equal_pinned':local.read_text(encoding='utf-8')==a.decode('utf-8'),
            'local_CRLF_count':local.read_bytes().count(b'\r\n')},
        'source_xml_sensor_tags':[x.tag for x in sensor], 'source_xml_tip_sensor_geoms':pad_geoms,
        'findings':[
            {'claim':'Official Hand2.2 identifies tip_sensor_frame as a fixed fingertip tactile-pad link; whole-link mesh collision is not a published functional facet mask.',
             'source':sources['current_README.md']},
            {'claim':'Inspected hardware manual describes skeleton, compliant tissue and friction-providing passive outer skin, but limits itself to a non-tactile2.2Beta sample and says tip hardness/skin scheme remain unsettled. No measured friction or stiffness values were found.',
             'source':sources['hardware_overview.md']},
            {'claim':'Official model documentation says soft exterior is represented rigidly; no soft deformation simulation. Meshes are convexified for collision.',
             'source':sources['model_integration.md']},
            {'claim':'SDK can deliver device-specific sensor positions and force-axis placement via complete format metadata: positions/point_rpy/base_xyz/base_rpy, linked to tip_sensor_frame and info digest. Numbers are read from a real device; no matching offline metadata was available in this audit.',
             'source':sources['SDK_fingertip.md']},
            {'claim':'SDK aggregate location is a fixed geometric centroid and the derived lever moment is not measured contact torque. Force-axis placement is not automatically a geometric outward-normal/facet identity.',
             'source':sources['SDK_fingertip.md']},
            {'claim':'First-generation hand/body-with-soft and current hand2_2 are distinct variants. Current README labels full-hand STEP as historical Beta assemblies not regenerated from current appearance models; these are not version-matched new functional-surface evidence.',
             'source':sources['current_README.md']}],
        'functional_contact_face_or_outward_normal_map':'UNKNOWN_NOT_FOUND_IN_INSPECTED_PINNED_OR_CURRENT_PUBLIC_SOURCES',
        'catalog_172_108_mm2_planes_functional_or_mounting':'UNKNOWN; area and planarity alone do not assign function',
        'measured_mu_compliance_hardness_thickness':'UNKNOWN_NOT_INFERRED_FROM_COLOR_NAMES_OR_PHOTOS',
        'frozen_model_changed':False,'new_model_FK_forward_LP_physics_NN_GPU_calls':0,
        'source_download_scope':{'text_and_git_metadata_bytes':manifest['total_source_downloaded_bytes'],'text_metadata_files':len(manifest['files']),
            'extra_binary_USD_sensor_sublayer_bytes':657,'extra_binary_USD_scope':'Saved but not semantically decoded; not used as negative semantic evidence.',
            'mesh_or_CAD_downloads':0,'SDK_install_or_device_access_or_external_messages':0},
        'network_source_gap':manifest['errors'],
        'action':'Keep functional facet semantics UNKNOWN. Matching real-device sensor format plus asset revision could provide installation points/axes; any geometric contact-surface choice before that must remain a declared engineering hypothesis, not calibrated hardware truth.'}
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'report':pin(dest),'current_commit':current['sha'],'pinned_commit':pinned['sha'],
        'five_mesh_blobs_unchanged':all(x['all_three_identical'] for x in mesh),'official_MJCF_unchanged':a==b,
        'local_XML_LF_equal':result['right_MJCF_comparison']['local_newline_normalized_text_equal_pinned'],
        'functional_face':result['functional_contact_face_or_outward_normal_map']}))

if __name__=='__main__':main()
