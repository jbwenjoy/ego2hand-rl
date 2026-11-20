"""Plot per-behavior acquisition yields and source/pose backend discrepancy."""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin
from supermimic_research.strict_json import write_json


def main(out):
    out.mkdir(exist_ok=False);root=ROOT/'runs'
    files=[root/n/'report.json' for n in ('pen_switch_sample_review_v1','pen_switch_group_oracle_v1','pen_gpu_pose_geometry_v1')]
    sample,cpu,gpu=[json.loads(f.read_bytes()) for f in files]
    fig,(a,b)=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    roles=['TRAIN_A','TRAIN_B','TEST_RESIDUAL','TEST_SUPPORT'];y=np.arange(len(roles))
    for name,dy,color in [('risk',-.17,'#247a8c'),('uniform',.17,'#be7844')]:
        h=[sample['summary'][r][name]['pulse_d23_membership'] for r in roles];bars=a.barh(y+dy,h,.30,label=name.capitalize(),color=color)
        a.bar_label(bars,padding=3,fontsize=9)
    a.set(yticks=y,yticklabels=['Train A','Train B','Residual test','Support test'],xlim=(0,6),xlabel='Sites with action-dependent switch / 12',title='Perturbed fingers 2 and 3')
    a.invert_yaxis();a.legend(loc='lower right')
    methods=['CPU, recorded poses','CPU, native group existence*','GPU, recorded poses','GPU, learned poses'];y=np.arange(len(methods))
    for role,dy,color in [('TEST_RESIDUAL',-.17,'#386999'),('TEST_SUPPORT',.17,'#bf573f')]:
        z=[cpu['summary'][role][0],cpu['summary'][role][1],gpu['summary'][role]['true_pose']['relative_error'],gpu['summary'][role]['secant']['relative_error']]
        bars=b.barh(y+dy,z,.30,color=color,label=role.split('_')[1].title());b.bar_label(bars,fmt='%.2f',padding=2,fontsize=8)
    b.set(yticks=y,yticklabels=methods,xlim=(0,3),xlabel='Capacity response error / zero-response error',title='Same action contrasts on the two test policies')
    b.axvline(1,color='#777',ls=':',lw=1);b.invert_yaxis();b.legend(loc='upper right')
    fig.suptitle('Pre-action risk sampling and native geometry checks')
    fig.text(.52,-.045,'* Post-hoc use of native contacts to replace missing/extra CPU groups.',fontsize=8)
    dest=out/'switch_summary.png';fig.savefig(dest,dpi=150,bbox_inches='tight');plt.close(fig)
    write_json(out/'report.json',dict(inputs=[file_pin(f) for f in files],figure=file_pin(dest)))
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
