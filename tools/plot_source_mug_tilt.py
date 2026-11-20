"""Compare source annotations, the reanchored simulation and the new reference."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from check_source_upright_mug import tilt, verify_file_pin


def render(run):
    root=run.parent
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    fig,axs=plt.subplots(1,2,figsize=(10.8,4),sharey=True)
    inputs=json.loads((root/'mug_annotated_windows_original_v1/intent.json').read_bytes())['inputs']
    qa=json.loads(verify_file_pin(inputs['binding']).read_bytes())['mug_free_qpos']
    for ax,row in zip(axs,json.loads((run/'report.json').read_bytes())['rows'],strict=True):
        c=row['clip'];item=json.loads(verify_file_pin(row['source']).read_bytes())['rows'][0]
        record=json.loads((root/f'mug_strict_extend_{c}_v1/report.json').read_bytes())
        with np.load(verify_file_pin(item['reference'])) as old,np.load(verify_file_pin(row['reference'])) as new,np.load(verify_file_pin(record['trace'])) as tr:
            t=old['t'];times=(old['source_time']-old['source_time'][0])*1e-9
            ax.axhspan(90,180,color='#fff0e6');ax.axhline(90,ls='--',c='#777',lw=1)
            ax.plot(t,tilt(old['body'][:,:3,:3]),c='#b34528',label='旧参考')
            ax.plot(t[1:],tilt(Rotation.from_quat(tr['qpos'][:,qa+np.array([4,5,6,3])]).as_matrix()),c='#975393',label='旧仿真实测')
            ax.plot(new['t'],tilt(new['body'][:,:3,:3]),lw=2.5,c='#18746a',label='修正参考')
            ax.scatter(times,tilt(old['source_obj'][:,:3,:3]),s=8,c='#213e55',label='原片标注',zorder=3)
        ax.set(ylim=(0,180),xlabel='参考时间（秒）',title=f'HOT3D {c:06d}');ax.grid(alpha=.15)
    axs[0].set_ylabel('杯口相对竖直向上的夹角（度）')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False)
    fig.suptitle('杯口方向核查　·　阴影区为杯口朝下，源世界以 +Z 为上')
    fig.tight_layout(rect=(0,.10,1,.93));fig.savefig(run/'cup_tilt.png',dpi=155);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);render(p.parse_args().run.resolve())
