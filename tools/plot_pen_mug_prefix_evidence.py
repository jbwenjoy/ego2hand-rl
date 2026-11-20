"""Render the paired pen-policy and HOT3D mug-prefix comparisons from saved reports."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
from matplotlib import pyplot as plt
import numpy as np

from supermimic_research.artifacts import verify_file_pin


ROOT = Path(__file__).resolve().parents[1] / 'runs'
PEN_ROWS = [('base', '从头执行策略'), ('nominal', '8 帧名义动作'), ('action_only', '8 帧动作搜索'),
            ('joint', '8 帧动作与初态联合搜索'), ('joint_init_only', '仅用联合搜索初态')]
MUG_ROWS = [('nominal', '名义物理参数'), ('mass_085', '质量与惯量 × 0.85'),
            ('mass_115', '质量与惯量 × 1.15'), ('sliding_080', '滑动摩擦 × 0.80'),
            ('sliding_120', '滑动摩擦 × 1.20')]
INK, MUTED, RULE = '#142e3d', '#435c6b', '#ccd8dd'
BLUE, GREEN = '#427aab', '#007d71'
SLATE, ORANGE = '#69798f', '#b95820'


def read(path):
    return json.loads(Path(path).read_bytes())


def select_font():
    families = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ('Microsoft YaHei', 'Noto Sans CJK SC', 'Source Han Sans CN', 'SimHei'):
        if candidate in families:
            return candidate
    raise RuntimeError('Install a Simplified Chinese font to render the figure.')


def pen_data(report):
    intent = read(verify_file_pin(report['intent']))
    selection = read(verify_file_pin(intent['selection']))['arms']
    for field in ('initial_hand_q', 'raw_prefix', 'pd_sequence'):
        np.testing.assert_array_equal(selection['nominal'][field], selection['joint'][field])
    if report['clip'] != 'clip_003' or intent['repetitions'] != 3 or intent['worlds'] != 1:
        raise ValueError('Expected the three W1 repetitions on clip_003')
    rows = []
    for arm, _ in PEN_ROWS:
        group = report['arms'][arm]
        cases = sorted(group['cases'], key=lambda c: c['rep'])
        if [c['rep'] for c in cases] != [0,1,2] or group['planned_prefix_controls'] != (
                0 if arm in ('base', 'joint_init_only') else 8):
            raise ValueError(f'Unexpected W1 repetition or prefix coverage for {arm}')
        rows.append([c['valid_s'] for c in cases])
    return np.array(rows), intent['selection']['sha256']


def mug_data(report):
    intent = read(verify_file_pin(report['intent']))
    if intent['common_controls'] != 24 or intent['full_reference_s'] != 2:
        raise ValueError('Expected the registered 24-control HOT3D observation window')
    by_case = {(row['case'], row['mode']): row for row in report['rows']}
    values, censored = [], []
    for case, _ in MUG_ROWS:
        pair = [by_case[case, mode] for mode in ('replay', 'support')]
        values.append([r['valid_s'] for r in pair])
        censored.append([r['right_censored'] for r in pair])
    values, censored = np.array(values), np.array(censored, dtype=bool)
    if not np.array_equal(np.isclose(values, .48), censored):
        raise ValueError('Censor flags disagree with the common observation window')
    return values, censored


def panel(figure, bottom, rows, durations, text, colors, names, ticks, upper, censor=None):
    """Chart and numeric columns share the data-space y coordinates."""
    chart = figure.add_axes([.258, bottom, .379, .222])
    table = figure.add_axes([.658, bottom, .311, .222])
    y = np.arange(len(rows))
    for ax in (chart, table):
        ax.set_ylim(len(rows)-.5, -.5)
        ax.set_facecolor('none')
        ax.spines[:].set_visible(False)
    chart.set_yticks(y, [s for _, s in rows], fontsize=11, color=INK)
    chart.tick_params(axis='y', length=0, pad=12)
    chart.set_xlim(0, upper)
    chart.set_xticks(ticks, [f'{t:.2f}' for t in ticks], fontsize=9, color=MUTED)
    chart.tick_params(axis='x', length=0, pad=5)
    chart.grid(axis='x', color=RULE, linewidth=.7)
    chart.set_axisbelow(True)
    chart.set_xlabel('有效前缀时长（秒）', fontsize=10, color=MUTED, labelpad=5)
    if censor is not None:
        chart.axvline(.48, ls=(0, (2,2)), color=MUTED, lw=1, zorder=2)
    table.set_axis_off()
    table.set_xlim(0, 1)
    for col, (color, name, xpos) in enumerate(zip(colors, names, [.23, .76])):
        table.text(xpos, 1.065, name, ha='center', va='bottom', color=color, weight='bold',
                   fontsize=10.5, transform=table.transAxes)
        offset = (-.155, .155)[col]
        chart.barh(y+offset, durations[:,col], height=.20, color=color, zorder=3)
        for row in y:
            table.text(xpos, row, text[row][col], va='center', ha='center', color=color,
                       fontsize=10.6, weight='medium')
            if censor is not None and censor[row,col]:
                chart.plot(durations[row,col], row+offset, marker='>', color=color, ms=7,
                           clip_on=False, zorder=4)
    for j in np.arange(len(rows)-1)+.5:
        table.axhline(j, lw=.5, color=RULE)


def render(original, balanced, mug, out):
    a, selection_a = pen_data(read(original))
    b, selection_b = pen_data(read(balanced))
    if selection_a != selection_b:
        raise ValueError('Pen policies must use the same frozen search prefixes')
    cup, censored = mug_data(read(mug))
    plt.rcParams.update({'font.family':select_font(), 'axes.unicode_minus':False, 'figure.facecolor':'white'})
    fig = plt.figure(figsize=(12.4,9.0))
    def label(x, y, value, **kwargs):
        fig.text(x, y, value, va='top', color=kwargs.pop('color', INK), **kwargs)

    label(.043,.965,'转笔启动与 HOT3D 杯子重定向',fontsize=23,weight='bold')
    label(.044,.910,'有效前缀取自连续通过验收的控制步，每一步为 0.02 秒。',fontsize=11.7,color=MUTED)
    fig.add_artist(plt.Line2D([.043,.958],[.862,.862],transform=fig.transFigure,color=RULE,lw=1.0))
    label(.044,.841,'转笔  /  同一条受控参考动作，比较两个冻结策略',fontsize=15.5,weight='bold')
    label(.044,.802,'clip_003，前缀结束后由对应策略接管。条形为中位数，右侧依次列出三次重复的结果（秒）。',
          fontsize=10.5,color=MUTED)
    label(.044,.766,'本次联合搜索选中了名义前缀，两行使用相同的初态和动作。',fontsize=9.5,color=MUTED)
    pen_labels = [[ ' / '.join(f'{v:.2f}' for v in data[i]) for data in (a,b)] for i in range(len(a))]
    panel(fig,.501,PEN_ROWS,np.column_stack((np.median(a,axis=1),np.median(b,axis=1))),
          pen_labels,(BLUE,GREEN),('原始策略','两轨迹均衡策略'),[0,.25,.5,.75,1.],1.02)
    fig.add_artist(plt.Line2D([.043,.958],[.423,.423],transform=fig.transFigure,color=RULE,lw=1.0))
    label(.044,.403,'杯子  /  HOT3D 第一视角参考，比较五种仿真物理先验',fontsize=15.5,weight='bold')
    label(.044,.364,'同一初态，在每种先验下比较两套冻结动作。统一观察 24 个控制步（0.48 秒），完整参考长 2 秒。',
          fontsize=10.5,color=MUTED)
    cup_labels = [[('≥ ' if censored[i,j] else '') + f'{v:.2f}' for j,v in enumerate(row)]
                  for i,row in enumerate(cup)]
    panel(fig,.075,MUG_ROWS,cup,cup_labels,(SLATE,ORANGE),('普通前馈','冻结搜索动作（shooting）'),
          [0,.12,.24,.36,.48],.495,censored)
    label(.044,.023,'杯子以质心误差 10 mm、姿态误差 10° 和桌面支持力 0.1 N 为上限。≥ 表示到窗口末仍通过验收（右删失）。',
          fontsize=9.6,color=MUTED)
    fig.savefig(out, dpi=180, metadata={'Title':'转笔启动与 HOT3D 杯子重定向'})
    plt.close(fig)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,default=ROOT/'pen_start_prefix_w1_v1/report.json')
    p.add_argument('--balanced',type=Path,default=ROOT/'pen_start_prefix_balanced_w1_v1/report.json')
    p.add_argument('--mug',type=Path,default=ROOT/'mug_frozen_sensitivity_v1/report.json')
    p.add_argument('--out',type=Path,default=ROOT/'pen_start_prefix_w1_v1/pen_mug_evidence.png')
    args=p.parse_args()
    render(args.original,args.balanced,args.mug,args.out)
