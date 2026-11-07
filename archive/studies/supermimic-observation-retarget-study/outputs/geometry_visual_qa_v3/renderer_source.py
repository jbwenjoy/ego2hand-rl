"""Render existing geometric IK artifacts only; never optimize or step physics."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def render(fitted, output):
    output.mkdir(parents=True, exist_ok=False)
    with np.load(fitted / 'geometric_retarget.npz', allow_pickle=False) as z:
        arrays = {key: z[key].copy() for key in ('time_s', 'fk_hand_points_w', 'morphology_normalized_target_points_w')}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(15, 5.5))
    frames = [0, len(arrays['time_s']) // 2, len(arrays['time_s']) - 1]
    points = np.r_[arrays['fk_hand_points_w'][frames].reshape(-1, 3), arrays['morphology_normalized_target_points_w'][frames].reshape(-1, 3)]
    center = (points.min(0) + points.max(0)) / 2
    radius = np.ptp(points, axis=0).max() * .55
    for column, frame in enumerate(frames):
        ax = fig.add_axes([.035 + .325 * column, .19, .24, .60], projection='3d')
        for key, color, style, label in (('fk_hand_points_w', '#1764ab', '-', 'Robot FK'),
                                        ('morphology_normalized_target_points_w', '#db761b', '--', 'Normalized target')):
            for finger in range(5):
                index = [0] + list(range(1 + 4 * finger, 5 + 4 * finger))
                p = arrays[key][frame, index]
                ax.plot(*p.T, color=color, linestyle=style, marker='o', markersize=2, label=label if finger == 0 else None)
        ax.set(xlim=(center[0] - radius, center[0] + radius), ylim=(center[1] - radius, center[1] + radius), zlim=(center[2] - radius, center[2] + radius),
               xlabel='x (m)', ylabel='y (m)', zlabel='z (m)', title=f"Frame {frame} | {arrays['time_s'][frame]:.2f}s")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):axis.set_major_locator(MaxNLocator(4))
        ax.set_box_aspect((1, 1, 1));ax.view_init(elev=25, azim=-45)
        if column == 0:ax.legend(loc='upper left', fontsize=8, bbox_to_anchor=(0, 1.10))
    fig.suptitle('Cached-landmark geometric IK only | nominal MJB palm anchor\nDashed targets use robot bone lengths; contact, dynamics and real-ego qualification remain open', y=.98)
    for extension in ('png', 'svg'):fig.savefig(output / ('geometry.' + extension), dpi=180, bbox_inches='tight', pad_inches=.25)
    plt.close(fig)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = {'scope': 'Artifact-only render; no new IK or physical run', 'source_npz_sha256': sha(fitted / 'geometric_retarget.npz'),
              'renderer_sha256': sha(Path(__file__)), 'frames': frames, 'figure_sha256': {ext: sha(output / ('geometry.' + ext)) for ext in ('png', 'svg')},
              'visual_review': 'pending'}
    (output / 'render.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fitted', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.fitted.resolve(strict=True), args.output.resolve())
