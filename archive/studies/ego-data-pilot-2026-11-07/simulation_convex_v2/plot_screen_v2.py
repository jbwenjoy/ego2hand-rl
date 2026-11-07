"""Scientific plot from saved screening trajectories; no simulator imports."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent

def main():
    report = json.loads((ROOT/'cpu_screen_v2/result.json').read_text())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for col, tau in enumerate((.02, .005, .0025)):
        for row in report['cases']:
            if row['timeconst_s'] != tau or row['test'] != 'drop':
                continue
            a = np.load(row['trajectory']['path'], allow_pickle=False)
            label = f"dt {row['dt_s']*1000:g} ms"
            axes[0, col].plot(a['tick_time_s'], 1000*a['com_world_m'][:, 2], label=label)
            axes[1, col].plot(a['tick_time_s'], 1000*a['penetration_m'])
        axes[0, col].set_title(f'Contact time constant {tau*1000:g} ms')
        axes[0, col].legend(fontsize=9)
        axes[1, col].axhline(2, color='firebrick', linestyle='--', linewidth=1)
        for ax in axes[:, col]:
            ax.set_xlim(.1, .3)
            ax.grid(alpha=.2)
        axes[1, col].set_ylim(-.2, 10.7)
        axes[1, col].set_xlabel('Simulation time (s), impact window')
    axes[0, 0].set_ylabel('COM height (mm)')
    axes[1, 0].set_ylabel('Generated-contact penetration (mm)')
    axes[1, 2].text(.297, 2.2, '2 mm engineering limit', ha='right', color='firebrick', fontsize=9)
    fig.suptitle('Actual convex-phone CPU screening: fixed geometry, mass 0.2 kg, friction 0.6\nAssumed parameters; no hand, retargeting, or real-parameter calibration', fontsize=13)
    output = ROOT/'cpu_screen_drop_comparison_v2.png'
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output)

if __name__ == '__main__':
    main()
