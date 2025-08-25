#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2 x 3 Grid — Motor Angle Tracking (Theta/Beta) for Ramp, Sinusoidal, Step
Columns: Ramp | Sinusoidal | Step
Rows:    Theta (deg) | Beta (deg)
Style and legends follow plot_static.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Style (kept consistent with the provided scripts)
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'Times New Roman',
    'font.size': 16,
    'axes.linewidth': 1.2,
    'legend.frameon': False,
})

SAMPLE_RATE = 1000  # Hz
time_axis = lambda n: np.arange(n) / SAMPLE_RATE

# --- Helper to load a case ---
def load_case(robot_file_old, robot_file_new, start_idx, end_idx):
    df_old = pd.read_csv(robot_file_old).iloc[start_idx:end_idx, :]
    df_new = pd.read_csv(robot_file_new).iloc[start_idx:end_idx, :]

    state_theta_old = np.array([df_old['state_theta_a'], df_old['state_theta_b'], df_old['state_theta_c'], df_old['state_theta_d']])
    state_beta_old  = np.array([df_old['state_beta_a'] , df_old['state_beta_b'] , df_old['state_beta_c'] , df_old['state_beta_d']])

    state_theta_new = np.array([df_new['state_theta_a'], df_new['state_theta_b'], df_new['state_theta_c'], df_new['state_theta_d']])
    state_beta_new  = np.array([df_new['state_beta_a'] , df_new['state_beta_b'] , df_new['state_beta_c'] , df_new['state_beta_d']])

    cmd_theta = np.array([df_old['cmd_theta_a'], df_old['cmd_theta_b'], df_old['cmd_theta_c'], df_old['cmd_theta_d']])
    cmd_beta  = np.array([df_old['cmd_beta_a'] , df_old['cmd_beta_b'] , df_old['cmd_beta_c'] , df_old['cmd_beta_d']])

    t = time_axis(end_idx - start_idx)
    return dict(
        t=t,
        cmd_theta=cmd_theta,
        cmd_beta=cmd_beta,
        state_theta_old=state_theta_old,
        state_beta_old=state_beta_old,
        state_theta_new=state_theta_new,
        state_beta_new=state_beta_new,
    )

def maybe_lowpass(arr_deg, cutoff_hz=5):
    """Use DataLoader.low_pass_filter if available to match sinusoidal script behavior; otherwise return original."""
    try:
        import DataLoader as DL  # plot_motor_sinusoidal.py used this namespace
        if hasattr(DL, 'low_pass_filter'):
            return DL.low_pass_filter(arr_deg, cutoff_hz)
    except Exception:
        pass
    return arr_deg

def apply_script_specific_adjustments(case_name, data):
    """Apply the little quirks used in each original script on channel a (index 0)."""
    if case_name == 'ramp':
        # cmd_theta[0][0] = cmd_theta[0][2]; cmd_theta[0][1] = cmd_theta[0][2]
        data['cmd_theta'][0][0] = data['cmd_theta'][0][2]
        data['cmd_theta'][0][1] = data['cmd_theta'][0][2]
    elif case_name == 'sinusoidal':
        # cmd_theta filtered in the original wave script
        data['cmd_theta'][0] = np.deg2rad(maybe_lowpass(np.rad2deg(data['cmd_theta'][0]), 5))
        # also did: cmd_theta[0][0] = cmd_theta[0][2]; cmd_theta[0][1] = cmd_theta[0][2]
        data['cmd_theta'][0][0] = data['cmd_theta'][0][2]
        data['cmd_theta'][0][1] = data['cmd_theta'][0][2]
    elif case_name == 'step':
        # cmd_theta[0][0] = cmd_theta[0][1]; state_theta_old[0][1350:] = inf
        data['cmd_theta'][0][0] = data['cmd_theta'][0][1]
        data['state_theta_old'][0][1350:] = np.inf
    return data

def main():
    # --- Define the three cases using the original file names and indices ---
    cases = [
        dict(name='step',       old='exp_data_final/0320_imp_step_old.csv',  new='exp_data_final/0320_imp_step_new.csv',  start=0,   end=3000),
        dict(name='ramp',       old='exp_data_final/0320_imp_ramp_old.csv',  new='exp_data_final/0320_imp_ramp_new.csv',  start=0,   end=3000),
        dict(name='sinusoidal', old='exp_data_final/0320_imp_wave_old.csv',  new='exp_data_final/0320_imp_wave_new.csv',  start=500, end=3500),
    ]

    # Y-axis configs mirrored from each original script
    ycfg = {
        'step': {
            'theta': dict(ylim=(0, 240), yticks=np.arange(0, 241, 60)),
            'beta':  dict(ylim=(-200, 600), yticks=np.arange(-200, 601, 200)),
        },
        'ramp': {
            'theta': dict(ylim=(0, 400), yticks=np.arange(0, 401, 70)),
            'beta':  dict(ylim=(-200, 50), yticks=np.arange(-200, 51, 50)),
        },
        'sinusoidal': {
            'theta': dict(ylim=(0, 480), yticks=np.arange(0, 481, 120)),
            'beta':  dict(ylim=(-200, 1000), yticks=np.arange(-200, 1001, 200)),
        }
    }

    fig, axs = plt.subplots(2, 3, figsize=(16, 6), sharex=False, sharey=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.22, wspace=0.28, hspace=0.7)
    fig.suptitle(r'\textbf{Comparison of Theta and Beta Angles under Different Trajectories}', fontsize=20, y=0.98)

    colors = ['#3C3C3C', "C2", 'C3']  # Command, Old, New (to match originals)
    lw = 1.5

    # Column titles
    col_titles = ['Step','Ramp', 'Sinusoidal']

    legend_handles = None

    for j, c in enumerate(cases):
        data = load_case(c['old'], c['new'], c['start'], c['end'])
        data = apply_script_specific_adjustments(c['name'], data)
        t = data['t']

        # Row 0: Theta
        ax = axs[0, j]
        h0 = ax.plot(t, np.rad2deg(data['cmd_theta'][0]), label=r'Command', color=colors[0], linestyle='-', linewidth=lw)
        h1 = ax.plot(t, np.rad2deg(data['state_theta_old'][0]), label=r'State (Traditional Method)', color=colors[1], linestyle='-.', linewidth=lw)
        h2 = ax.plot(t, np.rad2deg(data['state_theta_new'][0]), label=r'State (Proposed Method)', color=colors[2], linestyle='--', linewidth=lw)
        ax.set_title(r'\textbf{$\theta$ (%s)}' % col_titles[j], pad=8, fontsize=18)
        ax.set_xlabel(r'\textbf{Time (s)}', fontsize=16)
        ax.set_ylabel(r'$\mathbf{\theta}$ \textbf{(deg)}', fontsize=16)
        ax.set_xticks(np.arange(0, 31, 5)/10)
        ax.grid(True)
        ax.set_ylim(*ycfg[c['name']]['theta']['ylim'])
        ax.set_yticks(ycfg[c['name']]['theta']['yticks'])
        ax.tick_params(axis='both', labelsize=12)

        if legend_handles is None:
            legend_handles = [h0[0], h1[0], h2[0]]

        # Row 1: Beta
        ax = axs[1, j]
        ax.plot(t, np.rad2deg(data['cmd_beta'][0]), label=r'Command', color=colors[0], linestyle='-', linewidth=lw)
        ax.plot(t, np.rad2deg(data['state_beta_old'][0]), label=r'State (Traditional Method)', color=colors[1], linestyle='-.', linewidth=lw)
        ax.plot(t, np.rad2deg(data['state_beta_new'][0]), label=r'State (Proposed Method)', color=colors[2], linestyle='--', linewidth=lw)
        ax.set_title(r'\textbf{$\beta$ (%s)}' % col_titles[j], pad=8, fontsize=18)
        ax.set_xlabel(r'\textbf{Time (s)}', fontsize=16)
        ax.set_ylabel(r'$\mathbf{\beta}$ \textbf{(deg)}', fontsize=16)
        ax.set_xticks(np.arange(0, 31, 5)/10)
        ax.grid(True)
        ax.set_ylim(*ycfg[c['name']]['beta']['ylim'])
        ax.set_yticks(ycfg[c['name']]['beta']['yticks'])
        ax.tick_params(axis='both', labelsize=16)

    labels = [h.get_label() for h in legend_handles]
    fig.legend(legend_handles, labels, loc='lower center', fontsize=16, ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.02))

    # Save
    plt.savefig('motor_angles_combined.pdf', bbox_inches='tight')
    # plt.savefig('motor_angles_combined.png', dpi=200, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
