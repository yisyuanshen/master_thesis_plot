#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2x3 Grid — LF only (command / estimate / measured)
Rows:   G | L | U
Cols:   Fx | Fy      (Fy falls back to Z if Y not present)
Style:  follow plot_all.py (Times, legend at bottom, gridlines, etc.)
"""

import numpy as np
import matplotlib.pyplot as plt

# If your environment has LaTeX, you can set text.usetex=True for identical look
plt.rcParams.update({
    'text.usetex': True,            # set True if LaTeX is installed
    'font.family': 'Times New Roman',
    'font.size': 16,
    'axes.linewidth': 1.2,
    'legend.frameon': True,
})

from DataLoader import DataLoader

SAMPLE_RATE = 1000
LF_IDX = 0  # Left-Front

# ---- dataset definitions (from your G / L / U scripts) ----
datasets = {
    'G': dict(robot='exp_data_final/sim_force_G.csv',
              force='exp_data_final/sim_force_G_force.csv',
              start=3500, end=15500, trim_edges=0),
    'L': dict(robot='exp_data_final/sim_force_L.csv',
              force='exp_data_final/sim_force_L_force.csv',
              start=3500, end=15500, trim_edges=0),
    'U': dict(robot='exp_data_final/sim_force_U.csv',
              force='exp_data_final/sim_force_U_force.csv',
              start=3000, end=16000, trim_edges=500),  # ±500 samples (as in your U script)
}

def load_case(robot, force, start, end, trim_edges=0):
    loader = DataLoader(sim=True)
    loader.trigger_idx = None
    loader.load_robot_data(robot, start_idx=start, end_idx=end)
    loader.load_sim_force_data(force, start_idx=start, end_idx=end)
    # Z-axis offsets (consistent with G/L/U scripts)
    loader.cmd_force_z -= 5.48
    loader.state_force_z -= (0.68*9.81 - 5.48)

    if trim_edges > 0:
        s, e = trim_edges, -trim_edges
        for name in ['sim_force_x','sim_force_z','cmd_force_x','cmd_force_z','state_force_x','state_force_z']:
            arr = getattr(loader, name)
            setattr(loader, name, arr[:, s:e])
        time_sim   = np.arange(loader.sim_force_x.shape[1])   / SAMPLE_RATE
        time_robot = np.arange(loader.state_force_x.shape[1]) / SAMPLE_RATE
    else:
        time_sim   = np.arange(loader.df_sim_force.shape[0]) / SAMPLE_RATE
        time_robot = np.arange(loader.df_robot.shape[0])     / SAMPLE_RATE

    return loader, time_sim, time_robot

def get_triplet(loader, comp: str):
    """
    Return (measured, estimated, commanded) for component at LF index.
    - comp in {'x', 'y'}
    - If 'y' not available, fallback to 'z' with sign conventions:
        measured = -sim_force_z, estimated = state_force_z, commanded = -cmd_force_z
      (matches original Z plotting behavior)
    - For true Y: measured = -sim_force_y (per plot_all), estimated = state_force_y, commanded = cmd_force_y
    - For X: measured/state/cmd used as is (LF needs no sign flip)
    """
    if comp == 'x':
        return loader.sim_force_x[LF_IDX], loader.state_force_x[LF_IDX], loader.cmd_force_x[LF_IDX]

    if comp == 'y':
        has_y = all(hasattr(loader, f'{k}_force_y') for k in ['sim','state','cmd'])
        if has_y:
            meas = -getattr(loader, 'sim_force_y')[LF_IDX]
            est  = getattr(loader, 'state_force_y')[LF_IDX]
            cmd  = getattr(loader, 'cmd_force_y')[LF_IDX]
        else:
            meas = -loader.sim_force_z[LF_IDX]
            est  =  loader.state_force_z[LF_IDX]
            cmd  = -loader.cmd_force_z[LF_IDX]
        return meas, est, cmd

    raise ValueError("comp must be 'x' or 'y'")

def main():
    fig, axs = plt.subplots(2, 3, figsize=(16, 5), sharex=False, sharey=False)
    # fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.18, wspace=0.30, hspace=0.90)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.28, wspace=0.35, hspace=1.25)
    fig.suptitle(r'\textbf{Force Control and Estimation in Static Simulation}', fontsize=20, y=0.98)

    row_keys = ['G','L','U']
    col_names = ['Fx','Fy']

    colors = dict(meas='C1', est='#3C3C3C', cmd='C0')
    styles = dict(meas='-', est=':', cmd='--')
    lw = 1.6

    legend_handles = None

    for i, k in enumerate(row_keys):
        loader, t_sim, t_robot = load_case(**datasets[k])
        for j, comp in enumerate(['x','y']):
            ax = axs[j, i]
            meas, est, cmd = get_triplet(loader, comp)
            ax.plot(t_sim,   meas, label='Measured (Ground Truth)',    color=colors['meas'], linestyle=styles['meas'], linewidth=lw)
            ax.plot(t_robot, est,  label='Estimated (State)', color=colors['est'],  linestyle=styles['est'],  linewidth=lw)
            ax.plot(t_robot, cmd,  label='Desired (Command)', color=colors['cmd'],  linestyle=styles['cmd'],  linewidth=lw)

            ax.grid(True, alpha=0.6)
            ax.set_xlabel(r'\textbf{Time (s)}' if plt.rcParams['text.usetex'] else 'Time (s)', fontsize=16)
            ax.set_ylabel(r'\textbf{Force (N)}' if plt.rcParams['text.usetex'] else 'Force (N)', fontsize=16)
            # ax.set_title(f'{k} — {col_names[j]}', pad=8, fontsize=18)
            ax.set_xticks([0,3,6,9,12])
            if comp == 'x':
                ax.set_ylim(-44,44)
                ax.set_yticks([-40,-20,0,20,40])
                if i == 0:
                    ax.set_title(r'$F_x$ \textbf{(Foot Tip)}', pad=8, fontsize=18)
                elif i == 1:
                    ax.set_title(r'$F_x$ \textbf{(Lower Rim)}', pad=8, fontsize=18)
                elif i == 2:
                    ax.set_title(r'$F_x$ \textbf{(Upper Rim)}', pad=8, fontsize=18)
            else:
                ax.set_ylim(48,92)
                ax.set_yticks([50,60,70,80,90])
                if i == 0:
                    ax.set_title(r'$F_y$ \textbf{(Foot Tip)}', pad=8, fontsize=18)
                elif i == 1:
                    ax.set_title(r'$F_y$ \textbf{(Lower Rim)}', pad=8, fontsize=18)
                elif i == 2:
                    ax.set_title(r'$F_y$ \textbf{(Upper Rim)}', pad=8, fontsize=18)                

            # Aim for ~5 x-ticks
            tmax = max(float(t_sim[-1]) if len(t_sim) else 0.0,
                       float(t_robot[-1]) if len(t_robot) else 0.0)
            if tmax > 0:
                nticks = 5
                step = max(0.5, round(tmax / (nticks - 1), 2))
                ax.set_xticks(np.arange(0, tmax + 1e-9, step))

            if legend_handles is None:
                legend_handles = [ax.lines[0], ax.lines[1], ax.lines[2]]

    labels = [h.get_label() for h in legend_handles]
    fig.legend(legend_handles, labels, loc='lower center', ncol=3, fontsize=16, frameon=True, bbox_to_anchor=(0.5, 0.02))

    # plt.savefig('GLU_FxFy_LF_2x3.png', dpi=200, bbox_inches='tight')
    plt.savefig('force_control_static.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
