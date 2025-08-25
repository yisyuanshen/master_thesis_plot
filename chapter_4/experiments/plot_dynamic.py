#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Integrated 2x4 plotting (Left Hind only), style aligned with plot_static.py.
#
# Layout:
# - 4 rows: h15, h18, h21, h24 (top to bottom)
# - 2 cols: Sim (left), Real (right)
# - Each subplot draws three curves: Measured, Estimated(State), Desired(Command)
# - State segmentation shading follows loader.state_rim for the Left Hind module.
#
# Notes:
# - This script assumes a DataLoader module compatible with the provided per-hXX scripts.
# - Start/End indices and offsets match those per-file scripts when possible.
# - Figure legend is placed at the bottom, shared by all subplots.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Global style (refer: plot_static)
plt.rcParams.update({
    'text.usetex': True,  # set to False if LaTeX not available
    'font.family': 'Times New Roman',
    'font.size': 16,
    'axes.linewidth': 1.2,
    'legend.frameon': False,
})

from DataLoader import DataLoader

# Left Hind foot index in arrays
LH = 3

# State color map
state_colors = {
    1: '#FF9797',   # Upper Rim
    2: '#FFE153',   # Lower Rim
    3: "#8396FF",   # Foot Tip (Point G)
    4: '#FFE153',
    5: '#FF9797',
}

linewidth = 1.5
colors = ['C1', "#3C3C3C", 'C0']  # measured, estimated, desired
sample_rate = 1000  # Hz

# Y-axis config per row
row_cfg = {
    "h15": {"ylim": (0, 120), "yticks": np.arange(0, 121, 30)},
    "h18": {"ylim": (0, 120), "yticks": np.arange(0, 121, 30)},
    "h21": {"ylim": (0, 120), "yticks": np.arange(0, 121, 30)},
    "h24": {"ylim": (0, 120), "yticks": np.arange(0, 121, 30)},
}

# File definitions per row (sim & real)
def cfg_for(tag: str):
    if tag not in ["h15", "h18", "h21", "h24"]:
        raise ValueError(f"Unsupported tag: {tag}")
    # Real files
    real_robot = f"exp_data_final/0612_force_{tag}.csv"
    real_vicon = f"exp_data_final/0612_force_{tag}_vicon.csv"
    real = {
        "robot": real_robot,
        "vicon": real_vicon,
        "start": 2000,
        "end": 10000,
        "cmd_offset": 0.68*9.81,  # subtract on cmd
    }
    # Simulation files
    sim_robot = f"exp_data_final/sim_force_{tag}.csv"
    sim_force = f"exp_data_final/sim_force_{tag}_force.csv"
    sim = {
        "robot": sim_robot,
        "sim_force": sim_force,
        "start": 4000,
        "end": 12000,
        "cmd_offset": 5.48,                # subtract on cmd
        "state_offset": (0.68*9.81-5.48),  # subtract on state
    }
    return sim, real

def shade_states(ax, state_seq, sr=sample_rate):
    start_idx = 0
    current = int(state_seq[0])
    n = len(state_seq)
    for i in range(n):
        if int(state_seq[i]) != current or i == n - 1:
            end_idx = i if i < n - 1 else n - 1
            ax.axvspan(start_idx/sr, end_idx/sr, color=state_colors.get(current, '#DDDDDD'), alpha=0.15)
            start_idx = i
            current = int(state_seq[i])

def draw_sim(ax, tag):
    """Left column: simulation (Measured=sim, Estimated=state, Desired=command)."""
    sim_cfg, _ = cfg_for(tag)
    loader = DataLoader(sim=False)
    loader.trigger_idx = None
    loader.load_robot_data(sim_cfg["robot"], start_idx=sim_cfg["start"], end_idx=sim_cfg["end"])
    loader.load_sim_force_data(sim_cfg["sim_force"], start_idx=sim_cfg["start"], end_idx=sim_cfg["end"])

    # Offsets
    loader.cmd_force_z -= sim_cfg["cmd_offset"]
    loader.state_force_z -= sim_cfg["state_offset"]

    t_sim = np.arange(loader.df_sim_force.shape[0]) / sample_rate
    t_robot = np.arange(loader.df_robot.shape[0]) / sample_rate

    # Plot Left Hind only
    ax.plot(t_sim, -loader.sim_force_z[LH], label=r'Measured (Ground Truth)', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(t_robot, loader.state_force_z[LH], label=r'Estimated (State)', color=colors[1], linestyle=':', linewidth=linewidth)
    ax.plot(t_robot, -loader.cmd_force_z[LH], label=r'Desired (Command)', color=colors[2], linestyle='--', linewidth=linewidth)

    shade_states(ax, loader.state_rim[LH])

def draw_real(ax, tag):
    """Right column: real (Measured=vicon, Estimated=state, Desired=command)."""
    _, real_cfg = cfg_for(tag)
    loader = DataLoader(sim=False)
    loader.trigger_idx = None
    loader.load_robot_data(real_cfg["robot"], start_idx=real_cfg["start"], end_idx=real_cfg["end"])
    loader.load_vicon_data(real_cfg["vicon"], start_idx=real_cfg["start"], end_idx=real_cfg["end"])

    loader.cmd_force_z -= real_cfg["cmd_offset"]

    t_vicon = np.arange(loader.df_vicon.shape[0]) / sample_rate
    t_robot = np.arange(loader.df_robot.shape[0]) / sample_rate

    ax.plot(t_vicon, -loader.vicon_force_z[LH], label=r'Measured (Ground Truth)', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(t_robot, loader.state_force_z[LH], label=r'Estimated (State)', color=colors[1], linestyle=':', linewidth=linewidth)
    ax.plot(t_robot, -loader.cmd_force_z[LH], label=r'Desired (Command)', color=colors[2], linestyle='--', linewidth=linewidth)

    shade_states(ax, loader.state_rim[LH])

def main():
    tags = ["h15", "h18", "h21", "h24"]
    fig, axs = plt.subplots(2, len(tags), figsize=(16, 6))  # 4x2

    # Column titles
    axs[0, 0].set_title(r'\textbf{Sim}', fontsize=16, pad=8)
    axs[0, 1].set_title(r'\textbf{Real}', fontsize=16, pad=8)

    legend_handles = None

    for i, tag in enumerate(tags):
        # Left: Sim
        draw_sim(axs[0, i], tag)
        # Right: Real
        draw_real(axs[1, i], tag)

        # Row labeling on the left y-axis
        # axs[i, 0].text(-0.07, 0.5, tag.upper(), transform=axs[i, 0].transAxes, rotation=90,
        #                va='center', ha='right', fontsize=14, fontweight='bold')

        # Axis formatting aligned with plot_static
        for j in range(2):
            ax = axs[j, i]
            ax.grid(True, alpha=0.6)
            ax.set_xlabel(r'\textbf{Time (s)}', fontsize=16)
            ax.set_ylabel(r'\textbf{Force (N)}', fontsize=16)
            ax.set_xticks(np.arange(0, 8.1, 2))

            # Y per row
            ycfg = row_cfg[tag]
            ax.set_ylim(*ycfg["ylim"])
            ax.set_yticks(ycfg["yticks"])

            ax.tick_params(axis='both', labelsize=16)

            if legend_handles is None:
                legend_handles = [ax.lines[0], ax.lines[1], ax.lines[2]]  # measured, estimated, desired

        # Titles per subplot
        if tag == "h15":
            axs[0, i].set_title(fr'\textbf{{Simulation (H=0.15m)}}', fontsize=18, pad=8)
            axs[1, i].set_title(fr'\textbf{{Real Robot (H=0.15m)}}', fontsize=18, pad=8)
        if tag == "h18":
            axs[0, i].set_title(fr'\textbf{{Simulation (H=0.18m)}}', fontsize=18, pad=8)
            axs[1, i].set_title(fr'\textbf{{Real Robot (H=0.18m)}}', fontsize=18, pad=8)
        if tag == "h21":
            axs[0, i].set_title(fr'\textbf{{Simulation (H=0.21m)}}', fontsize=18, pad=8)
            axs[1, i].set_title(fr'\textbf{{Real Robot (H=0.21m)}}', fontsize=18, pad=8)
        if tag == "h24":
            axs[0, i].set_title(fr'\textbf{{Simulation (H=0.24m)}}', fontsize=18, pad=8)
            axs[1, i].set_title(fr'\textbf{{Real Robot (H=0.24m)}}', fontsize=18, pad=8)

    # plt.tight_layout(rect=[0, 0.09, 1, 1])
    # plt.subplots_adjust(wspace=0.3, hspace=1)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.28, wspace=0.35, hspace=1)
    fig.suptitle(r'\textbf{Vertical Force Control and Estimation in Dynamic Locomotion}', fontsize=20, y=0.98)

    # Bottom legend (shared)
    # labels = [h.get_label() for h in legend_handles]
    lines = [legend_handles[0], Patch(facecolor=state_colors[1], edgecolor='none', alpha=0.3, label='Upper Rim'),
             legend_handles[1], Patch(facecolor=state_colors[2], edgecolor='none', alpha=0.3, label='Lower Rim'),
             legend_handles[2], Patch(facecolor=state_colors[3], edgecolor='none', alpha=0.3, label='Foot Tip (Point G)')]

    labels = [line.get_label() for line in lines]
    fig.legend(lines, labels, loc='lower center', ncol=3, fontsize=16, frameon=True, bbox_to_anchor=(0.5, 0.02))

    # Save outputs
    out_png = "LH_2x4_SimReal_h15_h24.png"
    out_pdf = "force_control_dynamic.pdf"
    # plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out_png} | {out_pdf}")

if __name__ == "__main__":
    main()
