from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

from DataLoader import DataLoader

# === Matplotlib styling for thesis-quality output ===
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'Times New Roman',
    'font.size': 14,
    'axes.linewidth': 1.2,
    'legend.frameon': False,
})

# === Data Loading ===
loader = DataLoader(sim=False)

# Data file paths (latest selection)
robot_file_paths = 'data/0612/0612_force_h24.csv'
vicon_file_paths = 'data/0612/force_h24.csv'

start_idx = 2000
end_idx = 10000
loader.trigger_idx = None

loader.load_robot_data(robot_file_paths, start_idx=start_idx, end_idx=end_idx)
loader.load_vicon_data(vicon_file_paths, start_idx=start_idx, end_idx=end_idx)

# === Force Filtering ===
loader.cmd_force_z -= 0.68*9.81

# === Time Axis ===
sample_rate = 1000  # Hz, change if different
time_vicon = np.arange(loader.df_vicon.shape[0]) / sample_rate
time_robot = np.arange(loader.df_robot.shape[0]) / sample_rate

# === Plotting ===
fig, axs = plt.subplots(2, 2, figsize=(12, 6))
colors = ['C1', "#3C3C3C", 'C0']

state_colors = {
    1: '#FF9797',
    2: '#FFE153',
    3: "#8396FF",
    4: '#FFE153',
    5: '#FF9797',
}

linewidth = 1.5

# --- Left Forelimb - Anterior-Posterior (X) ---
ax = axs[0, 0]
ax.plot(time_vicon, -loader.vicon_force_z[0], label=r'Measured (Vicon)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_z[0], label=r'Estimated (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.plot(time_robot, -loader.cmd_force_z[0], label=r'Desired (Command)', color=colors[2], linestyle='--', linewidth=linewidth)
ax.set_title(r'\textbf{Vertical Force on LF Module}', fontsize=18)

start_idx = 0
end_idx = 0
current_state = loader.state_rim[0][0]
for i in range(loader.df_robot.shape[0]):
    if loader.state_rim[0][i] != current_state or i == loader.df_robot.shape[0] - 1:
        end_idx = i
        ax.axvspan(start_idx/sample_rate, end_idx/sample_rate, color=state_colors[current_state], alpha=0.15)
        start_idx = i
        current_state = loader.state_rim[0][i]

# --- Right Hindlimb - Anterior-Posterior (X) ---
ax = axs[0, 1]
ax.plot(time_vicon, -loader.vicon_force_z[1], label=r'Measured (Vicon)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_z[1], label=r'Estimated (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.plot(time_robot, -loader.cmd_force_z[1], label=r'Desired (Command)', color=colors[2], linestyle='--', linewidth=linewidth)
ax.set_title(r'\textbf{Vertical Force on RF Module}', fontsize=18)

start_idx = 0
end_idx = 0
current_state = loader.state_rim[1][0]
for i in range(loader.df_robot.shape[0]):
    if loader.state_rim[1][i] != current_state or i == loader.df_robot.shape[0] - 1:
        end_idx = i
        ax.axvspan(start_idx/sample_rate, end_idx/sample_rate, color=state_colors[current_state], alpha=0.15)
        start_idx = i
        current_state = loader.state_rim[1][i]

# --- Left Forelimb - Vertical (Z) ---
ax = axs[1, 0]
ax.plot(time_vicon, -loader.vicon_force_z[2], label=r'Measured (Vicon)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_z[2], label=r'Estimated (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.plot(time_robot, -loader.cmd_force_z[2], label=r'Desired (Command)', color=colors[2], linestyle='--', linewidth=linewidth)
ax.set_title(r'\textbf{Vertical Force on LH Module}', fontsize=18)

start_idx = 0
end_idx = 0
current_state = loader.state_rim[2][0]
for i in range(loader.df_robot.shape[0]):
    if loader.state_rim[2][i] != current_state or i == loader.df_robot.shape[0] - 1:
        end_idx = i
        ax.axvspan(start_idx/sample_rate, end_idx/sample_rate, color=state_colors[current_state], alpha=0.15)
        start_idx = i
        current_state = loader.state_rim[2][i]

# --- Right Hindlimb - Vertical (Z) ---
ax = axs[1, 1]
ax.plot(time_vicon, -loader.vicon_force_z[3], label=r'Measured (Vicon)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_z[3], label=r'Estimated (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.plot(time_robot, -loader.cmd_force_z[3], label=r'Desired (Command)', color=colors[2], linestyle='--', linewidth=linewidth)
ax.set_title(r'\textbf{Vertical Force on RH Module}', fontsize=18)

start_idx = 0
end_idx = 0
current_state = loader.state_rim[3][0]
for i in range(loader.df_robot.shape[0]):
    if loader.state_rim[3][i] != current_state or i == loader.df_robot.shape[0] - 1:
        end_idx = i
        ax.axvspan(start_idx/sample_rate, end_idx/sample_rate, color=state_colors[current_state], alpha=0.15)
        start_idx = i
        current_state = loader.state_rim[3][i]

# === Axis Formatting ===
for i in range(2):
    for j in range(2):
        axs[i, j].set_xlabel(r'\textbf{Time (s)}', fontsize=16)
        axs[i, j].set_ylabel(r'\textbf{Force (N)}', fontsize=16)
        axs[i, j].tick_params(axis='both', labelsize=16)
        axs[i, j].set_ylim([0, 120])
        axs[i, j].set_yticks(np.arange(0, 121, 30))
        # axs[i, j].legend(loc='upper right', fontsize=18)
        ax.set_xticks(np.arange(0, 8.1, 2))
        axs[i, j].grid(True)

plt.tight_layout(rect=[0, 0.09, 1, 1])

plt.subplots_adjust(wspace=0.25, hspace=0.7)

lines = [axs[0, 0].lines[0], axs[0, 0].lines[1], axs[0, 0].lines[2],
         Patch(facecolor=state_colors[3], edgecolor='none', alpha=0.3, label='Foot Tip (Point G)')]

labels = [line.get_label() for line in lines]

fig.legend(lines, labels, loc='lower center', fontsize=14, ncol=5, frameon=True, bbox_to_anchor=(0.5, 0))

# === Save as vector PDF (for LaTeX or printing) ===
# plt.savefig('force_comparison_plot.pdf', format='pdf', bbox_inches='tight')

plt.show()
