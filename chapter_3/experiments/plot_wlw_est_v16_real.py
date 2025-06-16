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
robot_file_paths = 'data/0612/0612_wlw_est_h16_v15_1.csv'
vicon_file_paths = 'data/0612/wlw_est_h16_v15_1.csv'

robot_file_paths = 'data/0612/0612_wlw_est_h18_v15.csv'
vicon_file_paths = 'data/0612/wlw_est_h18_v15.csv'

start_idx = 5000
end_idx = 7000
loader.trigger_idx = None

loader.load_robot_data(robot_file_paths, start_idx=start_idx, end_idx=end_idx)
loader.load_vicon_data(vicon_file_paths, start_idx=start_idx, end_idx=end_idx)

# === Force Filtering ===
loader.vicon_force_z = np.where(loader.vicon_force_z >= 0, 0, loader.vicon_force_z)
loader.state_force_z = np.where(loader.state_force_z <= 0, 0, loader.state_force_z)
loader.state_force_z = np.where(loader.vicon_force_z > -2, 0, loader.state_force_z)

loader.state_force_x = np.where(((loader.vicon_force_x < 2) & (loader.vicon_force_x > -2)), 0, loader.state_force_x)

# === Time Axis ===
sample_rate = 1000  # Hz, change if different
time_vicon = np.arange(loader.df_vicon.shape[0]) / sample_rate
time_robot = np.arange(loader.df_robot.shape[0]) / sample_rate

# === Plotting ===
fig, axs = plt.subplots(2, 2, figsize=(12, 6))
colors = ['C1', "#3C3C3C"]
linewidth = 1.5

# --- Left Forelimb - Anterior-Posterior (X) ---
ax = axs[0, 0]
ax.plot(time_vicon, -loader.vicon_force_x[0], label=r'Measured (Vicon)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_x[0], label=r'Estimated (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.set_title(r'\textbf{Horizontal Force on LF Module}', fontsize=18)
ax.set_ylim([-50, 75])
ax.set_yticks(np.arange(-50, 76, 25))

# --- Right Hindlimb - Anterior-Posterior (X) ---
ax = axs[0, 1]
ax.plot(time_vicon, -loader.vicon_force_x[3], label=r'Measured (Vicon)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_x[3], label=r'Estimated (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.set_title(r'\textbf{Horizontal Force on RH Module}', fontsize=18)
ax.set_ylim([-50, 75])
ax.set_yticks(np.arange(-50, 76, 25))

# --- Left Forelimb - Vertical (Z) ---
ax = axs[1, 0]
ax.plot(time_vicon, -loader.vicon_force_z[0], label=r'Measured (Vicon)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_z[0], label=r'Estimated (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.set_title(r'\textbf{Vertical Force on LF Module}', fontsize=18)
ax.set_ylim([-20, 220])
ax.set_yticks(np.arange(0, 201, 50))

# --- Right Hindlimb - Vertical (Z) ---
ax = axs[1, 1]
ax.plot(time_vicon, -loader.vicon_force_z[3], label=r'Measured (Vicon)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_z[3], label=r'Estimated (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.set_title(r'\textbf{Vertical Force on RH Module}', fontsize=18)
ax.set_ylim([-20, 220])
ax.set_yticks(np.arange(0, 201, 50))

# === Axis Formatting ===
for i in range(2):
    for j in range(2):
        axs[i, j].set_xlabel(r'\textbf{Time (s)}', fontsize=16)
        axs[i, j].set_ylabel(r'\textbf{Force (N)}', fontsize=16)
        axs[i, j].tick_params(axis='both', labelsize=16)
        # axs[i, j].legend(loc='upper right', fontsize=18)
        ax.set_xticks(np.arange(0, 2.1, 0.5))
        axs[i, j].grid(True)
        
plt.tight_layout(rect=[0, 0.09, 1, 1])

plt.subplots_adjust(wspace=0.25, hspace=0.7)

lines = [axs[0, 0].lines[0], axs[0, 0].lines[1]]
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='lower center', fontsize=16, ncol=2, frameon=True, bbox_to_anchor=(0.5, 0))

# === Save as vector PDF (for LaTeX or printing) ===
# plt.savefig('force_comparison_plot.pdf', format='pdf', bbox_inches='tight')

plt.show()
