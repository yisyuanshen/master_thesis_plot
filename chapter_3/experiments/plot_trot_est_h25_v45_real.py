import matplotlib.pyplot as plt
import numpy as np

from DataLoader import DataLoader

# Style
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'Times New Roman',
    'font.size': 14,
    'axes.linewidth': 1.2,
    'legend.frameon': False,
})

# Load Data
loader = DataLoader(sim=False)
loader.cutoff_freq = 30

robot_file_paths = 'exp_data/real/0617_trot_h25_v45_open_1.csv'
vicon_file_paths = 'exp_data/real/0617_trot_h25_v45_open_1_vicon.csv'

start_idx = 2300
end_idx = 3300
loader.trigger_idx = None

loader.load_robot_data(robot_file_paths, start_idx=start_idx, end_idx=end_idx)
loader.load_vicon_data(vicon_file_paths, start_idx=start_idx, end_idx=end_idx)

# Data Process
loader.data_process()
# loader.vicon_force_z = np.where(loader.vicon_force_z >= 0, 0, loader.vicon_force_z)
# loader.state_force_z = np.where(loader.state_force_z <= 0, 0, loader.state_force_z)
# loader.state_force_z = np.where(loader.vicon_force_z > -2, 0, loader.state_force_z)

# loader.state_force_x = np.where(loader.state_force_z == 0, 0, loader.state_force_x)

# Time
sample_rate = 1000  # Hz, change if different
time_vicon = np.arange(loader.df_vicon.shape[0]) / sample_rate
time_robot = np.arange(loader.df_robot.shape[0]) / sample_rate

# Plot
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
colors = ['C1', "#3C3C3C"]
linewidth = 1.5

# Force
ax = axs[0, 0]
ax.plot(time_vicon, -loader.vicon_force_x[1], label=r'Measured GRF (Vicon)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_x[1], label=r'Estimated GRF (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.set_title(r'\textbf{Horizontal GRF on Right Front Module}', fontsize=18)
ax.set_ylim([-30, 30])
ax.set_yticks(np.arange(-30, 31, 15))

ax = axs[0, 1]
ax.plot(time_vicon, -loader.vicon_force_x[3], label=r'Measured GRF (Vicon)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_x[3], label=r'Estimated GRF (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.set_title(r'\textbf{Horizontal GRF on Left Hind Module}', fontsize=18)
ax.set_ylim([-50, 50])
ax.set_yticks(np.arange(-50, 51, 25))

ax = axs[1, 0]
ax.plot(time_vicon, -loader.vicon_force_z[1], label=r'Measured GRF (Vicon)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_z[1], label=r'Estimated GRF (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.set_title(r'\textbf{Vertical GRF on Right Front Module}', fontsize=18)
ax.set_ylim([-25, 205])
ax.set_yticks(np.arange(0, 181, 60))

ax = axs[1, 1]
ax.plot(time_vicon, -loader.vicon_force_z[3], label=r'Measured GRF (Vicon)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_z[3], label=r'Estimated GRF (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.set_title(r'\textbf{Vertical GRF on Left Hind Module}', fontsize=18)
ax.set_ylim([-25, 205])
ax.set_yticks(np.arange(0, 181, 60))

# Format
for i in range(2):
    for j in range(2):
        axs[i, j].set_xlabel(r'\textbf{Time (s)}', fontsize=16)
        axs[i, j].set_ylabel(r'\textbf{Force (N)}', fontsize=16)
        axs[i, j].tick_params(axis='both', labelsize=16)
        # axs[i, j].legend(loc='upper right', fontsize=18)
        axs[i, j].set_xticks(np.arange(0, 101, 25)/100)
        axs[i, j].grid(True)
        
plt.tight_layout(rect=[0, 0.07, 1, 1])

plt.subplots_adjust(wspace=0.25, hspace=0.5)

lines = [axs[0, 0].lines[0], axs[0, 0].lines[1]]
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='lower center', fontsize=16, ncol=2, frameon=True, bbox_to_anchor=(0.5, 0))

# save
plt.savefig('real_trot_est_h25_v45_result.pdf', format='pdf', bbox_inches='tight')

plt.show()

loader.compute_rmse()
