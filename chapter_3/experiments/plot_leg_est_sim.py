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
loader = DataLoader(sim=True)

# Data file paths (latest selection)
robot_file_paths = 'data/sim_leg_est.csv'
sim_force_file_paths = 'data/sim_force_plate_leg.csv'

start_idx = 12000
end_idx = 24000

loader.load_robot_data(robot_file_paths, start_idx=start_idx, end_idx=end_idx)
loader.load_sim_force_data(sim_force_file_paths, start_idx=start_idx, end_idx=end_idx)

# === Force Filtering ===


# === Time Axis ===
sample_rate = 1000  # Hz, change if different
time_sim_force = np.arange(loader.df_sim_force.shape[0]) / sample_rate
time_robot = np.arange(loader.df_robot.shape[0]) / sample_rate

# === Plotting ===
fig, axs = plt.subplots(2, 1, figsize=(12, 8))
colors = ['C1', "#3C3C3C"]
linewidth = 1.5

# --- Left Forelimb - Anterior-Posterior (X) ---
ax = axs[0]
ax.plot(time_sim_force, -loader.sim_force_z[0], label=r'Measured (Sim)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_z[0], label=r'Estimated (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.set_title(r'\textbf{Vertical Force on LF Module}', fontsize=18)
# ax.set_ylim([-50, 75])
# ax.set_yticks(np.arange(-50, 76, 25))

state_colors = {
    1: '#FF9797',
    2: '#FFE153',
    3: "#8396FF",
    4: '#FFE153',
    5: '#FF9797',
    0: '#FFFFFF'
}

start_idx = 0
end_idx = 0
current_state = loader.state_rim[0][0]
for i in range(loader.df_robot.shape[0]):
    if loader.state_rim[0][i] != current_state or i == loader.df_robot.shape[0] - 1:
        end_idx = i
        ax.axvspan(start_idx/sample_rate, end_idx/sample_rate, color=state_colors[current_state], alpha=0.3)
        start_idx = i
        current_state = loader.state_rim[0][i]

# --- Right Hindlimb - Anterior-Posterior (X) ---
loader.legmodel.contact_map(loader.state_theta[0], loader.state_beta[0])
ax = axs[1]
ax.plot(time_robot, abs((-loader.sim_force_z[0]-loader.state_force_z[0])/loader.sim_force_z[0])*100, label=r'Estimated (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.set_title(r'\textbf{Horizontal Force on RH Module}', fontsize=18)
ax.set_ylim([-0, 100])
ax.set_yticks(np.arange(-0, 101, 20))


# === Axis Formatting ===
for i in range(2):
    axs[i].set_xlabel(r'\textbf{Time (s)}', fontsize=16)
    axs[i].set_ylabel(r'\textbf{Force (N)}', fontsize=16)
    axs[i].tick_params(axis='both', labelsize=16)
    # axs[i, j].legend(loc='upper right', fontsize=18)
    # ax.set_xticks(np.arange(0, 2.1, 0.5))
    axs[i].grid(True)
        
plt.tight_layout(rect=[0, 0.09, 1, 1])

plt.subplots_adjust(wspace=0.25, hspace=0.7)

lines = [axs[0].lines[0], axs[0].lines[1],
         Patch(facecolor=state_colors[1], edgecolor='none', alpha=0.3, label='Upper Rim'),
         Patch(facecolor=state_colors[2], edgecolor='none', alpha=0.3, label='Lower Rim'),
         Patch(facecolor=state_colors[3], edgecolor='none', alpha=0.3, label='Foot Tip (Point G)')]

labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='lower center', fontsize=16, ncol=2, frameon=True, bbox_to_anchor=(0.5, 0))

# === Save as vector PDF (for LaTeX or printing) ===
# plt.savefig('force_comparison_plot.pdf', format='pdf', bbox_inches='tight')

plt.show()
