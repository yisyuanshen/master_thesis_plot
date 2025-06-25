from matplotlib.patches import Patch
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
loader = DataLoader(sim=True)

robot_file_paths = 'exp_data/sim/sim_force_G.csv'
sim_force_file_paths = 'exp_data/sim/sim_force_G_force.csv'

start_idx = 3500
end_idx = 15500
loader.trigger_idx = None

loader.load_robot_data(robot_file_paths, start_idx=start_idx, end_idx=end_idx)
loader.load_sim_force_data(sim_force_file_paths, start_idx=start_idx, end_idx=end_idx)

# Data Process
loader.cmd_force_z -= 5.48
loader.state_force_z -= (0.68*9.81-5.48)

# Time
sample_rate = 1000  # Hz, change if different
time_sim = np.arange(loader.df_sim_force.shape[0]) / sample_rate
time_robot = np.arange(loader.df_robot.shape[0]) / sample_rate

# Plot Z
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
colors = ['C1', "#3C3C3C", 'C0']

linewidth = 1.5

# Force
ax = axs[0, 0]
ax.plot(time_sim, -loader.sim_force_z[0], label=r'Measured GRF (Sim)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_z[0], label=r'Estimated GRF (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.plot(time_robot, -loader.cmd_force_z[0], label=r'Desired GRF (Command)', color=colors[2], linestyle='--', linewidth=linewidth)
ax.set_title(r'\textbf{Vertical GRF on Left Front Module}', fontsize=18)
ax.set_ylim([50, 90])
ax.set_yticks(np.arange(50, 91, 10))

ax = axs[0, 1]
ax.plot(time_sim, -loader.sim_force_z[1], label=r'Measured GRF (Sim)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_z[1], label=r'Estimated GRF (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.plot(time_robot, -loader.cmd_force_z[1], label=r'Desired GRF (Command)', color=colors[2], linestyle='--', linewidth=linewidth)
ax.set_title(r'\textbf{Vertical GRF on Right Front Module}', fontsize=18)
ax.set_ylim([20, 60])
ax.set_yticks(np.arange(20, 61, 10))

ax = axs[1, 0]
ax.plot(time_sim, -loader.sim_force_z[3], label=r'Measured GRF (Sim)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_z[3], label=r'Estimated GRF (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.plot(time_robot, -loader.cmd_force_z[3], label=r'Desired GRF (Command)', color=colors[2], linestyle='--', linewidth=linewidth)
ax.set_title(r'\textbf{Vertical GRF on Left Hind Module}', fontsize=18)
ax.set_ylim([20, 60])
ax.set_yticks(np.arange(20, 61, 10))

ax = axs[1, 1]
ax.plot(time_sim, -loader.sim_force_z[2], label=r'Measured GRF (Sim)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_z[2], label=r'Estimated GRF (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.plot(time_robot, -loader.cmd_force_z[2], label=r'Desired GRF (Command)', color=colors[2], linestyle='--', linewidth=linewidth)
ax.set_title(r'\textbf{Vertical GRF on Right Hind Module}', fontsize=18)
ax.set_ylim([50, 90])
ax.set_yticks(np.arange(50, 91, 10))

# Format
for i in range(2):
    for j in range(2):
        axs[i, j].set_xlabel(r'\textbf{Time (s)}', fontsize=16)
        axs[i, j].set_ylabel(r'\textbf{Force (N)}', fontsize=16)
        axs[i, j].tick_params(axis='both', labelsize=16)
        # axs[i, j].legend(loc='upper right', fontsize=18)
        axs[i, j].set_xticks(np.arange(0, 12.1, 2))
        axs[i, j].grid(True)

plt.tight_layout(rect=[0, 0.07, 1, 1])

plt.subplots_adjust(wspace=0.25, hspace=0.5)

lines = [axs[0, 0].lines[0], axs[0, 0].lines[1], axs[0, 0].lines[2]]

labels = [line.get_label() for line in lines]

fig.legend(lines, labels, loc='lower center', fontsize=14, ncol=5, frameon=True, bbox_to_anchor=(0.5, 0))

# save
# plt.savefig('.pdf', format='pdf', bbox_inches='tight')

plt.show()


# Plot X
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
colors = ['C1', "#3C3C3C", 'C0']

linewidth = 1.5

# Force
ax = axs[0, 0]
ax.plot(time_sim, loader.sim_force_x[0], label=r'Measured GRF (Sim)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_x[0], label=r'Estimated GRF (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.plot(time_robot, loader.cmd_force_x[0], label=r'Desired GRF (Command)', color=colors[2], linestyle='--', linewidth=linewidth)
ax.set_title(r'\textbf{Horizontal GRF on Left Front Module}', fontsize=18)
ax.set_ylim([-40, 40])
ax.set_yticks(np.arange(-40, 41, 20))

ax = axs[0, 1]
ax.plot(time_sim, loader.sim_force_x[1], label=r'Measured GRF (Sim)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_x[1], label=r'Estimated GRF (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.plot(time_robot, -loader.cmd_force_x[1], label=r'Desired GRF (Command)', color=colors[2], linestyle='--', linewidth=linewidth)
ax.set_title(r'\textbf{Horizontal GRF on Right Front Module}', fontsize=18)
ax.set_ylim([-40, 40])
ax.set_yticks(np.arange(-40, 41, 20))

ax = axs[1, 0]
ax.plot(time_sim, loader.sim_force_x[3], label=r'Measured GRF (Sim)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_x[3], label=r'Estimated GRF (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.plot(time_robot, loader.cmd_force_x[3], label=r'Desired GRF (Command)', color=colors[2], linestyle='--', linewidth=linewidth)
ax.set_title(r'\textbf{Horizontal GRF on Left Hind Module}', fontsize=18)
ax.set_ylim([-40, 40])
ax.set_yticks(np.arange(-40, 41, 20))

ax = axs[1, 1]
ax.plot(time_sim, loader.sim_force_x[2], label=r'Measured GRF (Sim)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_x[2], label=r'Estimated GRF (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.plot(time_robot, -loader.cmd_force_x[2], label=r'Desired GRF (Command)', color=colors[2], linestyle='--', linewidth=linewidth)
ax.set_title(r'\textbf{Horizontal GRF on Right Hind Module}', fontsize=18)
ax.set_ylim([-40, 40])
ax.set_yticks(np.arange(-40, 41, 20))

# Format
for i in range(2):
    for j in range(2):
        axs[i, j].set_xlabel(r'\textbf{Time (s)}', fontsize=16)
        axs[i, j].set_ylabel(r'\textbf{Force (N)}', fontsize=16)
        axs[i, j].tick_params(axis='both', labelsize=16)
        # axs[i, j].legend(loc='upper right', fontsize=18)
        axs[i, j].set_xticks(np.arange(0, 12.1, 2))
        axs[i, j].grid(True)

plt.tight_layout(rect=[0, 0.07, 1, 1])

plt.subplots_adjust(wspace=0.25, hspace=0.5)

lines = [axs[0, 0].lines[0], axs[0, 0].lines[1], axs[0, 0].lines[2]]

labels = [line.get_label() for line in lines]

fig.legend(lines, labels, loc='lower center', fontsize=14, ncol=5, frameon=True, bbox_to_anchor=(0.5, 0))

# save
# plt.savefig('.pdf', format='pdf', bbox_inches='tight')

plt.show()


# RMSE
from sklearn.metrics import mean_squared_error

def compute_rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

# state
rmse_results = {'X': [], 'Z': []}

for i in range(4):
    rmse_z = compute_rmse(loader.cmd_force_z[i], loader.sim_force_z[i])
    rmse_results['Z'].append(rmse_z)

    rmse_x = compute_rmse(loader.cmd_force_x[i], loader.sim_force_x[i] if i in [0, 3] else -loader.sim_force_x[i])
    rmse_results['X'].append(rmse_x)

print('\ncmd:')
module_names = ['LF', 'RF', 'RH', 'LH']
for i, name in enumerate(module_names):
    print(f"[{name}] RMSE Z: {rmse_results['Z'][i]:.3f} N | RMSE X: {rmse_results['X'][i]:.3f} N")

print(f'\nZ = {np.average(rmse_results['Z']):.2f}, X = {np.average(rmse_results['X']):.2f}')

# state
rmse_results = {'X': [], 'Z': []}

for i in range(4):
    rmse_z = compute_rmse(loader.state_force_z[i], -loader.sim_force_z[i])
    rmse_results['Z'].append(rmse_z)

    rmse_x = compute_rmse(loader.state_force_x[i], loader.sim_force_x[i])
    rmse_results['X'].append(rmse_x)

print('\nstate:')
module_names = ['LF', 'RF', 'RH', 'LH']
for i, name in enumerate(module_names):
    print(f"[{name}] RMSE Z: {rmse_results['Z'][i]:.3f} N | RMSE X: {rmse_results['X'][i]:.3f} N")

print(f'\nZ = {np.average(rmse_results['Z']):.2f}, X = {np.average(rmse_results['X']):.2f}')
