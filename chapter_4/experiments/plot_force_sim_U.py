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
robot_file_paths = 'data/sim_force_U.csv'
sim_force_file_paths = 'data/sim_force_plate_U.csv'

start_idx = 3500
end_idx = 15500
loader.trigger_idx = None

loader.load_robot_data(robot_file_paths, start_idx=start_idx, end_idx=end_idx)
loader.load_sim_force_data(sim_force_file_paths, start_idx=start_idx, end_idx=end_idx)

# === Force Filtering ===

loader.cmd_force_z -= 6.41
# loader.state_force_z -= sum([loader.sim_force_z[i][1000]-loader.state_force_z[i][1000] for i in range(4)])/4
# loader.state_force_z -= 2.358586007859552

# === Time Axis ===
sample_rate = 1000  # Hz, change if different
time_sim_force = np.arange(loader.df_sim_force.shape[0]) / sample_rate
time_robot = np.arange(loader.df_robot.shape[0]) / sample_rate

# === Plotting ===
fig, axs = plt.subplots(2, 2, figsize=(12, 6))
colors = ['C1', "#3C3C3C", 'C0']

linewidth = 1.5

# --- Left Forelimb - Anterior-Posterior (X) ---
ax = axs[0, 0]
ax.plot(time_sim_force, -loader.sim_force_z[0], label=r'Measured (Sim)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_z[0], label=r'Estimated (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.plot(time_robot, -loader.cmd_force_z[0], label=r'Desired (Command)', color=colors[2], linestyle='--', linewidth=linewidth)
ax.set_title(r'\textbf{Vertical Force on LF Module}', fontsize=18)
ax.set_ylim([40, 90])
ax.set_yticks(np.arange(40, 91, 10))

# --- Right Hindlimb - Anterior-Posterior (X) ---
ax = axs[0, 1]
ax.plot(time_sim_force, -loader.sim_force_z[1], label=r'Measured (Sim)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_z[1], label=r'Estimated (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.plot(time_robot, -loader.cmd_force_z[1], label=r'Desired (Command)', color=colors[2], linestyle='--', linewidth=linewidth)
ax.set_title(r'\textbf{Vertical Force on RF Module}', fontsize=18)
ax.set_ylim([20, 70])
ax.set_yticks(np.arange(20, 71, 10))

# --- Left Forelimb - Vertical (Z) ---
ax = axs[1, 0]
ax.plot(time_sim_force, -loader.sim_force_z[3], label=r'Measured (Sim)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_z[3], label=r'Estimated (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.plot(time_robot, -loader.cmd_force_z[3], label=r'Desired (Command)', color=colors[2], linestyle='--', linewidth=linewidth)
ax.set_title(r'\textbf{Vertical Force on LH Module}', fontsize=18)
ax.set_ylim([40, 90])
ax.set_yticks(np.arange(40, 91, 10))

# --- Right Hindlimb - Vertical (Z) ---
ax = axs[1, 1]
ax.plot(time_sim_force, -loader.sim_force_z[2], label=r'Measured (Sim)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_z[2], label=r'Estimated (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.plot(time_robot, -loader.cmd_force_z[2], label=r'Desired (Command)', color=colors[2], linestyle='--', linewidth=linewidth)
ax.set_title(r'\textbf{Vertical Force on RH Module}', fontsize=18)
ax.set_ylim([20, 70])
ax.set_yticks(np.arange(20, 71, 10))

# === Axis Formatting ===
for i in range(2):
    for j in range(2):
        axs[i, j].set_xlabel(r'\textbf{Time (s)}', fontsize=16)
        axs[i, j].set_ylabel(r'\textbf{Force (N)}', fontsize=16)
        axs[i, j].tick_params(axis='both', labelsize=16)
        # axs[i, j].legend(loc='upper right', fontsize=18)
        axs[i, j].set_xticks(np.arange(0, 12.1, 2))
        axs[i, j].grid(True)

plt.tight_layout(rect=[0, 0.09, 1, 1])

plt.subplots_adjust(wspace=0.25, hspace=0.7)

lines = [axs[0, 0].lines[0], axs[0, 0].lines[1], axs[0, 0].lines[2]]

labels = [line.get_label() for line in lines]

fig.legend(lines, labels, loc='lower center', fontsize=14, ncol=5, frameon=True, bbox_to_anchor=(0.5, 0))

# === Save as vector PDF (for LaTeX or printing) ===
# plt.savefig('force_comparison_plot.pdf', format='pdf', bbox_inches='tight')

plt.show()



# === Plotting ===
fig, axs = plt.subplots(2, 2, figsize=(12, 6))
colors = ['C1', "#3C3C3C", 'C0']

linewidth = 1.5

# --- Left Forelimb - Anterior-Posterior (X) ---
ax = axs[0, 0]
ax.plot(time_sim_force, loader.sim_force_x[0], label=r'Measured (Sim)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_x[0], label=r'Estimated (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.plot(time_robot, loader.cmd_force_x[0], label=r'Desired (Command)', color=colors[2], linestyle='--', linewidth=linewidth)
ax.set_title(r'\textbf{Horizontal Force on LF Module}', fontsize=18)
# ax.set_ylim([50, 90])
# ax.set_yticks(np.arange(50, 91, 10))

# --- Right Hindlimb - Anterior-Posterior (X) ---
ax = axs[0, 1]
ax.plot(time_sim_force, -loader.sim_force_x[1], label=r'Measured (Sim)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, -loader.state_force_x[1], label=r'Estimated (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.plot(time_robot, loader.cmd_force_x[1], label=r'Desired (Command)', color=colors[2], linestyle='--', linewidth=linewidth)
ax.set_title(r'\textbf{Horizontal Force on RF Module}', fontsize=18)

# --- Left Forelimb - Vertical (Z) ---
ax = axs[1, 0]
ax.plot(time_sim_force, loader.sim_force_x[3], label=r'Measured (Sim)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_x[3], label=r'Estimated (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.plot(time_robot, loader.cmd_force_x[3], label=r'Desired (Command)', color=colors[2], linestyle='--', linewidth=linewidth)
ax.set_title(r'\textbf{Horizontal Force on LH Module}', fontsize=18)

# --- Right Hindlimb - Vertical (Z) ---
ax = axs[1, 1]
ax.plot(time_sim_force, -loader.sim_force_x[2], label=r'Measured (Sim)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, -loader.state_force_x[2], label=r'Estimated (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.plot(time_robot, loader.cmd_force_x[2], label=r'Desired (Command)', color=colors[2], linestyle='--', linewidth=linewidth)
ax.set_title(r'\textbf{Horizontal Force on RH Module}', fontsize=18)

# === Axis Formatting ===
for i in range(2):
    for j in range(2):
        axs[i, j].set_xlabel(r'\textbf{Time (s)}', fontsize=16)
        axs[i, j].set_ylabel(r'\textbf{Force (N)}', fontsize=16)
        axs[i, j].tick_params(axis='both', labelsize=16)
        # axs[i, j].legend(loc='upper right', fontsize=18)
        axs[i, j].set_ylim([-40, 40])
        axs[i, j].set_yticks(np.arange(-40, 41, 20))
        axs[i, j].set_xticks(np.arange(0, 12.1, 2))
        axs[i, j].grid(True)

plt.tight_layout(rect=[0, 0.09, 1, 1])

plt.subplots_adjust(wspace=0.25, hspace=0.7)

lines = [axs[0, 0].lines[0], axs[0, 0].lines[1], axs[0, 0].lines[2]]

labels = [line.get_label() for line in lines]

fig.legend(lines, labels, loc='lower center', fontsize=14, ncol=5, frameon=True, bbox_to_anchor=(0.5, 0))

# === Save as vector PDF (for LaTeX or printing) ===
# plt.savefig('force_comparison_plot.pdf', format='pdf', bbox_inches='tight')

plt.show()


# === Compute RMSE ===
from sklearn.metrics import mean_squared_error

def compute_rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

rmse_results = {
    'X': [],
    'Z': []
}

for i in range(4):
    # --- Z axis ---
    rmse_z = compute_rmse(loader.cmd_force_z[i], loader.sim_force_z[i])
    rmse_results['Z'].append(rmse_z)

    # --- X axis ---
    rmse_x = compute_rmse(loader.cmd_force_x[i], loader.sim_force_x[i] if i in [0, 3] else -loader.sim_force_x[i])
    rmse_results['X'].append(rmse_x)

print('\ncmd:')
# === Print RMSE ===
module_names = ['LF', 'RF', 'RH', 'LH']
for i, name in enumerate(module_names):
    print(f"[{name}] RMSE Z: {rmse_results['Z'][i]:.3f} N | RMSE X: {rmse_results['X'][i]:.3f} N")
    
rmse_results = {
    'X': [],
    'Z': []
}

for i in range(4):
    # --- Z axis ---
    rmse_z = compute_rmse(loader.state_force_z[i], -loader.sim_force_z[i])
    rmse_results['Z'].append(rmse_z)

    # --- X axis ---
    rmse_x = compute_rmse(loader.state_force_x[i], loader.sim_force_x[i])
    rmse_results['X'].append(rmse_x)

print('\nstate:')
# === Print RMSE ===
module_names = ['LF', 'RF', 'RH', 'LH']
for i, name in enumerate(module_names):
    print(f"[{name}] RMSE Z: {rmse_results['Z'][i]:.3f} N | RMSE X: {rmse_results['X'][i]:.3f} N")
