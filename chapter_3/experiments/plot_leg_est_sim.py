from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

from DataLoader import DataLoader
import LegModel_

# Style
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'Times New Roman',
    'font.size': 18,
    'axes.linewidth': 1.2,
    'legend.frameon': False,
})

# Load Data
loader = DataLoader(sim=True)
loader.cutoff_freq = 5

robot_file_paths = 'exp_data_final/sim_leg.csv'
sim_force_file_paths = 'exp_data_final/sim_leg_force.csv'

start_idx = 18000
end_idx = 30000

loader.load_robot_data(robot_file_paths, start_idx=start_idx, end_idx=end_idx)
loader.load_sim_force_data(sim_force_file_paths, start_idx=start_idx, end_idx=end_idx)

# Data Process
loader.state_force_z -= (0.68*9.81-5.48)

leg = LegModel_.LegModel()
eta_array = np.array(list(zip(loader.state_theta[0], loader.state_beta[0])))

leg.forward(eta_array)
print('Forward Kinematics Done')
leg.calculate_alpha()
print('Alpha Calculation Done')
leg.calculate_p()
print('P Calculation Done')
leg.calculate_jacobian()
print('Jacobian Calculation Done')

rcond = leg.jacobian_rcond
alpha = leg.alpha

# Time
sample_rate = 1000  # Hz, change if different
time_sim_force = np.arange(loader.df_sim_force.shape[0]) / sample_rate
time_robot = np.arange(loader.df_robot.shape[0]) / sample_rate

# Plot
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
colors = ['C1', "#3C3C3C", 'C2', 'C3']
linewidth = 1.5

# Force
ax = axs[0]
ax.plot(time_sim_force, -loader.sim_force_z[0], label=r'Measured GRF (Ground Truth)', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader.state_force_z[0], label=r'Estimated GRF (State)', color=colors[1], linestyle=':', linewidth=linewidth)
ax.set_title(r'\textbf{Vertical Ground Reaction Force (GRF)}', fontsize=18)
ax.set_ylabel(r'\textbf{Force (N)}', fontsize=16)
ax.set_ylim([-100, 200])
ax.set_yticks(np.arange(-100, 201, 100))

state_colors = {
    1: '#FF9797',
    2: '#FFE153',
    3: "#FFE153",
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

# Error
loader.legmodel.contact_map(loader.state_theta[0], loader.state_beta[0])
ax = axs[1]
ax.plot(time_robot, ((loader.state_force_z[0]-(-loader.sim_force_z[0]))/loader.state_force_z[0])*100, label=r'Estimation Error', color=colors[2], linestyle='-', linewidth=linewidth)
ax.set_title(r'\textbf{Estimation Error and Jacobian Condition Number}', fontsize=18)
ax.set_ylabel(r'\textbf{Estimation Error (\%)}', fontsize=16)
ax.set_ylim([-50, 150])
ax.set_yticks(np.arange(-50, 151, 50))

state_colors = {
    1: '#FF9797',
    2: '#FFE153',
    3: "#FFE153",
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

ax_twin = ax.twinx()
ax_twin.set_ylabel(r'\textbf{Condition Number $({\kappa_2})$}', fontsize=16, labelpad=15)
ax_twin.plot(time_robot, 1/rcond, label=r'Condition Number', color=colors[3], linestyle='--', linewidth=linewidth)
ax_twin.set_ylim([-10, 90])
ax_twin.set_yticks(np.arange(-0, 81, 20))

# Format
for i in range(2):
    axs[i].set_xlabel(r'\textbf{Time (s)}', fontsize=16)
    axs[i].tick_params(axis='both', labelsize=16)
    # axs[i, j].legend(loc='upper right', fontsize=18)
    ax.set_xticks(np.arange(0, 13, 2))
    axs[i].grid(True)
        
plt.tight_layout(rect=[0, 0.11, 1, 1])

plt.subplots_adjust(wspace=0.25, hspace=0.5)

lines = [axs[0].lines[0], axs[1].lines[0],
         axs[0].lines[1], ax_twin.lines[0],
         Patch(facecolor=state_colors[1], edgecolor='none', alpha=0.3, label='Upper Rim'),
         Patch(facecolor=state_colors[2], edgecolor='none', alpha=0.3, label='Lower Rim')]

labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='lower center', fontsize=16, ncol=3, frameon=True, bbox_to_anchor=(0.5, 0))

# === Save as vector PDF (for LaTeX or printing) ===
plt.savefig('sim_leg_result.pdf', format='pdf', bbox_inches='tight')

plt.show()

force = []
for i in range(0, 6000, 1):
    if ((1/rcond)[i] <= 5) :
        force.append(abs((loader.state_force_z[0]-(-loader.sim_force_z[0]))/loader.sim_force_z[0])[i]*100)

print(round(np.average(force), 2))