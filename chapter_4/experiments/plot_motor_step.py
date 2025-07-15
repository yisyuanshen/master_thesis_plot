from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

robot_file_paths_old = 'exp_data_final/0320_imp_step_old.csv'
robot_file_paths_new = 'exp_data_final/0320_imp_step_new.csv'

start_idx = 0
end_idx = 3000

df_old = pd.read_csv(robot_file_paths_old).iloc[start_idx:end_idx, :]
df_new = pd.read_csv(robot_file_paths_new).iloc[start_idx:end_idx, :]

state_theta_old = np.array([df_old['state_theta_a'], df_old['state_theta_b'], df_old['state_theta_c'], df_old['state_theta_d']])
state_beta_old  = np.array([df_old['state_beta_a'] , df_old['state_beta_b'] , df_old['state_beta_c'] , df_old['state_beta_d']])

state_theta_new = np.array([df_new['state_theta_a'], df_new['state_theta_b'], df_new['state_theta_c'], df_new['state_theta_d']])
state_beta_new  = np.array([df_new['state_beta_a'] , df_new['state_beta_b'] , df_new['state_beta_c'] , df_new['state_beta_d']])

cmd_theta = np.array([df_old['cmd_theta_a'], df_old['cmd_theta_b'], df_old['cmd_theta_c'], df_old['cmd_theta_d']])
cmd_beta  = np.array([df_old['cmd_beta_a'] , df_old['cmd_beta_b'] , df_old['cmd_beta_c'] , df_old['cmd_beta_d']])

cmd_theta[0][0] = cmd_theta[0][1]
state_theta_old[0][1350:] = np.inf

# Time
sample_rate = 1000  # Hz, change if different
time_robot = np.arange(end_idx-start_idx) / sample_rate

# Plot Z
fig, axs = plt.subplots(1, 2, figsize=(12, 8))
colors = ['#3C3C3C', "C0", 'C1']

linewidth = 1.5

# Theta and Beta
ax = axs[0]
ax.plot(time_robot, np.rad2deg(cmd_theta[0]), label=r'Command', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, np.rad2deg(state_theta_old[0]), label=r'State (Traditional Method)', color=colors[1], linestyle='-.', linewidth=linewidth)
ax.plot(time_robot, np.rad2deg(state_theta_new[0]), label=r'State (Proposed Method)', color=colors[2], linestyle='--', linewidth=linewidth)
ax.set_title(r'\textbf{Theta Angle Comparison}', fontsize=18)
ax.set_ylim([0, 240])
ax.set_yticks(np.arange(0, 241, 60))
ax.set_xlabel(r'\textbf{Time (s)}', fontsize=16)
ax.set_ylabel(r'$\mathbf{\theta}$ \textbf{(deg)}', fontsize=16)
ax.tick_params(axis='both', labelsize=16)
ax.set_xticks(np.arange(0, 31, 5)/10)
ax.grid(True)

ax = axs[1]
ax.plot(time_robot, np.rad2deg(cmd_beta[0]), label=r'Command', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, np.rad2deg(state_beta_old[0]), label=r'State (Traditional Method)', color=colors[1], linestyle='-.', linewidth=linewidth)
ax.plot(time_robot, np.rad2deg(state_beta_new[0]), label=r'State (Proposed Method)', color=colors[2], linestyle='--', linewidth=linewidth)
ax.set_title(r'\textbf{Beta Angle Comparison}', fontsize=18)
ax.set_ylim([-200, 600])
ax.set_yticks(np.arange(-200, 601, 200))
ax.set_xlabel(r'\textbf{Time (s)}', fontsize=16)
ax.set_ylabel(r'$\mathbf{\beta}$ \textbf{(deg)}', fontsize=16)
ax.tick_params(axis='both', labelsize=16)
ax.set_xticks(np.arange(0, 31, 5)/10)
ax.grid(True)

plt.tight_layout(rect=[0, 0.07, 1, 1])

plt.subplots_adjust(wspace=0.25, hspace=0.5)

lines = [axs[0].lines[0], axs[0].lines[1], axs[0].lines[2]]

labels = [line.get_label() for line in lines]

fig.legend(lines, labels, loc='lower center', fontsize=14, ncol=5, frameon=True, bbox_to_anchor=(0.5, 0))

# save
# plt.savefig('.pdf', format='pdf', bbox_inches='tight')

plt.show()
