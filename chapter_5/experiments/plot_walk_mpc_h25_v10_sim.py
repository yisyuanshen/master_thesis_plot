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

odom = False

# Load Data
loader_o = DataLoader(sim=True)
loader_c = DataLoader(sim=True)

loader_o.cutoff_freq = 5
loader_c.cutoff_freq = 5

robot_file_paths_o = 'exp_data/sim/sim_walk_h25_v10_open.csv'
sim_force_file_paths_o = 'exp_data/sim/sim_walk_h25_v10_open_force.csv'

robot_file_paths_c = 'exp_data/sim/sim_walk_h25_v10_closed.csv'
sim_force_file_paths_c = 'exp_data/sim/sim_walk_h25_v10_closed_force.csv'

start_idx = 5500
end_idx = 29500

loader_o.trigger_idx = None
loader_o.load_robot_data(robot_file_paths_o, start_idx=start_idx, end_idx=end_idx)
loader_o.load_sim_force_data(sim_force_file_paths_o, start_idx=start_idx, end_idx=end_idx)

loader_c.trigger_idx = None
loader_c.load_robot_data(robot_file_paths_c, start_idx=start_idx, end_idx=end_idx)
loader_c.load_sim_force_data(sim_force_file_paths_c, start_idx=start_idx, end_idx=end_idx)


# === Force Filtering ===
loader_o.sim_force_z = np.where(loader_o.sim_force_z >= 0, 0, loader_o.sim_force_z)
loader_o.state_force_z = np.where(loader_o.state_force_z <= 0, 0, loader_o.state_force_z)
loader_o.state_force_z = np.where(loader_o.sim_force_z > -2, 0, loader_o.state_force_z)
loader_o.state_force_x = np.where(loader_o.state_force_z == 0, 0, loader_o.state_force_x)

loader_c.sim_force_z = np.where(loader_c.sim_force_z >= 0, 0, loader_c.sim_force_z)
loader_c.state_force_z = np.where(loader_c.state_force_z <= 0, 0, loader_c.state_force_z)
loader_c.state_force_z = np.where(loader_c.sim_force_z > -2, 0, loader_c.state_force_z)
loader_c.state_force_x = np.where(loader_c.state_force_z == 0, 0, loader_c.state_force_x)

# Time
sample_rate = 1000  # Hz, change if different
time_sim = np.arange(loader_o.df_sim_force.shape[0]) / sample_rate
time_robot = np.arange(loader_o.df_robot.shape[0]) / sample_rate

# Plot
fig, axs = plt.subplots(4, 2, figsize=(12, 8))
colors = ['C2', 'C3', 'C5', 'C9']
linewidth = 1.5

ax = axs[0, 0]
if odom:
    ax.plot(time_robot, loader_o.odom_pos_x, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot, loader_c.odom_pos_x, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    ax.plot(time_sim, loader_o.sim_pos_x, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_sim, loader_c.sim_pos_x, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Position X}', fontsize=18)
if odom:
    ax.set_ylim([-0.4, 2.4])
    ax.set_yticks(np.arange(0, 3, 1))
else:
    ax.set_ylim([-0.4, 2.4])
    ax.set_yticks(np.arange(0, 21, 10)/10)
ax.set_ylabel(r'\textbf{Position (m)}', fontsize=16)


ax = axs[0, 1]
if odom:
    ax.plot(time_robot, loader_o.odom_pos_z, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot, loader_c.odom_pos_z, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    ax.plot(time_sim, loader_o.sim_pos_z, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_sim, loader_c.sim_pos_z, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Position Z}', fontsize=18)
if odom:
    ax.set_ylim([0.242, 0.254])
    ax.set_yticks(np.arange(242, 255, 4)/1000)
else:
    ax.set_ylim([0.240, 0.255])
    ax.set_yticks(np.arange(240, 256, 5)/1000)
ax.set_ylabel(r'\textbf{Position (m)}', fontsize=16)


ax = axs[1, 0]
if odom:
    ax.plot(time_robot, loader_o.odom_vel_x, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot, loader_c.odom_vel_x, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    ax.plot(time_sim[:-1], loader_o.sim_vel_x, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_sim[:-1], loader_c.sim_vel_x, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)
    
ax.set_title(r'\textbf{Velocity X}', fontsize=18)
if odom:
    ax.set_ylim([-0.02, 0.22])
    ax.set_yticks(np.arange(0, 3, 1)/10)
else:
    ax.set_ylim([-0.02, 0.22])
    ax.set_yticks(np.arange(0, 3, 1)/10)
ax.set_ylabel(r'\textbf{Velocity (m/s)}', fontsize=16)


ax = axs[1, 1]
if odom:
    ax.plot(time_robot, loader_o.odom_vel_z, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot, loader_c.odom_vel_z, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    ax.plot(time_sim[:-1], loader_o.sim_vel_z, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_sim[:-1], loader_c.sim_vel_z, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Velocity Z}', fontsize=18)
if odom:
    ax.set_ylim([-0.06, 0.06])
    ax.set_yticks(np.arange(-6, 7, 6)/100)
else:
    ax.set_ylim([-0.05, 0.05])
    ax.set_yticks(np.arange(-5, 6, 5)/100)
ax.set_ylabel(r'\textbf{Velocity (m/s)}', fontsize=16)


ax = axs[2, 0]
if odom:
    ax.plot(time_robot, loader_o.imu_roll, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot, loader_c.imu_roll, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    ax.plot(time_sim, loader_o.imu_roll, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_sim, loader_c.imu_roll, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Roll}', fontsize=18)
ax.set_ylim([-2, 2])
ax.set_yticks(np.arange(-2, 3, 2))
ax.set_ylabel(r'\textbf{Angle (deg)}', fontsize=16)


ax = axs[2, 1]
if odom:
    ax.plot(time_robot, loader_o.imu_pitch, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot, loader_c.imu_pitch, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    ax.plot(time_sim, loader_o.imu_pitch, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_sim, loader_c.imu_pitch, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Pitch}', fontsize=18)
ax.set_ylim([-2, 1])
ax.set_yticks(np.arange(-2, 2, 1))
ax.set_ylabel(r'\textbf{Angle (deg)}', fontsize=16)


ax = axs[3, 0]
if odom:
    ax.plot(time_robot[:-1], loader_o.imu_roll_rate, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot[:-1], loader_c.imu_roll_rate, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    ax.plot(time_sim[:-1], loader_o.imu_roll_rate, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_sim[:-1], loader_c.imu_roll_rate, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Roll Rate}', fontsize=18)
ax.set_ylim([-18, 18])
ax.set_yticks(np.arange(-18, 19, 18))
ax.set_ylabel(r'\textbf{Rate (deg/s)}', fontsize=16)


ax = axs[3, 1]
if odom:
    ax.plot(time_robot[:-1], loader_o.imu_pitch_rate, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot[:-1], loader_c.imu_pitch_rate, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    ax.plot(time_sim[:-1], loader_o.imu_pitch_rate, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_sim[:-1], loader_c.imu_pitch_rate, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Pitch Rate}', fontsize=18)
ax.set_ylim([-15, 15])
ax.set_yticks(np.arange(-15, 16, 15))
ax.set_ylabel(r'\textbf{Rate (deg/s)}', fontsize=16)


# Format
for i in range(4):
    for j in range(2):
        axs[i, j].set_xlabel(r'\textbf{Time (s)}', fontsize=16)
        axs[i, j].tick_params(axis='both', labelsize=16)
        # axs[i, j].legend(loc='upper right', fontsize=18)
        axs[i, j].set_xticks(np.arange(0, 25, 4))
        axs[i, j].grid(True)
        
plt.tight_layout(rect=[0, 0.06, 1, 1])

plt.subplots_adjust(wspace=0.25, hspace=1.2)

lines = [axs[0, 0].lines[0], axs[0, 0].lines[1]]
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='lower center', fontsize=16, ncol=2, frameon=True, bbox_to_anchor=(0.5, 0))

# save
# if odom:
#     plt.savefig('sim_walk_odom_h25_v10_result.pdf', format='pdf', bbox_inches='tight')
# else:
#     plt.savefig('sim_walk_truth_h25_v10_result.pdf', format='pdf', bbox_inches='tight')

plt.show()