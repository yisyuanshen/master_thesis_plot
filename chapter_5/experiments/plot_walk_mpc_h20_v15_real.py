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
loader_o = DataLoader(sim=False)
loader_c = DataLoader(sim=False)

loader_o.cutoff_freq = 10
loader_c.cutoff_freq = 10

robot_file_paths_o = 'exp_data/real/0617_walk_h20_v15_open.csv'
vicon_file_paths_o = 'exp_data/real/0617_walk_h20_v15_open_vicon.csv'

robot_file_paths_c = 'exp_data/real/0617_walk_h20_v15_closed.csv'
vicon_file_paths_c = 'exp_data/real/0617_walk_h20_v15_closed_vicon.csv'

start_idx = 2500
end_idx = 18500

loader_o.trigger_idx = None
loader_o.load_robot_data(robot_file_paths_o, start_idx=start_idx, end_idx=end_idx)
loader_o.load_vicon_data(vicon_file_paths_o, start_idx=start_idx, end_idx=end_idx)

loader_c.trigger_idx = None
loader_c.load_robot_data(robot_file_paths_c, start_idx=start_idx, end_idx=end_idx)
loader_c.load_vicon_data(vicon_file_paths_c, start_idx=start_idx, end_idx=end_idx)


# Data Process
loader_o.vicon_force_z = np.where(loader_o.vicon_force_z >= 0, 0, loader_o.vicon_force_z)
loader_o.state_force_z = np.where(loader_o.state_force_z <= 0, 0, loader_o.state_force_z)
loader_o.state_force_z = np.where(loader_o.vicon_force_z > -2, 0, loader_o.state_force_z)
loader_o.state_force_x = np.where(loader_o.state_force_z == 0, 0, loader_o.state_force_x)

loader_c.vicon_force_z = np.where(loader_c.vicon_force_z >= 0, 0, loader_c.vicon_force_z)
loader_c.state_force_z = np.where(loader_c.state_force_z <= 0, 0, loader_c.state_force_z)
loader_c.state_force_z = np.where(loader_c.vicon_force_z > -2, 0, loader_c.state_force_z)
loader_c.state_force_x = np.where(loader_c.state_force_z == 0, 0, loader_c.state_force_x)

# Time
sample_rate = 1000  # Hz, change if different
time_vicon = np.arange(loader_o.df_vicon.shape[0]) / sample_rate
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
    ax.plot(time_vicon, loader_o.vicon_pos_x, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_vicon, loader_c.vicon_pos_x, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Position X}', fontsize=18)
if odom:
    ax.set_ylim([-0.4, 2.4])
    ax.set_yticks(np.arange(0, 3, 1))
else:
    ax.set_ylim([-0.4, 2.4])
    ax.set_yticks(np.arange(0, 3, 1))
ax.set_ylabel(r'\textbf{Position (m)}', fontsize=16)


ax = axs[0, 1]
if odom:
    ax.plot(time_robot, loader_o.odom_pos_z, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot, loader_c.odom_pos_z, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    ax.plot(time_vicon, loader_o.vicon_pos_z, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_vicon, loader_c.vicon_pos_z, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Position Z}', fontsize=18)
if odom:
    ax.set_ylim([0.194, 0.203])
    ax.set_yticks(np.arange(194, 204, 3)/1000)
else:
    ax.set_ylim([0.190, 0.205])
    ax.set_yticks(np.arange(190, 206, 5)/1000)
ax.set_ylabel(r'\textbf{Position (m)}', fontsize=16)


ax = axs[1, 0]
if odom:
    ax.plot(time_robot, loader_o.odom_vel_x, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot, loader_c.odom_vel_x, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    ax.plot(time_vicon[:-1], loader_o.vicon_vel_x, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_vicon[:-1], loader_c.vicon_vel_x, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)
    
ax.set_title(r'\textbf{Velocity X}', fontsize=18)
if odom:
    ax.set_ylim([-0.15, 0.3])
    ax.set_yticks(np.arange(-15, 31, 15)/100)
else:
    ax.set_ylim([-0.08, 0.38])
    ax.set_yticks(np.arange(0, 31, 15)/100)
ax.set_ylabel(r'\textbf{Velocity (m/s)}', fontsize=16)


ax = axs[1, 1]
if odom:
    ax.plot(time_robot, loader_o.odom_vel_z, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot, loader_c.odom_vel_z, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    ax.plot(time_vicon[:-1], loader_o.vicon_vel_z, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_vicon[:-1], loader_c.vicon_vel_z, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Velocity Z}', fontsize=18)
if odom:
    ax.set_ylim([-0.2, 0.2])
    ax.set_yticks(np.arange(-20, 21, 20)/100)
else:
    ax.set_ylim([-0.2, 0.2])
    ax.set_yticks(np.arange(-20, 21, 20)/100)
ax.set_ylabel(r'\textbf{Velocity (m/s)}', fontsize=16)


ax = axs[2, 0]
if odom:
    ax.plot(time_robot, loader_o.imu_roll, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot, loader_c.imu_roll, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    ax.plot(time_vicon, loader_o.vicon_roll, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_vicon, loader_c.vicon_roll, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Roll}', fontsize=18)
if odom:
    ax.set_ylim([-4, 8])
    ax.set_yticks(np.arange(-4, 9, 4))
else:
    ax.set_ylim([-4, 8])
    ax.set_yticks(np.arange(-4, 9, 4))
ax.set_ylabel(r'\textbf{Angle (deg)}', fontsize=16)


ax = axs[2, 1]
if odom:
    ax.plot(time_robot, loader_o.imu_pitch, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot, loader_c.imu_pitch, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    ax.plot(time_vicon, loader_o.vicon_pitch, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_vicon, loader_c.vicon_pitch, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Pitch}', fontsize=18)
if odom:
    ax.set_ylim([-6, 3])
    ax.set_yticks(np.arange(-6, 4, 3))
else:
    ax.set_ylim([-6, 3])
    ax.set_yticks(np.arange(-6, 4, 3))
ax.set_ylabel(r'\textbf{Angle (deg)}', fontsize=16)


ax = axs[3, 0]
if odom:
    ax.plot(time_robot[:-1], loader_o.imu_roll_rate, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot[:-1], loader_c.imu_roll_rate, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    ax.plot(time_vicon[:-1], loader_o.vicon_roll_rate, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_vicon[:-1], loader_c.vicon_roll_rate, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Roll Rate}', fontsize=18)
if odom:
    ax.set_ylim([-50, 50])
    ax.set_yticks(np.arange(-50, 51, 50))
else:
    ax.set_ylim([-50, 50])
    ax.set_yticks(np.arange(-50, 51, 50))
ax.set_ylabel(r'\textbf{Rate (deg/s)}', fontsize=16)


ax = axs[3, 1]
if odom:
    ax.plot(time_robot[:-1], loader_o.imu_pitch_rate, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot[:-1], loader_c.imu_pitch_rate, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    ax.plot(time_vicon[:-1], loader_o.vicon_pitch_rate, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_vicon[:-1], loader_c.vicon_pitch_rate, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Pitch Rate}', fontsize=18)
if odom:
    ax.set_ylim([-30, 60])
    ax.set_yticks(np.arange(-30, 61, 30))
else:
    ax.set_ylim([-35, 70])
    ax.set_yticks(np.arange(-35, 71, 35))
ax.set_ylabel(r'\textbf{Rate (deg/s)}', fontsize=16)


# Format
for i in range(4):
    for j in range(2):
        axs[i, j].set_xlabel(r'\textbf{Time (s)}', fontsize=16)
        axs[i, j].tick_params(axis='both', labelsize=16)
        # axs[i, j].legend(loc='upper right', fontsize=18)
        axs[i, j].set_xticks(np.arange(0, 17, 2))
        axs[i, j].grid(True)
        
plt.tight_layout(rect=[0, 0.06, 1, 1])

plt.subplots_adjust(wspace=0.25, hspace=1.2)

lines = [axs[0, 0].lines[0], axs[0, 0].lines[1]]
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='lower center', fontsize=16, ncol=2, frameon=True, bbox_to_anchor=(0.5, 0))

# save
if odom:
    plt.savefig('real_walk_odom_h20_v15_result.pdf', format='pdf', bbox_inches='tight')
else:
    plt.savefig('real_walk_vicon_h20_v15_result.pdf', format='pdf', bbox_inches='tight')

plt.show()
