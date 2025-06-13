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

odom = True

# === Data Loading ===
loader_o = DataLoader(sim=False)
loader_c = DataLoader(sim=False)

# Data file paths (latest selection)
robot_file_paths_o = 'data/0612/0612_walk_open_h25_v10.csv'
vicon_file_paths_o = 'data/0612/walk_open_h25_v10.csv'

robot_file_paths_c = 'data/0612/0612_walk_closed_h25_v10.csv'
vicon_file_paths_c = 'data/0612/walk_closed_h25_v10.csv'

start_idx = 1000
end_idx = 25000

loader_o.trigger_idx = None
loader_o.load_robot_data(robot_file_paths_o, start_idx=start_idx, end_idx=end_idx)
loader_o.load_vicon_data(vicon_file_paths_o, start_idx=start_idx, end_idx=end_idx)

loader_c.trigger_idx = None
loader_c.load_robot_data(robot_file_paths_c, start_idx=start_idx, end_idx=end_idx)
loader_c.load_vicon_data(vicon_file_paths_c, start_idx=start_idx, end_idx=end_idx)


# === Force Filtering ===
loader_o.vicon_force_z = np.where(loader_o.vicon_force_z >= 0, 0, loader_o.vicon_force_z)
loader_o.state_force_z = np.where(loader_o.state_force_z <= 0, 0, loader_o.state_force_z)
loader_o.state_force_z = np.where(loader_o.vicon_force_z > -2, 0, loader_o.state_force_z)
loader_o.state_force_x = np.where(((loader_o.vicon_force_x < 2) & (loader_o.vicon_force_x > -2)), 0, loader_o.state_force_x)

loader_c.vicon_force_z = np.where(loader_c.vicon_force_z >= 0, 0, loader_c.vicon_force_z)
loader_c.state_force_z = np.where(loader_c.state_force_z <= 0, 0, loader_c.state_force_z)
loader_c.state_force_z = np.where(loader_c.vicon_force_z > -2, 0, loader_c.state_force_z)
loader_c.state_force_x = np.where(((loader_c.vicon_force_x < 2) & (loader_c.vicon_force_x > -2)), 0, loader_c.state_force_x)

# === Time Axis ===
sample_rate = 1000  # Hz, change if different
time_vicon = np.arange(loader_o.df_vicon.shape[0]) / sample_rate
time_robot = np.arange(loader_o.df_robot.shape[0]) / sample_rate

# === Plotting ===
fig, axs = plt.subplots(4, 2, figsize=(12, 12))
colors = ['C2', "C3"]
linewidth = 1.5

ax = axs[0, 0]
if odom:
    ax.plot(time_robot, loader_o.odom_pos_x, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot, loader_c.odom_pos_x, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    pass    
ax.set_title(r'\textbf{Position X}', fontsize=18)
# ax.set_ylim([-40, 40])
# ax.set_yticks(np.arange(-40, 41, 20))

ax = axs[0, 1]
if odom:
    ax.plot(time_robot, loader_o.odom_pos_z, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot, loader_c.odom_pos_z, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    pass    
ax.set_title(r'\textbf{Position Z}', fontsize=18)
# ax.set_ylim([-40, 40])
# ax.set_yticks(np.arange(-40, 41, 20))

ax = axs[1, 0]
if odom:
    ax.plot(time_robot, loader_o.odom_vel_x, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot, loader_c.odom_vel_x, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    pass    
ax.set_title(r'\textbf{Velocity X}', fontsize=18)
# ax.set_ylim([-40, 40])
# ax.set_yticks(np.arange(-40, 41, 20))

ax = axs[1, 1]
if odom:
    ax.plot(time_robot, loader_o.odom_vel_z, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot, loader_c.odom_vel_z, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    pass    
ax.set_title(r'\textbf{Velocity Z}', fontsize=18)
# ax.set_ylim([-40, 40])
# ax.set_yticks(np.arange(-40, 41, 20))

ax = axs[2, 0]
if odom:
    ax.plot(time_robot, loader_o.imu_roll, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot, loader_c.imu_roll, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    pass    
ax.set_title(r'\textbf{Roll}', fontsize=18)
# ax.set_ylim([-40, 40])
# ax.set_yticks(np.arange(-40, 41, 20))

ax = axs[2, 1]
if odom:
    ax.plot(time_robot, loader_o.imu_pitch, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot, loader_c.imu_pitch, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    pass    
ax.set_title(r'\textbf{Pitch}', fontsize=18)
# ax.set_ylim([-40, 40])
# ax.set_yticks(np.arange(-40, 41, 20))

ax = axs[3, 0]
if odom:
    ax.plot(time_robot[:-1], loader_o.imu_roll_rate, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot[:-1], loader_c.imu_roll_rate, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    pass    
ax.set_title(r'\textbf{Roll Rate}', fontsize=18)
# ax.set_ylim([-40, 40])
# ax.set_yticks(np.arange(-40, 41, 20))

ax = axs[3, 1]
if odom:
    ax.plot(time_robot[:-1], loader_o.imu_pitch_rate, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot[:-1], loader_c.imu_pitch_rate, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    pass    
ax.set_title(r'\textbf{Pitch Rate}', fontsize=18)
# ax.set_ylim([-40, 40])
# ax.set_yticks(np.arange(-40, 41, 20))



# === Axis Formatting ===
for i in range(4):
    for j in range(2):
        axs[i, j].set_xlabel(r'\textbf{Time (s)}', fontsize=16)
        axs[i, j].set_ylabel(r'\textbf{Force (N)}', fontsize=16)
        axs[i, j].tick_params(axis='both', labelsize=16)
        # axs[i, j].legend(loc='upper right', fontsize=18)
        # ax.set_xticks(np.arange(0, 5.1, 1))
        axs[i, j].grid(True)
        
plt.tight_layout(rect=[0, 0.06, 1, 1])

plt.subplots_adjust(wspace=0.25, hspace=0.7)

lines = [axs[0, 0].lines[0], axs[0, 0].lines[1]]
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='lower center', fontsize=16, ncol=2, frameon=True, bbox_to_anchor=(0.5, 0))

# === Save as vector PDF (for LaTeX or printing) ===
plt.savefig('test.pdf', format='pdf', bbox_inches='tight')

# plt.show()
