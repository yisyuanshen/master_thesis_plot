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

robot_file_paths_o = 'exp_data_final/sim_trot_h20_v60_open_phase.csv'
sim_force_file_paths_o = 'exp_data_final/sim_trot_h20_v60_open_phase_force.csv'

robot_file_paths_c = 'exp_data_final/sim_trot_h20_v60_closed_phase.csv'
sim_force_file_paths_c = 'exp_data_final/sim_trot_h20_v60_closed_phase_force.csv'

start_idx = 5250
end_idx = 11250

loader_o.trigger_idx = None
loader_o.load_robot_data(robot_file_paths_o, start_idx=start_idx, end_idx=end_idx)
loader_o.load_sim_force_data(sim_force_file_paths_o, start_idx=start_idx, end_idx=end_idx)

loader_c.trigger_idx = None
loader_c.load_robot_data(robot_file_paths_c, start_idx=start_idx, end_idx=end_idx)
loader_c.load_sim_force_data(sim_force_file_paths_c, start_idx=start_idx, end_idx=end_idx)

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
    ax.set_ylim([-0.5, 3.5])
    ax.set_yticks(np.arange(0, 31, 15)/10)
else:
    ax.set_ylim([-0.5, 3.5])
    ax.set_yticks(np.arange(0, 31, 15)/10)
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
    ax.set_ylim([0.150, 0.225])
    ax.set_yticks(np.arange(150, 226, 25)/1000)
else:
    ax.set_ylim([0.150, 0.225])
    ax.set_yticks(np.arange(150, 226, 25)/1000)
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
    ax.set_ylim([-0.5, 1])
    ax.set_yticks(np.arange(-5, 11, 5)/10)
else:
    ax.set_ylim([-0.5, 1])
    ax.set_yticks(np.arange(-5, 11, 5)/10)
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
    ax.set_ylim([-0.15, 0.3])
    ax.set_yticks(np.arange(-15, 31, 15)/100)
else:
    ax.set_ylim([-0.3, 0.3])
    ax.set_yticks(np.arange(-3, 4, 3)/10)
ax.set_ylabel(r'\textbf{Velocity (m/s)}', fontsize=16)


ax = axs[2, 0]
if odom:
    ax.plot(time_robot, loader_o.imu_roll, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot, loader_c.imu_roll, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    ax.plot(time_sim, loader_o.imu_roll, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_sim, loader_c.imu_roll, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Roll}', fontsize=18)
ax.set_ylim([-20, 20])
ax.set_yticks(np.arange(-20, 21, 20))
ax.set_ylabel(r'\textbf{Angle (deg)}', fontsize=16)


ax = axs[2, 1]
if odom:
    ax.plot(time_robot, loader_o.imu_pitch, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot, loader_c.imu_pitch, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    ax.plot(time_sim, loader_o.imu_pitch, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_sim, loader_c.imu_pitch, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Pitch}', fontsize=18)
ax.set_ylim([-10, 10])
ax.set_yticks(np.arange(-10, 11, 10))
ax.set_ylabel(r'\textbf{Angle (deg)}', fontsize=16)


ax = axs[3, 0]
if odom:
    ax.plot(time_robot[:-1], loader_o.imu_roll_rate, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot[:-1], loader_c.imu_roll_rate, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    ax.plot(time_sim[:-1], loader_o.imu_roll_rate, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_sim[:-1], loader_c.imu_roll_rate, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Roll Rate}', fontsize=18)
ax.set_ylim([-120, 120])
ax.set_yticks(np.arange(-120, 121, 120))
ax.set_ylabel(r'\textbf{Rate (deg/s)}', fontsize=16)


ax = axs[3, 1]
if odom:
    ax.plot(time_robot[:-1], loader_o.imu_pitch_rate, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot[:-1], loader_c.imu_pitch_rate, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    ax.plot(time_sim[:-1], loader_o.imu_pitch_rate, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_sim[:-1], loader_c.imu_pitch_rate, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Pitch Rate}', fontsize=18)
ax.set_ylim([-150, 150])
ax.set_yticks(np.arange(-150, 151, 150))
ax.set_ylabel(r'\textbf{Rate (deg/s)}', fontsize=16)


# Format
for i in range(4):
    for j in range(2):
        axs[i, j].set_xlabel(r'\textbf{Time (s)}', fontsize=16)
        axs[i, j].tick_params(axis='both', labelsize=16)
        # axs[i, j].legend(loc='upper right', fontsize=18)
        axs[i, j].set_xticks(np.arange(0, 7, 1))
        axs[i, j].grid(True)
        
plt.tight_layout(rect=[0, 0.06, 1, 1])

plt.subplots_adjust(wspace=0.25, hspace=1.2)

lines = [axs[0, 0].lines[0], axs[0, 0].lines[1]]
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='lower center', fontsize=16, ncol=2, frameon=True, bbox_to_anchor=(0.5, 0))

# save
# if odom:
#     plt.savefig('sim_trot_odom_h20_v45_result.pdf', format='pdf', bbox_inches='tight')
# else:
#     plt.savefig('sim_trot_truth_h20_v45_result.pdf', format='pdf', bbox_inches='tight')

plt.show()




def compute_rmse_std(predicted, target):
    print(round(np.mean(predicted), 3), round(np.mean(target), 3))
    err = np.array(predicted) - np.array(target)
    return np.sqrt(np.mean(err**2)), np.std(err)

# 統一長度：取所有序列的最小長度
lengths = [
    len(loader_o.odom_pos_x), len(loader_o.odom_vel_x),
    len(loader_o.odom_pos_z), len(loader_o.odom_vel_z),
    len(loader_o.imu_roll),   len(loader_o.imu_pitch),
    len(loader_o.imu_roll_rate), len(loader_o.imu_pitch_rate),
    len(loader_c.odom_pos_x), len(loader_c.odom_vel_x),
    len(loader_c.odom_pos_z), len(loader_c.odom_vel_z),
    len(loader_c.imu_roll),   len(loader_c.imu_pitch),
    len(loader_c.imu_roll_rate), len(loader_c.imu_pitch_rate),
]
length = min(lengths)

# 建立 ground truth
dt = 1.0 / sample_rate
target_vel_x = []
for i in range(750):
    target_vel_x.append(0)
for i in range(4500):
    target_vel_x.append(0.6)
for i in range(749):
    target_vel_x.append(0)


target_pos_x     = np.cumsum(target_vel_x) * dt
ground_truth = {
    'Position X':      target_pos_x,
    'Position Z':      np.full(length, 0.2),
    'Velocity X':      target_vel_x,
    'Velocity Z':      np.zeros(length),
    'IMU Roll':        np.zeros(length),
    'IMU Pitch':       np.zeros(length),
    'IMU Roll Rate':   np.zeros(length),
    'IMU Pitch Rate':  np.zeros(length),
}

# 擷取預測值
def get_series(loader, key):
    if odom:
        d = {
            'Position X':    loader.odom_pos_x,
            'Position Z':    loader.odom_pos_z,
            'Velocity X':    loader.odom_vel_x,
            'Velocity Z':    loader.odom_vel_z,
            'IMU Roll':      loader.imu_roll,
            'IMU Pitch':     loader.imu_pitch,
            'IMU Roll Rate': loader.imu_roll_rate,
            'IMU Pitch Rate':loader.imu_pitch_rate,
        }
    else:
        d = {
            'Position X':    loader.sim_pos_x,
            'Position Z':    loader.sim_pos_z,
            'Velocity X':    loader.sim_vel_x,
            'Velocity Z':    loader.sim_vel_z,
            'IMU Roll':      loader.imu_roll,
            'IMU Pitch':     loader.imu_pitch,
            'IMU Roll Rate': loader.imu_roll_rate,
            'IMU Pitch Rate':loader.imu_pitch_rate,
        }
    return d[key][:length]

open_data   = {k: get_series(loader_o, k) for k in ground_truth}
closed_data = {k: get_series(loader_c, k) for k in ground_truth}

# 計算並印出表格
results = []
for key in ground_truth:
    gt = ground_truth[key][:length]
    po = open_data[key]
    pc = closed_data[key]
    # 再次對齊（保險）
    N = min(len(gt), len(po), len(pc))
    po, pc, gt = po[:N], pc[:N], gt[:N]
    ormse, ostd = compute_rmse_std(po, gt)
    crmse, cstd = compute_rmse_std(pc, gt)
    results.append((key, ormse, ostd, crmse, cstd))

print(f"\n==== RMSE & STD Results (odom = {odom}) ====\n")
print("{:<18s} {:>12s} {:>12s} {:>14s} {:>12s}"
      .format("State","Open RMSE","Closed RMSE","Open STD","Closed STD"))
print("-"*70)
for state, ormse, ostd, crmse, cstd in results:
    print(f"{state:<18s} {ormse:12.5f} {crmse:14.5f} {ostd:12.5f} {cstd:12.5f}")

print()

for state, ormse, ostd, crmse, cstd in results:
    print(f"{ormse:.3f}&{crmse:.3f}&{ostd:.3f}&{cstd:.3f}")