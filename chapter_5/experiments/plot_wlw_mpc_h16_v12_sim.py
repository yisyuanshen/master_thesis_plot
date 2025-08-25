import matplotlib.pyplot as plt
import numpy as np

from DataLoader import DataLoader

# Style
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'Times New Roman',
    'font.size': 18,
    'axes.linewidth': 1.2,
    'legend.frameon': False,
})

odom = True

# Load Data
loader_o = DataLoader(sim=True)
loader_c = DataLoader(sim=True)

loader_o.cutoff_freq = 5
loader_c.cutoff_freq = 5

robot_file_paths_o = 'exp_data_final/sim_wlw_h16_v12_open.csv'
sim_force_file_paths_o = 'exp_data_final/sim_wlw_h16_v12_open_force.csv'

robot_file_paths_c = 'exp_data_final/sim_wlw_h16_v12_closed.csv'
sim_force_file_paths_c = 'exp_data_final/sim_wlw_h16_v12_closed_force.csv'

start_idx = 5500
end_idx = 26500

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
fig, axs = plt.subplots(4, 2, figsize=(12, 9))
colors = ['C2', 'C3', 'C5', 'C9']
linewidth = 1.5

ax = axs[0, 0]
if odom:
    ax.plot(time_robot, loader_o.odom_pos_x, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot, loader_c.odom_pos_x, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    ax.plot(time_sim, loader_o.sim_pos_x, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_sim, loader_c.sim_pos_x, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Position X}', fontsize=20)
if odom:
    ax.set_ylim([-0.5, 3.5])
    ax.set_yticks(np.arange(0, 31, 15)/10)
else:
    ax.set_ylim([-0.5, 3.5])
    ax.set_yticks(np.arange(0, 31, 15)/10)
ax.set_ylabel(r'\textbf{Position (m)}', fontsize=18)


ax = axs[0, 1]
if odom:
    ax.plot(time_robot, loader_o.odom_pos_z, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot, loader_c.odom_pos_z, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    ax.plot(time_sim, loader_o.sim_pos_z, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_sim, loader_c.sim_pos_z, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Position Z}', fontsize=20)
if odom:
    ax.set_ylim([0.150, 0.165])
    ax.set_yticks(np.arange(150, 166, 5)/1000)
else:
    ax.set_ylim([0.150, 0.165])
    ax.set_yticks(np.arange(150, 166, 5)/1000)
ax.set_ylabel(r'\textbf{Position (m)}', fontsize=18)


ax = axs[1, 0]
if odom:
    ax.plot(time_robot, loader_o.odom_vel_x, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot, loader_c.odom_vel_x, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    ax.plot(time_sim[:-1], loader_o.sim_vel_x, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_sim[:-1], loader_c.sim_vel_x, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)
    
ax.set_title(r'\textbf{Velocity X}', fontsize=20)
if odom:
    ax.set_ylim([-0.05, 0.35])
    ax.set_yticks(np.arange(0, 31, 15)/100)
else:
    ax.set_ylim([-0.05, 0.35])
    ax.set_yticks(np.arange(0, 31, 15)/100)
ax.set_ylabel(r'\textbf{Velocity (m/s)}', fontsize=18)


ax = axs[1, 1]
if odom:
    ax.plot(time_robot, loader_o.odom_vel_z, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot, loader_c.odom_vel_z, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    ax.plot(time_sim[:-1], loader_o.sim_vel_z, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_sim[:-1], loader_c.sim_vel_z, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Velocity Z}', fontsize=20)
if odom:
    ax.set_ylim([-0.05, 0.05])
    ax.set_yticks(np.arange(-5, 6, 5)/100)
else:
    ax.set_ylim([-0.05, 0.05])
    ax.set_yticks(np.arange(-5, 6, 5)/100)
ax.set_ylabel(r'\textbf{Velocity (m/s)}', fontsize=18)


ax = axs[2, 0]
if odom:
    ax.plot(time_robot, loader_o.imu_roll, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot, loader_c.imu_roll, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    ax.plot(time_sim, loader_o.imu_roll, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_sim, loader_c.imu_roll, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Roll}', fontsize=20)
ax.set_ylim([-1, 1])
ax.set_yticks(np.arange(-1, 2, 1))
ax.set_ylabel(r'\textbf{Angle (deg)}', fontsize=18)


ax = axs[2, 1]
if odom:
    ax.plot(time_robot, loader_o.imu_pitch, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot, loader_c.imu_pitch, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    ax.plot(time_sim, loader_o.imu_pitch, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_sim, loader_c.imu_pitch, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Pitch}', fontsize=20)
ax.set_ylim([-1.5, 1.5])
ax.set_yticks(np.arange(-15, 16, 15)/10)
ax.set_ylabel(r'\textbf{Angle (deg)}', fontsize=18)


ax = axs[3, 0]
if odom:
    ax.plot(time_robot[:-1], loader_o.imu_roll_rate, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot[:-1], loader_c.imu_roll_rate, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    ax.plot(time_sim[:-1], loader_o.imu_roll_rate, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_sim[:-1], loader_c.imu_roll_rate, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Roll Rate}', fontsize=20)
ax.set_ylim([-9, 9])
ax.set_yticks(np.arange(-9, 10, 9))
ax.set_ylabel(r'\textbf{Rate (deg/s)}', fontsize=18)


ax = axs[3, 1]
if odom:
    ax.plot(time_robot[:-1], loader_o.imu_pitch_rate, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
    ax.plot(time_robot[:-1], loader_c.imu_pitch_rate, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)
else:
    ax.plot(time_sim[:-1], loader_o.imu_pitch_rate, label=r'Open Loop', color=colors[2], linestyle='-', linewidth=linewidth)
    ax.plot(time_sim[:-1], loader_c.imu_pitch_rate, label=r'Closed Loop', color=colors[3], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Pitch Rate}', fontsize=20)
ax.set_ylim([-12, 12])
ax.set_yticks(np.arange(-12, 13, 12))
ax.set_ylabel(r'\textbf{Rate (deg/s)}', fontsize=18)


# Format
for i in range(4):
    for j in range(2):
        axs[i, j].set_xlabel(r'\textbf{Time (s)}', fontsize=18)
        axs[i, j].tick_params(axis='both', labelsize=18)
        # axs[i, j].legend(loc='upper right', fontsize=18)
        axs[i, j].set_xticks(np.arange(0, 22, 3))
        axs[i, j].grid(True)
        
plt.tight_layout(rect=[0, 0.06, 1, 0.92])

plt.subplots_adjust(wspace=0.25, hspace=1.5)

lines = [axs[0, 0].lines[0], axs[0, 0].lines[1]]
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='lower center', fontsize=16, ncol=2, frameon=True, bbox_to_anchor=(0.5, 0))

fig.suptitle(r'\textbf{Simulation Results (\#7)}', fontsize=22, y=0.98)
# save
if odom:
    plt.savefig('sim_walk_odom_h20_v10_result.pdf', format='pdf', bbox_inches='tight')
# else:
#     plt.savefig('sim_walk_truth_h20_v10_result.pdf', format='pdf', bbox_inches='tight')

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
for i in range(500):
    target_vel_x.append(0)
for i in range(1000):
    target_vel_x.append(0.12/1000*i)
for i in range(18000):
    target_vel_x.append(0.12)
for i in range(1000):
    target_vel_x.append(0.12-0.12/1000*i)
for i in range(499):
    target_vel_x.append(0)


target_pos_x     = np.cumsum(target_vel_x) * dt
ground_truth = {
    'Position X':      target_pos_x,
    'Position Z':      np.full(length, 0.16),
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