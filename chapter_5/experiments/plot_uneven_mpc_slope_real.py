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

loader_o.cutoff_freq = 2
loader_c.cutoff_freq = 2

robot_file_paths_o = 'exp_data_final/0716_slope_open_1.csv'
robot_file_paths_c = 'exp_data_final/0716_slope_closed_3.csv'

start_idx = 2000
end_idx = 26000

loader_o.trigger_idx = None
loader_o.load_robot_data(robot_file_paths_o, start_idx=start_idx, end_idx=end_idx)

loader_c.trigger_idx = None
loader_c.load_robot_data(robot_file_paths_c, start_idx=start_idx, end_idx=end_idx)

# Time
sample_rate = 1000  # Hz, change if different
time_robot = np.arange(loader_o.df_robot.shape[0]) / sample_rate

# Plot
fig, axs = plt.subplots(4, 2, figsize=(12, 8))
colors = ['C2', 'C3', 'C5', 'C9']
linewidth = 1.5 


ax = axs[0, 0]
ax.plot(time_robot, loader_o.odom_pos_x, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader_c.odom_pos_x, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Position X}', fontsize=18)
ax.set_ylim([-0.4, 2.4])
ax.set_yticks(np.arange(0, 3, 1))
ax.set_ylabel(r'\textbf{Position (m)}', fontsize=16)


ax = axs[0, 1]
ax.plot(time_robot, loader_o.odom_pos_z, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader_c.odom_pos_z, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Position Z}', fontsize=18)
ax.set_ylim([0.21, 0.27])
ax.set_yticks(np.arange(210, 271, 20)/1000)
ax.set_ylabel(r'\textbf{Position (m)}', fontsize=16)


ax = axs[1, 0]
ax.plot(time_robot, loader_o.odom_vel_x, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader_c.odom_vel_x, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Velocity X}', fontsize=18)
ax.set_ylim([-0.3, 0.6])
ax.set_yticks(np.arange(-30, 61, 30)/100)
ax.set_ylabel(r'\textbf{Velocity (m/s)}', fontsize=16)


ax = axs[1, 1]
ax.plot(time_robot, loader_o.odom_vel_z, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader_c.odom_vel_z, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Velocity Z}', fontsize=18)
ax.set_ylim([-0.2, 0.1])
ax.set_yticks(np.arange(-20, 11, 10)/100)
ax.set_ylabel(r'\textbf{Velocity (m/s)}', fontsize=16)


ax = axs[2, 0]
ax.plot(time_robot, loader_o.imu_roll, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader_c.imu_roll, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Roll}', fontsize=18)
ax.set_ylim([-18, 18])
ax.set_yticks(np.arange(-18, 19, 18))
ax.set_ylabel(r'\textbf{Angle (deg)}', fontsize=16)


ax = axs[2, 1]
ax.plot(time_robot, loader_o.imu_pitch, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot, loader_c.imu_pitch, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Pitch}', fontsize=18)
ax.set_ylim([-18, 18])
ax.set_yticks(np.arange(-18, 19, 18))
ax.set_ylabel(r'\textbf{Angle (deg)}', fontsize=16)


ax = axs[3, 0]
ax.plot(time_robot[:-1], loader_o.imu_roll_rate, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot[:-1], loader_c.imu_roll_rate, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Roll Rate}', fontsize=18)
ax.set_ylim([-45, 45])
ax.set_yticks(np.arange(-45, 46, 45))
ax.set_ylabel(r'\textbf{Rate (deg/s)}', fontsize=16)


ax = axs[3, 1]
ax.plot(time_robot[:-1], loader_o.imu_pitch_rate, label=r'Open Loop', color=colors[0], linestyle='-', linewidth=linewidth)
ax.plot(time_robot[:-1], loader_c.imu_pitch_rate, label=r'Closed Loop', color=colors[1], linestyle='--', linewidth=linewidth)

ax.set_title(r'\textbf{Pitch Rate}', fontsize=18)
ax.set_ylim([-65, 65])
ax.set_yticks(np.arange(-65, 66, 65))
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
#     plt.savefig('real_walk_odom_h25_v10_result.pdf', format='pdf', bbox_inches='tight')
# else:
#     plt.savefig('real_walk_vicon_h25_v10_result.pdf', format='pdf', bbox_inches='tight')

plt.show()




def compute_rmse_std(predicted, target):
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
for i in range(250):
    target_vel_x.append(0)
for i in range(1000):
    target_vel_x.append(0.1/1000*i)
for i in range(20500):
    target_vel_x.append(0.1)
for i in range(1000):
    target_vel_x.append(0.1-0.1/1000*i)
for i in range(249):
    target_vel_x.append(0)


target_pos_x     = np.cumsum(target_vel_x) * dt
ground_truth = {
    'Position X':      target_pos_x,
    'Position Z':      np.full(length, 0.25),
    'Velocity X':      target_vel_x,
    'Velocity Z':      np.zeros(length),
    'IMU Roll':        np.zeros(length),
    'IMU Pitch':       np.zeros(length),
    'IMU Roll Rate':   np.zeros(length),
    'IMU Pitch Rate':  np.zeros(length),
}

# 擷取預測值
def get_series(loader, key):
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

print('\n\n')
for i in [2, 5, 6, 7, 8, 9, 10, 11]:
    print(i, '   ', np.average(loader_o.df_robot[f'i_{i}'].to_numpy()*loader_o.df_robot[f'v_{i}'].to_numpy()))

print('\n\n')
for i in [2, 5, 6, 7, 8, 9, 10, 11]:
    print(i, '   ', np.average(loader_c.df_robot[f'i_{i}'].to_numpy()*loader_c.df_robot[f'v_{i}'].to_numpy()))

print('\n\n')
print('Open Loop: ', sum([np.average(loader_o.df_robot[f'i_{i}'].to_numpy()*loader_o.df_robot[f'v_{i}'].to_numpy()) for i in [2, 5, 6, 7, 8, 9, 10, 11]]))
print('Closed Loop:', sum([np.average(loader_c.df_robot[f'i_{i}'].to_numpy()*loader_c.df_robot[f'v_{i}'].to_numpy()) for i in [2, 5, 6, 7, 8, 9, 10, 11]]))