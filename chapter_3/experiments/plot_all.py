#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRF 3x4 Grid Plotter — suptitle, per-subplot axis labels, framed bottom legend
(with SIM Fy measured negated, lighter legend frame)
-------------------------------------------------------------------------------
Layout (left→right): walk | trot | wlw
Layout (top→bottom): sim Fx | sim Fy | real Fx | real Fy
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from DataLoader import DataLoader

# -----------------------------
# Matplotlib style
# -----------------------------
plt.rcParams.update({
    'text.usetex': True,            # Disable if LaTeX is not installed
    'font.family': 'Times New Roman',
    'font.size': 16,
    'axes.linewidth': 1.2,
    'legend.frameon': False,        # We'll explicitly style the bottom figure legend
})

SAMPLE_RATE = 1000  # Hz
RF_IDX = 1
LH_IDX = 3

# Color & linestyle mapping (four distinct colors)
COL_RF_MEAS = 'C0'   # RF measured
COL_RF_EST  = 'C1'   # RF estimated
COL_LH_MEAS = 'C2'   # LH measured
COL_LH_EST  = 'C3'   # LH estimated
LS_MEAS = '-'        # measured line style
LS_EST  = '--'        # estimated line style
LW = 1.4

# Friendly gait display names
GAIT_NAME = {'walk': 'walking', 'trot': 'trotting', 'wlw': 'wheel-like-walking'}

# -----------------------------
# Dataset configurations (from your original scripts)
# -----------------------------
cfg = {
    # WALK
    ('walk', 'sim'): dict(
        sim=True,
        robot_file='exp_data_final/sim_walk_h20_v10_open.csv',
        force_file='exp_data_final/sim_walk_h20_v10_open_force.csv',
        vicon_file=None,
        start=15500, end=18000, trigger=None
    ),
    ('walk', 'real'): dict(
        sim=False,
        robot_file='exp_data_final/0617_walk_h20_v10_open.csv',
        force_file=None,
        vicon_file='exp_data_final/0617_walk_h20_v10_open_vicon.csv',
        start=14500, end=18500, trigger=6282 + 818 - 787  # 6313
    ),
    # TROT
    ('trot', 'sim'): dict(
        sim=True,
        robot_file='exp_data_final/sim_trot_h20_v45_open.csv',
        force_file='exp_data_final/sim_trot_h20_v45_open_force.csv',
        vicon_file=None,
        start=7800, end=8600, trigger=None
    ),
    ('trot', 'real'): dict(
        sim=False,
        robot_file='exp_data_final/0617_trot_h20_v45_open.csv',
        force_file=None,
        vicon_file='exp_data_final/0617_trot_h20_v45_open_vicon.csv',
        start=5550, end=6550, trigger=5218 + 10  # 5228
    ),
    # WLW
    ('wlw', 'sim'): dict(
        sim=True,
        robot_file='exp_data_final/sim_wlw_h16_v12_est.csv',
        force_file='exp_data_final/sim_wlw_h16_v12_est_force.csv',
        vicon_file=None,
        start=13000, end=15500, trigger=None
    ),
    ('wlw', 'real'): dict(
        sim=False,
        robot_file='exp_data_final/0626_wlw_h16_v12_open.csv',
        force_file=None,
        vicon_file='exp_data_final/0626_wlw_h16_v12_open_vicon.csv',
        start=13200, end=16200, trigger=None
    ),
}

def _best_attr(loader, base: str, comp_pref: str):
    """
    Robust attribute fetcher.
    Tries the preferred component first (e.g., 'y'), then falls back to 'z' if absent.
    This is to tolerate datasets where lateral/forward axis naming differs.
    """
    candidates = [comp_pref]
    if comp_pref == 'y':
        candidates.append('z')  # fallback if datasets used 'z' instead of 'y'
    for c in candidates:
        name = f'{base}_{c}'
        if hasattr(loader, name):
            return getattr(loader, name), c
    raise AttributeError(f'Missing attribute for {base}_{{{"/".join(candidates)}}}')

# -----------------------------
# Helper to load and preprocess one dataset
# -----------------------------
def load_dataset(*, sim: bool, robot_file: str, force_file: str | None, vicon_file: str | None,
                 start: int, end: int, trigger):
    loader = DataLoader(sim=sim)
    loader.cutoff_freq = 20
    loader.trigger_idx = trigger

    loader.load_robot_data(robot_file, start_idx=start, end_idx=end)
    if sim:
        assert force_file is not None, "Sim dataset requires sim force file."
        loader.load_sim_force_data(force_file, start_idx=start, end_idx=end)
    else:
        assert vicon_file is not None, "Real dataset requires Vicon file."
        loader.load_vicon_data(vicon_file, start_idx=start, end_idx=end)

    loader.data_process()

    # Time bases
    time_robot = np.arange(loader.df_robot.shape[0]) / SAMPLE_RATE
    if sim:
        time_meas = np.arange(loader.df_sim_force.shape[0]) / SAMPLE_RATE
    else:
        time_meas = np.arange(loader.df_vicon.shape[0]) / SAMPLE_RATE

    return loader, time_meas, time_robot

# -----------------------------
# Retrieve all six loaders
# -----------------------------
datasets = {}
for gait in ('walk', 'trot', 'wlw'):
    for domain in ('sim', 'real'):
        params = cfg[(gait, domain)]
        loader, t_meas, t_robot = load_dataset(**params)
        datasets[(gait, domain)] = (loader, t_meas, t_robot)

# -----------------------------
# Accessor for measured & estimated components with sign conventions
# -----------------------------
def get_components(loader, domain: str, comp: str):
    """
    comp in {'x','y'}
    - REAL (Vicon): measured = -vicon_force_{comp}
    - SIM:          measured =  sim_force_{comp};  BUT if comp=='y', measured *= -1  (per your request)
    - Estimated always from state_force_{comp}
    If the requested '{comp}' is not available, this function falls back to 'z' for robustness.
    Returns: (measured_array, estimated_array, actual_comp_char)
    """
    if comp not in ('x', 'y'):
        raise ValueError("comp must be 'x' or 'y'")

    # measured
    if domain == 'real':
        meas_raw, used_comp = _best_attr(loader, 'vicon_force', comp)
        meas = -meas_raw
    elif domain == 'sim':
        meas, used_comp = _best_attr(loader, 'sim_force', comp)
        if comp == 'y':
            meas = -meas  # NEGATE SIM Fy measured
    else:
        raise ValueError("domain must be 'real' or 'sim'")

    # estimated
    try:
        est, _ = _best_attr(loader, 'state_force', used_comp)
    except AttributeError:
        est, _ = _best_attr(loader, 'state_force', comp)

    return meas, est, used_comp

def _title_for(domain: str, comp_char: str, gait_key: str) -> str:
    mode = 'simulation' if domain == 'sim' else 'real robot'
    gait = GAIT_NAME[gait_key]
    disp_comp = 'x' if comp_char == 'x' else 'y'
    if mode == 'simulation':
        return rf'\textbf{{$F_{{{disp_comp}}}$ in {mode} ({gait})}}'
    else:
        return rf'\textbf{{$F_{{{disp_comp}}}$ on {mode} ({gait})}}'

# ============================================================================
# (新增) 可客製 12 張子圖的 ticks；預設為空，不改變原行為
# ============================================================================
# 需要的額外 import（僅用於刻度設定）
from matplotlib.ticker import MultipleLocator, FixedLocator, FormatStrFormatter

# TICK 設定表：
#   key: (domain, comp, gait)
#       - domain ∈ {'sim','real'}，comp ∈ {'x','y'}，gait ∈ {'walk','trot','wlw'}
#       - 支援萬用鍵 '*'（通用→半通用→最專一 的優先序合併）
#   欄位：
#       x_ticks / y_ticks : 明確指定主刻度位置 (list/np.array)
#       x_major / y_major : 等距主刻度間距 (float)
#       x_minor / y_minor : 等距次刻度間距 (float)
#       xlim / ylim       : (min, max)
#       x_format/y_format : 例 '%.2f'
TICK_CFG: dict[tuple[str, str, str], dict] = {
    ('sim','x','walk'):dict(y_major=20, ylim=(-40, 40),
                            x_ticks=[0,0.5,1,1.5,2,2.5],
                            x_format='%.1f'),
    ('sim','y','walk'):dict(y_major=50, ylim=(-50, 150),
                            x_ticks=[0,0.5,1,1.5,2,2.5],
                            x_format='%.1f'),
    ('real','x','walk'):dict(y_major=30, ylim=(-30, 60),
                            x_ticks=[0,1,2,3,4],
                            x_format='%.0f'),
    ('real','y','walk'):dict(y_major=50, ylim=(-50, 150),
                            x_ticks=[0,1,2,3,4],
                            x_format='%.0f'),
    
    ('sim','x','trot'):dict(y_major=15, ylim=(-30, 30),
                            x_ticks=[0,0.2,0.4,0.6,0.8],
                            x_format='%.1f'),
    ('sim','y','trot'):dict(y_major=50, ylim=(-50, 150),
                            x_ticks=[0,0.2,0.4,0.6,0.8],
                            x_format='%.1f'),
    ('real','x','trot'):dict(y_major=35, ylim=(-70, 70),
                            x_ticks=[0,0.2,0.4,0.6,0.8,1],
                            x_format='%.1f'),
    ('real','y','trot'):dict(y_major=80, ylim=(-80, 240),
                            x_ticks=[0,0.2,0.4,0.6,0.8,1],
                            x_format='%.1f'),
    
    ('sim','x','wlw'):dict(y_major=50, ylim=(-100, 100),
                            x_ticks=[0,0.5,1,1.5,2,2.5],
                            x_format='%.1f'),
    ('sim','y','wlw'):dict(y_major=60, ylim=(-60, 180),
                            x_ticks=[0,0.5,1,1.5,2,2.5],
                            x_format='%.1f'),
    ('real','x','wlw'):dict(y_major=35, ylim=(-70, 70),
                            x_ticks=[0,1,2,3,4],
                            x_format='%.0f'),
    ('real','y','wlw'):dict(y_major=50, ylim=(-50, 150),
                            x_ticks=[0,1,2,3,4],
                            x_format='%.0f'),
    
    # 預設為空：完全不覆寫 → 行為與原程式一致
    # --- 以下提供示例，若要啟用請取消註解 ---
    # ('*', '*', '*'): dict(x_major=0.2, y_major=50),  # 全域例子
    # ('sim', 'y', 'wlw'): dict(y_major=20, ylim=(-160, 160),
    #                           x_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
    #                           x_format='%.2f'),
    # ('real', 'x', 'trot'): dict(xlim=(0.0, 1.2), x_minor=0.05),
}

def _resolve_tick_cfg(domain: str, comp: str, gait: str):
    """通用→半通用→最專一合併（後者覆蓋前者）。若皆無則回傳 None。"""
    order = [
        ('*', '*', '*'),
        (domain, '*', '*'),
        ('*', comp, '*'),
        ('*', '*', gait),
        (domain, comp, '*'),
        (domain, '*', gait),
        ('*', comp, gait),
        (domain, comp, gait),
    ]
    merged = {}
    for key in order:
        if key in TICK_CFG:
            merged.update(TICK_CFG[key])
    return merged or None

def _apply_ticks(ax, cfg: dict | None):
    """將設定套用到單一 Axes。若 cfg 為 None 則不動作。"""
    if not cfg:
        return

    # 範圍
    if 'xlim' in cfg: ax.set_xlim(*cfg['xlim'])
    if 'ylim' in cfg: ax.set_ylim(*cfg['ylim'])

    # 主刻度（固定或等距其一）
    if 'x_ticks' in cfg:
        ax.xaxis.set_major_locator(FixedLocator(cfg['x_ticks']))
    elif 'x_major' in cfg:
        ax.xaxis.set_major_locator(MultipleLocator(cfg['x_major']))

    if 'y_ticks' in cfg:
        ax.yaxis.set_major_locator(FixedLocator(cfg['y_ticks']))
    elif 'y_major' in cfg:
        ax.yaxis.set_major_locator(MultipleLocator(cfg['y_major']))

    # 次刻度
    if 'x_minor' in cfg:
        ax.xaxis.set_minor_locator(MultipleLocator(cfg['x_minor']))
    if 'y_minor' in cfg:
        ax.yaxis.set_minor_locator(MultipleLocator(cfg['y_minor']))

    # 格式
    if 'x_format' in cfg:
        ax.xaxis.set_major_formatter(FormatStrFormatter(cfg['x_format']))
    if 'y_format' in cfg:
        ax.yaxis.set_major_formatter(FormatStrFormatter(cfg['y_format']))

# -----------------------------
# Plot grid: 4 rows x 3 cols
# Rows: sim Fx | sim Fy | real Fx | real Fy
# Cols: walk | trot | wlw
# -----------------------------
fig, axs = plt.subplots(4, 3, figsize=(16, 9), sharex=False, sharey=False)
fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.15, wspace=0.35, hspace=1.1)
fig.suptitle(r'\textbf{Contact Force Estimation Experimental Results}', fontsize=20, y=0.98)

# Plotter for one cell
from matplotlib.ticker import MaxNLocator  # (保留你原本的重複 import，不影響功能)
def plot_cell(ax, loader, t_meas, t_robot, domain: str, comp: str, title: str, gait_key: str):
    meas, est, _ = get_components(loader, domain, comp)

    # RF (meas/est)
    ax.plot(t_meas, meas[RF_IDX], color=COL_RF_MEAS, linestyle=LS_MEAS, linewidth=LW, label='Measured')
    ax.plot(t_robot, est[RF_IDX], color=COL_RF_EST,  linestyle=LS_EST,  linewidth=LW, label='Estimated')
    # LH (meas/est)
    # ax.plot(t_meas, meas[LH_IDX], color=COL_LH_MEAS, linestyle=LS_MEAS, linewidth=LW, label='Measured (Left Hind)')
    # ax.plot(t_robot, est[LH_IDX], color=COL_LH_EST,  linestyle=LS_EST,  linewidth=LW, label='Estimated (Left Hind)')

    # Titles & axes
    ax.set_title(title, fontsize=16, pad=8)
    ax.set_xlabel(r'\textbf{Time (s)}', fontsize=16)
    ax.set_ylabel(r'\textbf{Force (N)}', fontsize=16)
    ax.grid(True, linewidth=0.7, alpha=0.6)

    # Ensure ticks are present on both axes
    ax.tick_params(axis='both', which='both', direction='out', length=4, width=1.0)

    # === 關鍵：解析並套用客製化 ticks，未指定者回退到原本自動規則 ===
    tick_cfg = _resolve_tick_cfg(domain, comp, gait_key)

    # 若沒有對 x/y 提供明確自訂，則退回原本的「自動」行為
    tmax = max(float(t_meas[-1]) if len(t_meas) else 0.0,
               float(t_robot[-1]) if len(t_robot) else 0.0)

    has_custom_x = tick_cfg and any(k in tick_cfg for k in ('x_ticks', 'x_major', 'xlim'))
    has_custom_y = tick_cfg and any(k in tick_cfg for k in ('y_ticks', 'y_major', 'ylim'))

    if (tmax > 0) and (not has_custom_x):
        nticks = 5
        step = max(0.1, round(tmax / (nticks - 1), 2))
        ticks = np.arange(0, tmax + 1e-9, step)
        ax.set_xticks(ticks)

    if not has_custom_y:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # 最後套用自訂（如有）
    _apply_ticks(ax, tick_cfg)

# Fill the grid
for j, gait in enumerate(('walk', 'trot', 'wlw')):
    # sim Fx
    loader, t_meas, t_robot = datasets[(gait, 'sim')]
    plot_cell(axs[0, j], loader, t_meas, t_robot, 'sim', 'x', _title_for('sim', 'x', gait), gait)
    # sim Fy
    plot_cell(axs[1, j], loader, t_meas, t_robot, 'sim', 'y', _title_for('sim', 'y', gait), gait)

    # real Fx
    loader, t_meas, t_robot = datasets[(gait, 'real')]
    plot_cell(axs[2, j], loader, t_meas, t_robot, 'real', 'x', _title_for('real', 'x', gait), gait)
    # real Fy
    plot_cell(axs[3, j], loader, t_meas, t_robot, 'real', 'y', _title_for('real', 'y', gait), gait)

# Bottom legend (lighter border similar to original)
handles, labels = axs[-1, -1].get_legend_handles_labels()
order = ['Measured', 'Estimated', 'Measured (Left Hind)', 'Estimated (Left Hind)']
label_to_handle = {lab: h for h, lab in zip(handles, labels)}
ordered_handles = [label_to_handle[k] for k in order if k in label_to_handle]

leg = fig.legend(
    ordered_handles,
    order,
    loc='lower center',
    ncol=4,
    fontsize=16,
    frameon=True,
    facecolor='white',
    # edgecolor='0.6',  # lighter border
    framealpha=None,
    columnspacing=1.6,
    handlelength=2.3,
    borderaxespad=0.4,
    bbox_to_anchor=(0.5, 0.01)
)
# further lighten and square the frame to approximate older Matplotlib look
frame = leg.get_frame()
# frame.set_linewidth(0.8)
# frame.set_edgecolor('0.6')
# frame.set_facecolor('white')
plt.savefig('force_est_LF.pdf', format='pdf', bbox_inches='tight')
plt.show()
