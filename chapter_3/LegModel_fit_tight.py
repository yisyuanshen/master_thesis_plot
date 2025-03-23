import coef
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MultipleLocator

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


class LegModel:
    """Legwheel kinematics model."""
    def __init__(self):
        self.R = 0.1
        self.n_HF = np.deg2rad(130)
        self.n_BC = np.deg2rad(101)
        self.theta = np.deg2rad(17)
        self.beta = np.deg2rad(0)
    
    @staticmethod
    def rot_matrix(beta):
        return np.array([
            [np.cos(beta), -np.sin(beta)],
            [np.sin(beta),  np.cos(beta)]
        ])
    
    def forward(self, theta, beta):
        self.theta = theta
        self.beta = beta
        rot_beta = self.rot_matrix(beta)
        self.O = np.array([0, 0])
        
        points = [
            "A_r", "A_l", "B_r", "B_l", "C_r", "C_l", "D_r", "D_l", 
            "E", "F_r", "F_l", "G", "H_r", "H_l", "U_r", "U_l", "L_r", "L_l"
        ]
        for pt in points:
            funcs = getattr(coef, f"{pt}_poly")
            value = rot_beta @ np.array([f(theta) for f in funcs])
            setattr(self, pt, value)
    
    def plot(self, ax, rim_color, idx):
        lw = 2

        # Helper to compute angle between two points from a common center
        def get_angle(center, pt):
            return np.rad2deg(np.arctan2(pt[1]-center[1], pt[0]-center[0]))
        
        # Draw arcs for joints
        HF_l_arc = patches.Arc(
            self.U_l, 2*self.R, 2*self.R, angle=0,
            theta1=get_angle(self.U_l, self.H_l),
            theta2=get_angle(self.U_l, self.F_l),
            edgecolor=rim_color, linewidth=lw
        )
        HF_r_arc = patches.Arc(
            self.U_r, 2*self.R, 2*self.R, angle=0,
            theta1=get_angle(self.U_r, self.F_r),
            theta2=get_angle(self.U_r, self.H_r),
            edgecolor=rim_color, linewidth=lw
        )
        FG_l_arc = patches.Arc(
            self.L_l, 2*self.R, 2*self.R, angle=0,
            theta1=get_angle(self.L_l, self.F_l),
            theta2=get_angle(self.L_l, self.G),
            edgecolor=rim_color, linewidth=lw
        )
        FG_r_arc = patches.Arc(
            self.L_r, 2*self.R, 2*self.R, angle=0,
            theta1=get_angle(self.L_r, self.G),
            theta2=get_angle(self.L_r, self.F_r),
            edgecolor=rim_color, linewidth=lw
        )
        for arc in (HF_l_arc, HF_r_arc, FG_l_arc, FG_r_arc):
            ax.add_patch(arc)
        
        # Add markers or support lines by joint index
        if idx == 0:
            ax.add_patch(patches.Circle(self.H_l, 0.007, edgecolor='gray', facecolor='lightgray', zorder=10))
        elif idx == 1:
            ax.add_patch(patches.Circle(self.F_l, 0.007, edgecolor='gray', facecolor='lightgray', zorder=10))
        elif idx == 2:
            # Choose support color based on rim_color
            sup_color = {'gray': '#00BFFF', 'silver': '#1E90FF', 'lightgray': '#4169E1'}.get(rim_color, 'blue')
            ax.add_patch(patches.FancyArrowPatch(self.H_l, self.U_l, color=sup_color,
                                                  linewidth=1.5, arrowstyle='-', linestyle=':', zorder=10))
            ax.add_patch(patches.FancyArrowPatch(self.F_l, self.U_l, color=sup_color,
                                                  linewidth=1.5, arrowstyle='-', linestyle=':', zorder=10))
            ax.add_patch(patches.Circle(self.U_l, 0.007, edgecolor='gray', facecolor='lightgray', zorder=20))
        elif idx == 3:
            sup_color = {'gray': '#32CD32', 'silver': '#3CB371', 'lightgray': '#2E8B57'}.get(rim_color, 'green')
            ax.add_patch(patches.FancyArrowPatch(self.F_l, self.L_l, color=sup_color,
                                                  linewidth=1.5, arrowstyle='-', linestyle=':', zorder=10))
            ax.add_patch(patches.FancyArrowPatch(self.G, self.L_l, color=sup_color,
                                                  linewidth=1.5, arrowstyle='-', linestyle=':', zorder=10))
            ax.add_patch(patches.Circle(self.L_l, 0.007, edgecolor='gray', facecolor='lightgray', zorder=20))
        elif idx == 4:
            ax.add_patch(patches.Circle(self.G, 0.007, edgecolor='gray', facecolor='lightgray', zorder=10))
        
        titles = {0: 'H', 1: 'F', 2: 'U', 3: 'L', 4: 'G'}
        if idx in titles:
            ax.set_title(titles[idx])
        
        ax.set_aspect('equal')
        ax.set_xlim(-0.22, 0.22)
        ax.set_ylim(-0.34, 0.14)
        ax.set_xticks([])
        ax.set_yticks([])


def compute_trajectories(model, theta_range_deg):
    traj_names = ['F_r', 'F_l', 'G', 'H_r', 'H_l', 'U_r', 'U_l', 'L_r', 'L_l']
    trajectories = {name: [] for name in traj_names}
    for theta_deg in theta_range_deg:
        theta_rad = np.deg2rad(theta_deg)
        model.forward(theta_rad, 0)
        for name in traj_names:
            trajectories[name].append(getattr(model, name))
    # Convert list of arrays to a single NumPy array per trajectory
    for name in trajectories:
        trajectories[name] = np.array(trajectories[name])
    return trajectories


def main():
    legmodel = LegModel()
    theta_vals_deg = np.arange(17, 140, 1)
    trajectories = compute_trajectories(legmodel, theta_vals_deg)
    
    # Define pose states and mapping for subplots and gradient colormaps
    pose_states = [(30, 'gray'), (75, 'silver'), (110, 'lightgray')]
    subplot_info = {
        0: ('H_l', 'red'),
        1: ('F_l', 'orange'),
        2: ('U_l', 'blue'),
        3: ('L_l', 'green'),
        4: ('G',   'purple')
    }
    gradient_cmaps = {
        0: LinearSegmentedColormap.from_list("red_cmap", ['#ffb3b3', '#ff0000']),
        1: LinearSegmentedColormap.from_list("orange_cmap", ['#ffd2a3', '#ff8c00']),
        2: LinearSegmentedColormap.from_list("blue_cmap", ['#9eced6', '#00008b']),
        3: LinearSegmentedColormap.from_list("green_cmap", ['#80d080', '#006400']),
        4: LinearSegmentedColormap.from_list("purple_cmap", ['#dcd6f0', '#800080'])
    }
    
    # Create figure with a GridSpec layout
    fig = plt.figure(figsize=(9, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.8], hspace=0)
    
    # Top section: 1x5 subplots
    gs_top = gs[0].subgridspec(1, 5, wspace=0.01)
    axs_top = [fig.add_subplot(gs_top[i]) for i in range(5)]
    
    # Draw each top subplot with static poses and trajectory (gradient)
    for idx, ax in enumerate(axs_top):
        # Plot static poses
        for angle_deg, rim_color in pose_states:
            legmodel.forward(np.deg2rad(angle_deg), 0)
            legmodel.plot(ax, rim_color, idx)
        
        # Plot trajectory for the joint with gradient
        traj_attr, _ = subplot_info[idx]
        traj = trajectories[traj_attr]
        points = traj.reshape(-1, 1, 2)
        segments = np.concatenate([points[1:], points[:-1]], axis=1)
        cmap = gradient_cmaps[idx]
        norm = Normalize(vmin=theta_vals_deg.min(), vmax=theta_vals_deg.max())
        theta_mid = (theta_vals_deg[:-1] + theta_vals_deg[1:]) / 2.0
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2, zorder=5)
        lc.set_array(theta_mid)
        ax.add_collection(lc)
        
        ax.set_aspect('equal')
        ax.set_xlim(-0.22, 0.22)
        ax.set_ylim(-0.34, 0.14)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Lower section: Plot distances to the origin vs. theta
    ax_bottom = fig.add_subplot(gs[1])
    joint_info = {
        'H': ('H_l', 'red'),
        'F': ('F_l', 'orange'),
        'U': ('U_l', 'blue'),
        'L': ('L_l', 'green'),
        'G': ('G',   'purple')
    }
    for joint, (attr, color) in joint_info.items():
        traj = trajectories[attr]
        lengths = np.linalg.norm(traj, axis=1)
        ax_bottom.plot(theta_vals_deg, lengths, label=joint, color=color)
    
    ax_bottom.set_xlabel(r'$\theta$ (deg)')
    ax_bottom.set_ylabel(r'Distance (m)')
    ax_bottom.set_title(r'Distance to Origin ($O$)', pad=10)
    ax_bottom.title.set_position([0.5, 1.15])
    ax_bottom.grid(True)
    # Set y-axis major ticks with a spacing of 0.05
    ax_bottom.yaxis.set_major_locator(MultipleLocator(0.05))
    ax_bottom.legend()
    
    # Adjust overall figure layout
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.tight_layout()
    fig.canvas.draw()
    
    # Adjust top subplots to align with the bottom subplot’s horizontal span
    renderer = fig.canvas.get_renderer()
    bottom_bbox = ax_bottom.get_tightbbox(renderer=renderer)
    bottom_bbox_in_fig = bottom_bbox.transformed(fig.transFigure.inverted())
    left_edge_in_fig  = bottom_bbox_in_fig.x0
    right_edge_in_fig = bottom_bbox_in_fig.x1
    
    top_leftmost = min(ax.get_position().x0 for ax in axs_top)
    top_rightmost = max(ax.get_position().x0 + ax.get_position().width for ax in axs_top)
    top_total_width = top_rightmost - top_leftmost
    desired_width = right_edge_in_fig - left_edge_in_fig
    scale = desired_width / top_total_width
    
    for ax in axs_top:
        pos = ax.get_position()
        new_left = left_edge_in_fig + scale * (pos.x0 - top_leftmost)
        new_width = scale * pos.width
        ax.set_position([new_left, pos.y0, new_width, pos.height])
    
    plt.show()


if __name__ == '__main__':
    main()
