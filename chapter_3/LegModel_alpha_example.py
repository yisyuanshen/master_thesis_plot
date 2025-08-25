import coef
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.collections as mcoll

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 22


def truncate_colormap(cmap, min_val=0.3, max_val=1.0, n=10000):
    new_colors = cmap(np.linspace(min_val, max_val, n))
    return plt.cm.colors.LinearSegmentedColormap.from_list('truncated', new_colors)


def draw_gradient_arc(ax, center, width, height, theta1, theta2, linewidth=2.5, cmap_name='Purples', num_points=10000, reversed=False):
    cmap_full = plt.get_cmap(cmap_name)
    if reversed:
        truncated_cmap = truncate_colormap(cmap_full, 0, 0.5)  # light-to-dark if reversed
    else:
        truncated_cmap = truncate_colormap(cmap_full, 0.5, 1)  # light-to-dark otherwise
    theta = np.linspace(np.deg2rad(theta1), np.deg2rad(theta2), num_points)
    x = center[0] + (width/2) * np.cos(theta)
    y = center[1] + (height/2) * np.sin(theta)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = mcoll.LineCollection(segments, cmap=truncated_cmap, linewidth=linewidth)
    lc.set_array(np.linspace(0, 1, len(segments)))
    ax.add_collection(lc)


class LegModel:
    def __init__(self):
        self.R = 0.1
        self.r = 0.0125
        self.radius_outer = self.R + self.r
        self.radius_inner = self.R - self.r

        self.n_HF = np.deg2rad(130)
        self.n_BC = np.deg2rad(101)

        self.theta = np.deg2rad(17)
        self.beta = np.deg2rad(0)

    def rot_matrix(self, beta):
        return np.array([[np.cos(beta), -np.sin(beta)],
                         [np.sin(beta),  np.cos(beta)]])

    def forward(self, theta, beta):
        self.theta = theta
        self.beta = beta
        rot_beta = self.rot_matrix(self.beta)

        # The following assumes that coef.A_r_poly etc. are defined in module coef.
        self.O   = np.array([0, 0])
        self.A_r = rot_beta @ np.array([poly(theta) for poly in coef.A_r_poly])
        self.A_l = rot_beta @ np.array([poly(theta) for poly in coef.A_l_poly])
        self.B_r = rot_beta @ np.array([poly(theta) for poly in coef.B_r_poly])
        self.B_l = rot_beta @ np.array([poly(theta) for poly in coef.B_l_poly])
        self.C_r = rot_beta @ np.array([poly(theta) for poly in coef.C_r_poly])
        self.C_l = rot_beta @ np.array([poly(theta) for poly in coef.C_l_poly])
        self.D_r = rot_beta @ np.array([poly(theta) for poly in coef.D_r_poly])
        self.D_l = rot_beta @ np.array([poly(theta) for poly in coef.D_l_poly])
        self.E   = rot_beta @ np.array([poly(theta) for poly in coef.E_poly])
        self.F_r = rot_beta @ np.array([poly(theta) for poly in coef.F_r_poly])
        self.F_l = rot_beta @ np.array([poly(theta) for poly in coef.F_l_poly])
        self.G   = rot_beta @ np.array([poly(theta) for poly in coef.G_poly])
        self.H_r = rot_beta @ np.array([poly(theta) for poly in coef.H_r_poly])
        self.H_l = rot_beta @ np.array([poly(theta) for poly in coef.H_l_poly])
        self.U_r = rot_beta @ np.array([poly(theta) for poly in coef.U_r_poly])
        self.U_l = rot_beta @ np.array([poly(theta) for poly in coef.U_l_poly])
        self.L_r = rot_beta @ np.array([poly(theta) for poly in coef.L_r_poly])
        self.L_l = rot_beta @ np.array([poly(theta) for poly in coef.L_l_poly])

    def plot_on_ax(self, ax):
        # Clear the axis and use it for plotting the leg model
        ax.clear()
        linewidth = 1.5

        # Draw axis arrows
        x_axis = patches.FancyArrowPatch([-0.145, 0], [0.145, 0],
                                         color='silver', linewidth=2,
                                         arrowstyle='-|>', mutation_scale=15)
        y_axis = patches.FancyArrowPatch([0, -0.145], [0, 0.145],
                                         color='silver', linewidth=2,
                                         arrowstyle='-|>', mutation_scale=15)
        ax.add_patch(x_axis)
        ax.add_patch(y_axis)

        # Draw joints (as circles)
        O   = patches.Circle(self.O, 0.003, edgecolor='gray', facecolor='lightgray')
        A_r = patches.Circle(self.A_r, 0.003, edgecolor='gray', facecolor='lightgray')
        A_l = patches.Circle(self.A_l, 0.003, edgecolor='gray', facecolor='lightgray')
        B_r = patches.Circle(self.B_r, 0.003, edgecolor='gray', facecolor='lightgray')
        B_l = patches.Circle(self.B_l, 0.003, edgecolor='gray', facecolor='lightgray')
        C_r = patches.Circle(self.C_r, 0.003, edgecolor='gray', facecolor='lightgray')
        C_l = patches.Circle(self.C_l, 0.003, edgecolor='gray', facecolor='lightgray')
        D_r = patches.Circle(self.D_r, 0.003, edgecolor='gray', facecolor='lightgray')
        D_l = patches.Circle(self.D_l, 0.003, edgecolor='gray', facecolor='lightgray')
        E   = patches.Circle(self.E,   0.003, edgecolor='gray', facecolor='lightgray')
        F_r = patches.Circle(self.F_r, 0.003, edgecolor='gray', facecolor='lightgray')
        F_l = patches.Circle(self.F_l, 0.003, edgecolor='gray', facecolor='lightgray')
        G   = patches.Circle(self.G,   0.003, edgecolor='gray', facecolor='lightgray')
        H_r = patches.Circle(self.H_r, 0.003, edgecolor='gray', facecolor='lightgray')
        H_l = patches.Circle(self.H_l, 0.003, edgecolor='gray', facecolor='lightgray')
        U_r = patches.Circle(self.U_r, 0.003, edgecolor='gray', facecolor='lightgray')
        U_l = patches.Circle(self.U_l, 0.003, edgecolor='gray', facecolor='lightgray')
        L_r = patches.Circle(self.L_r, 0.003, edgecolor='gray', facecolor='lightgray')
        L_l = patches.Circle(self.L_l, 0.003, edgecolor='gray', facecolor='lightgray')

        # HF_l (left leg hinge) arcs and arrows
        HF_l_outer = patches.Arc(self.U_l, 2*self.radius_outer, 2*self.radius_outer, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.H_l[1]-self.U_l[1],
                                                              self.H_l[0]-self.U_l[0])),
                                 theta2=np.rad2deg(np.arctan2(self.F_l[1]-self.U_l[1],
                                                              self.F_l[0]-self.U_l[0])),
                                 edgecolor='red', facecolor='red', linewidth=linewidth)
        HF_l_inner = patches.Arc(self.U_l, 2*self.radius_inner, 2*self.radius_inner, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.H_l[1]-self.U_l[1],
                                                              self.H_l[0]-self.U_l[0])),
                                 theta2=np.rad2deg(np.arctan2(self.F_l[1]-self.U_l[1],
                                                              self.F_l[0]-self.U_l[0])),
                                 edgecolor='red', linewidth=linewidth)
        HF_l_upper = patches.FancyArrowPatch((self.H_l-self.U_l)*self.radius_outer/self.R+self.U_l,
                                             (self.H_l-self.U_l)*self.radius_inner/self.R+self.U_l,
                                             color='red', linewidth=linewidth,
                                             arrowstyle='-', shrinkA=0, shrinkB=0)
        HF_l_lower = patches.Arc(self.F_l, 2*self.r, 2*self.r, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.F_l[1]-self.U_l[1],
                                                              self.F_l[0]-self.U_l[0])),
                                 theta2=np.rad2deg(np.arctan2(self.F_l[1]-self.U_l[1],
                                                              self.F_l[0]-self.U_l[0]))-180,
                                 edgecolor='red', linewidth=linewidth)
        for patch in [HF_l_outer, HF_l_inner, HF_l_upper, HF_l_lower]:
            ax.add_patch(patch)

        # HF_r (right leg hinge) arcs and arrows
        HF_r_outer = patches.Arc(self.U_r, 2*self.radius_outer, 2*self.radius_outer, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.F_r[1]-self.U_r[1],
                                                              self.F_r[0]-self.U_r[0])),
                                 theta2=np.rad2deg(np.arctan2(self.H_r[1]-self.U_r[1],
                                                              self.H_r[0]-self.U_r[0])),
                                 edgecolor='green', linewidth=linewidth)
        HF_r_inner = patches.Arc(self.U_r, 2*self.radius_inner, 2*self.radius_inner, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.F_r[1]-self.U_r[1],
                                                              self.F_r[0]-self.U_r[0])),
                                 theta2=np.rad2deg(np.arctan2(self.H_r[1]-self.U_r[1],
                                                              self.H_r[0]-self.U_r[0])),
                                 edgecolor='green', linewidth=linewidth)
        HF_r_upper = patches.FancyArrowPatch((self.H_r-self.U_r)*self.radius_inner/self.R+self.U_r,
                                             (self.H_r-self.U_r)*self.radius_outer/self.R+self.U_r,
                                             color='green', linewidth=linewidth,
                                             arrowstyle='-', shrinkA=0, shrinkB=0)
        HF_r_lower = patches.Arc(self.F_r, 2*self.r, 2*self.r, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.F_r[1]-self.U_r[1],
                                                              self.F_r[0]-self.U_r[0]))-180,
                                 theta2=np.rad2deg(np.arctan2(self.F_r[1]-self.U_r[1],
                                                              self.F_r[0]-self.U_r[0])),
                                 edgecolor='green', linewidth=linewidth)
        for patch in [HF_r_outer, HF_r_inner, HF_r_upper, HF_r_lower]:
            ax.add_patch(patch)
        
        # FG_r (front right leg) arcs and arrows
        FG_r_outer = patches.Arc(self.L_r, 2*self.radius_outer, 2*self.radius_outer, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.G[1]-self.L_r[1],
                                                              self.G[0]-self.L_r[0])),
                                 theta2=np.rad2deg(np.arctan2(self.F_r[1]-self.L_r[1],
                                                              self.F_r[0]-self.L_r[0])),
                                 edgecolor='orange', linewidth=linewidth)
        FG_r_inner = patches.Arc(self.L_r, 2*self.radius_inner, 2*self.radius_inner, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.G[1]-self.L_r[1],
                                                              self.G[0]-self.L_r[0])),
                                 theta2=np.rad2deg(np.arctan2(self.F_r[1]-self.L_r[1],
                                                              self.F_r[0]-self.L_r[0])),
                                 edgecolor='orange', linewidth=linewidth)
        FG_r_upper = patches.Arc(self.F_r, 2*self.r, 2*self.r, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.F_r[1]-self.L_r[1],
                                                              self.F_r[0]-self.L_r[0])),
                                 theta2=np.rad2deg(np.arctan2(self.F_r[1]-self.L_r[1],
                                                              self.F_r[0]-self.L_r[0]))-180,
                                 edgecolor='orange', linewidth=linewidth)
        FG_r_lower = patches.Arc(self.G, 2*self.r, 2*self.r, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.G[1]-self.L_r[1],
                                                              self.G[0]-self.L_r[0]))-180,
                                 theta2=np.rad2deg(np.arctan2(self.G[1]-self.L_r[1],
                                                              self.G[0]-self.L_r[0])),
                                 edgecolor='orange', linewidth=linewidth)
        for patch in [FG_r_outer, FG_r_inner, FG_r_upper, FG_r_lower]:
            ax.add_patch(patch)

        # FG_l (front left leg) arcs and arrows
        FG_l_outer = patches.Arc(self.L_l, 2*self.radius_outer, 2*self.radius_outer, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.F_l[1]-self.L_l[1],
                                                              self.F_l[0]-self.L_l[0])),
                                 theta2=np.rad2deg(np.arctan2(self.G[1]-self.L_l[1],
                                                              self.G[0]-self.L_l[0])),
                                 edgecolor='blue', linewidth=linewidth)
        FG_l_inner = patches.Arc(self.L_l, 2*self.radius_inner, 2*self.radius_inner, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.F_l[1]-self.L_l[1],
                                                              self.F_l[0]-self.L_l[0])),
                                 theta2=np.rad2deg(np.arctan2(self.G[1]-self.L_l[1],
                                                              self.G[0]-self.L_l[0])),
                                 edgecolor='blue', linewidth=linewidth)
        FG_l_upper = patches.Arc(self.F_l, 2*self.r, 2*self.r, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.F_l[1]-self.L_l[1],
                                                              self.F_l[0]-self.L_l[0]))-180,
                                 theta2=np.rad2deg(np.arctan2(self.F_l[1]-self.L_l[1],
                                                              self.F_l[0]-self.L_l[0])),
                                 edgecolor='blue', linewidth=linewidth)
        FG_l_lower = patches.Arc(self.G, 2*self.r, 2*self.r, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.G[1]-self.L_l[1],
                                                              self.G[0]-self.L_l[0])),
                                 theta2=np.rad2deg(np.arctan2(self.G[1]-self.L_l[1],
                                                              self.G[0]-self.L_l[0]))-180,
                                 edgecolor='blue', linewidth=linewidth)
        for patch in [FG_l_outer, FG_l_inner, FG_l_upper, FG_l_lower]:
            ax.add_patch(patch)

        # Linkages (example: connecting joints with arrows)
        OB_r = patches.FancyArrowPatch(self.O, self.B_r, color='black', linewidth=linewidth,
                                       arrowstyle='-', shrinkA=0, shrinkB=0)
        OB_l = patches.FancyArrowPatch(self.O, self.B_l, color='black', linewidth=linewidth,
                                       arrowstyle='-', shrinkA=0, shrinkB=0)
        AE_r = patches.FancyArrowPatch(self.A_r, self.E, color='black', linewidth=linewidth,
                                       arrowstyle='-', shrinkA=0, shrinkB=0)
        AE_l = patches.FancyArrowPatch(self.A_l, self.E, color='black', linewidth=linewidth,
                                       arrowstyle='-', shrinkA=0, shrinkB=0)
        DC_r = patches.FancyArrowPatch(self.D_r, self.C_r, color='black', linewidth=linewidth,
                                       arrowstyle='-', shrinkA=0, shrinkB=0)
        DC_l = patches.FancyArrowPatch(self.D_l, self.C_l, color='black', linewidth=linewidth,
                                       arrowstyle='-', shrinkA=0, shrinkB=0)
        for patch in [OB_r, OB_l, AE_r, AE_l, DC_r, DC_l]:
            ax.add_patch(patch)

        # Draw gradient arcs with arrows for annotation
        cmap_name = 'coolwarm'
        # Arc 1: centered at O from -90 to 78 degrees
        draw_gradient_arc(ax, center=self.O,
                          width=2*self.R*1.3, height=2*self.R*1.3,
                          theta1=-90, theta2=78, linewidth=5,
                          cmap_name=cmap_name, reversed=False)
        arrow_angle = 80
        arrow_start = (self.O[0] + self.R*1.3 * np.cos(np.deg2rad(arrow_angle)),
                       self.O[1] + self.R*1.3 * np.sin(np.deg2rad(arrow_angle)))
        arrow_end = (self.O[0] + self.R*1.3 * np.cos(np.deg2rad(arrow_angle+1)),
                     self.O[1] + self.R*1.3 * np.sin(np.deg2rad(arrow_angle+1)))
        arrow_color = plt.get_cmap(cmap_name)(0.999)
        alpha_arrow = patches.FancyArrowPatch(arrow_start, arrow_end,
                                               color=arrow_color,
                                               arrowstyle='-|>', mutation_scale=35,
                                               linewidth=0.1)
        ax.add_patch(alpha_arrow)

        # Arc 2: centered at G from -90 to 70 degrees
        draw_gradient_arc(ax, center=self.G,
                          width=2*self.r*1.7, height=2*self.r*1.7,
                          theta1=-90, theta2=70, linewidth=3,
                          cmap_name=cmap_name, reversed=False)
        arrow_angle = 70
        arrow_start = (self.G[0] + self.r*1.72 * np.cos(np.deg2rad(arrow_angle)),
                       self.G[1] + self.r*1.72 * np.sin(np.deg2rad(arrow_angle)))
        arrow_end = (self.G[0] + self.r*1.9 * np.cos(np.deg2rad(arrow_angle+20)),
                     self.G[1] + self.r*1.9 * np.sin(np.deg2rad(arrow_angle+20)))
        arrow_color = plt.get_cmap(cmap_name)(0.999)
        alpha_arrow = patches.FancyArrowPatch(arrow_start, arrow_end,
                                               color=arrow_color,
                                               arrowstyle='-|>', mutation_scale=25,
                                               linewidth=0.1)
        ax.add_patch(alpha_arrow)

        # Arc 3: centered at O from -258 to -90 degrees (reversed colormap)
        draw_gradient_arc(ax, center=self.O,
                          width=2*self.R*1.3, height=2*self.R*1.3,
                          theta1=-258, theta2=-90, linewidth=5,
                          cmap_name=cmap_name, reversed=True)
        arrow_angle = -260
        arrow_start = (self.O[0] + self.R*1.3 * np.cos(np.deg2rad(arrow_angle)),
                       self.O[1] + self.R*1.3 * np.sin(np.deg2rad(arrow_angle)))
        arrow_end = (self.O[0] + self.R*1.3 * np.cos(np.deg2rad(arrow_angle-1)),
                     self.O[1] + self.R*1.3 * np.sin(np.deg2rad(arrow_angle-1)))
        arrow_color = plt.get_cmap(cmap_name)(0.001)
        alpha_arrow = patches.FancyArrowPatch(arrow_start, arrow_end,
                                               color=arrow_color,
                                               arrowstyle='-|>', mutation_scale=35,
                                               linewidth=0.1)
        ax.add_patch(alpha_arrow)
        
        for patch in [O,  F_r, F_l, G, H_r, H_l]:
            ax.add_patch(patch)

        # Set aspect, limits, labels, and title for the main plot
        ax.set_aspect('equal')
        ax.set_xlim(-0.15, 0.15)
        ax.set_ylim(-0.15, 0.15)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(r'Definition of Contact Angle ($\alpha$)', pad=10)
        ax.tick_params(axis='both', labelsize=12)

        # Add a colorbar
        norm = plt.Normalize(-np.pi, np.pi)
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_name), norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.1, pad=0.04)
        cbar.set_label(r'$\alpha$ (rad)', fontsize=14)

    def plot_alpha_example(self, ax, alpha):
        ax.clear()
        linewidth = 2

        
        # Draw arcs for joints
        HF_l_arc = patches.Arc(
            self.U_l, 2*self.R, 2*self.R, angle=0,
            theta1=np.rad2deg(np.arctan2(self.H_l[1]-self.U_l[1], self.H_l[0]-self.U_l[0])),
            theta2=np.rad2deg(np.arctan2(self.F_l[1]-self.U_l[1], self.F_l[0]-self.U_l[0])),
            edgecolor='black', linewidth=linewidth
        )
        HF_r_arc = patches.Arc(
            self.U_r, 2*self.R, 2*self.R, angle=0,
            theta1=np.rad2deg(np.arctan2(self.F_r[1]-self.U_r[1], self.F_r[0]-self.U_r[0])),
            theta2=np.rad2deg(np.arctan2(self.H_r[1]-self.U_r[1], self.H_r[0]-self.U_r[0])),
            edgecolor='black', linewidth=linewidth
        )
        FG_l_arc = patches.Arc(
            self.L_l, 2*self.R, 2*self.R, angle=0,
            theta1=np.rad2deg(np.arctan2(self.F_l[1]-self.L_l[1], self.F_l[0]-self.L_l[0])),
            theta2=np.rad2deg(np.arctan2(self.G[1]-self.L_l[1], self.G[0]-self.L_l[0])),
            edgecolor='black', linewidth=linewidth
        )
        FG_r_arc = patches.Arc(
            self.L_r, 2*self.R, 2*self.R, angle=0,
            theta1=np.rad2deg(np.arctan2(self.G[1]-self.L_r[1], self.G[0]-self.L_r[0])),
            theta2=np.rad2deg(np.arctan2(self.F_r[1]-self.L_r[1], self.F_r[0]-self.L_r[0])),
            edgecolor='black', linewidth=linewidth
        )

        # Linkages (example: connecting joints with arrows)
        OB_r = patches.FancyArrowPatch(self.O, self.B_r, color='black', linewidth=linewidth,
                                       arrowstyle='-', shrinkA=0, shrinkB=0)
        OB_l = patches.FancyArrowPatch(self.O, self.B_l, color='black', linewidth=linewidth,
                                       arrowstyle='-', shrinkA=0, shrinkB=0)
        AE_r = patches.FancyArrowPatch(self.A_r, self.E, color='black', linewidth=linewidth,
                                       arrowstyle='-', shrinkA=0, shrinkB=0)
        AE_l = patches.FancyArrowPatch(self.A_l, self.E, color='black', linewidth=linewidth,
                                       arrowstyle='-', shrinkA=0, shrinkB=0)
        DC_r = patches.FancyArrowPatch(self.D_r, self.C_r, color='black', linewidth=linewidth,
                                       arrowstyle='-', shrinkA=0, shrinkB=0)
        DC_l = patches.FancyArrowPatch(self.D_l, self.C_l, color='black', linewidth=linewidth,
                                       arrowstyle='-', shrinkA=0, shrinkB=0)
            
        if alpha == -30:
            ground = patches.FancyArrowPatch([-0.155, -0.115], [0.155, -0.115], color='silver', linewidth=2.5, arrowstyle='-', mutation_scale=0)
            ax.add_patch(ground)
            ax.set_xlim(-0.16, 0.16)
            ax.set_ylim(-0.135, 0.155)
            
            for patch in [OB_r, OB_l, AE_r, AE_l, DC_r, DC_l]:
                ax.add_patch(patch)

            for arc in (HF_l_arc, HF_r_arc, FG_l_arc, FG_r_arc):
                ax.add_patch(arc)
            
            P = patches.Circle(np.array([0.0199, -0.115]), 0.01, edgecolor='red', facecolor='pink')
            ax.add_patch(P)
            
            
        if alpha == 60:
            ground = patches.FancyArrowPatch([-0.155, -0.162], [0.155, -0.162], color='silver', linewidth=2.5, arrowstyle='-', mutation_scale=0)
            ax.add_patch(ground)
            ax.set_xlim(-0.16, 0.16)
            ax.set_ylim(-0.182, 0.108)
            
            for patch in [OB_r, OB_l, AE_r, AE_l, DC_r, DC_l]:
                ax.add_patch(patch)

            for arc in (HF_l_arc, HF_r_arc, FG_l_arc, FG_r_arc):
                ax.add_patch(arc)
            
            P = patches.Circle(np.array([-0.0368, -0.162]), 0.01, edgecolor='red', facecolor='pink')
            ax.add_patch(P)
            

        if alpha == 120:
            ground = patches.FancyArrowPatch([-0.185, -0.112], [0.125, -0.112], color='silver', linewidth=2.5, arrowstyle='-', mutation_scale=0)
            ax.add_patch(ground)
            ax.set_xlim(-0.19, 0.13)
            ax.set_ylim(-0.132, 0.158)
            
            for patch in [OB_r, OB_l, AE_r, AE_l, DC_r, DC_l]:
                ax.add_patch(patch)

            for arc in (HF_l_arc, HF_r_arc, FG_l_arc, FG_r_arc):
                ax.add_patch(arc)
            
            P = patches.Circle(np.array([-0.014, -0.112]), 0.01, edgecolor='red', facecolor='pink')
            ax.add_patch(P)
        

        ax.set_aspect('equal')
        # ax.set_xlabel('X (m)')
        # ax.set_ylabel('Y (m)')
        # ax.set_title(r'Definition of Contact Angle ($\alpha$)')
        # ax.tick_params(axis='both', labelsize=12)


# --- Main: create a figure with one main plot and three vertical subplots on the right ---
if __name__ == '__main__':
    legmodel = LegModel()
    legmodel.forward(np.deg2rad(17), np.deg2rad(0))

    # Create overall figure with GridSpec:
    fig = plt.figure(figsize=(10.25, 7))
    outer = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.1)

    # Left main axis for the leg model
    ax_main = fig.add_subplot(outer[0])
    legmodel.plot_on_ax(ax_main)

    # Right: create a nested GridSpec for three vertical subplots
    inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[1], hspace=0.4)
    ax_sub1 = fig.add_subplot(inner[0])
    ax_sub2 = fig.add_subplot(inner[1])
    ax_sub3 = fig.add_subplot(inner[2])

    # Example content for the right-hand subplots
    legmodel.forward(np.deg2rad(30), np.deg2rad(35))
    legmodel.plot_alpha_example(ax_sub1, -30)
    ax_sub1.set_title(r'$\alpha=-\frac{1}{2}\pi$', pad=12)
    
    legmodel.forward(np.deg2rad(54), np.deg2rad(-13))
    legmodel.plot_alpha_example(ax_sub2, 60)
    ax_sub2.set_title(r'$\alpha=\frac{1}{3}\pi$', pad=12)
    
    legmodel.forward(np.deg2rad(45), np.deg2rad(-102))
    legmodel.plot_alpha_example(ax_sub3, 120)
    ax_sub3.set_title(r'$\alpha=\frac{2}{3}\pi$', pad=12)

    ax_sub1.set_xticks([])
    ax_sub1.set_yticks([])
    ax_sub2.set_xticks([])
    ax_sub2.set_yticks([])
    ax_sub3.set_xticks([])
    ax_sub3.set_yticks([])
    
    fig.subplots_adjust(left=0.1, right=1, top=0.9, bottom=0.08)
    plt.show()
