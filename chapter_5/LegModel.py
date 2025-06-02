import coef
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12



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
        return np.array([[np.cos(beta), -np.sin(beta)], [np.sin(beta), np.cos(beta)]])
        
    def forward(self, theta, beta):
        self.theta = theta
        self.beta = beta
        
        rot_beta = self.rot_matrix(self.beta)
        
        self.O = np.array([0, 0])
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
        
    def plot(self, ax, color, linestyle, label):
        linewidth = 2.5
        
        # Axis
        x_axis = patches.FancyArrowPatch([-0.15, 0], [0.19, 0], color='silver', linewidth=2, arrowstyle='-|>', mutation_scale=15)
        y_axis = patches.FancyArrowPatch([0, -0.21], [0, 0.13], color='silver', linewidth=2, arrowstyle='-|>', mutation_scale=15)
        
        # Joints
        O   = patches.Circle(self.O  , 0.003, edgecolor='gray', facecolor='lightgray')
        A_r = patches.Circle(self.A_r, 0.003, edgecolor='gray', facecolor='lightgray')
        A_l = patches.Circle(self.A_l, 0.003, edgecolor='gray', facecolor='lightgray')
        B_r = patches.Circle(self.B_r, 0.003, edgecolor='gray', facecolor='lightgray')
        B_l = patches.Circle(self.B_l, 0.003, edgecolor='gray', facecolor='lightgray')
        C_r = patches.Circle(self.C_r, 0.003, edgecolor='gray', facecolor='lightgray')
        C_l = patches.Circle(self.C_l, 0.003, edgecolor='gray', facecolor='lightgray')
        D_r = patches.Circle(self.D_r, 0.003, edgecolor='gray', facecolor='lightgray')
        D_l = patches.Circle(self.D_l, 0.003, edgecolor='gray', facecolor='lightgray')
        E   = patches.Circle(self.E  , 0.003, edgecolor='gray', facecolor='lightgray')
        F_r = patches.Circle(self.F_r, 0.003, edgecolor='gray', facecolor='lightgray')
        F_l = patches.Circle(self.F_l, 0.003, edgecolor='gray', facecolor='lightgray')
        G   = patches.Circle(self.G  , 0.003, edgecolor='gray', facecolor='lightgray')
        H_r = patches.Circle(self.H_r, 0.003, edgecolor='gray', facecolor='lightgray')
        H_l = patches.Circle(self.H_l, 0.003, edgecolor='gray', facecolor='lightgray')
        U_r = patches.Circle(self.U_r, 0.003, edgecolor='gray', facecolor='lightgray')
        U_l = patches.Circle(self.U_l, 0.003, edgecolor='gray', facecolor='lightgray')
        L_r = patches.Circle(self.L_r, 0.003, edgecolor='gray', facecolor='lightgray')
        L_l = patches.Circle(self.L_l, 0.003, edgecolor='gray', facecolor='lightgray')
        
        # HF_l
        HF_l_outer = patches.Arc(self.U_l, 2*self.radius_outer, 2*self.radius_outer, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.H_l[1] - self.U_l[1], self.H_l[0] - self.U_l[0])),
                                 theta2=np.rad2deg(np.arctan2(self.F_l[1] - self.U_l[1], self.F_l[0] - self.U_l[0])),
                                 edgecolor=color, facecolor=color, linewidth=linewidth, linestyle=linestyle)
        
        HF_l_inner = patches.Arc(self.U_l, 2*self.radius_inner, 2*self.radius_inner, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.H_l[1] - self.U_l[1], self.H_l[0] - self.U_l[0])),
                                 theta2=np.rad2deg(np.arctan2(self.F_l[1] - self.U_l[1], self.F_l[0] - self.U_l[0])),
                                 edgecolor=color, linewidth=linewidth, linestyle=linestyle)
        
        HF_l_upper = patches.FancyArrowPatch((self.H_l-self.U_l)*self.radius_outer/self.R+self.U_l,
                                             (self.H_l-self.U_l)*self.radius_inner/self.R+self.U_l,
                                             color=color, linewidth=linewidth, arrowstyle='-', shrinkA=0, shrinkB=0, linestyle=linestyle)
        
        HF_l_lower = patches.Arc(self.F_l, 2*self.r, 2*self.r, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.F_l[1] - self.U_l[1], self.F_l[0] - self.U_l[0])),
                                 theta2=np.rad2deg(np.arctan2(self.F_l[1] - self.U_l[1], self.F_l[0] - self.U_l[0]))-180,
                                 edgecolor=color, linewidth=linewidth, linestyle=linestyle)
        
        # HF_r
        HF_r_outer = patches.Arc(self.U_r, 2*self.radius_outer, 2*self.radius_outer, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.F_r[1] - self.U_r[1], self.F_r[0] - self.U_r[0])),
                                 theta2=np.rad2deg(np.arctan2(self.H_r[1] - self.U_r[1], self.H_r[0] - self.U_r[0])),
                                 edgecolor=color, linewidth=linewidth, linestyle=linestyle)
        
        HF_r_inner = patches.Arc(self.U_r, 2*self.radius_inner, 2*self.radius_inner, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.F_r[1] - self.U_r[1], self.F_r[0] - self.U_r[0])),
                                 theta2=np.rad2deg(np.arctan2(self.H_r[1] - self.U_r[1], self.H_r[0] - self.U_r[0])),
                                 edgecolor=color, linewidth=linewidth, linestyle=linestyle)
        
        HF_r_upper = patches.FancyArrowPatch((self.H_r-self.U_r)*self.radius_inner/self.R+self.U_r,
                                             (self.H_r-self.U_r)*self.radius_outer/self.R+self.U_r,
                                             color=color, linewidth=linewidth, arrowstyle='-', shrinkA=0, shrinkB=0, linestyle=linestyle)

        HF_r_lower = patches.Arc(self.F_r, 2*self.r, 2*self.r, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.F_r[1] - self.U_r[1], self.F_r[0] - self.U_r[0]))-180,
                                 theta2=np.rad2deg(np.arctan2(self.F_r[1] - self.U_r[1], self.F_r[0] - self.U_r[0])),
                                 edgecolor=color, linewidth=linewidth, linestyle=linestyle)

        # FG_l
        FG_l_outer = patches.Arc(self.L_l, 2*self.radius_outer, 2*self.radius_outer, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.F_l[1] - self.L_l[1], self.F_l[0] - self.L_l[0])),
                                 theta2=np.rad2deg(np.arctan2(self.G[1] - self.L_l[1], self.G[0] - self.L_l[0])),
                                 edgecolor=color, linewidth=linewidth, linestyle=linestyle)
        
        FG_l_inner = patches.Arc(self.L_l, 2*self.radius_inner, 2*self.radius_inner, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.F_l[1] - self.L_l[1], self.F_l[0] - self.L_l[0])),
                                 theta2=np.rad2deg(np.arctan2(self.G[1] - self.L_l[1], self.G[0] - self.L_l[0])),
                                 edgecolor=color, linewidth=linewidth, linestyle=linestyle)
        
        FG_l_upper = patches.Arc(self.F_l, 2*self.r, 2*self.r, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.F_l[1] - self.L_l[1], self.F_l[0] - self.L_l[0]))-180,
                                 theta2=np.rad2deg(np.arctan2(self.F_l[1] - self.L_l[1], self.F_l[0] - self.L_l[0])),
                                 edgecolor=color, linewidth=linewidth, linestyle=linestyle)

        FG_l_lower = patches.Arc(self.G, 2*self.r, 2*self.r, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.G[1] - self.L_l[1], self.G[0] - self.L_l[0])),
                                 theta2=np.rad2deg(np.arctan2(self.G[1] - self.L_l[1], self.G[0] - self.L_l[0]))-180,
                                 edgecolor=color, linewidth=linewidth, linestyle=linestyle)
        
        # FG_r
        FG_r_outer = patches.Arc(self.L_r, 2*self.radius_outer, 2*self.radius_outer, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.G[1] - self.L_r[1], self.G[0] - self.L_r[0])),
                                 theta2=np.rad2deg(np.arctan2(self.F_r[1] - self.L_r[1], self.F_r[0] - self.L_r[0])),
                                 edgecolor=color, linewidth=linewidth, linestyle=linestyle)
        
        FG_r_inner = patches.Arc(self.L_r, 2*self.radius_inner, 2*self.radius_inner, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.G[1] - self.L_r[1], self.G[0] - self.L_r[0])),
                                 theta2=np.rad2deg(np.arctan2(self.F_r[1] - self.L_r[1], self.F_r[0] - self.L_r[0])),
                                 edgecolor=color, linewidth=linewidth, linestyle=linestyle)
        
        FG_r_upper = patches.Arc(self.F_r, 2*self.r, 2*self.r, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.F_r[1] - self.L_r[1], self.F_r[0] - self.L_r[0])),
                                 theta2=np.rad2deg(np.arctan2(self.F_r[1] - self.L_r[1], self.F_r[0] - self.L_r[0]))-180,
                                 edgecolor=color, linewidth=linewidth, linestyle=linestyle)

        FG_r_lower = patches.Arc(self.G, 2*self.r, 2*self.r, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.G[1] - self.L_r[1], self.G[0] - self.L_r[0]))-180,
                                 theta2=np.rad2deg(np.arctan2(self.G[1] - self.L_r[1], self.G[0] - self.L_r[0])),
                                 edgecolor=color, linewidth=linewidth, linestyle=linestyle)

        # Linkages
        OB_r = patches.FancyArrowPatch(self.O  , self.B_r, color=color, linewidth=linewidth, arrowstyle='-', shrinkA=0, shrinkB=0, linestyle=linestyle)
        OB_l = patches.FancyArrowPatch(self.O  , self.B_l, color=color, linewidth=linewidth, arrowstyle='-', shrinkA=0, shrinkB=0, linestyle=linestyle)
        AE_r = patches.FancyArrowPatch(self.A_r, self.E  , color=color, linewidth=linewidth, arrowstyle='-', shrinkA=0, shrinkB=0, linestyle=linestyle)
        AE_l = patches.FancyArrowPatch(self.A_l, self.E  , color=color, linewidth=linewidth, arrowstyle='-', shrinkA=0, shrinkB=0, linestyle=linestyle)
        DC_r = patches.FancyArrowPatch(self.D_r, self.C_r, color=color, linewidth=linewidth, arrowstyle='-', shrinkA=0, shrinkB=0, linestyle=linestyle)
        DC_l = patches.FancyArrowPatch(self.D_l, self.C_l, color=color, linewidth=linewidth, arrowstyle='-', shrinkA=0, shrinkB=0, linestyle=linestyle)
        DC_l = patches.FancyArrowPatch(self.D_l, self.D_l, color=color, linewidth=linewidth, arrowstyle='-', shrinkA=0, shrinkB=0, linestyle='-', label=label)
        
        ax.add_patch(HF_r_outer)
        ax.add_patch(HF_r_inner)
        ax.add_patch(HF_r_upper)
        ax.add_patch(HF_r_lower)
        
        ax.add_patch(HF_l_outer)
        ax.add_patch(HF_l_inner)
        ax.add_patch(HF_l_upper)
        ax.add_patch(HF_l_lower)

        ax.add_patch(FG_r_outer)
        ax.add_patch(FG_r_inner)
        ax.add_patch(FG_r_upper)
        ax.add_patch(FG_r_lower)
        
        ax.add_patch(FG_l_outer)
        ax.add_patch(FG_l_inner)
        ax.add_patch(FG_l_upper)
        ax.add_patch(FG_l_lower)
        
        ax.add_patch(OB_r)
        ax.add_patch(OB_l)
        ax.add_patch(AE_r)
        ax.add_patch(AE_l)
        ax.add_patch(DC_r)
        ax.add_patch(DC_l)
        
if __name__ == '__main__':
    legmodel = LegModel()
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    
    legmodel.forward(np.deg2rad(80.97806522725013), np.deg2rad(28.07248693585296))
    legmodel.plot(ax, 'skyblue', '--', 'Target Pose')
    
    legmodel.forward(np.deg2rad(68.5916156878487), np.deg2rad(46.872978650772204))    
    legmodel.plot(ax, 'black', '-', 'Current Pose')
         
         
    ground = patches.FancyArrowPatch([-0.1, -0.2] , [0.2, -0.2], color='darkred', linewidth=2.5, arrowstyle='-', shrinkA=0, shrinkB=0, linestyle='-', label='Ground')
    obstacle_h = patches.FancyArrowPatch([0.05, -0.15] , [0.19, -0.15], color='darkgreen', linewidth=2.5, arrowstyle='-', shrinkA=0, shrinkB=0, linestyle='-', label='Obstacle')
    obstacle_vl = patches.FancyArrowPatch([0.05, -0.15] , [0.05, -0.2], color='darkgreen', linewidth=2.5, arrowstyle='-', shrinkA=0, shrinkB=0, linestyle='-')
    obstacle_vr = patches.FancyArrowPatch([0.19, -0.15] , [0.19, -0.2], color='darkgreen', linewidth=2.5, arrowstyle='-', shrinkA=0, shrinkB=0, linestyle='-')
    
    ax.add_patch(ground)
    ax.add_patch(obstacle_h)
    ax.add_patch(obstacle_vl)
    ax.add_patch(obstacle_vr)

    arow = patches.FancyArrowPatch([0.1, -0.15], [0.1, -0.105], color='red', linewidth=3, arrowstyle='-|>', mutation_scale=25, linestyle='-')
    ax.add_patch(arow)
    
    arow = patches.FancyArrowPatch([0.1, -0.2], [0.1, -0.155], color='blue', linewidth=3, arrowstyle='-|>', mutation_scale=25, linestyle='-')
    ax.add_patch(arow)

    # ax.legend(loc='upper left')
    ax.set_aspect('equal')
    ax.set_xlim(-0.21, 0.21)
    ax.set_ylim(-0.22, 0.14)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # ax.set_xlabel('X (m)')
    # ax.set_ylabel('Y (m)')
    # ax.set_title('')
    
    # plt.xticks(np.arange(-2, 4, 1)/20, fontsize=12)
    # plt.yticks(np.arange(-4, 3, 1)/20, fontsize=12)
    # plt.savefig('LegPlot.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    