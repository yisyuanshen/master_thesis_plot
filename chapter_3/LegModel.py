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
        
    def plot(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.clear()
        
        linewidth = 1.5
        
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
                                 edgecolor='red', facecolor='red', linewidth=linewidth)
        
        HF_l_inner = patches.Arc(self.U_l, 2*self.radius_inner, 2*self.radius_inner, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.H_l[1] - self.U_l[1], self.H_l[0] - self.U_l[0])),
                                 theta2=np.rad2deg(np.arctan2(self.F_l[1] - self.U_l[1], self.F_l[0] - self.U_l[0])),
                                 edgecolor='red', linewidth=linewidth)
        
        HF_l_upper = patches.FancyArrowPatch((self.H_l-self.U_l)*self.radius_outer/self.R+self.U_l,
                                             (self.H_l-self.U_l)*self.radius_inner/self.R+self.U_l,
                                             color='red', linewidth=linewidth, arrowstyle='-', shrinkA=0, shrinkB=0)
        
        HF_l_lower = patches.Arc(self.F_l, 2*self.r, 2*self.r, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.F_l[1] - self.U_l[1], self.F_l[0] - self.U_l[0])),
                                 theta2=np.rad2deg(np.arctan2(self.F_l[1] - self.U_l[1], self.F_l[0] - self.U_l[0]))-180,
                                 edgecolor='red', linewidth=linewidth)
        
        # HF_r
        HF_r_outer = patches.Arc(self.U_r, 2*self.radius_outer, 2*self.radius_outer, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.F_r[1] - self.U_r[1], self.F_r[0] - self.U_r[0])),
                                 theta2=np.rad2deg(np.arctan2(self.H_r[1] - self.U_r[1], self.H_r[0] - self.U_r[0])),
                                 edgecolor='green', linewidth=linewidth)
        
        HF_r_inner = patches.Arc(self.U_r, 2*self.radius_inner, 2*self.radius_inner, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.F_r[1] - self.U_r[1], self.F_r[0] - self.U_r[0])),
                                 theta2=np.rad2deg(np.arctan2(self.H_r[1] - self.U_r[1], self.H_r[0] - self.U_r[0])),
                                 edgecolor='green', linewidth=linewidth)
        
        HF_r_upper = patches.FancyArrowPatch((self.H_r-self.U_r)*self.radius_inner/self.R+self.U_r,
                                             (self.H_r-self.U_r)*self.radius_outer/self.R+self.U_r,
                                             color='green', linewidth=linewidth, arrowstyle='-', shrinkA=0, shrinkB=0)

        HF_r_lower = patches.Arc(self.F_r, 2*self.r, 2*self.r, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.F_r[1] - self.U_r[1], self.F_r[0] - self.U_r[0]))-180,
                                 theta2=np.rad2deg(np.arctan2(self.F_r[1] - self.U_r[1], self.F_r[0] - self.U_r[0])),
                                 edgecolor='green', linewidth=linewidth)

        # FG_l
        FG_l_outer = patches.Arc(self.L_l, 2*self.radius_outer, 2*self.radius_outer, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.F_l[1] - self.L_l[1], self.F_l[0] - self.L_l[0])),
                                 theta2=np.rad2deg(np.arctan2(self.G[1] - self.L_l[1], self.G[0] - self.L_l[0])),
                                 edgecolor='blue', linewidth=linewidth)
        
        FG_l_inner = patches.Arc(self.L_l, 2*self.radius_inner, 2*self.radius_inner, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.F_l[1] - self.L_l[1], self.F_l[0] - self.L_l[0])),
                                 theta2=np.rad2deg(np.arctan2(self.G[1] - self.L_l[1], self.G[0] - self.L_l[0])),
                                 edgecolor='blue', linewidth=linewidth)
        
        FG_l_upper = patches.Arc(self.F_l, 2*self.r, 2*self.r, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.F_l[1] - self.L_l[1], self.F_l[0] - self.L_l[0]))-180,
                                 theta2=np.rad2deg(np.arctan2(self.F_l[1] - self.L_l[1], self.F_l[0] - self.L_l[0])),
                                 edgecolor='blue', linewidth=linewidth)

        FG_l_lower = patches.Arc(self.G, 2*self.r, 2*self.r, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.G[1] - self.L_l[1], self.G[0] - self.L_l[0])),
                                 theta2=np.rad2deg(np.arctan2(self.G[1] - self.L_l[1], self.G[0] - self.L_l[0]))-180,
                                 edgecolor='blue', linewidth=linewidth)
        
        # FG_r
        FG_r_outer = patches.Arc(self.L_r, 2*self.radius_outer, 2*self.radius_outer, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.G[1] - self.L_r[1], self.G[0] - self.L_r[0])),
                                 theta2=np.rad2deg(np.arctan2(self.F_r[1] - self.L_r[1], self.F_r[0] - self.L_r[0])),
                                 edgecolor='orange', linewidth=linewidth)
        
        FG_r_inner = patches.Arc(self.L_r, 2*self.radius_inner, 2*self.radius_inner, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.G[1] - self.L_r[1], self.G[0] - self.L_r[0])),
                                 theta2=np.rad2deg(np.arctan2(self.F_r[1] - self.L_r[1], self.F_r[0] - self.L_r[0])),
                                 edgecolor='orange', linewidth=linewidth)
        
        FG_r_upper = patches.Arc(self.F_r, 2*self.r, 2*self.r, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.F_r[1] - self.L_r[1], self.F_r[0] - self.L_r[0])),
                                 theta2=np.rad2deg(np.arctan2(self.F_r[1] - self.L_r[1], self.F_r[0] - self.L_r[0]))-180,
                                 edgecolor='orange', linewidth=linewidth)

        FG_r_lower = patches.Arc(self.G, 2*self.r, 2*self.r, angle=0,
                                 theta1=np.rad2deg(np.arctan2(self.G[1] - self.L_r[1], self.G[0] - self.L_r[0]))-180,
                                 theta2=np.rad2deg(np.arctan2(self.G[1] - self.L_r[1], self.G[0] - self.L_r[0])),
                                 edgecolor='orange', linewidth=linewidth)

        # Linkages
        OB_r = patches.FancyArrowPatch(self.O  , self.B_r, color='black', linewidth=linewidth, arrowstyle='-', shrinkA=0, shrinkB=0)
        OB_l = patches.FancyArrowPatch(self.O  , self.B_l, color='black', linewidth=linewidth, arrowstyle='-', shrinkA=0, shrinkB=0)
        AE_r = patches.FancyArrowPatch(self.A_r, self.E  , color='black', linewidth=linewidth, arrowstyle='-', shrinkA=0, shrinkB=0)
        AE_l = patches.FancyArrowPatch(self.A_l, self.E  , color='black', linewidth=linewidth, arrowstyle='-', shrinkA=0, shrinkB=0)
        DC_r = patches.FancyArrowPatch(self.D_r, self.C_r, color='black', linewidth=linewidth, arrowstyle='-', shrinkA=0, shrinkB=0)
        DC_l = patches.FancyArrowPatch(self.D_l, self.C_l, color='black', linewidth=linewidth, arrowstyle='-', shrinkA=0, shrinkB=0)

        # Support lines
        beta_line = patches.FancyArrowPatch((self.O-self.G)*0.6, self.G*1.2, color='purple', linewidth=1, arrowstyle='-', shrinkA=0, shrinkB=0, linestyle='-.')
        U_l_upper = patches.FancyArrowPatch(self.H_l, self.U_l, color='red'   , linewidth=linewidth, arrowstyle='-', shrinkA=0, shrinkB=0, linestyle=':')
        U_l_lower = patches.FancyArrowPatch(self.F_l, self.U_l, color='red'   , linewidth=linewidth, arrowstyle='-', shrinkA=0, shrinkB=0, linestyle=':')
        L_l_upper = patches.FancyArrowPatch(self.F_l, self.L_l, color='blue'  , linewidth=linewidth, arrowstyle='-', shrinkA=0, shrinkB=0, linestyle=':')
        L_l_lower = patches.FancyArrowPatch(self.G  , self.L_l, color='blue'  , linewidth=linewidth, arrowstyle='-', shrinkA=0, shrinkB=0, linestyle=':')
        L_r_upper = patches.FancyArrowPatch(self.F_r, self.L_r, color='orange', linewidth=linewidth, arrowstyle='-', shrinkA=0, shrinkB=0, linestyle=':')
        L_r_lower = patches.FancyArrowPatch(self.G  , self.L_r, color='orange', linewidth=linewidth, arrowstyle='-', shrinkA=0, shrinkB=0, linestyle=':')
        U_r_upper = patches.FancyArrowPatch(self.H_r, self.U_r, color='green' , linewidth=linewidth, arrowstyle='-', shrinkA=0, shrinkB=0, linestyle=':')
        U_r_lower = patches.FancyArrowPatch(self.F_r, self.U_r, color='green' , linewidth=linewidth, arrowstyle='-', shrinkA=0, shrinkB=0, linestyle=':')
        
        # Other annotations
        phi_l_line = patches.FancyArrowPatch(self.O, 0.7*self.R*np.array([np.cos(np.deg2rad(17+90)), np.sin(np.deg2rad(17+90))]), color='black' , linewidth=1, arrowstyle='-', shrinkA=0, shrinkB=0, linestyle=':')
        phi_r_line = patches.FancyArrowPatch(self.O, 0.7*self.R*np.array([np.cos(np.deg2rad(-17+90)), np.sin(np.deg2rad(-17+90))]), color='black' , linewidth=1, arrowstyle='-', shrinkA=0, shrinkB=0, linestyle=':')
        
        beta_arc    = patches.Arc(self.O, 1.8*self.R, 1.8*self.R, angle=0, theta1=90                      , theta2=90+np.rad2deg(self.beta)           , edgecolor='black', linewidth=linewidth, linestyle='--')
        theta_arc   = patches.Arc(self.O, 1.2*self.R, 1.2*self.R, angle=0, theta1=90+np.rad2deg(self.beta), theta2=90+np.rad2deg(self.beta+self.theta), edgecolor='black', linewidth=linewidth, linestyle='--')
        theta_0_arc = patches.Arc(self.O, 1.2*self.R, 1.2*self.R, angle=0, theta1=90                      , theta2=107                                , edgecolor='black', linewidth=linewidth, linestyle='--')
        phi_l_arc   = patches.Arc(self.O, 0.6*self.R, 0.6*self.R, angle=0, theta1=107                     , theta2=90+np.rad2deg(self.beta+self.theta), edgecolor='black', linewidth=linewidth, linestyle='--')
        phi_r_arc   = patches.Arc(self.O, 0.6*self.R, 0.6*self.R, angle=0, theta1=90+np.rad2deg(self.beta-self.theta), theta2=73                      , edgecolor='black', linewidth=linewidth, linestyle='--')



        ax.add_patch(x_axis)
        ax.add_patch(y_axis)
         
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
        
        ax.add_patch(beta_line)
        ax.add_patch(U_l_upper)
        ax.add_patch(U_l_lower)
        ax.add_patch(L_l_upper)
        ax.add_patch(L_l_lower)
        ax.add_patch(L_r_upper)
        ax.add_patch(L_r_lower)
        ax.add_patch(U_r_upper)
        ax.add_patch(U_r_lower)
        
        ax.add_patch(O)
        # ax.add_patch(A_r)
        # ax.add_patch(A_l)
        # ax.add_patch(B_r)
        # ax.add_patch(B_l)
        # ax.add_patch(C_r)
        # ax.add_patch(C_l)
        # ax.add_patch(D_r)
        # ax.add_patch(D_l)
        # ax.add_patch(E)
        
        ax.add_patch(F_r)
        ax.add_patch(F_l)
        ax.add_patch(G)
        ax.add_patch(H_r)
        ax.add_patch(H_l)
        ax.add_patch(U_r)
        ax.add_patch(U_l)
        ax.add_patch(L_r)
        ax.add_patch(L_l)
        
        ax.add_patch(phi_l_line)
        ax.add_patch(phi_r_line)
        
        ax.add_patch(beta_arc)
        ax.add_patch(theta_arc)
        ax.add_patch(theta_0_arc)
        ax.add_patch(phi_l_arc)
        ax.add_patch(phi_r_arc)
        
        # Text
        ax.text(self.O[0]-0.003, self.O[1]-0.008, r'$O$', ha='right', va='top')
        ax.text(self.U_l[0]+0.003, self.U_l[1]+0.008, r'$U_L$', ha='left', va='center')
        ax.text(self.U_r[0]-0.005, self.U_r[1]+0.008, r'$U_R$', ha='right', va='center')
        ax.text(self.L_l[0]-0.008, self.L_l[1]+0.003, r'$L_L$', ha='right', va='center')
        ax.text(self.L_r[0]+0.002, self.L_r[1]+0.008, r'$L_R$', ha='left', va='bottom')
        ax.text(self.H_l[0]-0.003, self.H_l[1]+0.005, r'$H_L$', ha='left', va='bottom')
        ax.text(self.H_r[0]-0.006, self.H_r[1]+0.005, r'$H_R$', ha='right', va='center')
        ax.text(self.F_l[0]-0.004, self.F_l[1]-0.020, r'$F_L$', ha='center', va='top')
        ax.text(self.F_r[0]+0.020, self.F_r[1]-0.008, r'$F_R$', ha='left', va='center')
        ax.text(self.G[0]+0.002, self.G[1]-0.018, r'$G$', ha='center', va='top')
        ax.text(-0.035, 0.030, r'$\phi_L$', ha='center', va='top')
        ax.text( 0.023, 0.049, r'$\phi_R$', ha='center', va='top')
        ax.text(-0.06,  0.040, r'$\theta$', ha='center', va='top')
        ax.text(-0.009, 0.077, r'$\theta_0$', ha='center', va='top')
        ax.text(-0.025, 0.103, r'$\beta$', ha='center', va='top')
        
        
        
        ax.set_aspect('equal')
        ax.set_xlim(-0.16, 0.20)
        ax.set_ylim(-0.22, 0.14)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Parameter Definitions of the Leg-Wheel Mechanism')
        
        plt.xticks(np.arange(-2, 4, 1)/20, fontsize=12)
        plt.yticks(np.arange(-4, 3, 1)/20, fontsize=12)
        # plt.savefig('LegPlot.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        
if __name__ == '__main__':
    legmodel = LegModel()
    
    legmodel.forward(np.deg2rad(71), np.deg2rad(30))

    print(legmodel.G)
    
    legmodel.plot()
