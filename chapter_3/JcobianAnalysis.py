import coef
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib.ticker import FuncFormatter

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

class LegModel:
    def __init__(self):
        self.R = 0.1
        self.r = 0.0125
        self.radius_outer = self.R + self.r
        self.radius_inner = self.R - self.r
        
        self.N = 0
        self.theta = None
        self.beta = None
        self.rim = None
        self.alpha = None
        self.P_theta = None
        self.P_theta_deriv = None
        self.jacobian = None
        self.jacobian_rcond = None

    def rot_matrix(self, beta):
        return np.array([[np.cos(beta), -np.sin(beta)], [np.sin(beta), np.cos(beta)]])
        
    def forward(self, eta: np.ndarray):
        self.N = eta.shape[0]
        self.theta = eta[:, 0]
        self.beta = eta[:, 1]

        rot_matrix = self.rot_matrix(eta[:, 1])
        rot_beta = rot_matrix.transpose((2, 0, 1))

        self.O = np.array([0, 0])
        self.A_r = np.einsum('nij,nj->ni', rot_beta, (np.array([poly(eta[:, 0]) for poly in coef.A_r_poly]).T))
        self.A_l = np.einsum('nij,nj->ni', rot_beta, (np.array([poly(eta[:, 0]) for poly in coef.A_l_poly]).T))
        self.B_r = np.einsum('nij,nj->ni', rot_beta, (np.array([poly(eta[:, 0]) for poly in coef.B_r_poly]).T))
        self.B_l = np.einsum('nij,nj->ni', rot_beta, (np.array([poly(eta[:, 0]) for poly in coef.B_l_poly]).T))
        self.C_r = np.einsum('nij,nj->ni', rot_beta, (np.array([poly(eta[:, 0]) for poly in coef.C_r_poly]).T))
        self.C_l = np.einsum('nij,nj->ni', rot_beta, (np.array([poly(eta[:, 0]) for poly in coef.C_l_poly]).T))
        self.D_r = np.einsum('nij,nj->ni', rot_beta, (np.array([poly(eta[:, 0]) for poly in coef.D_r_poly]).T))
        self.D_l = np.einsum('nij,nj->ni', rot_beta, (np.array([poly(eta[:, 0]) for poly in coef.D_l_poly]).T))
        self.E   = np.einsum('nij,nj->ni', rot_beta, (np.array([poly(eta[:, 0]) for poly in coef.E_poly]).T))
        self.F_r = np.einsum('nij,nj->ni', rot_beta, (np.array([poly(eta[:, 0]) for poly in coef.F_r_poly]).T))
        self.F_l = np.einsum('nij,nj->ni', rot_beta, (np.array([poly(eta[:, 0]) for poly in coef.F_l_poly]).T))
        self.G   = np.einsum('nij,nj->ni', rot_beta, (np.array([poly(eta[:, 0]) for poly in coef.G_poly]).T))
        self.H_r = np.einsum('nij,nj->ni', rot_beta, (np.array([poly(eta[:, 0]) for poly in coef.H_r_poly]).T))
        self.H_l = np.einsum('nij,nj->ni', rot_beta, (np.array([poly(eta[:, 0]) for poly in coef.H_l_poly]).T))
        self.U_r = np.einsum('nij,nj->ni', rot_beta, (np.array([poly(eta[:, 0]) for poly in coef.U_r_poly]).T))
        self.U_l = np.einsum('nij,nj->ni', rot_beta, (np.array([poly(eta[:, 0]) for poly in coef.U_l_poly]).T))
        self.L_r = np.einsum('nij,nj->ni', rot_beta, (np.array([poly(eta[:, 0]) for poly in coef.L_r_poly]).T))
        self.L_l = np.einsum('nij,nj->ni', rot_beta, (np.array([poly(eta[:, 0]) for poly in coef.L_l_poly]).T))
        
    def calculate_alpha(self):
        rims = []
        alphas = []
        
        for i in range(self.N):
            contact_height= []
            contact_height.append(self.U_l[i,1]-self.radius_outer if self.H_l[i,0] <= self.U_l[i,0] <= self.F_l[i,0] else 0)
            contact_height.append(self.L_l[i,1]-self.radius_outer if self.F_l[i,0] <= self.L_l[i,0] <= self.G[i,0]   else 0)
            contact_height.append(self.G[i,1]-self.r)
            contact_height.append(self.L_r[i,1]-self.radius_outer if self.G[i,0]   <= self.L_r[i,0] <= self.F_r[i,0] else 0)
            contact_height.append(self.U_r[i,1]-self.radius_outer if self.F_r[i,0] <= self.U_r[i,0] <= self.H_r[i,0] else 0)
            rims.append(np.argmin(contact_height) if min(contact_height) != 0 else np.nan)
            
            if rims[-1] == 0:
                UH = self.H_l[i]-self.U_l[i]
                alphas.append(-np.pi+abs(np.arccos(np.clip(-UH[1] / np.linalg.norm(UH), -1.0, 1.0))))
            elif rims[-1] == 1:
                LF = self.F_l[i]-self.L_l[i]
                alphas.append(-np.deg2rad(50)+abs(np.arccos(np.clip(-LF[1] / np.linalg.norm(LF), -1.0, 1.0))))
            elif rims[-1] == 2:
                LG = self.G[i]-self.L_r[i]
                alphas.append(abs(np.arccos(np.clip(-LG[1] / np.linalg.norm(LG), -1.0, 1.0))))
            elif rims[-1] == 3:
                LG = self.G[i]-self.L_r[i]
                alphas.append(abs(np.arccos(np.clip(-LG[1] / np.linalg.norm(LG), -1.0, 1.0))))
            elif rims[-1] == 4:
                UF = self.F_r[i]-self.U_r[i]
                alphas.append(np.deg2rad(50)+abs(np.arccos(np.clip(-UF[1] / np.linalg.norm(UF), -1.0, 1.0))))
            else:
                alphas.append(np.nan)
        
        self.rim = np.array(rims)
        self.alpha = np.array(alphas)   

    def calculate_p(self):
        U_l_coefs = np.array([ coef.U_x_coef, coef.U_y_coef])
        U_r_coefs = U_l_coefs*np.array([[-1], [1]])
        L_l_coefs = np.array([ coef.L_x_coef, coef.L_y_coef])
        L_r_coefs = L_l_coefs*np.array([[-1], [1]])
        H_l_coefs = np.array([ coef.H_x_coef, coef.H_y_coef])
        H_r_coefs = H_l_coefs*np.array([[-1], [1]])
        G_coefs   = np.array([ coef.G_x_coef, coef.G_y_coef])
        
        UH_l_coefs = H_l_coefs - U_l_coefs
        LG_l_coefs = G_coefs   - L_l_coefs
        LG_r_coefs = G_coefs   - L_r_coefs
        UH_r_coefs = H_r_coefs - U_r_coefs
        
        P_poly_coefs = np.zeros((self.N, 2, 8))
        
        ratio_outer = 1 + self.r / self.R
        ratio_rim = self.r / self.R
        
        mask0 = (self.rim == 0)
        mask1 = (self.rim == 1)
        mask2 = (self.rim == 2)
        mask3 = (self.rim == 3)
        mask4 = (self.rim == 4)
        
        if np.any(mask0):
            rot_alpha = self.rot_matrix(self.alpha[mask0]+np.pi).transpose((2, 0, 1))
            temp = np.einsum('nij,jk->nik', rot_alpha, UH_l_coefs)
            P_poly_coefs[mask0] = ratio_outer * temp + U_l_coefs[None, :, :]
        
        if np.any(mask1):
            rot_alpha = self.rot_matrix(self.alpha[mask1]).transpose((2, 0, 1))
            temp = np.einsum('nij,jk->nik', rot_alpha, LG_l_coefs)
            P_poly_coefs[mask1] = ratio_outer * temp + L_l_coefs[None, :, :]
        
        if np.any(mask2):
            rot_alpha = self.rot_matrix(self.alpha[mask2]).transpose((2, 0, 1))
            temp = np.einsum('nij,jk->nik', rot_alpha, LG_l_coefs)
            P_poly_coefs[mask2] = ratio_rim * temp + G_coefs[None, :, :]
            
        if np.any(mask3):
            rot_alpha = self.rot_matrix(self.alpha[mask3]).transpose((2, 0, 1))
            temp = np.einsum('nij,jk->nik', rot_alpha, LG_r_coefs)
            P_poly_coefs[mask3] = ratio_outer * temp + L_r_coefs[None, :, :]
            
        if np.any(mask4):
            rot_alpha = self.rot_matrix(self.alpha[mask4]-np.pi).transpose((2, 0, 1))
            temp = np.einsum('nij,jk->nik', rot_alpha, UH_r_coefs)
            P_poly_coefs[mask4] = ratio_outer * temp + U_r_coefs[None, :, :]
        
        print('P_poly Coefficients Calculation Done')
        
        P_poly_deriv_coefs = P_poly_coefs[..., 1:] * np.arange(1, 8)
        
        P_theta = np.zeros((self.N, 2))
        P_theta_deriv = np.zeros((self.N, 2))

        for k in range(7, -1, -1):
            P_theta = P_theta * self.theta[:, None] + P_poly_coefs[..., k]

        for k in range(6, -1, -1):
            P_theta_deriv = P_theta_deriv * self.theta[:, None] + P_poly_deriv_coefs[..., k]

        self.P_theta = P_theta
        self.P_theta_deriv = P_theta_deriv
           
    def calculate_jacobian(self):
        px = self.P_theta[:, 0]
        py = self.P_theta[:, 1]
        px_d = self.P_theta_deriv[:, 0]
        py_d = self.P_theta_deriv[:, 1]
        
        cos_beta = np.cos(self.beta)
        sin_beta = np.sin(self.beta)
        
        A = px_d * cos_beta - py_d * sin_beta
        B = px * sin_beta + py * cos_beta
        C = px_d * sin_beta + py_d * cos_beta
        D = px * cos_beta - py * sin_beta
        
        J11 =  0.5 * (A - B)
        J12 = -0.5 * (A + B)
        J21 =  0.5 * (C + D)
        J22 =  0.5 * (D - C)
        
        self.jacobian = np.stack((
            np.stack((J11, J12), axis=-1),
            np.stack((J21, J22), axis=-1)
        ), axis=1)
        
        self.jacobian_rcond = np.array([
            np.nan if np.isnan(self.rim[i]) else 1 / np.linalg.cond(self.jacobian[i])
            for i in range(self.N)
        ])
        
    def plot_leg(self, ax, theta, beta, offset=(0, 0), x_scale=1.5):
        color = 'black'
        alpha = 0.6
        
        # Compute leg coordinates for a single sample
        self.forward(np.array([[theta, beta]]))
        self.calculate_alpha()
        self.calculate_p()
        
        # Create a transformation with x-axis scaling and translation offset
        # x_scale magnifies the x-axis; set x_scale > 1 to stretch, or < 1 to compress.
        trans_offset = transforms.Affine2D().scale(x_scale, 1).translate(*offset) + ax.transData
        
        linewidth = 1.25
        
        # Draw arcs for joints (extract first element since arrays are (1,2))
        HF_l_arc = patches.Arc(
            self.U_l[0], 2*self.R, 2*self.R, angle=0,
            theta1=np.rad2deg(np.arctan2(self.H_l[0][1] - self.U_l[0][1],
                                        self.H_l[0][0] - self.U_l[0][0])),
            theta2=np.rad2deg(np.arctan2(self.F_l[0][1] - self.U_l[0][1],
                                        self.F_l[0][0] - self.U_l[0][0])),
            edgecolor=color, linewidth=linewidth,
            transform=trans_offset, alpha=alpha
        )
        HF_r_arc = patches.Arc(
            self.U_r[0], 2*self.R, 2*self.R, angle=0,
            theta1=np.rad2deg(np.arctan2(self.F_r[0][1] - self.U_r[0][1],
                                        self.F_r[0][0] - self.U_r[0][0])),
            theta2=np.rad2deg(np.arctan2(self.H_r[0][1] - self.U_r[0][1],
                                        self.H_r[0][0] - self.U_r[0][0])),
            edgecolor=color, linewidth=linewidth,
            transform=trans_offset, alpha=alpha
        )
        FG_l_arc = patches.Arc(
            self.L_l[0], 2*self.R, 2*self.R, angle=0,
            theta1=np.rad2deg(np.arctan2(self.F_l[0][1] - self.L_l[0][1],
                                        self.F_l[0][0] - self.L_l[0][0])),
            theta2=np.rad2deg(np.arctan2(self.G[0][1] - self.L_l[0][1],
                                        self.G[0][0] - self.L_l[0][0])),
            edgecolor=color, linewidth=linewidth,
            transform=trans_offset, alpha=alpha
        )
        FG_r_arc = patches.Arc(
            self.L_r[0], 2*self.R, 2*self.R, angle=0,
            theta1=np.rad2deg(np.arctan2(self.G[0][1] - self.L_r[0][1],
                                        self.G[0][0] - self.L_r[0][0])),
            theta2=np.rad2deg(np.arctan2(self.F_r[0][1] - self.L_r[0][1],
                                        self.F_r[0][0] - self.L_r[0][0])),
            edgecolor=color, linewidth=linewidth,
            transform=trans_offset, alpha=alpha
        )
        
        # Linkages (connecting joints with arrows, extracting coordinates with [0])
        OB_r = patches.FancyArrowPatch(self.O, self.B_r[0], color=color, linewidth=linewidth,
                                        arrowstyle='-', shrinkA=0, shrinkB=0, transform=trans_offset, alpha=alpha)
        OB_l = patches.FancyArrowPatch(self.O, self.B_l[0], color=color, linewidth=linewidth,
                                        arrowstyle='-', shrinkA=0, shrinkB=0, transform=trans_offset, alpha=alpha)
        AE_r = patches.FancyArrowPatch(self.A_r[0], self.E[0], color=color, linewidth=linewidth,
                                        arrowstyle='-', shrinkA=0, shrinkB=0, transform=trans_offset, alpha=alpha)
        AE_l = patches.FancyArrowPatch(self.A_l[0], self.E[0], color=color, linewidth=linewidth,
                                        arrowstyle='-', shrinkA=0, shrinkB=0, transform=trans_offset, alpha=alpha)
        DC_r = patches.FancyArrowPatch(self.D_r[0], self.C_r[0], color=color, linewidth=linewidth,
                                        arrowstyle='-', shrinkA=0, shrinkB=0, transform=trans_offset, alpha=alpha)
        DC_l = patches.FancyArrowPatch(self.D_l[0], self.C_l[0], color=color, linewidth=linewidth,
                                        arrowstyle='-', shrinkA=0, shrinkB=0, transform=trans_offset, alpha=alpha)
        
        
        # Add all patches to the axis
        for patch in [OB_r, OB_l, AE_r, AE_l, DC_r, DC_l]:
            ax.add_patch(patch)
        
        for arc in (HF_l_arc, HF_r_arc, FG_l_arc, FG_r_arc):
            ax.add_patch(arc)
        
        rot_beta = self.rot_matrix(self.beta[0])
        contact_p = rot_beta @ self.P_theta[0]*(np.linalg.norm(self.P_theta[0])-self.r)/np.linalg.norm(self.P_theta[0])
        ground = patches.FancyArrowPatch([contact_p[0]-0.1, contact_p[1]], [contact_p[0]+0.1, contact_p[1]], color='gray', 
                                        linewidth=1, arrowstyle='-', mutation_scale=0,
                                        transform=trans_offset)
        ax.add_patch(ground)
        
        P = patches.Circle(contact_p, 0.015, edgecolor='green', facecolor='lime', transform=trans_offset)
        ax.add_patch(P)

     
if __name__ == '__main__':
    leg = LegModel()
    
    sampling_rate = 500
    theta_array = np.repeat(np.deg2rad(np.linspace(17, 160, sampling_rate)), sampling_rate)
    beta_array = np.tile(np.deg2rad(np.linspace(-180, 180, sampling_rate)), sampling_rate)
    eta_array = np.array(list(zip(theta_array, beta_array)))

    leg.forward(eta_array)
    print('Forward Kinematics Done')

    leg.calculate_alpha()
    print('Alpha Calculation Done')
    
    leg.calculate_p()
    print('P Calculation Done')

    leg.calculate_jacobian()
    print('Jacobian Calculation Done')
    
    rcond_array = leg.jacobian_rcond
    rim_array = leg.rim
    
    boundary = {'N4':[], '43':[], '32':[], '21':[], '10':[], '0N':[]}
    eta_array = eta_array.reshape(sampling_rate, sampling_rate, 2)
    rim_array = rim_array.reshape(sampling_rate, sampling_rate, 1)
    
    for i in range(sampling_rate):
        for j in range(sampling_rate-1):
            if np.isnan(rim_array[i, j]) and rim_array[i, j+1] == 4:
                boundary['N4'].append(eta_array[i, j])
            if rim_array[i, j] == 4 and rim_array[i, j+1] == 3:
                boundary['43'].append(eta_array[i, j])
            if rim_array[i, j] == 3 and rim_array[i, j+1] == 2:
                boundary['32'].append(eta_array[i, j])
            if rim_array[i, j] == 2 and rim_array[i, j+1] == 1:
                boundary['21'].append(eta_array[i, j])
            if rim_array[i, j] == 1 and rim_array[i, j+1] == 0:
                boundary['10'].append(eta_array[i, j])
            if rim_array[i, j] == 0 and np.isnan(rim_array[i, j+1]):
                boundary['0N'].append(eta_array[i, j])
            
            
    beta_grid = beta_array.reshape(sampling_rate, sampling_rate)
    theta_grid = theta_array.reshape(sampling_rate, sampling_rate)
    rcond_grid = rcond_array.reshape(sampling_rate, sampling_rate)
    
    fig, ax = plt.subplots(figsize=(11, 6))
    mesh = ax.pcolormesh(beta_grid, theta_grid, rcond_grid, shading='auto', cmap='coolwarm')
    cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', pad=0.04, fraction=0.1)
    cbar.set_label(r'Reciprocal Jacobian Condition Number $(\frac{1}{\kappa_2})$')
    cbar.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    
    for key, value in boundary.items():
        value = np.array(value)
        plt.plot(value[:, 1], value[:, 0], color='dimgray', linestyle='--', linewidth=1.5, label=key)
    
    eta_leg = [[np.deg2rad(60),  np.deg2rad(0)], [np.deg2rad(90), np.deg2rad(0)], [np.deg2rad(120), np.deg2rad(0)],
               [np.deg2rad(30),  np.deg2rad(-30)], [np.deg2rad(30),  np.deg2rad(30)],
               [np.deg2rad(30),  np.deg2rad(-130)], [np.deg2rad(30),  np.deg2rad(-80)],
               [np.deg2rad(30),  np.deg2rad(80)], [np.deg2rad(30),  np.deg2rad(130)],
               [np.deg2rad(60),  np.deg2rad(-120)], [np.deg2rad(60),  np.deg2rad(-70)],
               [np.deg2rad(60),  np.deg2rad(70)], [np.deg2rad(60),  np.deg2rad(120)],
               [np.deg2rad(90),  np.deg2rad(-110)], [np.deg2rad(90),  np.deg2rad(-60)],
               [np.deg2rad(90),  np.deg2rad(60)], [np.deg2rad(90),  np.deg2rad(110)],
               [np.deg2rad(120),  np.deg2rad(-80)], [np.deg2rad(120),  np.deg2rad(80)]]
    
    for eta in eta_leg:
        leg.plot_leg(ax, eta[0], eta[1], offset=(eta[1], eta[0]))
    
    ax.text(np.deg2rad(0), np.deg2rad(145), r'$\bf{G\ Point}$', fontweight='bold', ha='center', va='top')
    ax.text(np.deg2rad(-100), np.deg2rad(140), r'$\bf{Right\ Upper\ Rim}$', fontweight='bold', ha='center', va='top')
    ax.text(np.deg2rad(100), np.deg2rad(140), r'$\bf{Left\ Upper\ Rim}$', ha='center', va='top')
    ax.text(np.deg2rad(-32), np.deg2rad(57), r'$\begin{array}{c}\bf{Right}\\\bf{Lower}\\\bf{Rim}\end{array}$', ha='center', va='top')
    ax.text(np.deg2rad(32), np.deg2rad(57), r'$\begin{array}{c}\bf{Left}\\\bf{Lower}\\\bf{Rim}\end{array}$', ha='center', va='top')
    
    ax.set_xticks(np.deg2rad([-180, -135, -90, -50, 0, 50, 90, 135, 180]))
    ax.set_yticks(np.deg2rad([17, 45, 90, 135, 160]))
    
    deg_formatter = FuncFormatter(lambda x, pos: f'{np.rad2deg(x):.0f}')
    ax.xaxis.set_major_formatter(deg_formatter)
    ax.yaxis.set_major_formatter(deg_formatter)

    ax.set_aspect(1.5)

    plt.xlabel(r'$\beta\ (deg)$')
    plt.ylabel(r'$\theta\ (deg)$')
    plt.title('Jacobian Reciprocal Condition Number Map')
    plt.tight_layout()

    # plt.savefig('Jacobian_kappa.pdf', format='pdf')
    # plt.savefig('Jacobian_kappa.png', format='png', dpi=600, transparent=False)
    plt.show()