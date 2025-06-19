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
            # contact_height.append(self.G[i,1]-self.r)
            contact_height.append(0)  # for wheel mode 
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
     
if __name__ == '__main__':
    legmodel = LegModel()