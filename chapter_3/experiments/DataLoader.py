import pandas as pd
import numpy as np
from LegModel import LegModel
from scipy.signal import butter, filtfilt
import ViconProcess

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class DataLoader:
    def __init__(self, sim, trigger_idx=None):
        self.cutoff_freq = 50
        self.sim = sim
        self.trigger_idx = trigger_idx
        
        self.legmodel = LegModel(sim=self.sim)
        
    
    def load_robot_data(self, file_path, start_idx=0, end_idx=-1):
        self.df_robot = pd.read_csv(file_path).iloc[start_idx:end_idx, :]
        
        self.state_force_x = np.array([self.df_robot['force_Fx_a'], self.df_robot['force_Fx_b'], self.df_robot['force_Fx_c'], self.df_robot['force_Fx_d']])
        self.state_force_z = np.array([self.df_robot['force_Fy_a'], self.df_robot['force_Fy_b'], self.df_robot['force_Fy_c'], self.df_robot['force_Fy_d']])
        self.state_force_x = self.low_pass_filter(self.state_force_x)
        self.state_force_z = self.low_pass_filter(self.state_force_z)
        
        self.cmd_force_x = np.array([self.df_robot['imp_cmd_Fx_a'], self.df_robot['imp_cmd_Fx_b'], self.df_robot['imp_cmd_Fx_c'], self.df_robot['imp_cmd_Fx_d']])
        self.cmd_force_z = np.array([self.df_robot['imp_cmd_Fy_a'], self.df_robot['imp_cmd_Fy_b'], self.df_robot['imp_cmd_Fy_c'], self.df_robot['imp_cmd_Fy_d']])
        self.cmd_force_x = self.low_pass_filter(self.cmd_force_x)
        self.cmd_force_z = self.low_pass_filter(self.cmd_force_z)
        
        self.state_theta = np.array([self.df_robot['state_theta_a'], self.df_robot['state_theta_b'], self.df_robot['state_theta_c'], self.df_robot['state_theta_d']])
        self.state_beta  = np.array([self.df_robot['state_beta_a'] , self.df_robot['state_beta_b'] , self.df_robot['state_beta_c'] , self.df_robot['state_beta_d']])

        self.cmd_theta = np.array([self.df_robot['cmd_theta_a'], self.df_robot['cmd_theta_b'], self.df_robot['cmd_theta_c'], self.df_robot['cmd_theta_d']])
        self.cmd_beta  = np.array([self.df_robot['cmd_beta_a'] , self.df_robot['cmd_beta_b'] , self.df_robot['cmd_beta_c'] , self.df_robot['cmd_beta_d']])
        
        self.odom_pos_x = self.df_robot['odom_pos_x']
        self.odom_pos_z = self.df_robot['odom_pos_z']
        
        self.odom_vel_x = self.df_robot['odom_vel_x']
        self.odom_vel_z = self.df_robot['odom_vel_z']
        self.odom_vel_x = self.low_pass_filter(self.odom_vel_x)
        self.odom_vel_z = self.low_pass_filter(self.odom_vel_z)
        
        self.sim_pos_x = self.df_robot['sim_pos_x']
        self.sim_pos_z = self.df_robot['sim_pos_z']
        
        imu_qx = self.df_robot['imu_orien_x']
        imu_qy = self.df_robot['imu_orien_y']
        imu_qz = self.df_robot['imu_orien_z']
        imu_qw = self.df_robot['imu_orien_w']
        imu_roll = np.arctan2(2 * (imu_qw * imu_qx + imu_qy * imu_qz), 1 - 2 * (imu_qx**2 + imu_qy**2))
        imu_pitch = np.arcsin(2 * (imu_qw * imu_qy - imu_qz * imu_qx))
        self.imu_roll = np.rad2deg(imu_roll)
        self.imu_pitch = np.rad2deg(imu_pitch)
        if not self.sim: self.imu_pitch *= -1
        
        self.imu_acc_x = self.df_robot['imu_lin_acc_x']
        self.imu_acc_z = self.df_robot['imu_lin_acc_z']
                
        state_leg_z = []
        for i in range(4):
            self.legmodel.contact_map(self.state_theta[i], self.state_beta[i])
            state_leg_z.append(self.legmodel.contact_p[:, 1])
        self.state_leg_z = np.array(state_leg_z)
        
        cmd_leg_z = []
        for i in range(4):
            self.legmodel.contact_map(self.state_theta[i], self.state_beta[i])
            cmd_leg_z.append(self.legmodel.contact_p[:, 1])
        self.cmd_leg_z = np.array(cmd_leg_z)
        
        state_rim = []
        for i in range(4):
            self.legmodel.contact_map(self.state_theta[i], self.state_beta[i])
            state_rim.append(self.legmodel.rim)
        self.state_rim = np.array(state_rim)
    
    
    def load_vicon_data(self, file_path, start_idx=0, end_idx=-1):
        self.df_vicon, trigger_idx = ViconProcess.read_csv(file_path)
        if self.trigger_idx is None: self.trigger_idx = trigger_idx

        self.df_vicon = self.df_vicon.iloc[self.trigger_idx+start_idx:self.trigger_idx+end_idx, :]
        
        self.vicon_pos_x = self.df_vicon['vicon_pos_x']
        self.vicon_pos_z = self.df_vicon['vicon_pos_z']
        self.vicon_pos_x -= self.vicon_pos_x.iloc[0]
        
        self.vicon_roll = self.df_vicon['vicon_roll']
        self.vicon_pitch = self.df_vicon['vicon_pitch']
        
        self.vicon_force_x = np.array([self.df_vicon['Fx_1'], self.df_vicon['Fx_4'], self.df_vicon['Fx_3'], self.df_vicon['Fx_2']])
        self.vicon_force_z = np.array([self.df_vicon['Fz_1'], self.df_vicon['Fz_4'], self.df_vicon['Fz_3'], self.df_vicon['Fz_2']])
        self.vicon_force_x = self.low_pass_filter(self.vicon_force_x)
        self.vicon_force_z = self.low_pass_filter(self.vicon_force_z)
        
        # model = LinearRegression()
        # model.fit(self.vicon_pos_x.to_numpy().reshape(-1, 1), self.vicon_pos_z.to_numpy())
        # line_fit = model.predict(self.vicon_pos_x.to_numpy().reshape(-1, 1))
        
        # self.vicon_pos_z -= line_fit
        self.vicon_pos_z -= self.vicon_pos_z.iloc[0]
        self.vicon_pos_z += self.odom_pos_z.iloc[0]
    
    
    def load_sim_force_data(self, file_path, start_idx=0, end_idx=-1, trigger_idx=0):
        self.df_sim_force = pd.read_csv(file_path).iloc[start_idx+trigger_idx:end_idx+trigger_idx, :]
        
        self.sim_force_x = np.array([self.df_sim_force['Fx_1'], self.df_sim_force['Fx_4'], self.df_sim_force['Fx_3'], self.df_sim_force['Fx_2']])
        self.sim_force_z = np.array([self.df_sim_force['Fz_1'], self.df_sim_force['Fz_4'], self.df_sim_force['Fz_3'], self.df_sim_force['Fz_2']])
        self.sim_force_x = self.low_pass_filter(self.sim_force_x)
        self.sim_force_z = self.low_pass_filter(self.sim_force_z)
        
    
    def low_pass_filter(self, data, fs=1000, order=2):
        nyquist = 0.5 * fs
        normal_cutoff = self.cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data, axis=-1)
        return filtered_data
    
    
    def data_process(self):
        if self.sim:
            self.sim_force_z = np.where(self.sim_force_z >= 0, 0, self.sim_force_z)
            self.state_force_z = np.where(self.state_force_z <= 0, 0, self.state_force_z)
            self.state_force_z = np.where(self.sim_force_z > -2, 0, self.state_force_z)

            self.state_force_x = np.where(self.state_force_z == 0, 0, self.state_force_x)
        else:
            self.vicon_force_z = np.where(self.vicon_force_z >= 0, 0, self.vicon_force_z)
            self.state_force_z = np.where(self.state_force_z <= 0, 0, self.state_force_z)
            self.state_force_z = np.where(self.vicon_force_z > -2, 0, self.state_force_z)

            self.state_force_x = np.where(self.state_force_z == 0, 0, self.state_force_x)
    
    
    def compute_rmse(self):
        self.data_process()
        
        if self.sim:
            rmse_x_rf = np.sqrt(mean_squared_error(self.state_force_x[1], self.sim_force_x[1]))
            rmse_z_rf = np.sqrt(mean_squared_error(self.state_force_z[1], -self.sim_force_z[1]))

            rmse_x_lh = np.sqrt(mean_squared_error(self.state_force_x[3], self.sim_force_x[3]))
            rmse_z_lh = np.sqrt(mean_squared_error(self.state_force_z[3], -self.sim_force_z[3]))

            print(f"RMSE: X = {((rmse_x_rf+rmse_x_lh)/2):.2f} N, Z = {((rmse_z_rf+rmse_z_lh)/2):.2f} N")
        else:
            rmse_x_rf = np.sqrt(mean_squared_error(self.state_force_x[1], -self.vicon_force_x[1]))
            rmse_z_rf = np.sqrt(mean_squared_error(self.state_force_z[1], -self.vicon_force_z[1]))

            rmse_x_lh = np.sqrt(mean_squared_error(self.state_force_x[3], -self.vicon_force_x[3]))
            rmse_z_lh = np.sqrt(mean_squared_error(self.state_force_z[3], -self.vicon_force_z[3]))

            print(f"RMSE: X = {((rmse_x_rf+rmse_x_lh)/2):.2f} N, Z = {((rmse_z_rf+rmse_z_lh)/2):.2f} N")
