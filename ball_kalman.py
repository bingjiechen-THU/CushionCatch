import numpy as np

class KalmanFilter3D:
    def __init__(self, delta_t, KD, Q, R, initial_state, g=9.81):
        # 时间步长
        self.delta_t = delta_t
        # 空气阻力系数
        self.KD = KD
        # 过程噪声协方差矩阵
        self.Q = Q
        # 测量噪声协方差矩阵
        self.R = R
        # 重力加速度
        self.g = g
        
        # 初始状态 (位置和速度) [x, y, z, vx, vy, vz]
        self.s_b = initial_state.reshape((6, 1))    # 转化成列向量
        
        # 初始协方差矩阵
        self.P = np.eye(6) * 0.1
        
        # 状态转移矩阵 (6x6单位矩阵)
        self.F = np.array([
            [1, 0, 0, self.delta_t, 0, 0],
            [0, 1, 0, 0, self.delta_t, 0],
            [0, 0, 1, 0, 0, self.delta_t],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # 测量矩阵 (假设我们可以直接测量位置和速度)
        self.H = np.eye(6)
    
    def predict_state(self):
        # 计算加速度
        b = self.s_b[0:3] 
        v = self.s_b[3:6]
        acc = np.array([[0], [0], [-self.g]]) - self.KD * np.linalg.norm(v) * v
        
        # 状态预测
        v_new = v + self.delta_t * acc
        b_new = b + self.delta_t * v + 0.5 * self.delta_t**2 * acc
        s_b_new = np.vstack((b_new, v_new))
        
        return s_b_new
    
    def update(self, z):
        z = z.reshape((6, 1))       # 转化成列向量

        # 预测步骤
        s_b_pred = self.predict_state()
        
        # 预测协方差
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # 计算卡尔曼增益
        K = P_pred @ self.H.T @ np.linalg.inv(self.H @ P_pred @ self.H.T + self.R)
        
        # 更新状态
        self.s_b = s_b_pred + K @ (z - self.H @ s_b_pred)
        
        # 更新协方差
        self.P = (np.eye(6) - K @ self.H) @ P_pred
    
    def get_state(self):
        # 返回当前预测的状态 (位置和速度), 转化为行向量
        position = np.array(self.s_b[0:3].flatten())
        velocity = np.array(self.s_b[3:6].flatten())
        return position, velocity

# # 示例使用
# # 定义滤波器参数 
# delta_t = 0.1
# KD = 0.0238
# Q = np.eye(6) * 0.01  # 过程噪声协方差
# R = np.eye(6) * 0.1   # 测量噪声协方差
# initial_state = np.array([1.1, 1.1, 2.2, 0, 0, 0])

# # 创建卡尔曼滤波器实例
# kf = KalmanFilter3D(delta_t, KD, Q, R, initial_state)

# # 假设我们得到一个新的测量值 (三维位置和速度)
# measure_value = np.array([1, 1, 2, 0.5, 0.5, 0.5])

# # 更新卡尔曼滤波器
# kf.update(measure_value)

# # 获取预测的状态
# position, velocity = kf.get_state()
# print("预测的三维位置:", position)
# print("预测的三维速度:", velocity)
