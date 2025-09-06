import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from Plot.plot_image import *
from scipy.interpolate import CubicSpline


class BallTrajectoryPredictor:
    def __init__(self, g=9.81, K_D=0.0238):
        self.g = g  # 重力加速度 (m/s^2)
        self.K_D = K_D  # 空气阻力系数
        self.time_points = None  # 保存时间点
        self.trajectory = None  # 保存位置轨迹
        self.velocity = None  # 保存速度轨迹

    def equations_of_motion(self, t, s_b):
        b = s_b[:3]  # 位置 (x, y, z)
        v = s_b[3:]  # 速度 (vx, vy, vz)
        
        # 计算加速度，考虑重力和空气阻力
        norm_v = np.linalg.norm(v)
        a = -np.array([0, 0, self.g]) - self.K_D * norm_v * v
        
        # 返回速度和加速度
        return np.hstack((v, a))

    def predict_trajectory(self, s_b_initial, t_span, t_eval=None):
        # 使用数值积分求解运动方程
        sol = solve_ivp(self.equations_of_motion, t_span, s_b_initial, method='RK45', t_eval=t_eval)
        
        # 保存时间点、位置和速度轨迹
        self.time_points = sol.t
        self.trajectory = sol.y[:3].T   # 位置数据
        self.velocity = sol.y[3:].T     # 速度数据

    def get_position_at_time(self, t):
        if self.time_points is None or self.trajectory is None:
            raise ValueError("You must call predict_trajectory before querying positions.")
        
        # 创建三次样条插值函数
        interpolators = [CubicSpline(self.time_points, self.trajectory[:, i], extrapolate=True) for i in range(3)]
        
        # 计算时间 t 对应的位置
        position = np.array([interp(t) for interp in interpolators])
        return position

    def get_velocity_at_time(self, t):
        if self.time_points is None or self.velocity is None:
            raise ValueError("You must call predict_trajectory before querying velocities.")
        
        # 创建三次样条插值函数
        interpolators = [CubicSpline(self.time_points, self.velocity[:, i], extrapolate=True) for i in range(3)]
        
        # 计算时间 t 对应的速度
        velocity = np.array([interp(t) for interp in interpolators])
        return velocity
    
# 使用示例
if __name__ == "__main__":
    # 新的一个估计状态
    b_initial = np.array([0, 0, 2])  # 位置 (x, y, z)
    v_initial = np.array([2, 2, 3])  # 速度 (vx, vy, vz)
    
    # 状态向量 [位置, 速度]
    s_b_initial = np.hstack((b_initial, v_initial))
    
    # 创建预测器实例
    predictor = BallTrajectoryPredictor(K_D=0.0238)
    
    # 时间范围，当前时间为0
    t_span = (0, 2)  # 从 t=0 到 t=2 秒
    
    # 预测轨迹
    predictor.predict_trajectory(s_b_initial, t_span, t_eval=np.linspace(0, 2, 100))
    
    # 查询任意时间点的位置和速度
    times_to_query = [0.57, 1.0, 1.5, 1.75]
    for t in times_to_query:
        pos = predictor.get_position_at_time(t)
        vel = predictor.get_velocity_at_time(t)
        print(f"时间: {t:.2f}s, 位置: {pos}, 速度: {vel}")

    # 提取 x, y, z 位置数据
    x = predictor.trajectory[:, 0]
    y = predictor.trajectory[:, 1]
    z = predictor.trajectory[:, 2]
    data = [x, y, z]
    
    # 绘制3D轨迹图
    plot3d(data, "./Plot/pos_3d_test.png")
