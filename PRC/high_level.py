import numpy as np
from scipy.optimize import minimize

class High_Level_Planner:
    def __init__(self, robot, predictor, end_cy_T, n_b=2, n_m=7, w_b=5.0, w_m=1.0, lambda_=0.4, 
                 q_b0=None, q_m0=None, 
                 R_coll=1.0, H_coll=1.5):
        # 设置参数
        self.w_b = w_b
        self.w_m = w_m
        self.lambda_ = lambda_
        self.robot = robot
        self.predictor = predictor
        self.end_cy_T = end_cy_T
        
        # robot机械臂以及底座的自由度
        self.n_b = n_b        
        self.n_m = n_m
        self.n = n_b+n_m

        # 初始配置
        self.q_b0 = q_b0 if q_b0 is not None else np.zeros(self.n_b)      
        self.q_m0 = q_m0 if q_m0 is not None else np.zeros(self.n_m)   

        # 半圆柱体空间参数
        self.R_coll = R_coll
        self.H_coll = H_coll
        
    def objective_function(self, x):

        q_bf = x[:self.n_b]          # 最终移动平台配置        
        q_mf = x[self.n_b : self.n]  # 最终机械臂关节配置
        t_f = x[self.n]              # 倾向于尽量使得最后捕捉时间慢一点

        cost = self.w_b * np.sum((self.q_b0 - q_bf)**2) + self.w_m * np.sum((self.q_m0 - q_mf)**2) - 10*t_f**2

        return cost

    # 对约束返回的理解：如果返回的是一个个单独的值，那么有多个返回是可以的，优化器会自动将其合并为一个列表
    # 但如果已经返回的是一个多维列表了，那么就不能再有多个返回并列，必须将这多个返回合并为一个列表
    def constraint_arm_base_collision(self, x):
        q_f = x[:self.n]     # 最终机器人配置
        wTe = self.robot.fkine(q_f).A
        ee_x = wTe[0, -1]
        ee_y = wTe[1, -1]
        ee_z = wTe[2, -1]
        # 确保末端执行器在半圆柱体内
        return self.R_coll**2 - (ee_x**2 + ee_y**2), ee_z, self.H_coll - ee_z
    
    def constraint_qf_box(self, x):
        q_f = x[:self.n]  # 最终机器人配置
        tf = x[self.n]    # 抓取时间
        # 定义最大抓取范围的box约束
        delta_q_max = np.hstack(([1,1], np.ones(self.n_m))) # 假设的最大变化量
        # delta_q = self.lambda_ * delta_q_max  # 根据比例参数计算
        delta_q = delta_q_max
        return delta_q - np.abs(q_f - np.hstack((self.q_b0, self.q_m0)))
    
    # 捕捉约束
    def constraint_catch_pose(self, x):
        q_f = x[:self.n]  # 最终机器人配置
        tf = x[self.n]    # 抓取时间

        # 位置约束,位置误差为0
        wTe_cy = (self.robot.fkine(q_f) @ self.end_cy_T).A
        cy_position = wTe_cy[:3, -1]           # 末端容器位置
        ball_position = self.predictor.get_position_at_time(tf)      # 球的位置
        position_error = np.linalg.norm(cy_position - ball_position)

        # 方向约束,球的速度方向和容器方向对齐
        ee_z_direction = wTe_cy[:3, 2]         # 末端容器的z轴方向向量
        ee_z_direction = ee_z_direction/np.linalg.norm(ee_z_direction)
        ball_vel = self.predictor.get_velocity_at_time(tf)          # 球的速度向量
        ball_vel = -ball_vel/np.linalg.norm(ball_vel)
        direction_error = 1 - np.dot(ee_z_direction, ball_vel)

        # 确保位置相同以及末端z轴方向与球速度方向相反
        return position_error, direction_error
    
    # 关节大小约束
    def constraint_qf_tf_limit(self, x):
        q_m_f = x[self.n_b:self.n]
        q_max = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        q_min = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        tf = x[self.n]              # 保证tf是大于0
        return np.hstack((q_max - q_m_f, q_m_f - q_min, tf))

    # 抓取高度约束
    def constraint_z(self, x):
        q_f = x[:self.n]  # 最终机器人配置
        tf = x[self.n]    # 抓取时间

        wTe_cy = (self.robot.fkine(q_f) @ self.end_cy_T).A
        cy_position = wTe_cy[:3, -1]           # 末端容器位置

        cons_z = cy_position[2] - 0.6

        return cons_z


    def optimize(self):
        # 初始猜测值（最终配置和抓取时间）
        x0 = np.hstack((self.q_b0, self.q_m0, 0.5))  # 初始猜测
        
        # 定义优化问题的约束
        constraints = [
            # {'type': 'ineq', 'fun': lambda x: self.constraint_arm_base_collision(x)},
            # {'type': 'ineq', 'fun': lambda x: self.constraint_qf_box(x)},            
            {'type': 'eq', 'fun': lambda x: self.constraint_catch_pose(x)},
            # {'type': 'ineq', 'fun': lambda x: self.constraint_qf_tf_limit(x)},
            {'type': 'ineq', 'fun': lambda x: self.constraint_z(x)}
        ]
        
        # 使用SLSQP算法求解优化问题
        result = minimize(self.objective_function, x0, method='SLSQP', constraints=constraints)
        
        # 输出优化结果
        q_f_opt = result.x[:self.n]
        t_f_opt = result.x[self.n]

        print("优化结果：", result.success)
        
        return q_f_opt, t_f_opt


# # 示例使用
# optimizer = high_level_planner()
# q_bf_opt, q_mf_opt, t_f_opt = optimizer.optimize()

# print("优化后的最终机械臂关节配置:", q_mf_opt)
# print("优化后的最终移动平台配置:", q_bf_opt)
# print("优化后的抓取时间 t_f:", t_f_opt)
