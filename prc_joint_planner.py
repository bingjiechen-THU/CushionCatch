import numpy as np

class PRCJointPlanner:
    """
    PRC (Polynomial Rate Control) 关节空间轨迹规划器。
    
    该规划器使用5次多项式为多关节机器人生成平滑的轨迹，
    确保在指定的运动学约束（最大速度和加速度）内，
    从初始关节配置移动到目标关节配置。
    """

    def __init__(self, qd_max, acc_max):
        """
        初始化 PRC 规划器。
        
        这个阶段只设置机器人的固定物理约束。
        
        参数:
        - qd_max (list or np.array): 每个关节的最大速度 (rad/s)。
        - acc_max (list or np.array): 每个关节的最大加速度 (rad/s^2)。
        """
        self.qd_max = np.asarray(qd_max)
        self.acc_max = np.asarray(acc_max)

    def _compute_t_prc(self, delta_q):
        """
        内部方法：计算每个关节独立运动到目标所需的最短时间。
        
        参数:
        - delta_q (np.array): 每个关节需要运动的绝对角度差。
        
        返回:
        - np.array: 每个关节的规划时间 (t_prc)。
        """
        t_prc = np.zeros_like(delta_q)
        for i in range(len(delta_q)):
            # 检查是否可以构建一个完整的加速-匀速-减速的梯形速度曲线
            if self.delta_q[i] < self.acc_max[i] * (self.qd_max[i] / self.acc_max[i])**2:
                # 速度曲线为三角形（只有加速和减速段）
                t_prc[i] = 2 * np.sqrt(delta_q[i] / self.acc_max[i])
            else:
                # 速度曲线为梯形（有匀速段）
                t_prc[i] = (2 * self.qd_max[i] / self.acc_max[i] + 
                            (delta_q[i] - self.acc_max[i] * (self.qd_max[i] / self.acc_max[i])**2) / self.qd_max[i])
        return t_prc

    def _generate_poly5_trajectory(self, t, T, q0, qca):
        """
        内部方法：使用5次多项式及其导数生成单个关节的位置、速度和加速度。
        
        位置: q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        速度: v(t) = a1 + 2*a2*t + 3*a3*t^2 + 4*a4*t^3 + 5*a5*t^4
        加速度: a(t) = 2*a2 + 6*a3*t + 12*a4*t^2 + 20*a5*t^3
        
        参数:
        - t (np.array): 时间序列。
        - T (float): 总规划时间。
        - q0 (float): 初始位置。
        - qca (float): 目标位置。
        
        返回:
        - tuple(np.array, np.array, np.array): 该关节的位置、速度和加速度轨迹。
        """
        # 边界条件：初始和最终速度、加速度均为0
        # q(0)=q0, q(T)=qca, v(0)=0, v(T)=0, a(0)=0, a(T)=0
        a0 = q0
        a1 = 0
        a2 = 0
        a3 = 10 * (qca - q0) / T**3
        a4 = -15 * (qca - q0) / T**4
        a5 = 6 * (qca - q0) / T**5

        # 计算位置 q(t)
        q_t = a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
        
        # 计算速度 v(t) - q(t)的一阶导数
        v_t = a1 + 2 * a2 * t + 3 * a3 * t**2 + 4 * a4 * t**3 + 5 * a5 * t**4
        
        # 计算加速度 a(t) - q(t)的二阶导数
        a_t = 2 * a2 + 6 * a3 * t + 12 * a4 * t**2 + 20 * a5 * t**3
        
        return q_t, v_t, a_t

    def get_trajectory(self, q_0, q_ca, delta_t, lambda_param=1.2):
        """
        执行完整的轨迹规划流程并返回结果。
        
        参数:
        - q_0 (list or np.array): 关节的初始配置 (rad)。
        - q_ca (list or np.array): 关节的目标配置 (rad)。
        - delta_t (float): 仿真步长时间 (s)。
        - lambda_param (float): 时间扩展系数，用于确保所有关节同步到达。默认为 1.2。
        
        返回:
        - tuple(np.array, np.array, np.array): 
            - q_trajectory: 关节位置轨迹 (N_joints x N_steps)
            - velocities: 关节速度轨迹 (N_joints x N_steps)
            - accelerations: 关节加速度轨迹 (N_joints x N_steps)
        """
        self.q_0 = np.asarray(q_0)
        self.q_ca = np.asarray(q_ca)
        self.delta_t = delta_t
        self.lambda_param = lambda_param
        
        # 1. 计算每个关节的角度差
        self.delta_q = np.abs(self.q_ca - self.q_0)
        print("delta_q=", self.delta_q)
        
        # 2. 计算每个关节独立运动所需的最短时间
        t_prc_individual = self._compute_t_prc(self.delta_q)
        
        # 3. 计算整体规划时间 T_prc（取最慢的关节时间并乘以扩展系数）
        T_prc = self.lambda_param * np.max(t_prc_individual)
        print("T_prc=", T_prc)

        # 4. 生成时间序列
        t_values = np.arange(0, T_prc, self.delta_t)
        
        # 5. 为每个关节生成轨迹
        q_list, v_list, a_list = [], [], []
        for i in range(len(self.q_0)):
            q_traj, v_traj, a_traj = self._generate_poly5_trajectory(
                t_values, T_prc, self.q_0[i], self.q_ca[i]
            )
            q_list.append(q_traj)
            v_list.append(v_traj)
            a_list.append(a_traj)
            
        # 6. 将轨迹列表转换为Numpy数组并转置为 (N_steps, N_joints)
        q_trajectory = np.array(q_list).T
        velocities = np.array(v_list).T
        accelerations = np.array(a_list).T
        
        return len(q_trajectory), q_trajectory, velocities, accelerations

# 示例使用
if __name__ == "__main__":
    # 设置9个自由度的关节参数
    qd_max = [1.0] * 9  # 最大速度 (rad/s)
    acc_max = [0.5] * 9  # 最大加速度 (rad/s^2)
    q_0 = [0.0] * 9    # 初始配置 (rad)
    q_ca = [1.5] * 9   # 目标配置 (rad)

    # 创建 PRC 规划器
    planner = PRCJointPlanner(qd_max, acc_max)

    # 获取轨迹结果
    steps, q_trajectory, velocities, accelerations = planner.get_trajectory(q_0, q_ca, delta_t=0.01, lambda_param=1.2)

    # 输出结果
    print(steps)
    # print("Joint Trajectories (Position):")
    # print(q_trajectory)
    print("Velocities:")
    print(velocities[0])
    # print("Accelerations:")
    # print(accelerations)
