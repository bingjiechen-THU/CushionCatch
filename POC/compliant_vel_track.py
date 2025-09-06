import cvxpy as cp
import numpy as np

class Vel_tracking:
    def __init__(self, gamma=0.1):
        self.gamma = gamma  # 控制增益

    def compute_joint_vel(self, v, jacobian, base_T, wTe_cy):
        # 根据雅可比矩阵列数定义优化变量维度
        jacobian = jacobian[:3]     # 只考虑线速度
        joint_velocities = cp.Variable(jacobian.shape[1])
        slack = cp.Variable(len(v))      # 三维松弛变量
        
        # 计算当前容器位姿
        [x_cy, y_cy, z_cy] = wTe_cy[0:3, 3]
        [x_base, y_base, z_base] = base_T[0:3, 3]


        # z轴CBF------------------考虑与地面的碰撞
        b_z = z_cy**2 - 0.1**2
        # 计算 b_z 的导数 
        z_dot = 2*z_cy * jacobian[2, :] @ joint_velocities

        # 自碰撞CBF----------------考虑与底座之间的碰撞
        xy_vel = jacobian[0:2, :] @ joint_velocities
        cy_base_xy = np.array([x_cy-x_base, y_cy-y_base])
        distance = np.linalg.norm(cy_base_xy)
        b_xy = np.sum(cy_base_xy**2) - 0.3**2
        
        constraints = []
        cbf_state = False

        if z_cy < 0.2:
            cbf_z = z_dot >= -self.gamma * b_z 
            constraints.append(cbf_z)
            cbf_state = True

        if distance < 0.6:
            direc_cons = xy_vel @ cy_base_xy >= -self.gamma * b_xy
            constraints.append(direc_cons)
            cbf_state = True                   


        if cbf_state:
            # 定义优化目标：最小化关节速度的平方和，并加入松弛变量的平方和
            constraints.append(jacobian @ joint_velocities == v + slack)
            objective = cp.Minimize(cp.sum_squares(joint_velocities) + cp.sum_squares(slack))
        else:
            constraints.append(jacobian @ joint_velocities == v)
            objective = cp.Minimize(cp.sum_squares(joint_velocities))


        # 定义并求解优化问题
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        return joint_velocities.value
