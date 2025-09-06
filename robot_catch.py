import swift
import spatialgeometry as sg
import spatialmath as sm
import qpsolvers as qp
import numpy as np

import scene
from plot_image import *
from ball_kalman import KalmanFilter3D
from ball_trajectory import BallTrajectoryPredictor
from high_level import High_Level_Planner
from compliance_learning.poc_traj_learner import POC_Traj_Learner
from compliant_vel_track import Vel_tracking
from prc_joint_planner import PRCJointPlanner

class Catcher():
    def __init__(self, initial_state) -> None:
        # ----------------------------------------机器人仿真环境变量
        self.env, self.robot = scene.swift_scene()  # 添加场景.  环境、robot、目标点、障碍物、目标点、动态障碍物1、动态障碍物2
        self.ball = sg.Sphere(radius=0.05, base=sm.SE3(3, 3, 3), color=(0.7, 0.2, 0.1, 1.0))
        self.env.add(self.ball)
        # 接球圆柱相对于末端的齐次变换矩阵
        self.end_cy_T = R = sm.SE3(np.array([
                                    [0,  0,  1,  0],
                                    [0,  1,  0,  0],
                                    [-1, 0,  0,  0.06],
                                    [0,  0,  0,  1]
                                ]))
        wTe = self.robot.fkine(self.robot.q)
        self.end_cy = sg.Cylinder(radius=0.06, length=0.2, base=wTe@self.end_cy_T, color=(0.3, 0.3, 0.3, 0.4))   # 圆柱容器
        self.env.add(self.end_cy)        
        self.position_error = 99

        # -----------------------------------------机器人捕捉过程变量
        self.sim_t = 0
        self.delta_t = 0.01
        KD = 0.0238
        Q = np.eye(6) * 0.01    # 过程噪声协方差
        R = np.eye(6) * 0.1     # 测量噪声协方差
        # 创建卡尔曼滤波器实例,根据球的初始状态建立模型
        self.kf = KalmanFilter3D(self.delta_t, KD, Q, R, initial_state)  
        
        # -----------------------------------------根据估计状态预测球的轨迹，并计算最佳捕捉点
        self.p_state = None
        self.v_state = None
        self.q_f_opt = None
        self.t_opt = None
        self.v_at_catch = None
        self.predictor = BallTrajectoryPredictor(K_D=0.0238)
        self.high_level_planner = High_Level_Planner(self.robot, self.predictor, self.end_cy_T,
                                                     q_b0=self.robot.q[:2],q_m0=self.robot.q[2:9])
        
        #-------------------------------------------创建到最佳捕捉点的关节空间轨迹规划器
        joint_max_vel = [2, 2, 2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]
        joint_max_acc = [10, 10, 15, 7.5, 10, 12.5, 15, 20, 20]      
        self.joint_planner = PRCJointPlanner(joint_max_vel, joint_max_acc)

        #-------------------------------------------创建柔顺控制速度的网络预测器
        self.poc_traj_learner = POC_Traj_Learner(load_path="./compliance_learning/ckpt/model_weights.pth")

        #--------------------------------------------创建柔顺控制的碰撞避免
        self.vel_tracking = Vel_tracking(gamma=0.1)


    def state_estimate(self, measure_value):
        # measure_value = np.array([[1.0], [1.0], [1.0], [0.5], [0.5], [0.5]])
        # 更新卡尔曼滤波器
        self.kf.update(measure_value)
        # 获取预测的状态
        position, velocity = self.kf.get_state()
        print("估计的三维位置:", position)
        print("估计的三维速度:", velocity)
        return position, velocity
    
    def get_desired_state(self, measure_value):
        self.p_state, self.v_state = self.state_estimate(measure_value)
        p_v_initial = np.hstack((self.p_state, self.v_state))
        t_span = (0, 2)
        t_eval=np.linspace(0, 2, 100)
        # t_span = (0, 10)
        # t_eval=np.linspace(0, 10, 100)        
        self.predictor.predict_trajectory(p_v_initial, t_span, t_eval)
        self.q_f_opt, self.t_opt = self.high_level_planner.optimize()
        self.desired_pose_ee = self.robot.fkine(self.q_f_opt)
        self.v_at_catch = self.predictor.get_velocity_at_time(self.t_opt)
        self.posi_at_catch = self.predictor.get_position_at_time(self.t_opt)
        print("high_level_planner:", self.q_f_opt, self.t_opt)        

    def get_joint_traj(self):
        self.con_step = 0       # 控制步
        self.prc_steps, self.joint_traj, self.joint_vel_traj, _ = self.joint_planner.get_trajectory(self.robot.q, self.q_f_opt, delta_t=0.01, lambda_param=1.2)
        # print("joint_trajectory:", self.joint_traj)

    def get_compliant_vel(self):
        inputs = np.hstack((self.v_at_catch, self.posi_at_catch))
        self.compliant_vel_traj = self.poc_traj_learner.predict_vel(inputs)
        self.compliant_step = 0
        self.comliant_total_steps = len(self.compliant_vel_traj)

    # ---------------------------------------------接球前以及接球时的仿真
    def robot_sim(self):        
        if self.con_step < self.prc_steps:
            self.robot.qd = self.joint_vel_traj[self.con_step]
            self.con_step += 1

        # 末端容器仿真
        self.end_cy.T = self.robot.fkine(self.robot.q) @ self.end_cy_T    
        # 球运动仿真
        self.ball.T = sm.SE3(self.predictor.get_position_at_time(self.sim_t))               # sm.SE3作用为从位置数据变换为3轴世界中的齐次变换矩阵
        self.position_error = np.linalg.norm((self.end_cy.T)[:3, -1] - (self.ball.T)[:3, -1])
        print("position_error=", self.position_error)
        self.env.step(self.delta_t)
        self.sim_t += self.delta_t

    # -----------------------------------------------------柔顺控制仿真
    def robot_sim_compliance(self):
        if self.compliant_step < self.comliant_total_steps:
            # self.robot.qd = self.pose_corr.compute_joint_velocities(
            #     self.compliant_vel_traj[self.compliant_step],
            #     self.robot.jacob0(self.robot.q), linear_only=True)
            wTe_cy = (self.robot.fkine(self.robot.q) @ self.end_cy_T).A
            base_T = self.robot._T
            self.robot.qd = self.vel_tracking.compute_joint_vel(
                self.compliant_vel_traj[self.compliant_step],
                self.robot.jacob0(self.robot.q), 
                base_T, wTe_cy
            )
            self.compliant_step += 1

        # 末端容器仿真
        self.end_cy.T = self.robot.fkine(self.robot.q) @ self.end_cy_T    
        # 球运动仿真
        self.ball.T = sm.SE3(self.end_cy.T[:3, -1])               # sm.SE3作用为从位置数据变换为3轴世界中的齐次变换矩阵
        self.env.step(self.delta_t*4)
        self.sim_t += self.delta_t*4

if __name__ == '__main__':

    # 初始化卡尔曼滤波器的状态
    # initial_state = np.array([0, 2, 2, 0, -1.5, 2])
    # initial_state = np.array([-2, -2, 3, 1, 1.5, 2])
    # initial_state = np.array([-2, -1, 2, 0.6, 1.5, 4])
    initial_state = np.array([3, 3, 3, -1, -2, 3])
    catcher = Catcher(initial_state)
    # 模拟动捕的数据
    measure_value = np.array([3, 3, 3, -1, -2, 3])
    catcher.get_desired_state(measure_value)
    catcher.get_joint_traj()
    catcher.get_compliant_vel()
    # 仿真循环
    while catcher.position_error > 5e-2:
        catcher.robot_sim()
    while catcher.compliant_step < catcher.comliant_total_steps:
        catcher.robot_sim_compliance()

    #-----------------------------------------------------抓捕信息打印以及绘图

    # 球轨迹数据
    pos = catcher.predictor.get_position_at_time(catcher.t_opt)
    vel = catcher.predictor.get_velocity_at_time(catcher.t_opt)
    print(f"时间: {catcher.t_opt:.2f}s, 位置: {pos}, 速度: {vel}")

    # 绘制3D轨迹图
    x = catcher.predictor.trajectory[:, 0]
    y = catcher.predictor.trajectory[:, 1]
    z = catcher.predictor.trajectory[:, 2]
    data = [x, y, z]
    plot3d(data, "./figures/pos_3d.png", pos)


    


