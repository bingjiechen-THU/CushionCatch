import spatialgeometry as sg
import spatialmath as sm
import qpsolvers as qp
import numpy as np

import Env.scene as scene
from Plot.plot_image import *

from PRC.ball_kalman import KalmanFilter3D
from PRC.ball_trajectory import BallTrajectoryPredictor
from PRC.high_level import High_Level_Planner

from PRC.prc_joint_planner import PRCJointPlanner
from POC.poc_traj_learner import POC_Traj_Learner
from POC.compliant_vel_track import Vel_tracking


class Catcher():
    """
    A class to simulate a robotic arm catching a ball.
    """
    def __init__(self, initial_state) -> None:
        # --- Robot Simulation Environment Variables ---
        self.env, self.robot = scene.swift_scene()  # Initialize the Swift environment and the robot
        self.ball = sg.Sphere(radius=0.05, base=sm.SE3(3, 3, 3), color=(0.7, 0.2, 0.1, 1.0))
        self.env.add(self.ball)

        # Homogeneous transformation matrix of the catching cylinder relative to the end-effector
        self.end_cy_T = sm.SE3(np.array([
            [0,  0,  1,  0],
            [0,  1,  0,  0],
            [-1, 0,  0,  0.06],
            [0,  0,  0,  1]
        ]))
        wTe = self.robot.fkine(self.robot.q)
        # Cylindrical container for catching
        self.end_cy = sg.Cylinder(
            radius=0.06, length=0.2, base=wTe @ self.end_cy_T, color=(0.3, 0.3, 0.3, 0.4)
        )
        self.env.add(self.end_cy)
        self.position_error = 99

        # --- Robot Catching Process Variables ---
        self.sim_t = 0
        self.delta_t = 0.01
        KD = 0.0238
        Q = np.eye(6) * 0.01    # Process noise covariance
        R = np.eye(6) * 0.1     # Measurement noise covariance
        # Create a Kalman filter instance based on the ball's initial state
        self.kf = KalmanFilter3D(self.delta_t, KD, Q, R, initial_state)

        # --- Trajectory Prediction and Optimal Catch Point Calculation ---
        self.p_state = None
        self.v_state = None
        self.q_f_opt = None
        self.t_opt = None
        self.v_at_catch = None
        self.predictor = BallTrajectoryPredictor(K_D=0.0238)
        self.high_level_planner = High_Level_Planner(
            self.robot, self.predictor, self.end_cy_T,
            q_b0=self.robot.q[:2], q_m0=self.robot.q[2:9]
        )

        # --- Joint Space Trajectory Planner ---
        joint_max_vel = [2, 2, 2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]
        joint_max_acc = [10, 10, 15, 7.5, 10, 12.5, 15, 20, 20]
        self.joint_planner = PRCJointPlanner(joint_max_vel, joint_max_acc)

        # --- Compliant Control Velocity Predictor (Neural Network) ---
        self.poc_traj_learner = POC_Traj_Learner(load_path="./Compliance_Learner/ckpt/model_weights.pth")

        # --- Compliant Control Collision Avoidance ---
        self.vel_tracking = Vel_tracking(gamma=0.1)

    def state_estimate(self, measure_value):
        """
        Estimates the ball's state (position and velocity) using a Kalman filter.
        """
        # Update the Kalman filter with the new measurement
        self.kf.update(measure_value)
        # Get the predicted state
        position, velocity = self.kf.get_state()
        print("Estimated 3D Position:", position)
        print("Estimated 3D Velocity:", velocity)
        return position, velocity

    def get_desired_state(self, measure_value):
        """
        Calculates the desired robot state for catching the ball.
        """
        self.p_state, self.v_state = self.state_estimate(measure_value)
        p_v_initial = np.hstack((self.p_state, self.v_state))
        t_span = (0, 2)
        t_eval = np.linspace(0, 2, 100)
        # Predict the ball's trajectory
        self.predictor.predict_trajectory(p_v_initial, t_span, t_eval)
        # Optimize the catch point
        self.q_f_opt, self.t_opt = self.high_level_planner.optimize()
        self.desired_pose_ee = self.robot.fkine(self.q_f_opt)
        self.v_at_catch = self.predictor.get_velocity_at_time(self.t_opt)
        self.posi_at_catch = self.predictor.get_position_at_time(self.t_opt)
        print("high_level_planner:", self.q_f_opt, self.t_opt)

    def get_joint_traj(self):
        """
        Generates the joint space trajectory to the optimal catch point.
        """
        self.con_step = 0  # Control step counter
        self.prc_steps, self.joint_traj, self.joint_vel_traj, _ = self.joint_planner.get_trajectory(
            self.robot.q, self.q_f_opt, delta_t=0.01, lambda_param=1.2
        )

    def get_compliant_vel(self):
        """
        Predicts the compliant velocity trajectory for the post-catch phase.
        """
        inputs = np.hstack((self.v_at_catch, self.posi_at_catch))
        self.compliant_vel_traj = self.poc_traj_learner.predict_vel(inputs)
        self.compliant_step = 0
        self.comliant_total_steps = len(self.compliant_vel_traj)

    # --- Pre-catch and Catch Simulation ---
    def robot_sim(self):
        """
        Simulates one step of the robot's movement before catching the ball.
        """
        if self.con_step < self.prc_steps:
            self.robot.qd = self.joint_vel_traj[self.con_step]
            self.con_step += 1

        # Simulate the end-effector container's movement
        self.end_cy.T = self.robot.fkine(self.robot.q) @ self.end_cy_T
        # Simulate the ball's movement
        self.ball.T = sm.SE3(self.predictor.get_position_at_time(self.sim_t))
        self.position_error = np.linalg.norm((self.end_cy.T)[:3, -1] - (self.ball.T)[:3, -1])
        print("position_error=", self.position_error)
        self.env.step(self.delta_t)
        self.sim_t += self.delta_t

    # --- Compliant Control Simulation ---
    def robot_sim_compliance(self):
        """
        Simulates one step of the robot's compliant movement after catching the ball.
        """
        if self.compliant_step < self.comliant_total_steps:
            wTe_cy = (self.robot.fkine(self.robot.q) @ self.end_cy_T).A
            base_T = self.robot._T
            self.robot.qd = self.vel_tracking.compute_joint_vel(
                self.compliant_vel_traj[self.compliant_step],
                self.robot.jacob0(self.robot.q),
                base_T, wTe_cy
            )
            self.compliant_step += 1

        # Simulate the end-effector container's movement
        self.end_cy.T = self.robot.fkine(self.robot.q) @ self.end_cy_T
        # The ball is now "in" the catcher, so it moves with the catcher
        self.ball.T = sm.SE3(self.end_cy.T[:3, -1])
        self.env.step(self.delta_t * 4)
        self.sim_t += self.delta_t * 4


if __name__ == '__main__':
    initial_state = np.array([3, 3, 3, -1, -2, 3])
    noise = np.random.normal(0, 0.05, size=initial_state.shape)
    measure_value = initial_state + noise

    catcher = Catcher(initial_state)
    catcher.get_desired_state(measure_value)
    catcher.get_joint_traj()
    catcher.get_compliant_vel()

    # Simulation loop for pre-catch phase
    while catcher.position_error > 5e-2:
        catcher.robot_sim()

    # Simulation loop for post-catch compliant phase
    while catcher.compliant_step < catcher.comliant_total_steps:
        catcher.robot_sim_compliance()

    # --- Print Catch Information and Plot Results ---

    # Ball trajectory data at the moment of catch
    pos = catcher.predictor.get_position_at_time(catcher.t_opt)
    vel = catcher.predictor.get_velocity_at_time(catcher.t_opt)
    print(f"Time: {catcher.t_opt:.2f}s, Position: {pos}, Velocity: {vel}")

    # Plot the 3D trajectory
    x = catcher.predictor.trajectory[:, 0]
    y = catcher.predictor.trajectory[:, 1]
    z = catcher.predictor.trajectory[:, 2]
    data = [x, y, z]
    plot3d(data, "./Plot/pos_3d.png", pos)
