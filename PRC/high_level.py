import numpy as np
from scipy.optimize import minimize


class High_Level_Planner:
    """
    High-level planner for determining the optimal robot configuration and time for catching an object.
    """

    def __init__(self, robot, predictor, end_cy_T, n_b=2, n_m=7, w_b=5.0, w_m=1.0,
                 q_b0=None, q_m0=None,
                 R_coll=1.0, H_coll=1.5):
        """
        Initializes the High_Level_Planner.

        Args:
            robot: The robot model.
            predictor: The object trajectory predictor.
            end_cy_T: Transformation from the end-effector to the container.
            n_b (int): Degrees of freedom of the mobile base.
            n_m (int): Degrees of freedom of the manipulator.
            w_b (float): Weight for the base movement cost.
            w_m (float): Weight for the manipulator movement cost.
            lambda_ (float): Scaling factor for joint movement limits.
            q_b0 (np.ndarray): Initial configuration of the mobile base.
            q_m0 (np.ndarray): Initial configuration of the manipulator.
            R_coll (float): Radius of the cylindrical collision space.
            H_coll (float): Height of the cylindrical collision space.
        """
        # Set parameters
        self.w_b = w_b
        self.w_m = w_m
        self.robot = robot
        self.predictor = predictor
        self.end_cy_T = end_cy_T

        # Degrees of freedom for the robot arm and base
        self.n_b = n_b
        self.n_m = n_m
        self.n = n_b + n_m

        # Initial configuration
        self.q_b0 = q_b0 if q_b0 is not None else np.zeros(self.n_b)
        self.q_m0 = q_m0 if q_m0 is not None else np.zeros(self.n_m)

        # Half-cylinder space parameters
        self.R_coll = R_coll
        self.H_coll = H_coll

    def objective_function(self, x):
        """
        Calculates the cost function to be minimized.
        """
        q_bf = x[:self.n_b]          # Final mobile platform configuration
        q_mf = x[self.n_b: self.n]  # Final manipulator joint configuration
        t_f = x[self.n]              # Capture time (objective prefers a later time)

        cost = self.w_b * np.sum((self.q_b0 - q_bf)**2) + self.w_m * np.sum((self.q_m0 - q_mf)**2) - 10 * t_f**2

        return cost

    def constraint_catch_pose(self, x):
        """
        Constraint for the catching pose (position and orientation).
        """
        q_f = x[:self.n]  # Final robot configuration
        tf = x[self.n]    # Grasping time

        # Position constraint: position error should be zero
        wTe_cy = (self.robot.fkine(q_f) @ self.end_cy_T).A
        cy_position = wTe_cy[:3, -1]           # End-effector container position
        ball_position = self.predictor.get_position_at_time(tf)      # Ball's position
        position_error = np.linalg.norm(cy_position - ball_position)

        # Orientation constraint: align container's z-axis with the opposite of ball's velocity
        ee_z_direction = wTe_cy[:3, 2]         # Z-axis vector of the container
        ee_z_direction = ee_z_direction / np.linalg.norm(ee_z_direction)
        ball_vel = self.predictor.get_velocity_at_time(tf)          # Ball's velocity vector
        ball_vel = -ball_vel / np.linalg.norm(ball_vel)
        direction_error = 1 - np.dot(ee_z_direction, ball_vel)

        # Ensure position match and correct orientation
        return position_error, direction_error

    def constraint_qf_tf_limit(self, x):
        """
        Joint and time limit constraints.
        """
        q_m_f = x[self.n_b:self.n]
        q_max = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        q_min = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        tf = x[self.n]              # Ensure tf is positive
        return np.hstack((q_max - q_m_f, q_m_f - q_min, tf))

    def constraint_z(self, x):
        """
        Minimum height constraint for the grasp.
        """
        q_f = x[:self.n]  # Final robot configuration
        wTe_cy = (self.robot.fkine(q_f) @ self.end_cy_T).A
        cy_position = wTe_cy[:3, -1]           # End-effector container position
        cons_z = cy_position[2] - 0.6
        return cons_z

    def optimize(self):
        """
        Runs the optimization process to find the optimal configuration and time.
        """
        # Initial guess (final configuration and grasping time)
        x0 = np.hstack((self.q_b0, self.q_m0, 0.5))

        # Define the constraints for the optimization problem
        constraints = [
            {'type': 'eq', 'fun': self.constraint_catch_pose},
            {'type': 'ineq', 'fun': self.constraint_qf_tf_limit},
            {'type': 'ineq', 'fun': self.constraint_z}
        ]

        # Solve the optimization problem using the SLSQP algorithm
        result = minimize(self.objective_function, x0, method='SLSQP', constraints=constraints)

        # Extract optimization results
        q_f_opt = result.x[:self.n]
        t_f_opt = result.x[self.n]

        print("Optimization successful:", result.success)

        return q_f_opt, t_f_opt
