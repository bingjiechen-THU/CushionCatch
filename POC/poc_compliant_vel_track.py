import cvxpy as cp
import numpy as np


class Vel_tracking:
    """
    Implements a velocity tracking controller using Quadratic Programming (QP)
    with Control Barrier Functions (CBFs) to ensure safety.
    """

    def __init__(self, gamma=0.1):
        """
        Initializes the velocity tracking controller.

        Args:
            gamma (float): Control gain for the CBF constraints.
        """
        self.gamma = gamma

    def compute_joint_vel(self, v, jacobian, base_T, wTe_cy):
        """
        Computes the optimal joint velocities to track a desired end-effector
        velocity while satisfying safety constraints.

        Args:
            v (np.ndarray): Desired 3D linear velocity of the end-effector.
            jacobian (np.ndarray): The geometric Jacobian matrix of the robot.
            base_T (np.ndarray): Homogeneous transformation matrix of the robot base.
            wTe_cy (np.ndarray): Homogeneous transformation matrix of the end-effector cylindrical container.

        Returns:
            np.ndarray: The computed optimal joint velocities, or None if the
                        problem could not be solved.
        """
        # Consider only the linear velocity part of the Jacobian.
        jacobian = jacobian[:3]
        # Define optimization variables based on Jacobian dimensions.
        joint_velocities = cp.Variable(jacobian.shape[1])
        # Slack variable to relax the velocity tracking objective if needed.
        slack = cp.Variable(len(v))

        # Extract current end-effector and base positions.
        [x_cy, y_cy, z_cy] = wTe_cy[0:3, 3]
        [x_base, y_base, z_base] = base_T[0:3, 3]

        # --- Z-axis CBF: Avoid collision with the ground ---
        # Barrier function h(x) = z^2 - z_min^2 >= 0
        b_z = z_cy**2 - 0.1**2
        # Time derivative of the barrier function: h_dot = 2*z*z_dot
        z_dot = 2 * z_cy * jacobian[2, :] @ joint_velocities

        # --- Self-collision CBF: Avoid collision with the base ---
        xy_vel = jacobian[0:2, :] @ joint_velocities
        cy_base_xy = np.array([x_cy - x_base, y_cy - y_base])
        distance = np.linalg.norm(cy_base_xy)
        # Barrier function h(x) = ||p_xy - p_base_xy||^2 - d_min^2 >= 0
        b_xy = np.sum(cy_base_xy**2) - 0.3**2

        constraints = []
        cbf_state = False

        # Activate Z-axis CBF if the end-effector is close to the ground.
        if z_cy < 0.2:
            # Constraint: h_dot + gamma * h >= 0
            cbf_z = z_dot >= -self.gamma * b_z
            constraints.append(cbf_z)
            cbf_state = True

        # Activate self-collision CBF if the end-effector is close to the base.
        if distance < 0.7:
            # Constraint: h_dot + gamma * h >= 0
            direc_cons = xy_vel @ cy_base_xy >= -self.gamma * b_xy
            constraints.append(direc_cons)
            cbf_state = True

        # Define the optimization problem based on whether any CBF is active.
        if cbf_state:
            # If CBFs are active, relax the tracking objective with a slack variable.
            # Objective: Minimize joint velocities and slack variable.
            constraints.append(jacobian @ joint_velocities == v + slack)
            objective = cp.Minimize(
                cp.sum_squares(joint_velocities) + cp.sum_squares(slack)
            )
        else:
            # If no CBFs are active, strictly enforce the tracking objective.
            # Objective: Minimize joint velocities.
            constraints.append(jacobian @ joint_velocities == v)
            objective = cp.Minimize(cp.sum_squares(joint_velocities))

        # Define and solve the Quadratic Programming problem.
        problem = cp.Problem(objective, constraints)
        problem.solve()

        return joint_velocities.value

