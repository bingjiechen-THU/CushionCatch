import numpy as np


class PRCJointPlanner:
    """
    PRC (Polynomial Rate Control) Joint Space Trajectory Planner.

    This planner generates smooth trajectories for multi-joint robots using
    a 5th-order polynomial. It ensures that the movement from an initial to a
    target joint configuration adheres to specified kinematic constraints
    (maximum velocity and acceleration).
    """

    def __init__(self, qd_max, acc_max):
        """
        Initializes the PRC planner.

        This stage sets the fixed physical constraints of the robot.

        Args:
            qd_max (list or np.array): Maximum velocity for each joint (rad/s).
            acc_max (list or np.array): Maximum acceleration for each joint (rad/s^2).
        """
        self.qd_max = np.asarray(qd_max)
        self.acc_max = np.asarray(acc_max)

    def _compute_t_prc(self, delta_q):
        """
        Internal method: Calculate the minimum time required for each joint
        to move to its target independently.

        Args:
            delta_q (np.array): The absolute angle difference for each joint to travel.

        Returns:
            np.array: The planned time (t_prc) for each joint.
        """
        t_prc = np.zeros_like(delta_q)
        for i in range(len(delta_q)):
            # Check if a full trapezoidal velocity profile (accel-constant-decel) can be formed.
            if self.delta_q[i] < self.acc_max[i] * (self.qd_max[i] / self.acc_max[i])**2:
                # Velocity profile is triangular (only acceleration and deceleration phases).
                t_prc[i] = 2 * np.sqrt(delta_q[i] / self.acc_max[i])
            else:
                # Velocity profile is trapezoidal (includes a constant velocity phase).
                t_prc[i] = (2 * self.qd_max[i] / self.acc_max[i] +
                            (delta_q[i] - self.acc_max[i] * (self.qd_max[i] / self.acc_max[i])**2) / self.qd_max[i])
        return t_prc

    def _generate_poly5_trajectory(self, t, T, q0, qca):
        """
        Internal method: Generate position, velocity, and acceleration for a
        single joint using a 5th-order polynomial and its derivatives.

        Position: q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        Velocity: v(t) = a1 + 2*a2*t + 3*a3*t^2 + 4*a4*t^3 + 5*a5*t^4
        Acceleration: a(t) = 2*a2 + 6*a3*t + 12*a4*t^2 + 20*a5*t^3

        Args:
            t (np.array): Time series.
            T (float): Total planning time.
            q0 (float): Initial position.
            qca (float): Target position.

        Returns:
            tuple(np.array, np.array, np.array): Position, velocity, and
            acceleration trajectories for the joint.
        """
        # Boundary conditions: initial and final velocity/acceleration are zero.
        # q(0)=q0, q(T)=qca, v(0)=0, v(T)=0, a(0)=0, a(T)=0
        a0 = q0
        a1 = 0
        a2 = 0
        a3 = 10 * (qca - q0) / T**3
        a4 = -15 * (qca - q0) / T**4
        a5 = 6 * (qca - q0) / T**5

        # Calculate position q(t)
        q_t = a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5

        # Calculate velocity v(t) - 1st derivative of q(t)
        v_t = a1 + 2 * a2 * t + 3 * a3 * t**2 + 4 * a4 * t**3 + 5 * a5 * t**4

        # Calculate acceleration a(t) - 2nd derivative of q(t)
        a_t = 2 * a2 + 6 * a3 * t + 12 * a4 * t**2 + 20 * a5 * t**3

        return q_t, v_t, a_t

    def get_trajectory(self, q_0, q_ca, delta_t, lambda_param=1.2):
        """
        Executes the full trajectory planning process and returns the result.

        Args:
            q_0 (list or np.array): Initial joint configuration (rad).
            q_ca (list or np.array): Target joint configuration (rad).
            delta_t (float): Simulation time step (s).
            lambda_param (float): Time extension factor to ensure all joints
                                  arrive synchronously. Defaults to 1.2.

        Returns:
            tuple(np.array, np.array, np.array):
                - q_trajectory: Joint position trajectory (N_steps x N_joints)
                - velocities: Joint velocity trajectory (N_steps x N_joints)
                - accelerations: Joint acceleration trajectory (N_steps x N_joints)
        """
        self.q_0 = np.asarray(q_0)
        self.q_ca = np.asarray(q_ca)
        self.delta_t = delta_t
        self.lambda_param = lambda_param

        # 1. Calculate the angle difference for each joint
        self.delta_q = np.abs(self.q_ca - self.q_0)
        print("delta_q=", self.delta_q)

        # 2. Calculate the minimum time required for each joint to move independently
        t_prc_individual = self._compute_t_prc(self.delta_q)

        # 3. Calculate the overall planning time T_prc (take the slowest joint's
        #    time and multiply by the extension factor)
        T_prc = self.lambda_param * np.max(t_prc_individual)
        print("T_prc=", T_prc)

        # 4. Generate the time series
        t_values = np.arange(0, T_prc, self.delta_t)

        # 5. Generate trajectories for each joint
        q_list, v_list, a_list = [], [], []
        for i in range(len(self.q_0)):
            q_traj, v_traj, a_traj = self._generate_poly5_trajectory(
                t_values, T_prc, self.q_0[i], self.q_ca[i]
            )
            q_list.append(q_traj)
            v_list.append(v_traj)
            a_list.append(a_traj)

        # 6. Convert trajectory lists to NumPy arrays and transpose to (N_steps, N_joints)
        q_trajectory = np.array(q_list).T
        velocities = np.array(v_list).T
        accelerations = np.array(a_list).T

        return len(q_trajectory), q_trajectory, velocities, accelerations


# Example usage
if __name__ == "__main__":
    # Set parameters for a 9-DOF robot
    qd_max = [1.0] * 9     # Max velocity (rad/s)
    acc_max = [0.5] * 9    # Max acceleration (rad/s^2)
    q_0 = [0.0] * 9        # Initial configuration (rad)
    q_ca = [1.5] * 9       # Target configuration (rad)

    # Create a PRC planner instance
    planner = PRCJointPlanner(qd_max, acc_max)

    # Get the trajectory results
    steps, q_trajectory, velocities, accelerations = planner.get_trajectory(
        q_0, q_ca, delta_t=0.01, lambda_param=1.2
    )

    # Print results
    print(steps)
    # print("Joint Trajectories (Position):")
    # print(q_trajectory)
    print("Velocities:")
    print(velocities[0])
    # print("Accelerations:")
    # print(accelerations)
