import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from Plot.plot_image import *


class BallTrajectoryPredictor:
    """
    Predicts the trajectory of a ball considering gravity and air drag.
    """

    def __init__(self, g=9.81, K_D=0.0238):
        """
        Initializes the BallTrajectoryPredictor.

        Args:
            g (float): Acceleration due to gravity (m/s^2).
            K_D (float): Air drag coefficient.
        """
        self.g = g
        self.K_D = K_D
        self.time_points = None  # Stores time points
        self.trajectory = None   # Stores position trajectory
        self.velocity = None     # Stores velocity trajectory

    def equations_of_motion(self, t, s_b):
        """
        Defines the differential equations of motion for the ball.

        Args:
            t (float): Current time.
            s_b (np.ndarray): State vector [x, y, z, vx, vy, vz].

        Returns:
            np.ndarray: The derivative of the state vector [vx, vy, vz, ax, ay, az].
        """
        v = s_b[3:]  # Velocity (vx, vy, vz)

        # Calculate acceleration, considering gravity and air drag
        norm_v = np.linalg.norm(v)
        a = -np.array([0, 0, self.g]) - self.K_D * norm_v * v

        # Return velocity and acceleration
        return np.hstack((v, a))

    def predict_trajectory(self, s_b_initial, t_span, t_eval=None):
        """
        Solves the equations of motion to predict the ball's trajectory.

        Args:
            s_b_initial (np.ndarray): Initial state vector [x, y, z, vx, vy, vz].
            t_span (tuple): Time interval for the simulation (start, end).
            t_eval (np.ndarray, optional): Times at which to store the computed solution.
        """
        # Use numerical integration to solve the equations of motion
        sol = solve_ivp(
            self.equations_of_motion, t_span, s_b_initial, method='RK45', t_eval=t_eval
        )

        # Store time points, position, and velocity trajectories
        self.time_points = sol.t
        self.trajectory = sol.y[:3].T  # Position data
        self.velocity = sol.y[3:].T    # Velocity data

    def get_position_at_time(self, t):
        """
        Interpolates the trajectory to find the position at a specific time.

        Args:
            t (float or np.ndarray): The time(s) to query.

        Returns:
            np.ndarray: The interpolated position(s) [x, y, z] at time t.
        """
        if self.time_points is None or self.trajectory is None:
            raise ValueError("You must call predict_trajectory before querying positions.")

        # Create cubic spline interpolation functions for each axis
        interpolators = [
            CubicSpline(self.time_points, self.trajectory[:, i], extrapolate=True)
            for i in range(3)
        ]

        # Calculate the position at time t
        position = np.array([interp(t) for interp in interpolators])
        return position

    def get_velocity_at_time(self, t):
        """
        Interpolates the trajectory to find the velocity at a specific time.

        Args:
            t (float or np.ndarray): The time(s) to query.

        Returns:
            np.ndarray: The interpolated velocity/velocities [vx, vy, vz] at time t.
        """
        if self.time_points is None or self.velocity is None:
            raise ValueError("You must call predict_trajectory before querying velocities.")

        # Create cubic spline interpolation functions for each axis
        interpolators = [
            CubicSpline(self.time_points, self.velocity[:, i], extrapolate=True)
            for i in range(3)
        ]

        # Calculate the velocity at time t
        velocity = np.array([interp(t) for interp in interpolators])
        return velocity


# Example usage
if __name__ == "__main__":
    # Initial state
    b_initial = np.array([0, 0, 2])  # Initial position (x, y, z) in meters
    v_initial = np.array([2, 2, 3])  # Initial velocity (vx, vy, vz) in m/s

    # State vector [position, velocity]
    s_b_initial = np.hstack((b_initial, v_initial))

    # Create a predictor instance
    predictor = BallTrajectoryPredictor(K_D=0.0238)

    # Time span for prediction (from t=0 to t=2 seconds)
    t_span = (0, 2)

    # Predict the trajectory
    predictor.predict_trajectory(s_b_initial, t_span, t_eval=np.linspace(0, 2, 100))

    # Query position and velocity at specific time points
    times_to_query = [0.57, 1.0, 1.5, 1.75]
    for t in times_to_query:
        pos = predictor.get_position_at_time(t)
        vel = predictor.get_velocity_at_time(t)
        print(f"Time: {t:.2f}s, Position: {pos}, Velocity: {vel}")

    # Extract x, y, z position data for plotting
    x = predictor.trajectory[:, 0]
    y = predictor.trajectory[:, 1]
    z = predictor.trajectory[:, 2]
    data = [x, y, z]

    # Plot the 3D trajectory
    plot3d(data, "./Plot/pos_3d_test.png")
