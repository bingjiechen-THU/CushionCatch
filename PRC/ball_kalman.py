import numpy as np


class KalmanFilter3D:
    """
    Implements a 3D Kalman Filter to track an object's position and velocity,
    considering gravity and air drag.
    """

    def __init__(self, delta_t, KD, Q, R, initial_state, g=9.81):
        """
        Initializes the Kalman Filter.

        Args:
            delta_t (float): Time step.
            KD (float): Air drag coefficient.
            Q (np.ndarray): Process noise covariance matrix.
            R (np.ndarray): Measurement noise covariance matrix.
            initial_state (np.ndarray): Initial state [x, y, z, vx, vy, vz].
            g (float, optional): Gravitational acceleration. Defaults to 9.81.
        """
        # Time step
        self.delta_t = delta_t
        # Air drag coefficient
        self.KD = KD
        # Process noise covariance matrix
        self.Q = Q
        # Measurement noise covariance matrix
        self.R = R
        # Gravitational acceleration
        self.g = g

        # Initial state vector [x, y, z, vx, vy, vz]
        self.s_b = initial_state.reshape((6, 1))  # Convert to column vector

        # Initial covariance matrix
        self.P = np.eye(6) * 0.1

        # State transition matrix (6x6)
        self.F = np.array([
            [1, 0, 0, self.delta_t, 0, 0],
            [0, 1, 0, 0, self.delta_t, 0],
            [0, 0, 1, 0, 0, self.delta_t],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Measurement matrix (assuming direct measurement of position and velocity)
        self.H = np.eye(6)

    def predict_state(self):
        """
        Predicts the next state based on the current state and motion model.

        Returns:
            np.ndarray: The predicted state vector.
        """
        # Calculate acceleration
        b = self.s_b[0:3]
        v = self.s_b[3:6]
        acc = np.array([[0], [0], [-self.g]]) - self.KD * np.linalg.norm(v) * v

        # Predict state
        v_new = v + self.delta_t * acc
        b_new = b + self.delta_t * v + 0.5 * self.delta_t ** 2 * acc
        s_b_new = np.vstack((b_new, v_new))

        return s_b_new

    def update(self, z):
        """
        Updates the state estimate with a new measurement.

        Args:
            z (np.ndarray): The new measurement vector [x, y, z, vx, vy, vz].
        """
        z = z.reshape((6, 1))  # Convert to column vector

        # Prediction step
        s_b_pred = self.predict_state()

        # Predict covariance
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Calculate Kalman gain
        K = P_pred @ self.H.T @ np.linalg.inv(self.H @ P_pred @ self.H.T + self.R)

        # Update state
        self.s_b = s_b_pred + K @ (z - self.H @ s_b_pred)

        # Update covariance
        self.P = (np.eye(6) - K @ self.H) @ P_pred

    def get_state(self):
        """
        Returns the current estimated state.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the position
                                           and velocity vectors.
        """
        # Return the current predicted state (position and velocity) as row vectors
        position = np.array(self.s_b[0:3].flatten())
        velocity = np.array(self.s_b[3:6].flatten())
        return position, velocity
