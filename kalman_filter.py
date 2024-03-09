from least_squares import jacobian_ext, distance_function
from filterpy.kalman import ExtendedKalmanFilter
from generate_ranges import FieldAssets
import matplotlib.pyplot as plt
import numpy as np


def state_jacobian(state: np.ndarray, l1: float):
    y = state[1]
    F = np.zeros((2, 2))
    F[0, 1] = y / np.sqrt(l1**2 + y**2)
    return F


def simple_state_transition(measurement: np.ndarray) -> np.ndarray:
    l1, l2, l3, l4 = measurement[:]
    y1 = (l1**2 - l2**2 + 60**2)/120
    y2 = (l4**2 - l3**2 + 60**2)/120
    x1 = np.sqrt(l1**2 - y1**2)
    x2 = np.sqrt(l4**2 - y2**2) + 100

    return np.array([x1, y1])


def self_prediction(estimator: ExtendedKalmanFilter) -> ExtendedKalmanFilter:
    state = estimator.x
    measurement = estimator.z
    estimator.F[0:2, 0:2] = state_jacobian(state, measurement[0])
    estimator.x = np.dot(estimator.F, state)
    estimator.P = np.dot(estimator.F, estimator.P).dot(estimator.F.T) + estimator.Q
    return estimator

 
if __name__ == "__main__":
    initial_position = np.array([50.0, 24.0])
    obj = FieldAssets(100, 60, initial_position)

    total_iterations = 150
    t = 0
    delta_t = 0.05

    # player's path for visualisation
    player_trajectory = np.zeros((initial_position.shape[0], total_iterations))
    # player_trajectory[:, 0] = initial_position

    # estimated trajectory, for visualisation
    initial_guess = np.array([49.0, 23.0, 0, 0])
    est_trajectory = np.zeros((initial_guess.shape[0], total_iterations))
    # est_trajectory[:, 0] = initial_guess

    # kalman filter config
    ekf_estimator = ExtendedKalmanFilter(dim_x=4, dim_z=4)

    # setting a high covariance for motion model because the model is definitely inaccurate
    ekf_estimator.x = initial_guess
    state_transition = 1.0*np.eye(4)
    state_transition[0, 2] = delta_t
    state_transition[1, 3] = delta_t  
    ekf_estimator.F = state_transition
    ekf_estimator.Q = 30.0*np.eye(4)
    ekf_estimator.P = 50.0*np.eye(4)

    # low covariance for measurement noise
    ekf_estimator.R = 0.6 * np.eye(4)

    for i in range(total_iterations):
        # save player position
        player_trajectory[:, i] = obj.getPosition()

        # get measurement, run KF update step, get new estimate
        distance_meas = obj.rangingGenerator()
        ekf_estimator.update(distance_meas, jacobian_ext, distance_function)
        est_trajectory[:, i] = ekf_estimator.x
        
        # hot-wiring the prediction step: did not work
        # ekf_estimator.x[0:2] = simple_state_transition(distance_meas)
        # ekf_estimator = self_prediction(ekf_estimator)

        # run kf prediction step, update player position
        ekf_estimator.predict()
        obj.alternativeRunning()
    
    # gimme that plot
    plt.figure(figsize=(10, 8))
    plt.plot(player_trajectory[0, :], player_trajectory[1, :], marker='x', label="Player trajectory")
    plt.plot(est_trajectory[0, :], est_trajectory[1, :], marker='+', label="Estimated trajectory")
    plt.title('Tracking a drunk player on a field')
    plt.legend()
    plt.grid(True)
    plt.show()