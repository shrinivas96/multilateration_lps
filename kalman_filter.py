from position_processor import hEucledian_distance_function, jacobian_ext, measurement_jacobian
from filterpy.kalman import ExtendedKalmanFilter
from generate_ranges import FieldAssets
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    initial_position = np.array([50.0, 24.0])
    obj = FieldAssets(100, 60, initial_position)

    total_iterations = 150
    t = 0
    delta_t = 0.05

    # player's path for visualisation
    player_trajectory = np.zeros((initial_position.shape[0], total_iterations))
    player_trajectory[:, 0] = obj.getPosition()

    # estimated trajectory, for visualisation
    initial_guess = np.array([49., 24., 0.25, 0.25])
    # initial_guess = np.array([1., 1., 0.25, 0.25])
    est_trajectory = np.zeros((initial_guess.shape[0], total_iterations))
    est_trajectory[:, 0] = initial_guess

    # kalman filter config
    ekf_estimator = ExtendedKalmanFilter(dim_x=4, dim_z=4)

    use_other_update = True

    ekf_estimator.x = initial_guess
    state_transition = 1.0*np.eye(4)
    state_transition[0, 2] = delta_t
    state_transition[1, 3] = delta_t
    ekf_estimator.F = state_transition

    # setting a high covariance for motion model because the model is definitely inaccurate
    ekf_estimator.Q = 30.0*np.eye(4)
    ekf_estimator.P = 50.0*np.eye(4)

    # low covariance for measurement noise
    ekf_estimator.R = 0.6 * np.eye(4)

    for i in range(1, total_iterations):

        # get measurement, run KF update step, get new estimate
        distance_meas = obj.rangingGenerator()
        if use_other_update:
            # TODO what about this approach?
            ekf_estimator.update(distance_meas, measurement_jacobian, 
                                 hEucledian_distance_function, 
                                 args=(obj.receiverPos,), hx_args=(obj.receiverPos,))
        else:
            ekf_estimator.update(distance_meas, jacobian_ext, hEucledian_distance_function, hx_args=(obj.receiverPos,))
        est_trajectory[:, i] = ekf_estimator.x

        # run kf prediction step
        ekf_estimator.predict()

        # update player position, save it
        obj.alternativeRunning()
        player_trajectory[:, i] = obj.getPosition()

    # gimme that plot
    plt.figure(figsize=(10, 8))
    plt.plot(player_trajectory[0, :], player_trajectory[1, :], marker='x', label="Player trajectory")
    plt.plot(est_trajectory[0, :], est_trajectory[1, :], marker='+', label="Estimated trajectory")
    plt.title('Tracking a drunk player on a field')
    plt.legend()
    plt.grid(True)
    plt.show()