from filterpy.kalman import ExtendedKalmanFilter
from generate_ranges import FieldAssets
from tools import EvaluateFunctions
import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np

if __name__ == "__main__":
    initial_position = np.array([50.0, 24.0])
    obj = FieldAssets(100, 60, initial_position)

    delta_t = 0.05

    # should be replaced by function that can run at 20 Hz
    total_iterations = 150

    # player's path for visualisation
    player_trajectory = np.zeros((initial_position.shape[0], total_iterations))
    player_trajectory[:, 0] = initial_position

    # estimated trajectory, for visualisation;
    initial_guess = np.array([1., 1., 1., 1.])
    est_trajectory = np.zeros((initial_guess.shape[0], total_iterations))
    est_trajectory[:, 0] = initial_guess
    lso_trajectory = np.copy(est_trajectory)


    # kalman filter configuration; initialise prior, and model matrices
    ekf_estimator = ExtendedKalmanFilter(dim_x=4, dim_z=4)
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

    func_handle = EvaluateFunctions(obj.receiverPos)

    for i in range(1, total_iterations):
        # get distance measurement from sensors
        distance_meas = obj.rangingGenerator()
        func_handle.update_measurement(distance_meas)

        # LSO estimate of position
        lso_state_res = optimize.least_squares(func_handle.residual_function,
                                            ekf_estimator.x,
                                            method='lm')
        lso_trajectory[:, i] = lso_state_res.x

        # kf estimate of position: feed LSO estimate
        ekf_estimator.x = lso_state_res.x
        ekf_estimator.update(distance_meas, func_handle.measurement_jacobian,
                                 func_handle.hExpected_distance_function)
        est_trajectory[:, i] = ekf_estimator.x

        # update, save player position
        obj.alternativeRunning()
        player_trajectory[:, i] = obj.getPosition()

        # run kf prediction step
        ekf_estimator.predict()

    # gimme that plot
    plt.figure(figsize=(10, 8))
    plt.plot(player_trajectory[0, :], player_trajectory[1, :], marker='x', label="Player trajectory")
    plt.plot(est_trajectory[0, :], est_trajectory[1, :], marker='o', label="LSO KF trajectory")
    plt.plot(lso_trajectory[0, :], lso_trajectory[1, :], marker='^', label="LSO trajectory")
    plt.title('Tracking a drunk player on a field')
    # plt.xlim((-5, 100.0))
    # plt.ylim((-5, 60.0))
    plt.legend()
    plt.grid(True)

    # plt.figure(figsize=(10, 8))
    # error = np.linalg.norm(player_trajectory - est_trajectory, axis=0)
    # iterations = np.arange(total_iterations)
    # plt.plot(iterations, error)
    # plt.xlabel("Time step")
    # plt.ylabel("Normed Error")
    # plt.title("Norm of difference between positions")
    # plt.grid(True)

    # plt.figure(figsize=(10, 8))
    # velocity = (est_trajectory[:, 1:] - est_trajectory[:, :-1])*dt
    # velocity = np.linalg.norm(velocity, axis=0)
    # velocity_iterations = np.arange(total_iterations-1)
    # plt.plot(velocity_iterations, velocity)
    # plt.xlabel("Time step")
    # plt.ylabel("Velocity")
    # plt.title("Velocity of the player")
    # plt.grid(True)

    plt.show()