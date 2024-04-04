from generate_ranges import FieldAssets, SimulatePlayerMovement
from position_processor_tools import EvaluateMeasurementFunctions, ExtendedKalmanFilter
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    initial_position = np.array([30.0, 20.0])
    field_obj = FieldAssets(100, 60)
    player_sim_obj = SimulatePlayerMovement(field_obj, initial_position)

    func_handle = EvaluateMeasurementFunctions(field_obj.receiver_positions)

    total_iterations = 150
    delta_t = 0.05

    # player's path for visualisation
    player_trajectory = np.zeros((initial_position.shape[0], total_iterations))
    player_trajectory[:, 0] = player_sim_obj.getPosition()

    # estimated trajectory, for visualisation
    initial_guess = np.array([48., 22., 0.25, 0.25])
    est_trajectory = np.zeros((initial_guess.shape[0], total_iterations))
    est_trajectory[:, 0] = initial_guess

    # state and measurement space dimension
    nState_dim = initial_guess.shape[0]
    mMeas_dim = len(field_obj.receiver_positions)

    # model properties
    state_transition = 1.0*np.eye(nState_dim)
    state_transition[0, 2] = delta_t
    state_transition[1, 3] = delta_t
    process_covariance = 50.0*np.eye(nState_dim)
    process_noise = 30.0*np.eye(nState_dim)
    meas_covariance = 0.6*np.eye(mMeas_dim)

    # kalman filter config
    ekf_estimator = ExtendedKalmanFilter(
        nState_dim,
        state_transition,
        state_transition,
        process_covariance,
        process_noise,
        mMeas_dim,
        func_handle.hExpected_distance_function,
        func_handle.measurement_jacobian,
        meas_covariance,
        initial_guess
    )

    for i in range(1, total_iterations):
        # get measurement, run KF update step, get new estimate
        distance_meas = player_sim_obj.rangingGenerator()

        func_handle.update_measurement(distance_meas)

        ekf_estimator.update(distance_meas)
        est_trajectory[:, i] = ekf_estimator.x

        # update player position, save it
        player_sim_obj.simulateRun()
        player_trajectory[:, i] = player_sim_obj.getPosition()
        
        # run kf prediction step
        ekf_estimator.predict()

    # gimme that plot
    plt.figure(figsize=(10, 8))
    plt.plot(player_trajectory[0, :], player_trajectory[1, :], marker='x', label="Player trajectory")
    plt.plot(est_trajectory[0, :], est_trajectory[1, :], marker='+', label="Estimated trajectory")
    plt.title('Tracking a drunk player on a field')
    plt.legend()
    plt.grid(True)
    plt.show()