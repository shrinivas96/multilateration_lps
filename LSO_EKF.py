from generate_ranges import FieldAssets, SimulatePlayerMovement
from tools import EvaluateFunctions, OptimiserWrappper
from tools import ExtendedKalmanFilter as EKF
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    initial_position = np.array([30.0, 20.0])
    field_obj = FieldAssets(100, 60)
    player_sim_obj = SimulatePlayerMovement(field_obj, initial_position)

    func_handle = EvaluateFunctions(field_obj.receiver_positions)
    opt_handle = OptimiserWrappper(func_handle.residual_function)

    total_iterations = 150
    delta_t = 0.05

    # player's path for visualisation
    player_trajectory = np.zeros((initial_position.shape[0], total_iterations))
    player_trajectory[:, 0] = initial_position

    # estimated trajectory, for visualisation;
    initial_guess = np.array([1., 1., 1., 1.])
    est_trajectory = np.zeros((initial_guess.shape[0], total_iterations))
    est_trajectory[:, 0] = initial_guess
    lso_trajectory = np.copy(est_trajectory)
    nState_dim = initial_guess.shape[0]

    # kalman filter configuration; initialise prior and model matrices
    state_transition = 1.0*np.eye(nState_dim)
    state_transition[0, 2] = delta_t
    state_transition[1, 3] = delta_t
    
    # setting a high covariance for motion model because the model is definitely inaccurate
    process_noise = 30.0*np.eye(nState_dim)
    process_covariance = 50.0*np.eye(nState_dim)
    
    # low covariance for measurement noise
    mMeas_dim = len(field_obj.receiver_positions)
    meas_covariance = 0.6*np.eye(mMeas_dim)

    ekf_handle = EKF(
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
        # get distance measurement from sensors
        distance_meas = player_sim_obj.rangingGenerator()
        func_handle.update_measurement(distance_meas)

        # LSO estimate of position
        lso_state_res = opt_handle.optimise(ekf_handle.x)
        lso_trajectory[:, i] = lso_state_res.x

        # kf estimate of position: feed LSO estimate
        ekf_handle.x = lso_state_res.x
        ekf_handle.update(distance_meas)
        # ekf_handle.update(distance_meas, func_handle.measurement_jacobian,
        #                      func_handle.hExpected_distance_function)
        est_trajectory[:, i] = ekf_handle.x

        # update, save player position
        player_sim_obj.simulateRun()
        player_trajectory[:, i] = player_sim_obj.getPosition()

        # run kf prediction step
        ekf_handle.predict()

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