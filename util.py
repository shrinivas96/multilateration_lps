from generate_ranges import SimulatePlayerMovement
from position_processor_tools import EvaluateMeasurementFunctions, OptimiserWrapper
from position_processor_tools import ExtendedKalmanFilter as EKF
import numpy as np


def setup_ekf(
    nState_dim: int,
    mMeas_dim: int,
    initial_guess: np.ndarray,
    meas_func_handle: EvaluateMeasurementFunctions,
    delta_t: float,
) -> EKF:
    """
    Simple setup function for the Extended Kalman Filter (EKF). 
    The main aim is just to move some of the clutter away from 
    the simulation function, and isolate it.
    
    Parameters:
    -----------
    nState_dim: int
        Dimension of the state to be estimated by the EKF.
    mMeas_dim: int
        Measurement dimension.
    initial_guess: np.ndarray
        Initial state starting point.
    meas_func_handle: EvaluateMeasurementFunctions
        Object that allows to evaluate measurement functions.
    delta_t: float
        Time between two consecutive measurements. Used in the constant velocity model, 
        to compute the next predicted state.

    Returns:
    --------
    An ExtendedKalmanFilter object initialised with the given properties.
    """
    state_transition = 1.0 * np.eye(nState_dim)
    state_transition[0, 2] = delta_t
    state_transition[1, 3] = delta_t

    # setting a high covariance for motion model because the model is definitely inaccurate
    process_noise = 30.0 * np.eye(nState_dim)
    process_covariance = 50.0 * np.eye(nState_dim)

    # low covariance for measurement noise
    meas_covariance = 0.6 * np.eye(mMeas_dim)

    return EKF(
        dim_state=nState_dim,
        process_model=state_transition,
        process_jacobian=state_transition,  # similar process model and jacobian, because state transition is linear
        cov_state=process_covariance,
        noise_state=process_noise,
        dim_meas=mMeas_dim,
        measurement_model=meas_func_handle.hExpected_distance_function,
        measurement_jacobian=meas_func_handle.measurement_jacobian,
        noise_meas=meas_covariance,
        initial_state=initial_guess,
    )


def ekf_periodic_lso(
    initial_position: np.ndarray,
    initial_guess: np.ndarray,
    anchors: dict[str, np.ndarray],
    player_sim_obj: SimulatePlayerMovement,
    total_iterations: int = 150,
    delta_t_s: float = 0.05,
    lso_periodicity: int = 20
) -> tuple[np.ndarray]:
    """
    Function that initialises all required variables, and overall simulates 
    the whole scenario of player movement and position estimate. We use least squares 
    optimisation (LSO) only some times. Namely, once before the iteration loop begins, and then 
    once periodically every few iterations. Rest of the times, we use the Extended Kalman Filter (EKF)
    to perform the state estimation.

    Parameters:
    -----------
    initial_position: np.ndarray
        The starting position where the player ahs been initialised.
    initial_guess: np.ndarray
        The inital state guess that will be used by LSO/EKF. 
    field_obj: FieldAssets
        The receivers that are placed through out the field. The dictionary contains their name and the location.
    player_sim_obj: SimulatePlayerMovement
        The player object. Simulates movement of the player and returns measurements.
    total_iterations: int, optional
        Total number of times to simulate the scenario. Defaults to 150
    delta_t: float, optional
        The time between two consecutive measurements, used for computing velocity. Defaults to 0.05s
    lso_periodicity: int, optional
        Periodicity of the LSO estimate. Defaults to 20. 
    """
    # objects for measurement functions and optimiser
    meas_func_handle = EvaluateMeasurementFunctions(anchors)

    # i noticed that the lso performs poorly when it uses the jacobian that I implemented.
    # in a purely lso setting it does much worse, maybe we can talk about it later.
    use_jac_in_lso = False
    if use_jac_in_lso:
        opt_handle = OptimiserWrapper(meas_func_handle.residual_function, meas_func_handle.measurement_jacobian)
    else:
        opt_handle = OptimiserWrapper(meas_func_handle.residual_function)
        

    # variables for results and plotting
    # player's path for visualisation
    player_trajectory = np.zeros((initial_position.shape[0], total_iterations))
    player_trajectory[:, 0] = initial_position

    # estimated trajectory, for visualisation;
    est_trajectory = np.zeros((initial_guess.shape[0], total_iterations))
    est_trajectory[:, 0] = initial_guess

    # kalman filter configuration
    nState_dim = initial_guess.shape[0]
    mMeas_dim = len(anchors)
    ekf_handle = setup_ekf(nState_dim, mMeas_dim, initial_guess, meas_func_handle, delta_t_s)

    # we run the lso once so that the initial guess comes close to the true state
    meas_func_handle.update_measurement(player_sim_obj.rangingGenerator())
    lso_state_res = opt_handle.optimise(ekf_handle.x)
    ekf_handle.x = lso_state_res.x

    for i in range(1, total_iterations):
        # get distance measurement from sensors
        distance_meas = player_sim_obj.rangingGenerator()
        meas_func_handle.update_measurement(distance_meas)
    
        # lso can be run periodically once every few 1000 iterations, as a helper to EKF
        if i % lso_periodicity == 0:
            lso_state_res = opt_handle.optimise(ekf_handle.x)
            ekf_handle.x = lso_state_res.x

        # kf estimate of position: feed LSO estimate
        ekf_handle.update(distance_meas)
        est_trajectory[:, i] = ekf_handle.x

        # update, save player position
        player_sim_obj.simulateRun()
        player_trajectory[:, i] = player_sim_obj.getPosition()

        # run kf prediction step
        ekf_handle.predict()

    return player_trajectory, est_trajectory
