from least_squares import jacobian_ext, distance_function
from filterpy.kalman import ExtendedKalmanFilter
from generate_ranges import FieldAssets
import matplotlib.pyplot as plt
import numpy as np


def state_jacobian(state: np.ndarray, l1: float):
    """
    Linearised circle equations, to use them as some sort of a state transition matrix. 
    Two non linear equations used here:
    (BL rec) x^2 + y^2 = l1^2       => x = sqrt(l1^2 - y^2)
    (TL rec) x^2 + (y-60)^2 = l2^2  => y = (l1^2 - l2^2 + c1) / c2
    (c1, c2 are some constants that we get after expanding and substituting.)
    """
    y = state[1]
    F = np.zeros((2, 2))
    F[0, 1] = y / np.sqrt(l1**2 + y**2)
    # ^^there is already a mistake here: there is a negative sign missing in the numerator, and y^2 should be negative in denominator
    # setting them to negative yields NaN or Inf error.
    return F


def simple_state_transition(measurement: np.ndarray) -> np.ndarray:
    """
    Simple solution to the position estimate based on equations of circles. Gives out two possible positions,
    but was observed that (x1, y1) was always the plausible one, albeit not the right solution due to presence of noise.
    This was used as the predicted state for the Prediction step of the Kalman Filter.
    """
    # equations have been solved on paper and hard-coded here
    l1, l2, l3, l4 = measurement[:]
    y1 = (l1**2 - l2**2 + 60**2)/120
    y2 = (l4**2 - l3**2 + 60**2)/120
    x1 = np.sqrt(l1**2 - y1**2)
    x2 = np.sqrt(l4**2 - y2**2) + 100

    return np.array([x1, y1])


def self_prediction(estimator: ExtendedKalmanFilter) -> ExtendedKalmanFilter:
    """
    Self implemented prediction step of the (Extended) Kalman Filter. Removed out of the one proided by the library,
    because I wanted to have my own state transition jacobian matrix that updates in each time step.
    """
    # get state and measurement 
    state = estimator.x
    measurement = estimator.z

    # update state transition matrix based on current state
    estimator.F[0:2, 0:2] = state_jacobian(state, measurement[0])

    # predicted state and covariance
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

    # estimated trajectory, for visualisation
    initial_guess = np.array([49.0, 23.0, 0, 0])
    est_trajectory = np.zeros((initial_guess.shape[0], total_iterations))

    # kalman filter config
    ekf_estimator = ExtendedKalmanFilter(dim_x=4, dim_z=4)

    # three approaches were tried:
    # 1: constant velocity model: adding velocity states, and multiply with dt to get position
    # 2: using solution from equations of circle as prediction step 
    # 3: linearising circle equations and using that as state transition matrix
    approach = 2

    ekf_estimator.x = initial_guess
    state_transition = 1.0*np.eye(4)
    if approach == 1:
        state_transition[0, 2] = delta_t
        state_transition[1, 3] = delta_t  
    ekf_estimator.F = state_transition

    # setting a high covariance for motion model because the model is definitely inaccurate
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
        

        if approach == 1:
            # run kf prediction step, update player position
            ekf_estimator.predict()
        if approach == 2:
            # sort of "hot-wiring" the prediction step: did not work!!
            ekf_estimator.x[0:2] = simple_state_transition(distance_meas)
        if approach == 3:
            ekf_estimator = self_prediction(ekf_estimator)

        
        obj.alternativeRunning()
    
    # gimme that plot
    plt.figure(figsize=(10, 8))
    plt.plot(player_trajectory[0, :], player_trajectory[1, :], marker='x', label="Player trajectory")
    plt.plot(est_trajectory[0, :], est_trajectory[1, :], marker='+', label="Estimated trajectory")
    plt.title('Tracking a drunk player on a field')
    plt.legend()
    plt.grid(True)
    plt.show()