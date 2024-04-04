from generate_ranges import FieldAssets, SimulatePlayerMovement
from helper import present_results
from util import ekf_periodic_lso
import numpy as np


if __name__ == "__main__":
    # simulate a field and place a player at the given location
    initial_position = np.array([30.0, 20.0])
    field_obj = FieldAssets(field_length_m=100, field_width_m=60)
    player_sim_obj = SimulatePlayerMovement(
        field_obj, initial_position, avg_speed_mps=5, sensor_noise_m=0.3
    )

    # number of iterations
    total_iterations = 150

    # time between consecutive measurements, is used in velocity estimate and EKF process model
    delta_t_s = 1 / field_obj.receiver_freq_hz

    # starting state
    initial_guess = np.array([1.0, 1.0, 1.0, 1.0])

    # how often do you want least squares optimisation to run
    ls_periodicity = 20

    # simulates the main movement, sensor measurement, estimation loop
    player_trajectory, est_trajectory = ekf_periodic_lso(
        initial_position,
        initial_guess,
        field_obj.receiver_positions,
        player_sim_obj,
        total_iterations,
        delta_t_s,
        ls_periodicity,
    )

    # how many plots do you want popping up
    show_error_plot = True
    show_velocity_plot = True

    # the first two rows of the arrays contain the position estimate
    present_results(
        player_trajectory[0:2, :],
        est_trajectory[0:2, :],
        delta_t_s,
        show_error_plot,
        show_velocity_plot,
    )
