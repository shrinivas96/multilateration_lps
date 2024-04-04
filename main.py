from tools import EvaluateFunctions, OptimiserWrappper
from generate_ranges import FieldAssets, SimulatePlayerMovement
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
	initial_position = np.array([30.0, 20.0])
	field_obj = FieldAssets(100, 60)
	player_sim_obj = SimulatePlayerMovement(field_obj, initial_position)

	total_iterations = 150
	delta_t = 0.5

	# player's path for visualisation
	player_trajectory = np.zeros((initial_position.shape[0], total_iterations))
	player_trajectory[:, 0] = initial_position

	# estimated trajectory, for visualisation; start very close
	initial_guess = initial_position + np.array([0.1, -0.2])
	est_trajectory = np.zeros((initial_guess.shape[0], total_iterations))
	est_trajectory[:, 0] = initial_guess

	# residuals = MemoizeJac(expMeas_and_measJacobian)
	# hJacobian = residuals.derivative

	func_handle = EvaluateFunctions(field_obj.receiver_positions)
	opt_handle = OptimiserWrappper(func_handle.residual_function)

	for i in range(1, total_iterations):
		# get distance from all sensors, estimate position
		distance_meas = player_sim_obj.rangingGenerator()
		func_handle.update_measurement(distance_meas)
		state_res = opt_handle.optimise(initial_guess)

		# update estimate for next iteration, save it
		initial_guess = state_res.x
		est_trajectory[:, i] = initial_guess

		# save player position
		player_sim_obj.simulateRun()
		player_trajectory[:, i] = player_sim_obj.getPosition()

	# gimme that plot
	plt.figure(figsize=(10, 8))
	plt.plot(player_trajectory[0, :], player_trajectory[1, :], marker='x', label="Player trajectory")
	plt.plot(est_trajectory[0, :], est_trajectory[1, :], marker='+', label="Estimated trajectory")
	plt.title('Tracking a drunk player on a field')
	# plt.xlim((-5, 100.0))
	# plt.ylim((-5, 60.0))
	plt.legend()
	plt.grid(True)

	plt.figure(figsize=(10, 8))
	error = np.linalg.norm(player_trajectory - est_trajectory, axis=0)
	iterations = np.arange(total_iterations)
	plt.plot(iterations, error)
	plt.xlabel("Time step")
	plt.ylabel("Normed Error")
	plt.title("Norm of difference between positions")
	plt.grid(True)

	plt.figure(figsize=(10, 8))
	velocity = (est_trajectory[:, 1:] - est_trajectory[:, :-1])*delta_t
	velocity = np.linalg.norm(velocity, axis=0)
	velocity_iterations = np.arange(total_iterations-1)
	plt.plot(velocity_iterations, velocity)
	plt.xlabel("Time step")
	plt.ylabel("Velocity")
	plt.title("Velocity of the player")
	plt.grid(True)

	plt.show()