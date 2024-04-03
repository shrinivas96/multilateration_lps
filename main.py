from position_processor import residual_function, measurement_jacobian
from scipy.optimize._optimize import MemoizeJac
from tools import EvaluateFunctions
from generate_ranges import FieldAssets
import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np

if __name__ == "__main__":
	initial_position = np.array([50.0, 24.0])
	obj = FieldAssets(100, 60, initial_position)

	dt = 0.5

	# should be replaced by function that can run at 20 Hz
	total_iterations = 150
	t = 0

	# player's path for visualisation
	player_trajectory = np.zeros((initial_position.shape[0], total_iterations))
	player_trajectory[:, 0] = initial_position

	# estimated trajectory, for visualisation; start very close
	initial_guess = initial_position + np.array([0.1, -0.2])
	est_trajectory = np.zeros((initial_guess.shape[0], total_iterations))
	est_trajectory[:, 0] = initial_guess

	# residuals = MemoizeJac(expMeas_and_measJacobian)
	# hJacobian = residuals.derivative

	func = EvaluateFunctions(obj.receiverPos)

	for i in range(1, total_iterations):
		# get distance from all sensors, estimate position
		distance_meas = obj.rangingGenerator()
		func.update_measurement(distance_meas)
		state_res = optimize.least_squares(func.residual_function,
									 		initial_guess,
									 		# jac=func.measurement_jacobian,
											method='lm')

		# update estimate for next iteration, save it
		initial_guess = state_res.x
		est_trajectory[:, i] = initial_guess

		# save player position
		obj.alternativeRunning()
		player_trajectory[:, i] = obj.getPosition()

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
	velocity = (est_trajectory[:, 1:] - est_trajectory[:, :-1])*dt
	velocity = np.linalg.norm(velocity, axis=0)
	velocity_iterations = np.arange(total_iterations-1)
	plt.plot(velocity_iterations, velocity)
	plt.xlabel("Time step")
	plt.ylabel("Velocity")
	plt.title("Velocity of the player")
	plt.grid(True)

	plt.show()