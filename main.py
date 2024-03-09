from least_squares import residual_function
from generate_ranges import FieldAssets
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np

def plot_circles():
	initial_position = np.array([50.0, 24.0])
	
	obj = FieldAssets(100, 60, initial_position)
	l1, l2, l3, l4 = obj.rangingGenerator()
	y1 = (l1**2 - l2**2 + 60**2)/120
	y2 = (l4**2 - l3**2 + 60**2)/120
	x1 = np.sqrt(l1**2 - y1**2)
	x2 = np.sqrt(l4**2 - y2**2) + 100
	print("Simple equation based solution: ({0}, {1}) and ({2}, {3})".format(x1, y1, x2, y2))

	radii = np.array(obj.rangingGenerator())

	receiverPos = {"BL": np.array([0, 0]), 
					"TL": np.array([0, 60]),
					"TR": np.array([100, 60]), 
					"BR": np.array([100, 0])}
	coordinates = np.array(list(receiverPos.values()))
	
	fig, ax = plt.subplots(figsize=(8, 8))
	plt.scatter(coordinates[:, 0], coordinates[:, 1])
	for i in range(len(coordinates)):
		circle = plt.Circle(coordinates[i], radii[i], color='b', fill=False)
		ax.add_artist(circle)
	
	# Set equal aspect ratio
	ax.set_aspect('equal')

	# Set limits based on coordinates and radii
	xlim = (coordinates[:, 0].min() - radii.max(), coordinates[:, 0].max() + radii.max())
	ylim = (coordinates[:, 1].min() - radii.max(), coordinates[:, 1].max() + radii.max())
	plt.xlim(xlim)
	plt.ylim(ylim)

	# Add labels and grid
	plt.title('Circles with Given Coordinates and Radii')
	plt.xlabel('X Coordinate')
	plt.ylabel('Y Coordinate')
	plt.grid(True)

	plt.show()


def write_to_disk(player_trajectory, est_trajectory, file_name):
	# print()
	file_object = open(file_name, "w")
	file_object.write("Current \t\t estimated \t\t error\n")
	for i in range(player_trajectory.shape[1]):
		file_object.write("{} \t\t {} \t\t {}\n".format(player_trajectory[:, i],
												  		est_trajectory[:, i],
														np.linalg.norm(
															player_trajectory[:, i] - est_trajectory[:, i]
														)))
	file_object.close()


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

	for i in range(1, total_iterations):
		# get distance from all sensors, estimate position
		distance_meas = obj.rangingGenerator()
		state_res = optimize.least_squares(residual_function, initial_guess, jac='3-point', args=(distance_meas, t))

		# update estimate for next iteration, save it
		initial_guess = state_res.x
		est_trajectory[:, i] = initial_guess

		# save player position
		player_trajectory[:, i] = obj.getPosition()
		obj.alternativeRunning()

	# write_to_disk(player_trajectory, est_trajectory, "results/player_tracking.txt")

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