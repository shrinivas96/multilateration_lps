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
	
	# should be replaced by function that can run at 20 Hz
	total_iterations = 1500
	t = 0

	# player's path for visualisation
	player_trajectory = np.zeros((initial_position.shape[0], total_iterations))
	player_trajectory[:, 0] = initial_position

	# estimated trajectory, for visualisation
	initial_guess = np.array([49.0, 23.0])
	est_trajectory = np.zeros((initial_guess.shape[0], total_iterations))
	est_trajectory[:, 0] = initial_guess

	for i in range(1, total_iterations):
		# distance from all sensors: to be estimated
		distance_meas = obj.rangingGenerator()
		state_res = optimize.least_squares(residual_function, initial_guess, method='lm', args=(distance_meas, t))
		initial_guess = state_res.x
		est_trajectory[:, i] = initial_guess

		# update player position, save it
		obj.whereAreYouRunning()
		player_trajectory[:, i] = obj.getPosition()

	# write_to_disk(player_trajectory, est_trajectory, "results/player_tracking.txt")

	# gimme that plot
	plt.figure(figsize=(10, 8))
	plt.plot(player_trajectory[0, :], player_trajectory[1, :], marker='x', label="Player trajectory")
	plt.plot(est_trajectory[0, :], est_trajectory[1, :], marker='+', label="Estimated trajectory")
	plt.title('Tracking a drunk player on a field')
	plt.legend()
	plt.grid(True)

	plt.figure(figsize=(10, 8))
	error = np.linalg.norm(player_trajectory - est_trajectory, axis=0)
	iterations = np.arange(total_iterations)
	plt.plot(iterations, error)
	plt.xlabel("Number of Iterations")
	plt.ylabel("Normed Error")
	plt.title("Norm of difference between positions")
	plt.grid(True)
	plt.show()