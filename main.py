from generate_ranges import FieldAssets
from scipy.optimize import least_squares
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
	print(x1, x2)
	print(y1, y2)

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


def error_function(p0, anchors, estimated_distances):
	return np.sqrt(np.sum((anchors - p0)**2, axis=1)) - estimated_distances


if __name__ == "__main__":
	# TODO: implement 20 Hz running cycle
	# TODO: process distance to find the position
	# plot_circles()
	initial_position = np.array([50.0, 24.0])
	obj = FieldAssets(100, 60, initial_position)
	print(obj.rangingGenerator())
	l1, l2, l3, l4 = obj.rangingGenerator()

	# # Initialize the estimated position of P0 (initial guess)
	# initial_guess = np.array([50, 50])

	# # Minimize the error function using least squares optimization
	# result = least_squares(error_function, initial_guess, args=(anchors, estimated_distances))

	# # Estimated coordinates of P0
	# p0_estimate = result.x