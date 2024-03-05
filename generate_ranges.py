"""
Ideally generates the ranges of where the players are. This is pseudo random.
All measurements are in SI units.
"""
import matplotlib.pyplot as plt
import numpy as np
import random
# random.seed(0)

class FieldAssets:
	def __init__(self, fieldLength, fieldWidth) -> None:
		# some aspects related to the field
		self.fieldLength = fieldLength
		self.fieldWidth = fieldWidth

		# holds the player/sensors position
		self.playerPos = np.array([fieldLength/2, fieldWidth/2])

		# assuming all functions will be run at 20 Hz,
		# AND given that avg player runs at 10 m/s,
		# the average distance a player will cover is
		self.avgDistCovered = 0.5

		# noise for distroting all measurements
		self.noise = 0.3
		
		# position of all the sensors
		self.receiverPos = {"BL": np.array([0, 0]), 
					  		"TL": np.array([0, 60]), 
							"TR": np.array([100, 60]), 
							"BR": np.array([100, 0])}

	def whereAreYouRunning(self):
		"""
		This fucntion will decide on a random direction and a pre-decided distance 
		that the player will cover in the span of 0.05 s. At the moment it's a fairly naive
		function that just keeps the player 0.5m away from the borders.
		"""
		# the player can go into one of these quadrants, 
		# depending on if these quadrants are close to the border
		# TODO: change these values to radians in the next iteration
		quadrants_available = [[45, 135], [135, 225], [225, 315], [315, 405]]
		x = self.playerPos[0]
		y = self.playerPos[1]
		
		# check which quadrant(s) can be removed 
		if x < 0.5:
			quadrants_available.pop(1)
		elif x > 99.5:
			quadrants_available.pop(3)
		if y < 0.5:
			quadrants_available.pop(2)
		elif y > 99.5:
			quadrants_available.pop(0)

		# add the rest of the quadrants to be randomly chosen from
		available_directions = []
		for ranges in quadrants_available:
			available_directions += list(range(ranges[0], ranges[1]))
		
		# choose a direction where the player goes
		direction = random.choice(available_directions) * (np.pi/180)

		# this is the new player position for this time step
		self.playerPos += np.array([self.avgDistCovered*np.cos(direction),
							  		self.avgDistCovered*np.sin(direction)])
		
	def rangingGenerator(self):
		"""
		This function returns the distance of the player to the 4 sensors placed at the 4 corners of the field.
		Calculate the l2 norm between two given points. The l2 norm is the Eucledian distance
		"""
		distances = []
		for rec_position in self.receiverPos.values():
			distances.append(np.linalg.norm(self.playerPos - rec_position) + random.uniform(-self.noise, self.noise))
		
		return distances
	
	def getPosition(self):
		return self.playerPos


if __name__ == '__main__':
	obj = FieldAssets(100, 60)
	print(obj.getPosition())
	print(obj.rangingGenerator())
	
	num_iterations = 50

	# Arrays to store x and y coordinates
	x_coords = np.zeros(num_iterations)
	y_coords = np.zeros(num_iterations)
	
	# Initial coordinates
	initial_coord = np.array([50, 30])

	# Initialize first point
	x_coords[0], y_coords[0] = initial_coord

	for i in range(1, num_iterations):
		print(obj.rangingGenerator())
		obj.whereAreYouRunning()
		new_coord = obj.getPosition()
		x_coords[i], y_coords[i] = new_coord

	

	plt.figure(figsize=(8, 6))
	plt.plot(x_coords, y_coords, marker='o')
	plt.title('Change of Coordinates')
	plt.xlabel('X Coordinate')
	plt.ylabel('Y Coordinate')
	plt.grid(True)
	plt.show()
	
	# original position
	# generate some sort of a big array or a variable that takes in the field size and geenrates a field space.
	# this is ideally the whole field that you will be working with. based on this field is where the position 
	# of the ball and players will randomly be generated