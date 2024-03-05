"""
Ideally generates the ranges of where the players are. This is pseudo random.
All measurements are in SI units.
"""
import matplotlib.pyplot as plt
import numpy as np
import random
from math import cos, sin, pi
# random.seed(0)

class FieldAssets:
	def __init__(self, fieldLength, fieldWidth) -> None:
		self.fieldLength = fieldLength
		self.fieldWidth = fieldWidth
		self.playerPos = np.array([fieldLength/2, fieldWidth/2])
		self.origin = np.array([0, 0])

		# assuming all functions will be run at 20 Hz,
		# AND given that avg player runs at 10 m/s,
		# the average distance a player will cover is
		self.avgDistCovered = 0.5
		
		self.receiverPos = {"BL": (0, 0), "TL": (0, 60), "TR": (100, 60), "BR": (100, 0)}

	def whereAreYouRunning(self):
		"""
		This fucntion will decide on a random direction and a pre-decided distance 
		that the player will cover in the span of 0.05 s. At the moment this is a dumb 
		function that just keeps the player 0.5m away from the borders.
		"""
		# the player can go into one of these quadrants, 
		# depending on if these quadrants are close to the border
		# change these values to radians in the next iteration
		available_range = [[45, 135], [135, 225], [225, 315], [315, 405]]
		x = self.playerPos[0]
		y = self.playerPos[1]
		
		# check which quadrant(s) can be removed 
		if x < 0.5:
			available_range.pop(1)
		elif x > 99.5:
			available_range.pop(3)
		if y < 0.5:
			available_range.pop(2)
		elif y > 99.5:
			available_range.pop(0)

		# add the rest of the quadrants to be randomly chosen from
		available_directions = []
		for ranges in available_range:
			available_directions += list(range(ranges[0], ranges[1]))
		
		# choose a direction where the player goes
		direction = random.choice(available_directions) * (np.pi/180)

		self.playerPos += np.array([self.avgDistCovered*np.cos(direction),
							  		self.avgDistCovered*np.sin(direction)])
		
	def getPosition(self):
		return self.playerPos

def generate_ranges():
	pass



if __name__ == '__main__':
	generate_ranges()
	obj = FieldAssets(100, 60)
	print(obj.getPosition())
	
	num_iterations = 1000

	# Arrays to store x and y coordinates
	x_coords = np.zeros(num_iterations)
	y_coords = np.zeros(num_iterations)
	
	# Initial coordinates
	initial_coord = np.array([50, 30])

	# Initialize first point
	x_coords[0], y_coords[0] = initial_coord

	for i in range(1, num_iterations):
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