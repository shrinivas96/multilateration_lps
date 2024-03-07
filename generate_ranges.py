import time
import matplotlib.pyplot as plt
import threading
import numpy as np
import random

class FieldAssets:
	def __init__(self, fieldLength, fieldWidth, initPos=None) -> None:
		# some aspects related to the field
		self.fieldLength = fieldLength
		self.fieldWidth = fieldWidth

		if initPos is None:
			initPos = np.array([fieldLength/2, fieldWidth/2])
		
		# holds the player/sensors position
		self.playerPos = initPos

		# assuming all functions will be run at 20 Hz,
		# AND given that avg player runs at 5 m/s,
		# the average distance a player will cover is
		self.avgDistCovered = 0.25

		# noise for distorting all measurements
		self.noise = 0.3
		
		# position of all the sensors
		self.receiverPos = {"BL": np.array([0, 0]), 
					  		"TL": np.array([0, 60]), 
							"TR": np.array([100, 60]), 
							"BR": np.array([100, 0])}
		
		self.interval = 1/20
		self.thread = None
		self.isRunning = False

	def whereAreYouRunning(self):
		"""
		This fucntion will decide on a random direction and a pre-decided distance 
		that the player will cover in the span of 0.05 s. At the moment it's a fairly naive
		function that just keeps the player 0.25m away from the borders.
		"""
		while self.isRunning:
			# the player can go into one of these quadrants,
			# depending on if these quadrants are close to the border
			quadrants_available = [[45, 135], [135, 225], [225, 315], [315, 405]]		# (x-axis is 0 deg)
			x = self.playerPos[0]
			y = self.playerPos[1]
			
			# check which quadrant(s) can be removed 
			# the remaining quadrants are the directions where the player can move
			if x < 0.25:
				quadrants_available.pop(1)
			elif x > 99.25:
				quadrants_available.pop(3)
			if y < 0.25:
				quadrants_available.pop(2)
			elif y > 59.25:
				quadrants_available.pop(0)

			# expand the rest of the quadrants to be randomly chosen out of
			available_directions = []
			for ranges in quadrants_available:
				available_directions += list(range(ranges[0], ranges[1]))
			
			# choose a direction where the player goes
			direction = random.choice(available_directions) * (np.pi/180)

			# this is the new player position for this time step
			self.playerPos += np.array([self.avgDistCovered*np.cos(direction),
							  		self.avgDistCovered*np.sin(direction)])

			time.sleep(self.interval)

	def startRunning(self):
		if not self.isRunning:
			self.isRunning = True
			self.thread = threading.Thread(target=self.whereAreYouRunning)
			self.thread.start()

	def stopRunning(self):
		self.isRunning = False
		if self.thread is not None:
			self.thread.join()

	def rangingGenerator(self):
		"""
		This function returns the erroneous distance of the player to the 4 sensors placed at the 4 corners of the field.
		Calculates the Eucledian distance between two given points.
		"""
		distances = []
		for rec_position in self.receiverPos.values():
			distances.append(np.linalg.norm(self.playerPos - rec_position) + random.uniform(-self.noise, self.noise))
		
		return np.array(distances)
	
	def getPosition(self):
		return self.playerPos


def threaded_run():
	initial_position = np.array([30., 20.])
	obj = FieldAssets(100, 60, initial_position)
	
	print("Initial player position", obj.getPosition())
	
	# should be replaced by function that can run at 20 Hz
	total_iterations = 15000

	# player's path for visualisation
	player_trajectory = np.zeros((initial_position.shape[0], total_iterations))
	player_trajectory[:, 0] = initial_position

	obj.startRunning()

	# time tracking
	time_history = np.zeros(total_iterations)
	time_history[0] = time.time()*1000.0
	for i in range(1, total_iterations):
		player_trajectory[:, i] = obj.getPosition()
		time_history[i] = time.time()*1000.0
	obj.stopRunning()

	player_transpose = player_trajectory.T
	print("{} \t\t {}".format(player_trajectory[:, 0], time_history[0]))
	for i in range(1, total_iterations):
		print("{} \t\t {} \t\t {}".format(player_trajectory[:, i], time_history[i], time_history[i] - time_history[i-1]))

def regular_run():
	initial_position = np.array([30., 20.])
	obj = FieldAssets(100, 60, initial_position)
	
	print("Initial player position", obj.getPosition())
	
	# should be replaced by function that can run at 20 Hz
	total_iterations = 1500

	# player's path for visualisation
	player_trajectory = np.zeros((initial_position.shape[0], total_iterations))
	player_trajectory[:, 0] = initial_position

	for i in range(1, total_iterations):
		# distance_meas = obj.rangingGenerator()		# distance from all sensors: to be estimated
		obj.whereAreYouRunning()						# update player position
		player_trajectory[:, i] = obj.getPosition()		# save new position

	# gimme that plot
	plt.figure(figsize=(10, 8))
	plt.plot(player_trajectory[0, :], player_trajectory[1, :], marker='x')
	plt.title('Drunk player on a field')
	plt.grid(True)
	plt.show()

if __name__ == '__main__':
	threaded_run()