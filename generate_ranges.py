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
            initPos = np.array([fieldLength / 2, fieldWidth / 2])

        # holds the player/sensors position
        self.playerPos = initPos

        # holds the last chosen direction so that there is more chances that the player goes there the next time
        self.lastQuadrant = None

        # assuming all functions will be run at 20 Hz,
        # AND given that avg player runs at 5 m/s,
        # the average distance a player will cover is
        self.avgDistCovered = 0.25

        # noise for distorting all measurements
        self.noise = 0.3

        # should the players speed be randomly changing?
        self.randomSpeeds = False

        # position of all the sensors
        self.receiverPos = {
            "BL": np.array([0, 0]),
            "TL": np.array([0, 60]),
            "TR": np.array([100, 60]),
            "BR": np.array([100, 0])
        }

    def whereAreYouRunning(self) -> None:
        """
        This fucntion will decide on a random direction and a pre-decided distance
        that the player will cover in the span of 0.05 s. At the moment it's a fairly naive
        function that just keeps the player 0.25m away from the borders.
        """
        # the player can go into one of these quadrants,
        # depending on if these quadrants are close to the border
        quadrants_available = [
            [45, 135],
            [135, 225],
            [225, 315],
            [315, 405],
        ]  # (x-axis is 0 deg)
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
        direction = random.choice(available_directions) * (np.pi / 180)

        # this is the new player position for this time step
        self.playerPos += np.array(
            [
                self.avgDistCovered * np.cos(direction),
                self.avgDistCovered * np.sin(direction),
            ]
        )

    def lessRandomRunning(self) -> None:
        """
        This fucntion does the same job as whereAreYouRunning except it is an attempt 
        to make the player run more in the same direction as before, yet preserving a fair bit of randomness.
        """
        # the player can go into one of these quadrants,
        # depending on if these quadrants are close to the border
        quadrants_available = [
            [45, 135],
            [135, 225],
            [225, 315],
            [315, 405],
        ]  # (x-axis is 0 deg)

        # based on last chosen quadrant, duplicate it for more chances to be selected
        if self.lastQuadrant is not None:
            quadrants_available += [self.lastQuadrant]*20

        # current player position
        x = self.playerPos[0]
        y = self.playerPos[1]

        # check which quadrant(s) can be removed based on distance from the border
        # the remaining quadrants are the directions where the player can move
        quads_to_be_ignored = []
        if x < 0.25:
            quads_to_be_ignored.append(1)
        elif x > 99.25:
            quads_to_be_ignored.append(3)
        if y < 0.25:
            quads_to_be_ignored.append(2)
        elif y > 59.25:
            quads_to_be_ignored.append(0)

        # expand the rest of the quadrants to be randomly chosen out of
        available_directions = []
        for quad in quadrants_available:
            if quad in quads_to_be_ignored:
                continue
            available_directions += list(range(quad[0], quad[1]))

        # choose a direction where the player goes
        direction_deg = random.choice(available_directions)
        direction = direction_deg * (np.pi / 180)

        # based on the chosen direction, mark it to be duplicated
        for quad in quadrants_available:
            if quad[0] < direction_deg < quad[1]:
                self.lastQuadrant = quad
                break

        # can the player run at different speeds? makes the problem much harder
        if self.randomSpeeds:
            dist_covered = random.uniform(0, self.avgDistCovered)
        else:
            dist_covered = self.avgDistCovered

        # this is the new player position for this time step
        self.playerPos += np.array(
            [
                dist_covered * np.cos(direction),
                dist_covered * np.sin(direction),
            ]
        )

    def alternativeRunning(self):
        """
        This function is a way to update the player position less randomly. It gives a defined path to the player, 
        in hopes that the LS solution will track better. It chooses a random direction at first and then goes more 
        in that direction until the borders are reached.
        """
        # the player can go into one of these quadrants,
        # depending on if these quadrants are close to the border
        quadrants_available = [
            [45, 135],
            [135, 225],
            [225, 315],
            [315, 405],
        ]  # (x-axis is 0 deg)

        # current player position
        x = self.playerPos[0]
        y = self.playerPos[1]

        # check which quadrant(s) can be removed based on distance from the border
        # the remaining quadrants are the directions where the player can move
        if x < 0.25:
            quadrants_available.pop(1)
        elif x > 99.25:
            quadrants_available.pop(3)
        if y < 0.25:
            quadrants_available.pop(2)
        elif y > 59.25:
            quadrants_available.pop(0)

        # if the last chosen quadrant has not been removed then that is the only direction the player will go towards
        if self.lastQuadrant in quadrants_available:
            quadrants_available = [self.lastQuadrant]

        # create an array of possible angles where the player can go
        available_directions = []
        for quad in quadrants_available:
            available_directions += list(range(quad[0], quad[1]))
        
        # choose a direction where the player goes
        direction_deg = random.choice(available_directions)
        direction = direction_deg * (np.pi / 180)

        # based on the chosen direction, mark it to be repeated
        for quad in quadrants_available:
            if quad[0] < direction_deg < quad[1]:
                self.lastQuadrant = quad
                break

        # can the player run at different speeds? makes the problem much harder
        if self.randomSpeeds:
            dist_covered = random.uniform(0, self.avgDistCovered)
        else:
            dist_covered = self.avgDistCovered

        # this is the new player position for this time step
        self.playerPos += np.array(
            [
                dist_covered * np.cos(direction),
                dist_covered * np.sin(direction),
            ]
        )

    def rangingGenerator(self) -> np.ndarray:
        """
        This function returns the noise-added distance of the player to the 4 sensors placed at the 4 corners of the field.
        Calculates the Eucledian distance between two given points: current position and each of the receivers.
        """
        # array to hold all four distances
        distances = []
        for rec_position in self.receiverPos.values():
            distances.append(
                np.linalg.norm(self.playerPos - rec_position)
                + random.uniform(-self.noise, self.noise)
            )

        return np.array(distances)

    def getPosition(self) -> np.ndarray:
        return self.playerPos
    
    def allowRandomSpeeds(self, allow: bool) -> None:
        self.randomSpeeds = allow


def threaded_run():
    """Deprecated, this will be deleted in the next iteration"""
    initial_position = np.array([30.0, 20.0])
    obj = FieldAssets(100, 60, initial_position)

    print("Initial player position", obj.getPosition())
    

    # should be replaced by function that can run at 20 Hz
    total_iterations = 15000

    # player's path for visualisation
    player_trajectory = np.zeros((initial_position.shape[0], total_iterations))
    player_trajectory[:, 0] = initial_position

    # time tracking
    time_history = np.zeros(total_iterations)
    time_history[0] = time.time() * 1000.0
    for i in range(1, total_iterations):
        player_trajectory[:, i] = obj.getPosition()
        time_history[i] = time.time() * 1000.0

    player_transpose = player_trajectory.T
    print("{} \t\t {}".format(player_trajectory[:, 0], time_history[0]))
    for i in range(1, total_iterations):
        print(
            "{} \t\t {} \t\t {}".format(
                player_trajectory[:, i],
                time_history[i],
                time_history[i] - time_history[i - 1],
            )
        )


def regular_run():
    initial_position = np.array([30.0, 20.0])
    obj = FieldAssets(100, 60, initial_position)

    print("Initial player position", obj.getPosition())

    # should be replaced by function that can run at 20 Hz
    total_iterations = 1500

    # player's path for visualisation
    player_trajectory = np.zeros((initial_position.shape[0], total_iterations))
    player_trajectory[:, 0] = initial_position

    for i in range(1, total_iterations):
        # distance_meas = obj.rangingGenerator()		# distance from all sensors: to be estimated
        obj.whereAreYouRunning()  # update player position
        player_trajectory[:, i] = obj.getPosition()  # save new position

    # gimme that plot
    plt.figure(figsize=(10, 8))
    plt.plot(player_trajectory[0, :], player_trajectory[1, :], marker="x")
    plt.title("Drunk player on a field")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    regular_run()
    