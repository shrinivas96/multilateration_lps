import matplotlib.pyplot as plt
import numpy as np

# for reproducible results
np.random.seed(5)

class FieldAssets:
    def __init__(
        self, fieldLength: float, fieldWidth: float, receiver_freq: int=20
    ) -> None:
        # some aspects related to the field
        self.fieldLength = fieldLength
        self.fieldWidth = fieldWidth

        # the frequency at which the receivers update measurements
        self.receiverFrequency = receiver_freq

        # position of all the receivers; defaults to the 4 corners of the field
        # bottom-left is always considered as the origin of the Cartesian coordinate system
        self.receiverPos = {
            "BL": np.array([0, 0]),
            "TL": np.array([0, self.fieldWidth]),
            "TR": np.array([self.fieldLength, self.fieldWidth]),
            "BR": np.array([self.fieldLength, 0]),
        }


class SimulatePlayerMovement:
    def __init__(
        self,
        field_obj: FieldAssets,
        initPos: np.ndarray | None,
        avg_speed: float = 5.0,
        sensor_noise: float = 0.3,
    ) -> None:
        
        # params related to the field where the player is located
        self.field_obj = field_obj

        # holds the player/sensors position
        if initPos is None:
            initPos = np.array([self.field_obj.fieldLength, self.field_obj.fieldWidth]) / 2
        self.playerPos = initPos

        # holds the last chosen direction so that there is more chances that the player goes there the next time
        self.lastQuadrant = None

        # assuming receivers provide measurements at x Hz,
        # AND given that an avg player runs at y m/s,
        # the average distance a player will cover is y/x m
        self.avgDistCovered = avg_speed / self.field_obj.receiverFrequency
        
        # the player should stay avgDistCovered distance away from all four sides of the field
        self.fieldBorder = self.avgDistCovered

        # noise for distorting all measurements
        self.noise = sensor_noise

        # randomness config
        self.allowRandomSpeeds = False               # allow for random player speeds; a max of avg_speed
        self.moreRandomMovement = False             # to make the player movement more random

    def simulateRun(self):
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

        # set limits to where the player cannot go
        # the player should stay this much distance away from all four sides of the field
        field_left_lim = self.fieldBorder
        field_right_lim = self.field_obj.fieldLength - self.fieldBorder
        field_top_lim = self.field_obj.fieldWidth - self.fieldBorder
        field_bottom_lim = self.fieldBorder

        # check which quadrant(s) can be removed
        # the remaining quadrants are the directions where the player can move
        if x < field_left_lim:
            quadrants_available.pop(1)
        elif x > field_right_lim:
            quadrants_available.pop(3)
        if y < field_bottom_lim:
            quadrants_available.pop(2)
        elif y > field_top_lim:
            quadrants_available.pop(0)

        # unless more randomness is asked for
        if not self.moreRandomMovement:
            # if the last chosen quadrant has not been removed then that is the only direction the player will go towards
            if self.lastQuadrant in quadrants_available:
                quadrants_available = [self.lastQuadrant]

        # create an array of possible angles where the player can go
        available_directions = []
        for quad in quadrants_available:
            # choose one random value out of each available quadrant
            available_directions.append(np.random.randint(quad[0], quad[1]))

        # choose a random direction where the player goes
        direction_deg = np.random.choice(available_directions)
        direction_rad = direction_deg * (np.pi / 180)

        # unless more randomness is asked for
        if not self.moreRandomMovement:
            # based on the chosen direction, mark it to be repeated
            for quad in quadrants_available:
                if quad[0] < direction_deg < quad[1]:
                    self.lastQuadrant = quad
                    break

        # can the player run at different speeds?
        if self.allowRandomSpeeds:
            dist_covered = np.random.uniform(0, self.avgDistCovered)
        else:
            dist_covered = self.avgDistCovered

        # this is the new player position for this time step
        self.playerPos += np.array(
            [
                dist_covered * np.cos(direction_rad),
                dist_covered * np.sin(direction_rad),
            ]
        )

    def rangingGenerator(self) -> np.ndarray:
        """
        This function returns the noise-added distance of the player to the 4 sensors placed at the 4 corners of the field.
        Calculates the Eucledian distance between two given points: current position and each of the receivers.
        """
        num_receivers = len(self.field_obj.receiverPos)
        
        # array to hold distances to all receivers
        distances = np.zeros((num_receivers,))
        for index, rec in enumerate(self.field_obj.receiverPos.values()):
            distances[index] = np.linalg.norm(self.playerPos - rec) + np.random.uniform(-self.noise, self.noise)
        
        return np.array(distances)
    
    def getPosition(self) -> np.ndarray:
        return self.playerPos


def regular_run():
    initial_position = np.array([30.0, 20.0])
    field_obj = FieldAssets(100, 60)
    player_sim_obj = SimulatePlayerMovement(field_obj, initial_position)

    print("Initial player position", player_sim_obj.getPosition())

    # should be replaced by function that can run at 20 Hz
    total_iterations = 150

    # player's path for visualisation
    player_trajectory = np.zeros((initial_position.shape[0], total_iterations))
    player_trajectory[:, 0] = initial_position

    for i in range(1, total_iterations):
        # distance_meas = player_sim_obj.rangingGenerator()		# distance from all sensors: to be estimated
        player_sim_obj.simulateRun()  # update player position
        player_trajectory[:, i] = player_sim_obj.getPosition()  # save new position

    # gimme that plot
    plt.figure(figsize=(10, 8))
    plt.plot(player_trajectory[0, :], player_trajectory[1, :], marker="x")
    plt.title("Drunk player on a field")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    regular_run()
