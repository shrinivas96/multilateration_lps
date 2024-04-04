import matplotlib.pyplot as plt
import numpy as np

# for reproducible results
np.random.seed(5)

class FieldAssets:
    def __init__(
        self, field_length_m: float, field_width_m: float, receiver_freq_hz: int=20
    ) -> None:
        """
        Class to hold some parameters of a simulated field. The field contains four receivers (in the default case)
        at the four corners. These receivers measure the distance to a sensor moving within the field, 
        with new measurements available at a certain frequency.

        Parameters:
        -----------
        fieldLength, fieldWidth: float
            The length and width of the field.
        receiver_freq: int, optional
            The receivers can measure distance to the sensor at this rate. Defaults to 20 Hz
        """
        # some aspects related to the field
        self.field_length_m = field_length_m
        self.field_width_m = field_width_m

        # the frequency at which the receivers update measurements
        self.receiver_freq_hz = receiver_freq_hz

        # position of all the receivers; defaults to the 4 corners of the field
        # bottom-left is always considered as the origin of the Cartesian coordinate system
        self.receiver_positions = {
            "BL": np.array([0, 0]),
            "TL": np.array([0, self.field_width_m]),
            "TR": np.array([self.field_length_m, self.field_width_m]),
            "BR": np.array([self.field_length_m, 0]),
        }


class SimulatePlayerMovement:
    def __init__(
        self,
        field_obj: FieldAssets,
        init_pos: np.ndarray | None,
        avg_speed_mps: float = 5.0,
        sensor_noise_m: float = 0.3,
    ) -> None:
        """
        Simulates a player moving on the field defined by FieldAssets.

        Parameters:
        -----------
        field_obj: FieldAssets
            The properties of the field where the player is located.
        init_pos: np.ndarray, optional
            Starting position of the player. Defaults to the centre of the field.
        avg_speed_mps: float, optional 
            Average speed of the player, defaults to 5 metres per second.
        sensor_noise_m: float, optional
            Noise, in metres, induced in the sensor measurements. The noise included is chosen from a 
            uniform distribution [-sensor_noise_m, sensor_noise_m]. Defaults to 0.3 m.
        """
        
        # params related to the field where the player is located
        self.field_obj = field_obj

        # holds the player/sensors position
        if init_pos is None:
            init_pos = np.array([self.field_obj.field_length_m, self.field_obj.field_width_m]) / 2
        self.player_pos = init_pos

        # holds the last chosen direction so that there is more chances that the player goes there the next time
        self.last_quadrant = None

        # assuming receivers provide measurements at x Hz,
        # AND given that an avg player runs at y m/s,
        # the average distance a player will cover is y/x m
        self.avg_dist_covered_m = avg_speed_mps / self.field_obj.receiver_freq_hz
        
        # the player should stay avgDistCovered distance away from all four sides of the field
        self.fieldBorder = self.avg_dist_covered_m

        # noise for distorting all measurements
        self.noise = sensor_noise_m

        # randomness config
        self.allow_random_speeds = False                # allow for random player speeds; a max of avg_speed
        self.more_random_movement = False               # to make the player movement more random

    def simulateRun(self):
        """
        Simulate the more-or-less random running of a player in the field. This function needs to be 
        called each time to move the player. Starts by choosing a random direction to move into, 
        and then moves a pre selected distance in that direction. 
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
        x = self.player_pos[0]
        y = self.player_pos[1]

        # set limits to where the player cannot go
        # the player should stay this much distance away from all four sides of the field
        field_left_lim = self.fieldBorder
        field_right_lim = self.field_obj.field_length_m - self.fieldBorder
        field_top_lim = self.field_obj.field_width_m - self.fieldBorder
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
        if not self.more_random_movement:
            # if the last chosen quadrant has not been removed then that is the only direction the player will go towards
            if self.last_quadrant in quadrants_available:
                quadrants_available = [self.last_quadrant]

        # create an array of possible angles where the player can go
        available_directions = []
        for quad in quadrants_available:
            # choose one random value out of each available quadrant
            available_directions.append(np.random.randint(quad[0], quad[1]))

        # choose a random direction where the player goes
        direction_deg = np.random.choice(available_directions)
        direction_rad = direction_deg * (np.pi / 180)

        # unless more randomness is asked for
        if not self.more_random_movement:
            # based on the chosen direction, mark it to be repeated
            for quad in quadrants_available:
                if quad[0] < direction_deg < quad[1]:
                    self.last_quadrant = quad
                    break

        # can the player run at different speeds?
        if self.allow_random_speeds:
            dist_covered = np.random.uniform(0, self.avg_dist_covered_m)
        else:
            dist_covered = self.avg_dist_covered_m

        # this is the new player position for this time step
        self.player_pos += np.array(
            [
                dist_covered * np.cos(direction_rad),
                dist_covered * np.sin(direction_rad),
            ]
        )

    def rangingGenerator(self) -> np.ndarray:
        """
        This function returns the noise-added distance of the player to the receivers placed in the field.
        Calculates the Eucledian distance between two given points: current position and each of the receivers, 
        and adds noise chosen from a uniform distribution.
        """
        num_receivers = len(self.field_obj.receiver_positions)
        
        # array to hold distances to all receivers
        distances = np.zeros((num_receivers,))
        for index, rec in enumerate(self.field_obj.receiver_positions.values()):
            distances[index] = np.linalg.norm(self.player_pos - rec) + np.random.uniform(-self.noise, self.noise)
        
        return np.array(distances)
    
    def getPosition(self) -> np.ndarray:
        """
        Returns the current position of the player.
        """
        return self.player_pos


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
