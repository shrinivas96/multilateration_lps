from generate_ranges import FieldAssets
from scipy import optimize
import numpy as np


def resExpMeas_and_measJacobian(
        state: np.ndarray,
        measurement: np.ndarray,
        anchors: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the residual function and the Jacobian of the expected measurement function, at the given state
    r_t = z_t - h(x_t)
    hJ = dh(x_t) / dx_t

    The main purpose to do it this way is because the computation for
    both anyway have similar components, so one iteration could return both.
    To use this function in a good way refer to: https://stackoverflow.com/a/72768031/6609148
    """
    # the current state (xp yp)^T
    xp, yp = state[0], state[1]

    # some config and preallocating for speed
    no_of_anchors = len(anchors)
    no_of_states = state.shape[0]
    hJacobian = np.zeros([no_of_anchors, no_of_states])
    distances = np.zeros((no_of_anchors,))

    # for every receiver
    for index, anchor in enumerate(anchors.values()):
        # get the x-, y- coordinate of the receiver
        xi, yi = anchor[0], anchor[1]

        # variables for repeating computations: distance between the two
        xpi = xp - xi
        ypi = yp - yi
        dist_p2i = np.sqrt(xpi**2 + ypi**2)

        # the expected distance measurement, and its jacobian
        distances[index] = dist_p2i
        hJacobian[index, 0:2] = np.divide([xpi, ypi], dist_p2i)

    residual = measurement - distances
    return residual, hJacobian


def hEucledian_distance_function(
    state: np.ndarray, anchors: dict[str, np.ndarray]
) -> np.ndarray:
    """
    Computes the Eucledian distance of a given state to the known positions of the receivers (anchors) around the field.
    This is the expected measurement function, the mapping from the state space
    to the measurement space, i.e. $z_t = h(x_t)$.
    """
    xp, yp = state[0], state[1]
    no_of_anchors = len(anchors)
    distances = np.zeros((no_of_anchors,))
    for index, pos in enumerate(anchors.values()):
        distances[index] = np.sqrt((xp - pos[0]) ** 2 + (yp - pos[1]) ** 2)
    return distances


def measurement_jacobian(
    state: np.ndarray, anchors: dict[str, np.ndarray]           # for experimenting: *args
) -> np.ndarray:
    """
    Computes the Jacobian matrix of the expected measurement function.
    Only the first 2 columns of the Jacobian are filled corresponding to the 2 position states.
    If there are more states (e.g. velocity states) then they are 0's in the matrix.

    The argument args exists because scipy optimise passes the same arguments to the jacobian
    function as it does to the residual function. While I do not need the second ardument: measurements
    """
    # anchors = args[-1]

    xp, yp = state[0], state[1]
    no_of_anchors = len(anchors)
    no_of_states = state.shape[0]
    hJacobian = np.zeros([no_of_anchors, no_of_states])
    for index, anchor in enumerate(anchors.values()):
        xi, yi = anchor[0], anchor[1]
        xpi = xp - xi
        ypi = yp - yi
        denominator = np.sqrt(xpi**2 + ypi**2)
        hJacobian[index, 0:2] = np.divide([xpi, ypi], denominator)
    return hJacobian


def residual_function(
    state: np.ndarray, measurement: np.ndarray, anchors: dict[str, np.ndarray]
) -> np.ndarray:
    """
    Computes the residual between:
     - measured distance between the given state and anchor locations; contains noise
     - Eucledian distance between the state and anchor locations
    """
    residual = measurement - hEucledian_distance_function(state, anchors)
    return residual


def jacobian_i(state: np.ndarray) -> np.ndarray:
    xp, yp = state[0], state[1]
    xp2, yp2 = xp**2, yp**2
    yp60 = (yp - 60) ** 2
    xp100 = (xp - 100) ** 2

    j11 = xp / np.sqrt(xp2 + yp2)
    j12 = yp / np.sqrt(xp2 + yp2)
    j21 = xp / np.sqrt(xp2 + yp60)
    j22 = (yp - 60) / np.sqrt(xp2 + yp60)
    j31 = (xp - 100) / np.sqrt(xp100 + yp60)
    j32 = (yp - 60) / np.sqrt(xp100 + yp60)
    j41 = (xp - 100) / np.sqrt(xp100 + yp2)
    j42 = yp / np.sqrt(xp100 + yp2)
    Ji = np.array([[j11, j12], [j21, j22], [j31, j32], [j41, j42]])
    return Ji


def jacobian_ext(state: np.ndarray) -> np.ndarray:
    """
    Augmenting the above jacobia matrix for the purposes of using in the Kalman filter framework.
    The KF framework has a constant velocity model, which means two more states have been added,
    and the jacobian needs to be bigger.
    """
    Je = np.zeros((state.shape[0], state.shape[0]))
    Je[:, 0:2] = jacobian_i(state)
    return Je


def scipy_least_squares():
    """
    The scipy optimize function wants you to have a function that returns the residual fuction.
    """
    initial_position = np.array([50.0, 24.0])
    obj = FieldAssets(100, 60, initial_position)
    measurement = np.array(obj.rangingGenerator())

    initial_guess = np.array([45.0, 20.0])
    state_res = optimize.least_squares(residual_function, initial_guess, method='lm', args=(measurement, obj.receiverPos))
    print(state_res.x)


if __name__ == "__main__":
    scipy_least_squares()
