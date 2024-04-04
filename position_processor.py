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
    both have similar components, so one iteration could return both.
    To use this function in an efficient way refer to: https://stackoverflow.com/a/72768031/6609148
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
     - measured distance between the given state and anchor locations; contains noise z_t
     - Eucledian distance between the state and anchor locations; exp measurements \hat{z}_t
    Residual r_t = z_t - \hat{z}_t
    """
    residual = measurement - hEucledian_distance_function(state, anchors)
    return residual


def scipy_least_squares():
    """
    The scipy optimize function wants you to have a function that returns the residual fuction.
    """
    initial_position = np.array([50.0, 24.0])
    obj = FieldAssets(100, 60, initial_position)
    measurement = np.array(obj.rangingGenerator())

    initial_guess = np.array([45.0, 20.0])
    state_res = optimize.least_squares(residual_function, initial_guess, method='lm', args=(measurement, obj.receiver_positions))
    print(state_res.x)


if __name__ == "__main__":
    scipy_least_squares()
