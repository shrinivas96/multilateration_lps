from generate_ranges import FieldAssets
import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np


def hEucledian_distance_function(state: np.ndarray, anchors: dict) -> np.ndarray:
    """
    Computes the Eucledian distance of a given state to the known positions of the receivers (anchors) around the field.
    This function is the mapping from the state space to the measurement space, i.e. $z_t = h(x_t)$.
    """
    xp, yp = state[0], state[1]
    no_of_anchors = len(anchors)
    distances = np.zeros((no_of_anchors,))
    for index, pos in enumerate(anchors.values()):
        distances[index] = np.sqrt((xp - pos[0])**2 + (yp - pos[1])**2)
    return distances


def measurement_jacobian(state: np.ndarray, anchors) -> np.ndarray:
    # TODO lets implement the jacobian here
    no_of_anchors = len(anchors)
    no_of_states = state.shape[0]
    hJacobian = np.zeros([no_of_anchors, no_of_states])
    for index, anchor in enumerate(anchors.values()):
        ...
    return hJacobian


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


def scipy_least_squares():
    """
    The scipy optimize function wants you to have a function that returns the residual fuction.
    """
    initial_position = np.array([50.0, 24.0])
    obj = FieldAssets(100, 60, initial_position)
    measurement = np.array(obj.rangingGenerator())

    initial_guess = np.array([45.0, 20.0])
    state_res = optimize.least_squares(residual_function, initial_guess, method='lm', args=(measurement,))
    print(state_res.x)


if __name__ == "__main__":
    scipy_least_squares()
