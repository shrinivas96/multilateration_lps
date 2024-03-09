from generate_ranges import FieldAssets
import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np


def distance_function(state: np.ndarray) -> np.ndarray:
    """
    Simple calculation of the non-linear distance function, given current state [xp yp]^T
    """
    xp, yp = state[0], state[1]
    xp2, yp2 = xp**2, yp**2
    yp60 = (yp - 60) ** 2
    xp100 = (xp - 100) ** 2
    return np.array(
        [
            np.sqrt(xp2 + yp2),
            np.sqrt(yp60 + xp2),
            np.sqrt(xp100 + yp60),
            np.sqrt(xp100 + yp2),
        ]
    )


def jacobian_i(state: np.ndarray) -> np.ndarray:
    """
    Function to evaluate the Jacobian at the current state [xp yp]^T
    """
    xp, yp = state[0], state[1]
    xp2, yp2 = xp**2, yp**2
    yp60 = (yp - 60) ** 2
    xp100 = (xp - 100) ** 2

    j11 = xp / np.sqrt(xp2 + yp2)
    j12 = yp / np.sqrt(xp2 + yp2)
    j21 = xp / np.sqrt(xp2 + yp60)
    j22 = yp - 60 / np.sqrt(xp2 + yp60)
    j31 = xp - 100 / np.sqrt(xp100 + yp60)
    j32 = yp - 60 / np.sqrt(xp100 + yp60)
    j41 = xp - 100 / np.sqrt(xp100 + yp2)
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


def residual_function(state: np.ndarray, measurement: np.ndarray, t: int) -> np.ndarray:
    """
    For scipy to find the minimum of this function, the first argument should be the one that we are trying to estimate,
    i.e., the state [xp yp]^T
    """
    residual = measurement - distance_function(state)
    return residual


if __name__ == "__main__":
    initial_position = np.array([50.0, 24.0])
    obj = FieldAssets(100, 60, initial_position)
    measurement = np.array(obj.rangingGenerator())
    t = 0

    initial_guess = np.array([45.0, 20.0])
    state_res = optimize.least_squares(residual_function, initial_guess, method='lm', args=(measurement, t))
    print(state_res.x)