from scipy.optimize import least_squares, OptimizeResult
from typing import Callable, Union
import numpy as np

TypeJac = Union[str, Callable, None]


class EvaluateFunctions:
    def __init__(self, anchors: dict[str, np.ndarray]) -> None:
        self.__anchors = anchors
        self.__num_anchors = len(self.__anchors)

    def update_measurement(self, measurement: np.ndarray) -> None:
        self.__measurement = measurement

    def measurement_jacobian(self, state: np.ndarray) -> np.ndarray:
        """
        Computes the Jacobian matrix of the expected measurement function.
        Only the first 2 columns of the Jacobian are filled corresponding to the 2 position states.
        If there are more states (e.g. velocity states) then they are 0's in the matrix.
        """
        xp, yp = state[0], state[1]
        no_of_states = state.shape[0]
        hJacobian = np.zeros([self.__num_anchors, no_of_states])
        for index, anchor in enumerate(self.__anchors.values()):
            xi, yi = anchor[0], anchor[1]
            xpi = xp - xi
            ypi = yp - yi
            denominator = np.sqrt(xpi**2 + ypi**2)
            hJacobian[index, 0:2] = np.divide([xpi, ypi], denominator)
        return hJacobian

    def hExpected_distance_function(self, state: np.ndarray) -> np.ndarray:
        """
        Computes the Eucledian distance of a given state to the known positions of the receivers (anchors) around the field.
        Expected measurement function, maps from state space to the measurement space, i.e. $\hat{z}_t = h(x_t)$.
        """
        xp, yp = state[0], state[1]
        expectedDistances = np.zeros((self.__num_anchors,))
        for index, anchor in enumerate(self.__anchors.values()):
            expectedDistances[index] = np.sqrt(
                (xp - anchor[0]) ** 2 + (yp - anchor[1]) ** 2
            )
        return expectedDistances

    def residual_function(self, state: np.ndarray) -> np.ndarray:
        """
        Computes the residual between:
        - measured distance between the given state and anchor locations, contains noise; z_t
        - Eucledian distance between the state and anchor locations, exp. measurements; \hat{z}_t
        Residual r_t = z_t - \hat{z}_t
        """
        residual = self.__measurement - self.hExpected_distance_function(state)
        return residual


class OptimiserWrappper:
    def __init__(
        self, residual_function: Callable, jacobian: TypeJac = None, method: str = "lm"
    ) -> None:
        """
        This is a wrapper class for the least squares function of scipy optimise.
        There are many arguments that can be passed to the least squares function,
        but at this moment this class only accepts the ones that are needed by me at the moment. They are:
          - residual function: the function that will be squared by the function and found the minimum of
          - jacobian: defaults to '2-point', that is implemented in least squares for general and lm methods
          - method: optimisation method, defaults to 'lm'
        """
        self.__residual_function = residual_function

        # default way to calculation jacobian that is preferred by lm method of least squares
        # for an invalid string value, the least squares method already throws an error, so we do not need to check that here
        if jacobian is None:
            jacobian = "2-point"
        self.__jacobian = jacobian

        self.__optimisation_method = method

    def optimise(self, initial_state: np.ndarray) -> OptimizeResult:
        return least_squares(
            fun=self.__residual_function,
            x0=initial_state,
            jac=self.__jacobian,  # type: ignore
            method=self.__optimisation_method,
        )
