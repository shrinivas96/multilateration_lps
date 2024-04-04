from scipy.optimize import least_squares, OptimizeResult
from typing import Callable, Union, Optional
import numpy as np

TypeJac = Union[str, Callable, None]
TypeModels = Union[np.ndarray, Callable]


class EvaluateMeasurementFunctions:
    def __init__(self, anchors: dict[str, np.ndarray]) -> None:
        """
        A class to group measurement functions (model and Jacobian), that depend on similar parameters. 
        Both are functions of a changing state, measurement and fixed anchor positions. The state could change more frequently, 
        depending on which estimation technique is used, for example, least squares. Thus, the measurement is set once and 
        reused until a new measurement is available. The anchors are set once and are expected not to change.

        Parameters
        ----------
        anchors : dict of str and np.ndarray
            A dictionary describing the receiver locations in the field and their Cartesian coordinates. 
            The number of anchors is the number of receivers/measurements available.
        """
        self.__anchors = anchors
        self.__num_anchors = len(self.__anchors)
        self.__measurement = np.zeros((self.__num_anchors,))

    def update_measurement(self, measurement: np.ndarray) -> None:
        """
        Updates the measurement needed to compute the residual function.

        Parameters
        ----------
        measurement : np.ndarray
            The new measurement for the residual function. Must be of dimension (m,), where m is the number of anchors set during initialisation.
        """
        self.__measurement = measurement

    def measurement_jacobian(self, state: np.ndarray) -> np.ndarray:
        """
        Computes and evaluates the Jacobian matrix of the expected measurement function, at the given state.
        Only the first 2 columns of the Jacobian are filled corresponding to the 2 position states.
        If there are more states (e.g. velocity states) then they are 0's in the matrix.

        Parameters
        ----------
        state: np.ndarray
            The state where this Jacobian matrix needs to be evaluated at. 
            The first two elements of this array should always be the position states [xp yp]^T.
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
        Computes the Eucledian distance of a given position to the known positions of the receivers (anchors) around the field.
        This is the expected measurement function and maps from state space to the measurement space, i.e. $\hat{z}_t = h(x_t)$.

        Parameters
        ----------
        state: np.ndarray
            The independent variable(s) that this measurement model is a function of.
            The first two elements of this array should always be the position states [xp yp]^T.
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

        Parameters
        ----------
        state: np.ndarray
            The independent variable(s) that needs to be estimated through the minimisation of this residual function.
        """
        residual = self.__measurement - self.hExpected_distance_function(state)
        return residual


class OptimiserWrappper:
    def __init__(
        self, residual_function: Callable, jacobian: TypeJac = None, method: str = "lm", args: tuple = ()
    ) -> None:
        """
        This is a wrapper for the least squares function of the Scipy Optimize library.
        There are many arguments that can be passed to the least squares function,
        but at this moment this class only accepts three parameters.
        
        Parameters
        ----------
        residual_function: callable function
            The residual function f(x) to be optimised. f(x) is an m-D real function of n real variables.
            Least squares currently finds a minimum of 0.5*sum(f_i(x)**2, i = 0, ..., m - 1).
            The functions first argument should be the state, and should return a numpy array of size (m,)
        jacobian: str or a callable function, optional
            Derivative of the non-linear model, w.r.t. the state to be estimated. 
            Defaults to '2-point' in least squares for all optmisation methods. Method 'lm' only works with '2-point'.
            
            If callable, the function's first argument should be the state, and should return a numpy array of size (m, n).
            Also, if callable, the additional arguments should be the same as accepted by the residual function.
        method: str, optional
            Optimisation method, passed to the least squares function. Defaults to 'lm'.
        args: tuple, optional
            The extra arguments to be passed to the residual and the Jacobian function. The least squares function passes the same arguments
            to both functions.
        """
        self.__residual_function = residual_function

        # default way to calculate Jacobian that is preferred by lm method of least squares
        # for an invalid string value, the least squares method already throws an error, so we do not need to check that here
        if jacobian is None:
            jacobian = "2-point"
        self.__jacobian = jacobian

        self.__optimisation_method = method

        self.__args = args

    def optimise(self, initial_state: np.ndarray) -> OptimizeResult:
        """
        Optimizes the selected residual function w.r.t. the given initial state using the specified optimisation method.

        Parameters
        ----------
        initial_state : np.ndarray
            The starting point for the state to be estimated.

        Returns
        -------
        OptimizeResult
            The results of the optimisation. The final state estimate can be accessed through the `x` field of this result.
            For a more detailed understanding of this object please refer to `scipy.optimize.OptimizeResult`.
        """
        return least_squares(
            fun=self.__residual_function,
            x0=initial_state,
            jac=self.__jacobian,  # type: ignore
            method=self.__optimisation_method,
            args=self.__args
        )


class ExtendedKalmanFilter:
    def __init__(
        self,
        dim_state: int,
        process_model: TypeModels,
        process_jacobian: TypeModels,
        cov_state: np.ndarray,
        noise_state: np.ndarray,
        dim_meas: int,
        measurement_model: Callable,
        measurement_jacobian: Callable,
        noise_meas: np.ndarray,
        initial_state: Optional[np.ndarray],
    ) -> None:
        """
        An Extended Kalman Filter class that implements the prediction-update steps. All of the configuration matrices
        are set during the intialisation, and are expected not to be changed during the predict-update cycle.
        There are currently no checks made to confirm if all the vector and matrix sizes are consistent with each other.

        Parameters:
        -----------
        dim_state: int
            Dimension of the state to be estimated.
        process_model: np.ndarray or a callable function
            Model describing the process that governs the evolution of the state.
        process_jacobian: np.ndarray or a callable function
            Derivative of the process model with respect to the state. Must be, or should return a square matrix with dimensions (dim_state, dim_state).
        cov_state: np.ndarray
            Initial state covariance matrix. Must be a square matrix with dimensions (dim_state, dim_state).
        noise_state: np.ndarray
            Process noise matrix. Must be a square matrix with dimensions (dim_state, dim_state).
        dim_meas: int
            Dimension of the measurement space.
        measurement_model: Callable
            Computes the expected measurement. Must return an array of dimension (dim_meas,)
        measurement_jacobian: Callable
            Derivative of the measurement model with respect to the state. Must return an ndarray of dimension (dim_meas, dim_state)
        noise_meas: np.ndarray
            Measurement noise matrix. Must be a square matrix with dimensions (dim_meas, dim_meas).
        initial_state: np.ndarray, optional
            Initial estimate of the state. Defaults to an array of ones.
        """
        if initial_state is None:
            initial_state = np.ones((self.__dim_state,), dtype=np.float64)

        # TODO: there should be a check here to confirm all the sizes of matrix/vectors play well together

        # config related to state, its transition, and Jacobian
        self.__dim_state = dim_state
        self.x = initial_state
        self.fProcess_model = process_model                 # non-linear process model
        self.FProcess_jacobian = process_jacobian           # Jacobian of process model
        self.PCov_state = cov_state                         # initial covraiance of state
        self.QProcess_noise = noise_state                   # process noise
        self.__I = np.eye(self.__dim_state)                 # identity matrix for covariance update

        # config related to measurement, its predicted function and Jacobian
        self.__dim_meas = dim_meas
        self.hMeas_model = measurement_model                # non-linear measurement model
        self.HMeas_jacobian = measurement_jacobian          # Jacobian of measurement model
        self.RMeas_noise = noise_meas                       # measurement noise

    def predict(self, pm_args: tuple = (), pm_jac_args: tuple = ()):
        """
        Performs the prediction step of the EKF.
        
        Parameters
        ----------
        pm_args, pm_jac_args: tuple, optional
            The arguments to be passed to the process model/Jacobian function set during initialisation. 
            Used only if both the model and Jacobian are callable functions.
        """

        # compute predicted next state
        if isinstance(self.fProcess_model, np.ndarray):
            hat_x_t = np.dot(self.fProcess_model, self.x)
        elif isinstance(self.fProcess_model, Callable):
            hat_x_t = self.fProcess_model(self.x, *pm_args)

        # compute process Jacobian given current state
        # don't update state before evaluating the Jacobian
        if isinstance(self.FProcess_jacobian, np.ndarray):
            F_t = self.FProcess_jacobian
        elif isinstance(self.FProcess_jacobian, Callable):
            F_t = self.FProcess_jacobian(self.x, *pm_jac_args)

        # rewritting variables in the commonly accepted EKF notation for clarity
        P_t = self.PCov_state
        Q_t = self.QProcess_noise

        # predicted state and covariance
        self.x = hat_x_t
        self.PCov_state = np.dot(np.dot(F_t, P_t), F_t.T) + Q_t

    def update(self, z: np.ndarray, mm_args: tuple = (), mm_jac_args: tuple = ()):
        """
        Performs the update step of the EKF.

        Parameters
        ----------
        z: numpy.array
            New measurement vector for this time step. Must be an array of size (dim_meas,)
        mm_args, mm_jac_args: tuple, optional
            The arguments to be passed to the measurement model/Jacobian function set during initialisation. 
            Used only if both the model and Jacobian are callable functions.
        """

        # TODO: callable functions could be an optional argument here but then if not passed just use the one set in init
        # TODO: residuals, measurement model and measurement jacobian could be np array arguments, add if check like above

        # predicted measurement based on measurement model
        hat_z_t = self.hMeas_model(self.x, *mm_args)

        # measurement residual
        residual = z - hat_z_t

        H_t = self.HMeas_jacobian(self.x, *mm_jac_args)

        # innovation
        P_t = self.PCov_state
        R_t = self.RMeas_noise
        S_t = np.dot(np.dot(H_t, P_t), H_t.T) + R_t
        inv_S_t = np.linalg.inv(S_t)

        # Kalman Gain
        K_t = np.dot(np.dot(P_t, H_t.T), inv_S_t)

        # update state estimate
        self.x = self.x + np.dot(K_t, residual)

        # update covariance estimate
        # known as the Joseph equation, said to be the more numerically stable and symmetric way to update covariance
        I_KH = self.__I - np.dot(K_t, H_t)
        self.PCov_state = np.dot(np.dot(I_KH, P_t), I_KH.T) + np.dot(np.dot(K_t, R_t), K_t.T)
