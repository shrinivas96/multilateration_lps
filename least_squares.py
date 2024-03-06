from generate_ranges import FieldAssets
import matplotlib.pyplot as plt
import numpy as np

def est_h_i(state: np.ndarray):
    xp, yp = state
    xp2, yp2 = xp**2, yp**2
    yp60 = (yp - 60)**2
    xp100 = (xp - 100)**2
    return np.array([np.sqrt(xp2 + yp2),
                     np.sqrt(yp60 + xp2),
                     np.sqrt(xp100 + yp60),
                     np.sqrt(xp100 + yp2)])

def jacobian_i(state: np.ndarray):
    xp, yp = state
    xp2, yp2 = xp**2, yp**2
    yp60 = (yp - 60)**2
    xp100 = (xp - 100)**2

    j11 = xp / np.sqrt(xp2 + yp2)
    j12 = yp / np.sqrt(xp2 + yp2)
    j21 = xp / np.sqrt(xp2 + yp60)
    j22 = yp - 60 / np.sqrt(xp2 + yp60)
    j31 = xp - 100 / np.sqrt(xp100 + yp60)
    j32 = yp - 60 / np.sqrt(xp100 + yp60)
    j41 = xp - 100 / np.sqrt(xp100 + yp2)
    j42 = yp / np.sqrt(xp100 + yp2)
    Ji = np.array([[j11, j12],
                   [j21, j22],
                   [j31, j32],
                   [j41, j42]])
    return Ji

def least_squares(
        measurement: np.ndarray, 
        initial_guess: np.ndarray,
        iteration_count: int):
    # some params
    alpha = 0.2
    state_history = np.zeros((initial_guess.shape[0], iteration_count))
    state_history[:, 0] = initial_guess

    for i in range(1, iteration_count):
        x_t, y_t = state_history[:, i-1]
        Ji = jacobian_i(state_history[:, i-1])
        Pi1 = np.dot(Ji.T, Ji)
        Pi = -np.dot(np.linalg.inv(Pi1), Ji.T)
        residual = measurement - est_h_i(state_history[:, i-1])
        state_history[:, i] = state_history[:, i-1] + alpha * np.dot(Pi, residual.T)
    
    return state_history


if __name__ == '__main__':
    initial_position = np.array([50.0, 24.0])
    obj = FieldAssets(100, 60, initial_position)
    measurement = np.array(obj.rangingGenerator())
    
    initial_guess = np.array([48., 22.])
    iteration_count = 1000
    state_trajectory = least_squares(measurement, initial_guess, iteration_count)

    plt.figure(figsize=(8, 6))
    plt.scatter(initial_position[0], initial_position[1])
    plt.plot(state_trajectory[0, :], state_trajectory[1, :], marker='o',
             linewidth=2, markersize=5)
    plt.title('Evolution of the state')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.show()