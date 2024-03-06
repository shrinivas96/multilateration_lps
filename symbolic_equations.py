import matplotlib.pyplot as plt
import sympy as sp
import numpy as np

t = sp.Symbol('t')

x1, y1 = sp.symbols('x_1 y_1', real=True, positive=True)
x2, y2 = sp.symbols('x_2 y_2', real=True, positive=True)
x3, y3 = sp.symbols('x_3 y_3', real=True, positive=True)
x4, y4 = sp.symbols('x_4 y_4', real=True, positive=True)

n1, n2, n3, n4 = 0, 0, 0, 0     # sp.symbols('n_1 n_2 n_3 n_4')

xp = sp.Function('x_p', real=True, positive=True)(t)
yp = sp.Function('y_p', real=True, positive=True)(t)

state = sp.Matrix([xp, yp])
constants = [(x1, 0), (y1, 0), (x2, 0), (y2, 60), (x3, 100), (y3, 60), (x4, 100), (y4, 0)]

g = sp.Matrix([((xp - x1)**2 + (yp - y1)**2)**(0.5) + n1,
               ((xp - x2)**2 + (yp - y2)**2)**(0.5) + n2,
               ((xp - x3)**2 + (yp - y3)**2)**(0.5) + n3,
               ((xp - x4)**2 + (yp - y4)**2)**(0.5) + n4])
gConstEvaled = g.subs(constants)
gPartDeriv = gConstEvaled.jacobian(state)

# gCPartDeriv = gPartDeriv.subs(constants)

# Number of iterations
num_iterations = 10

alpha = 0.2
measurements = sp.Matrix([55.18701752483591, 61.55713628575031, 61.32000781174319, 55.65164061739057])
init_state_guess = sp.Matrix([20, 20])
states_history = [init_state_guess]


# Arrays to store x and y coordinates
x_coords = np.zeros(num_iterations)
y_coords = np.zeros(num_iterations)

# Initialize first point
x_coords[0], y_coords[0] = init_state_guess


for i in range(1, num_iterations):
    curr_state = states_history[i-1]
    x_t, y_t = curr_state
    print(x_t, y_t)
    state_eval = [(xp, x_t), (yp, y_t)]
    Ji = gPartDeriv.subs(state_eval)
    Pi1 = Ji.T * Ji
    Pi = -Pi1.inv() * Ji.T
    res = measurements - gConstEvaled.subs(state_eval)
    new_state = states_history[i-1] + alpha*Pi*res
    sNewState = sp.simplify(new_state)
    states_history.append(new_state)
    x_coords[i], y_coords[i] = new_state


plt.figure(figsize=(8, 6))
plt.plot(x_coords, y_coords, marker='o')
plt.title('Change of Coordinates')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.show()