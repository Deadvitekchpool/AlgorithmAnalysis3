import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import minimize

alpha = random.uniform(0, 1)
beta = random.uniform(0, 1)

x_k = []
y_k = []
a = []

for i in range(0, 101):
	d = np.random.normal(0, 1)
	temp_x = i / 100
	temp_y = alpha * temp_x + beta + d
	x_k.append(temp_x)
	y_k.append(temp_y)

x_k = np.array(x_k)
y_k = np.array(y_k)

def f(x, a, b):
	return a / (1 + b * x)

def D(params):
	global x_k, y_k

	a, b = params

	return np.sum((f(x_k, a, b) - y_k)**2)

def D_lm(params):
	global x_k, y_k

	a, b = params

	return f(x_k, a, b) - y_k

def jac_rational(params):
	a, b = params
	global x_k, y_k

	n = float(len(x_k))

	dy_da = (2 / n) * np.sum((1 / (1 + b * x_k)) * (y_k - (a / (1 + b * x_k))))
	dy_db = (-2 / n) * np.sum((y_k - (a / (1 + b * x_k))) * (x_k * a / (b * x_k + 1)**2))

	return (dy_da, dy_db) 

def gradient_descent_step(iters):
	global x_k, y_k

	m, c = 0, 0
	L = 0.001

	for i in range(iters): 
		Y_pred = m / (1 + x_k * c)
		D_c = np.sum(-2 * m * x_k * (m - y_k * (c * x_k + 1)) / (c * x_k + 1)**3)
		D_m = np.sum(-2 *(c * x_k * y_k - m + y_k ) / (c * x_k + 1)**2)
		m = m - L * D_m
		c = c - L * D_c

	return m, c


gd = gradient_descent_step(100000)

cg = minimize(D, [1, 1], method="CG", tol=0.001)

newton = minimize(D, [1, 1], method="BFGS", tol=0.001)

lm = least_squares(D_lm, [1, 1], method="lm", xtol=0.001, ftol=0.001)

print("GD")
print(gd)
print("CG")
print(cg)
print("NEWTON")
print(newton)
print("LM")
print("Number of function calls: " + str(lm.nfev))

plt.title("Rational approximant")
plt.scatter(x_k, y_k, label="Generated data", color='orange')
plt.plot(x_k, alpha / (1 + beta * x_k), "r", label="Generative line")
plt.plot(x_k, gd[0] / (1 + gd[1] * x_k), label="Gradient descent", color='black')
plt.plot(x_k, cg.x[0] / (1 + cg.x[1] * x_k), label="Conjugate gradient", color='green')
plt.plot(x_k, newton.x[0] / (1 + newton.x[1] * x_k), label="Newton method", color='purple')
plt.plot(x_k, lm.x[0] / (1 + lm.x[1] * x_k), label="Levenberg-Marquardt method", color='blue')
plt.grid()
plt.legend()
plt.show()