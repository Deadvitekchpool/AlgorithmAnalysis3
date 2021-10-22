import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import least_squares

alpha = random.uniform(0, 1)
beta = random.uniform(0, 1)

x_k = []
y_k = []

for i in range(0, 101):
	d = np.random.normal(0, 1)
	temp_x = i / 100
	temp_y = alpha * temp_x + beta + d
	x_k.append(temp_x)
	y_k.append(temp_y)

x_k = np.array(x_k)
y_k = np.array(y_k)

def f(x, a, b):
	return a * x + b

def D(params):
	global x_k, y_k

	a, b = params

	return np.sum((f(x_k, a, b) - y_k)**2)

def D_lm(params):
	global x_k, y_k

	a, b = params

	return f(x_k, a, b) - y_k

def grad_desc(iters, L):
	global x_k, y_k
	m = 0
	c = 0
	n = float(len(x_k))

	for i in range(iters): 
	    Y_pred = m * x_k + c

	    D_m = (-2 / n) * np.sum(x_k * (y_k - Y_pred))
	    D_c = (-2 / n) * np.sum(y_k - Y_pred)

	    m = m - L * D_m
	    c = c - L * D_c

	return (m, c)

gd = grad_desc(100000, 0.0001)

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

plt.title("Linear approximant")
plt.scatter(x_k, y_k, label="Generated data", color='orange')
plt.plot(x_k, alpha * x_k + beta, "r", label="Generative line")
plt.plot(x_k, gd[0] * x_k + gd[1], label="Gradient descent", color='black')
plt.plot(x_k, cg.x[0] * x_k + cg.x[1], label="Conjugate gradient", color='green')
plt.plot(x_k, newton.x[0] * x_k + newton.x[1], label="Newton method", color='purple')
plt.plot(x_k, lm.x[0] * x_k + lm.x[1], label="Levenberg-Marquardt method", color='blue')
plt.grid()
plt.legend()
plt.show()