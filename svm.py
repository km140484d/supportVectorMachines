import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

# df = pd.read_csv('data.csv', header=None)
# df.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'y']
# df.insert(0, 'one', 1)
# df = df.sample(frac=1)
#
# boundary_index = round(df.shape[0] * 0.8)       # uzeto je da je 80% skup za treniranje, a 20% skup za testiranje
# X, Y = df.iloc[:, 0:6].to_numpy(), df['y'].to_numpy()
#
# # polinomijalna regresija
# predictors = 5                                  # inicijalan broj prediktora 5 + 1 (kolona sa 1)
# J_arr, theta_arr, predictors_arr = [], [], []

x_neg = np.array([[3, 4], [1, 4], [2, 3]])
y_neg = np.array([-1, -1, -1])
x_pos = np.array([[6, -1], [7, -1], [5, -3]])
y_pos = np.array([1, 1, 1])
x1 = np.linspace(-10, 10)
x = np.vstack((np.linspace(-10, 10), np.linspace(-10, 10)))

# Data for the next section
X = np.vstack((x_pos, x_neg))
y = np.concatenate((y_pos, y_neg))

print(X, y)

# Parameters guessed by inspection
w = np.array([1, -1]).reshape(-1, 1)
b = -3

m,n = X.shape
y = y.reshape(-1,1) * 1.
X_dash = y * X
H = np.dot(X_dash , X_dash.T) * 1.

#Converting into cvxopt format
P = cvxopt_matrix(H)
print("P", P)
q = cvxopt_matrix(-np.ones((m, 1)))
G = cvxopt_matrix(-np.eye(m))
h = cvxopt_matrix(np.zeros(m))
A = cvxopt_matrix(y.reshape(1, -1))
b = cvxopt_matrix(np.zeros(1))

#Setting solver parameters (change default to decrease tolerance)
cvxopt_solvers.options['show_progress'] = False
cvxopt_solvers.options['abstol'] = 1e-10
cvxopt_solvers.options['reltol'] = 1e-10
cvxopt_solvers.options['feastol'] = 1e-10

#Run solver
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x'])
