import numpy as np
import matplotlib.pyplot as plt
from plt_overfit import overfit_example, output
from lab_utils_common import sigmoid
np.set_printoptions(precision=8)

x_train = np.array([[1, 2, 3]])

def compute_cost_linear_reg(X, y, w, b, lambda_=1):
    m, n = X.shape

    cost = 0
    for i in range(m):
        f_wb = np.dot(X[i], w) + b
        cost += (f_wb - y[i])**2

    cost /= (2*m)

    reg_cost = 0
    for j in range(n):
        reg_cost += w[j]**2

    reg_cost *= (lambda_/(2*m))

    total_cost = cost + reg_cost

    return total_cost