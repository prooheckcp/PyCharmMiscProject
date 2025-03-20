import numpy as np
import matplotlib.pyplot as plt
from plt_overfit import overfit_example, output
from lab_utils_common import sigmoid
np.set_printoptions(precision=8)

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

def compute_cost_logistic_reg(X, y, w, b, lambda_=1):
    m, n = X.shape

    cost = 0
    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)

    cost/= m

    cost_reg = 0
    for j in range(n):
        cost_reg += w[j] ** 2
    cost_reg *= (lambda_/(2*m))

    return cost + cost_reg

np.random.seed(1)
X_tmp = np.random.rand(5,6)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print("Regularized cost:", cost_tmp)


