import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math

# Data represents the population and profits times 10,000
x_train, y_train = load_data()

def f_wb(x, w, b):
    return x * w + b


def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities)
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0]

    # You need to return this variable correctly
    total_cost = 0

    ### START CODE HERE ###
    prediction = x * w + b
    total_cost = np.sum((prediction - y)**2) / (2 * m)
    ### END CODE HERE ###

    return total_cost


def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities)
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b
     """

    # Number of training examples
    m = x.shape[0]

    # You need to return the following variables correctly
    dj_dw = 0
    dj_db = 0

    ### START CODE HERE ###
    prediction = x * w + b
    dj_db = (np.sum(prediction - y))/m
    dj_dw = (np.sum((prediction - y) * x))/m

    ### END CODE HERE ###

    return dj_dw, dj_db

final_cost = compute_cost(x_train, y_train, 2, 0);
dj_dw, dj_db = compute_gradient(x_train, y_train, 2, 0);
print(f"d_dw = {dj_dw}, d_db = {dj_db}")

# print x_train
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')
plt.show()