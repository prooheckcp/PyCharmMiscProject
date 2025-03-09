import math, copy
import numpy as np
import matplotlib.pyplot as plt

# Input
x_train = np.array([1.0, 2.0]);

# Output
y_train = np.array([300.0, 500.0]);

# Global variables
W = 0
B = 0

def model(x):
    return W * x + B;

def compute_cost(x, y):
    length = len(x);
    cost = 0.0;

    for i in range(length):
        f_wb = model(x[i]);
        cost += (f_wb - y[i])**2;

    return cost / (2 * length);







