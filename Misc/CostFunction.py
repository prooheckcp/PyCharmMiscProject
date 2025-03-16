import numpy as np
import matplotlib.pyplot as plt

# Let's hard code our 'w' and 'b' for now
w = 200
b = 100

# Input
x_train = np.array([1.0, 2.0]);

# Output
y_train = np.array([300.0, 500.0]);

def model(x):
    return w * x + b;

# Computes the cost function for linear regression
def compute_cost(x, y):
    cost_sum = 0;
    dataset_size = len(x);

    for i in range(dataset_size):
        cost_sum += pow((model(x[i]) - y[i]), 2);

    return cost_sum / (2 * dataset_size);

print(f"Cost function: {compute_cost(x_train, y_train)}");

