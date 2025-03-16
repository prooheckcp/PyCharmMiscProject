import copy, math
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

# Our offset in the linear regression model
b_init = 785.1811367994083
# One value for each feature in x_train
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

def model(x, w, b: float):
    return np.dot(x, w) + b

def compute_cost(x, y, w, b):
    m = len(x)
    cost_sum = 0
    for i in range(m):
        cost_sum += (model(x[i], w, b) - y[i])**2

    return cost_sum / (m * 2)

def gradient_derivative(x, y, w, b):
    m = len(x)
    total_sum_dw = 0
    total_sum_db = 0

    for i in range(m):
        y_hat = model(x[i], w, b)
        total_sum_dw += (y_hat - y[i]) * x[i];
        total_sum_db += y_hat - y[i];

    dj_dw = total_sum_dw / m;
    dj_db = total_sum_db / m;

    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    j_history = []
    w = copy.deepcopy(w_in);
    b = b_in

    for i in range(num_iters):
        # Get current tangent lines
        dj_dw, dj_db = gradient_function(x, y, w, b);

        # Update parameters using w, b, alpha, and gradient
        w -= alpha * dj_dw
        b -= alpha * dj_db

        # Save cost J at each iteration
        if i < 10000:
            j_history.append(cost_function(x, y, w, b));

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {j_history[-1]:8.2f}   ")

    return w, b, j_history;


# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent
w_final, b_final, J_hist = gradient_descent(
    x_train,
    y_train,
    initial_w,
    initial_b,
    compute_cost,
    gradient_derivative,
    alpha,
    iterations
)

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")

for i in range(len(x_train)):
    print(f"prediction: {np.dot(x_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step')
plt.show()