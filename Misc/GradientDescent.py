import math, copy
import numpy as np
import matplotlib.pyplot as plt

# Input
x_train = np.array([1.0, 2.0]);

# Output
y_train = np.array([300.0, 500.0]);

# Initial variables
initial_w = 0
initial_b = 100

def model(x, w, b):
    return w * x + b;

def compute_cost(x, y, w, b):
    length = len(x);
    cost = 0.0;

    for i in range(length):
        f_wb = model(x[i], w, b);
        cost += (f_wb - y[i])**2;

    return cost / (2 * length);

def compute_gradient(x, y, w, b):
    length = len(x);

    cost_w = 0.0;
    cost_b = 0.0;

    for i in range(length):
        f_wb = model(x[i], w, b) - y[i]
        cost_w += f_wb * x[i]
        cost_b += f_wb

    dj_wb = cost_w / length;
    dj_db = cost_b / length;

    return dj_wb, dj_db;

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    b = b_in;
    w = w_in;
    j_history = [];
    p_history = [];

    for i in range(num_iters):
        dj_w, dj_b = gradient_function(x, y, w, b);

        w -= alpha * dj_w;
        b -= alpha * dj_b;

        if i < 100000:
            j_history.append(cost_function(x, y, w, b));
            p_history.append([w, b]);

        # Print cost every at intervals 10 times or as many iteration if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(
                f"Iteration {i:4}: Cost {j_history[-1]:0.2e}",
                f"dj_dw: {dj_w: 0.3e}, dj_db: {dj_b: 0.3e}",
                f"w: {w: 0.3e}, b:{b: 0.5e}"
            );

    return w, b, j_history, p_history;

iterations = 10000
tmp_alpha = 1.0e-2

final_w, final_b, j_history, p_history = gradient_descent(
    x_train, y_train,  # Our data
    initial_w, initial_b, # Our initial point
    tmp_alpha, iterations, # Our dynamic values
    compute_cost, compute_gradient # Our model functions
);

print(f"(w, b) found by gradient descent: ({final_w:8.4f}, {final_b:8.4f})")

# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))

print(len(j_history))
ax2.plot(1000 + np.arange(len(j_history[1000:])), j_history[1000:])
ax2.set_title("Cost vs. iteration (end)")
ax2.set_ylabel('Cost')
ax2.set_xlabel('iteration step')

ax1.plot(j_history[:100])
ax1.set_title("Cost vs. iteration(start)");
ax1.set_ylabel('Cost');
ax1.set_xlabel('iteration step');
plt.show()

print(f"1000 sqft house prediction {final_w*1.0 + final_b:0.1f} Thousand dollars")
print(f"1200 sqft house prediction {final_w*1.2 + final_b:0.1f} Thousand dollars")
print(f"2000 sqft house prediction {final_w*2.0 + final_b:0.1f} Thousand dollars")


