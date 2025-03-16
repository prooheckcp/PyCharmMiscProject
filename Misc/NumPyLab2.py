import numpy as np    # it is an unofficial standard to use np for numpy
import time

def my_dot(a, b):
    c = 0;
    for i in range(len(a)):
        c += a[i] * b[i];

    return c;

# test 1-D
X = np.array([[1],[2],[3],[4]])
w = np.array([2])
c = np.dot(X[1], w)

print(f"X[1] has shape {X[1].shape}")
print(f"w has shape {w.shape}")
print(f"c has shape {c.shape}")

a = np.zeros((2, 5))
print(f"a shape = {a.shape}, a = {a}")

a = np.array([
    [5, 4, 3],
    [3, 2 ,1],
    [1, 0, -1]
])

print(f"First one: {np.arange(20)}")
b = np.arange(20).reshape(-1, 10)
print(f"b = \n{b}")

#access 5 consecutive elements (start:stop:step)
print("a[0, 2:7:1] = ", b[0, 2:7:1], ",  a[0, 2:7:1].shape =", b[0, 2:7:1].shape, "a 1-D array")

#access 5 consecutive elements (start:stop:step) in two rows
print("b[:, 2:7:1] = 123\n", b[:, 2:7:1], ",  \n123b[:, 2:7:1].shape =", b[:, 2:7:1].shape, "a 2-D array")

# access all elements
print("ALL ELEMENTS a[:,:] = \n", b[:,:], ",  a[:,:].shape =", b[:,:].shape)