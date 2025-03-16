import numpy as np

fileData = np.genfromtxt("data.txt", delimiter=",")
print(fileData > 5)

h1 = np.ones((2, 3))
h2 = np.zeros((2, 2))

print(h1)
print(h2)