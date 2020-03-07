

import numpy as np

A = np.matrix([[3, 1], [1, 2]])
vals, vecs = np.linalg.eig(A)

idx = vals.argsort()[::-1]
vals = vals[idx]
vecs = vecs[:,idx]

print(vals)

print((5 + np.sqrt(5))/2)
print((5 - np.sqrt(5))/2)
print((1 + np.sqrt(5))/2)
print((1 - np.sqrt(5))/2)


print(vecs)

print(vecs[0, 0]/vecs[1, 0])
print(vecs[0, 1]/vecs[1, 1])