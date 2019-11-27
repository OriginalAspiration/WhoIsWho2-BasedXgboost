import time
import json
import pickle

import numpy as np
import xgboost as xgb

a = []
for i in range (5):
    a.append(np.ones(3))
a.append(np.ones(3) + 1)
a.append(np.ones(3) - 5)
print(a)

a = np.sum(a, axis=0) / len(a)

b = []
for i in range (5):
    b.append(np.ones(3))
b.append(np.ones(3) + 1)
b.append(np.ones(3) - 5)
print(b)

b = np.sum(b, axis=0) / len(b)
print(b)

c.append(a)
c.append(b)
a = a / np.max(np.abs(a))

print(a)
print(type(a))

