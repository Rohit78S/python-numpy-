import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
from math import log
# basic numpy 
# 3D array (2 blocks, each with 2 rows and 3 columns)
s = np.array([[[1, 2, 3], [4, 6, 5]],
              [[7, 8, 9], [10, 11, 12]]])

# Another copy of the same 3D array
s1 = np.array([[[1, 2, 3], [4, 6, 5]],
               [[7, 8, 9], [10, 11, 12]]])

# Reshape s into a new shape (2, 3, 2)
# Original shape: (2, 2, 3), reshaped to (2, 3, 2)
y = s.reshape(2, 3, 2)

# Flatten s into 1D array
y1 = s.reshape(-1)

# Number of dimensions of s
x = s.ndim

# Data type of array elements
z = s.dtype

# Split s into 1 sub-array along the first axis (axis=0 by default)
k = np.split(s, 1)

# Indices of elements where s is even
l = np.where(s % 2 == 0)

# Create a view of s (shares memory with s)
j = s.view()

# Modify the first block of the view
# Since j shares memory, original s is also affected
j[0] = 31

# Create a copy of s (independent, does not share memory)
m = s.copy()

# Concatenate s and s1 along axis=1 (rows)
d = np.concatenate((s, s1), axis=1)

# ================= Printing outputs =================
print("Concatenated array along axis=1:\n", d)
print("Flattened array:\n", y1)
print("Number of dimensions:", x)
print("Data type:", z)
print("Reshaped array (2,3,2):\n", y)
print("Sorted s along last axis:\n", np.sort(s))
print("Split array:\n", k)
print("Indices of even numbers:\n", l)
print("First 3 blocks of first dimension, first row:\n", s[0:3, 0:1])
print("View after modification:\n", j)
print("Copy (unaffected by view):\n", m)

# Iterate over all elements in 3D array and print them individually
for block in s:
    for row in block:
        for element in row:
            print(element)
