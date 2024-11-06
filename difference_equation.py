#y(n) = 0.3x(n) + 0.4x(n-1) + 0.3x(n-2)

import numpy as np
import matplotlib.pyplot as plt

def convolve(h, x):
    len_x = len(x)
    len_h = len(h)
    len_y = len_x + len_h - 1 
    y = [0]*len_y

    for i in range(len_y):
        for k in range(len_x):
            if(i - k >= 0 and i - k < len_h):
                y[i] += x[k] * h[i-k]
    return y

x = np.array([1, 2, 3, 4, 5])
h = np.array([0.3, 0.4, 0.3])

y = convolve(h, x)
print('y(n) = ', y)
