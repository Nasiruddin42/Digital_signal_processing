import numpy as np
import matplotlib.pyplot as plt


def convulation(x, h):
    len_x = len(x)
    len_h = len(h)
    len_y = len_x + len_h -1
    y = [0]*len_y

    for i in range(len_y):
        for k in range(len_x):
            if(i-k >= 0 and i-k < len_h):
                y[i] += x[k] * h[i-k]
    return y 

x = [1, 2, 3, 1]
h = [1, 2, 1, -1]

plt.figure(figsize=(7,5))
plt.subplots_adjust(wspace=0.5)
plt.subplots_adjust(hspace=0.5)

plt.subplot(2,2,1)
plt.stem(x)
plt.title('x(n)')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')

plt.subplot(2,2,2)
plt.stem(h)
plt.title('h(n)')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')

#convulation
y = convulation(x, h)

plt.subplot(2,2,3)
plt.stem(y)
plt.title('y(n) = x(n)*h(n)')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')

#crossrelation
y = convulation(x, h[::-1])

plt.subplot(2,2,4)
plt.stem(y)
plt.title('y(n)= x(n).h(n)')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')

plt.show()