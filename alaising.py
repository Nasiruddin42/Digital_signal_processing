import numpy as np
import matplotlib.pyplot as plt

#Show that 50 Hz is an alias of the frequency 10 Hz, when sampling at 40 Hz. Assume the signals are
#cos(2π.10t) and cos(2π.50t)

n = np.arange(0, 1, 0.001)
signal = np.cos(2 * np.pi * 10 * n)

plt.figure(figsize=(10, 4))
plt.subplots_adjust(hspace=0.5)
plt.subplots_adjust(wspace=0.5)

plt.subplot(2,2,1)
plt.plot(n, signal)
plt.title('x(n)=cos(2pi.10.t)')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')

n = np.arange(0, 1, 0.001)
signal1 = np.cos(2 * np.pi * 50 * n)

plt.subplot(2, 2, 2)
plt.plot(n, signal1)
plt.title('x(n)= 2cos(2pi.50.t)')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')

fs = 40
n = np.arange(0, 1, 1/fs)
xn = np.cos(2 * np.pi * 10 * n)

plt.subplot(2, 2, 3)
plt.stem(n, xn)
plt.title('x(n)= cos(2pi(10/fs). t)')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')

n = np.arange(0, 1, 1/fs)
xn = np.cos(2 * np.pi * 50 * n)

plt.subplot(2, 2, 4)
plt.stem(n, xn)
plt.title('x(n)= cos(2pi(50/fs). t)')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')

plt.show()
