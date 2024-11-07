import numpy as np
import matplotlib .pyplot as plt

t = np.arange(0, 0.008, 0.00001)
x = 5 * np.sin(2 * np.pi * 500 * t + np.pi/2)

plt.figure(figsize=(10,6))

plt.subplot(3,2,1)
plt.plot(t, x)
plt.title('x(n) = 5sin(2π.500t + π/2)')
plt.xlabel('Times(s)')
plt.ylabel('Amplitude')

fs = 3750
n = np.arange(0, 0.008, 1/fs)
x_n = 5 * np.sin(2 * np.pi * 500 * n + np.pi/2)

plt.subplot(3,2,2)
plt.stem(x_n)
plt.title('x(n) = 5sin(2π.(500/3750)n + π/2)')
plt.xlabel('n')
plt.ylabel('x(n)')

#Upsampling by factor of L = 2
L = 2
upsample_x = np.zeros(L* len(x_n))
upsample_x[::L] = x_n

plt.subplot(3,2,3)
plt.stem(upsample_x)
plt.title('Upsampled Sequence(with zeros)')
plt.xlabel('n')
plt.ylabel('Amplitude')

#Downsampling by factor of M = 2
M = 2
downsample_x = x_n[::M]

plt.subplot(3,2,4)
plt.stem(downsample_x)
plt.title('Downsampled sequence')
plt.xlabel('n')
plt.ylabel('Amplitude')

#Smooth of upsampling
for i in range(1, len(upsample_x)-1, 2):
    upsample_x[i] = upsample_x[i-1]+upsample_x[i+1] / 2

plt.subplot(3,2,5)
plt.stem(upsample_x)
plt.title('Upsampled sequence(smooth)')
plt.xlabel('n')
plt.ylabel('Amplitude')


plt.show()