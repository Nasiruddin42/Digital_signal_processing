#1. Write a python program for generating a composite signal (you could use sine or cosine waves). 
#   The parameters including the signal frequencies of 40 Hz, 80 Hz, 160 Hz with 
#   the amplitudes of 10, 20, and 40 respectively, and the signal length should be limited to 512 in samples.
#2. Plot the generated signal.
#3. Do standard sampling by following the Nyquist rate.
#4. Perform under sampling and over sampling too. Use Subplot function to show the original, sampled, under sampled, and over sampled signal.
#5. Then perform N=512 point DFT, show the magnitude and phase spectrum.

import numpy as np
import matplotlib.pyplot as plt
import cmath as cm
import math

sampling_rate = 4000
s_length = 512
t = np.arange(s_length)/sampling_rate
x = 10*np.sin(2*np.pi*40*t) + 20*np.sin(2*np.pi*80*t) + 40*np.sin(2*np.pi*160*t)

plt.figure(figsize=(10,6))
plt.subplots_adjust(wspace=(0.5))
plt.subplots_adjust(hspace=(0.9))

plt.subplot(3,2,1)
plt.plot(t,x)
plt.title('composite signal(512 samples)')
plt.xlabel('Times(s)')
plt.ylabel('Amplitude')

sampling_rate = 320 # nyquist rate = 2*F(max)
t = np.arange(s_length)/sampling_rate
x = 10*np.sin(2*np.pi*40*t) + 20*np.sin(2*np.pi*80*t) + 40*np.sin(2*np.pi*160*t)

plt.subplot(3,2,2)
plt.plot(t,x)
plt.title('sampled signal(Nyquist rate)')
plt.xlabel('Times(s)')
plt.ylabel('Amplitude')

sampling_rate = 640 # over sampling
t = np.arange(s_length)/sampling_rate
x = 10*np.sin(2*np.pi*40*t) + 20*np.sin(2*np.pi*80*t) + 40*np.sin(2*np.pi*160*t)

plt.subplot(3,2,3)
plt.plot(t,x)
plt.title('sampled signal(over sampling))')
plt.xlabel('Times(s)')
plt.ylabel('Amplitude')

sampling_rate = 160 # Down sampling
t = np.arange(s_length)/sampling_rate
x = 10*np.sin(2*np.pi*40*t) + 20*np.sin(2*np.pi*80*t) + 40*np.sin(2*np.pi*160*t)

plt.subplot(3,2,4)
plt.plot(t,x)
plt.title('sampled signal(under sampling)')
plt.xlabel('Times(s)')
plt.ylabel('Amplitude')

#512 point DFT
def dft(x_n, N):
    x_m = np.zeros(N, dtype = np.complex128)
    for m in range(N):
        for n in range(N):
            x_m[m] += x_n[n] * np.exp(-2j * np.pi * n * m / N)

    return x_m 


N = 512
n = np.arange(0, 1, 1/N)
x = 10 * np.sin(2 * np.pi * 40 * n) + 20 * np.sin(2 * np.pi * 80 * n) + 40 * np.sin(2 * np.pi * 160 * n)

x_m = dft(x, N)

plt.subplot(3,2,5)
plt.stem(np.abs(x_m))
plt.title('Magntitude spectrum')
plt.xlabel('m')
plt.ylabel('x(m)')

#phase spectrum
def phase(x_m, N):
    x_phase = []
    for z in x_m:
        phase = cm.phase(round(z.real) + round(z.imag) * 1j)
        x_phase.append(math.degrees(phase))

    return x_phase

x_phase = phase(x_m, N)

plt.subplot(3,2,6)
plt.stem(x_phase)
plt.title('phase spectrum')
plt.xlabel('n')
plt.ylabel('angel in (degree)')


plt.show()