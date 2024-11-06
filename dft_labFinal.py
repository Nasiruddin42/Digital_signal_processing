import numpy as np
import matplotlib.pyplot as plt
import cmath as cm
import math

def dft(x_n, N):
    x_m = np.zeros(N, dtype = np.complex128)
    for m in range(N):
        for n in range(N):
            x_m[m] += x_n[n] * np.exp(-2j * np.pi * m * n / N)
    return x_m

def phase(x_m, N):
    x_phase = []
    for z in (x_m):
        phase = cm.phase(round(z.real)+ round(z.imag) * 1j)
        x_phase.append(math.degrees(phase))
    return x_phase

def idft(x_m, N):
    x_n = np.zeros(N, dtype = np.complex128)
    for n in range(N):
        for m in range(N):
            x_n[n] += x_m[m] * np.exp(2j * np.pi * n * m / N)
        x_n[n] = x_n[n] / N 
    return x_n 

N = 8
n = np.arange(0, 0.001, 0.00001)
x = np.sin(2 * np.pi * 1000 * n) + 0.5 * np.sin(2 * np.pi * 2000 * n + 3*np.pi/4)

plt.figure(figsize=(8,6))
plt.subplots_adjust(hspace=0.8)
plt.subplots_adjust(wspace=0.5)

#input signal
plt.subplot(3,2,1)
plt.plot(n, x)
plt.title('Input signal')
plt.xlabel('Time(n)')
plt.ylabel('Amplitude')

fs = 8000
n = np.arange(0, 1, 1/fs)
x_n = np.sin(2 * np.pi * 1000 * n) + 0.5 * np.sin(2 * np.pi * 2000 * n + 3*np.pi/4)
x_n = x_n[0:N]

#sample signal
plt.subplot(3,2,2)
plt.stem(x_n)
plt.plot(x_n)
plt.title('Sampled signal')
plt.xlabel('n')
plt.ylabel('Amplitude')

#magnitude spectrum
x_m = dft(x_n, N)

plt.subplot(3,2,3)
plt.stem(np.abs(x_m))
plt.title('Magnitude spectrum')
plt.xlabel('m(kHz)')
plt.ylabel('Amplitude')

#power spectrum
plt.subplot(3,2,4)
plt.stem(np.abs(x_m)**2)
plt.title('Power Spectrum')
plt.xlabel('m(kHz)')
plt.ylabel('Amplitude')

#phase spectrum
x_phase = phase(x_m, N)

plt.subplot(3,2,5)
plt.stem(x_phase)
plt.title('Phase spectrum')
plt.xlabel('m(kHz)')
plt.ylabel('Angle(in degree)')

#dft reconstruction
x_n = idft(x_m, N)

plt.subplot(3,2,6)
plt.stem(x_n.real)
plt.plot(x_n.real)
plt.title('Idft reconstruction')
plt.xlabel('n')
plt.ylabel('Amplitude')

plt.show()