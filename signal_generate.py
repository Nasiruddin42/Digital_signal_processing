import numpy as np
import matplotlib.pyplot as plt

#sine wave
t = np.arange(0, 1, 0.001)
signal = np.sin(2*np.pi*2*t)

plt.figure(figsize=(10, 5))
plt.subplots_adjust(hspace=0.5)
plt.subplots_adjust(wspace=0.5)

plt.subplot(3,3,1)
plt.plot(t, signal)
plt.title('Sine wave')
plt.xlabel('Time')
plt.ylabel('Amplitude')

#cosine wave

t = np.arange(0, 1, 0.001)
signal1 = np.cos(2*np.pi*2*t)

plt.subplot(3,3,2)
plt.plot(t, signal1)
plt.title('Cosine wave')
plt.xlabel('Time')
plt.ylabel('Amplitude')

#unit impulse sequence 
def uni_impulse(shift, n):
    delta = []
    for i in n :
        if(i == shift):
            delta.append(1)
        else:
            delta.append(0)
    return delta 

n = np.arange(-5, 6, 1)
delta = uni_impulse(0, n)

plt.subplot(3,3,3)
plt.stem(n, delta)
plt.title('Unit impulse response')
plt.xlabel('n')
plt.ylabel('δ(n)')

#unit step sequence
def uni_step(shift, n):
    step = []
    for i in n:
        if(i < shift):
            step.append(0)
        else:
            step.append(1)
    return step 

n = np.arange(-5, 6, 1)
step = uni_step(0, n)

plt.subplot(3,3,4)
plt.stem(n, step)
plt.title('Unit step sequence')
plt.xlabel('n')
plt.ylabel('δ(n)')

#unit ramp sequence
def uni_ramp(shift, n):
    ramp = []
    j = 0
    for i in n:
        if(i < shift):
            ramp.append(0)
        else:
            ramp.append(j)
            j+=1
    return ramp

n = np.arange(-5, 6, 1)
ramp = uni_ramp(0, n)

plt.subplot(3,3,5)
plt.stem(n, ramp)
plt.title('Unit ramp signal')
plt.xlabel('n')
plt.ylabel('δ(n)')

#Exponential signal when (0<a<1)
n = np.arange(0, 20, 1)
a = 0.8
exp = a**n

plt.subplot(3,3,6)
plt.stem(n, exp)
plt.title('Exponential (0<a<1)')
plt.xlabel('n')
plt.ylabel('exp(n)')

#Exponential sinal when (a>1)
n = np.arange(0, 20, 1)
a = 1.2
exp = a**n

plt.subplot(3,3,7)
plt.stem(n, exp)
plt.title('Exponential signal (a>1)')
plt.xlabel('n')
plt.ylabel('exp(n)')


plt.show()