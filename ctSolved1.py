import numpy as np
import matplotlib.pyplot as plt

sampling_rate = 360
num_sampling = 512

fre1, amp1 = 40, 10
fre2, amp2 = 80, 20
fre3, amp3 = 160, 40

t = np.arange(num_sampling)/sampling_rate 

signal1 = amp1*np.sin(2*np.pi*fre1*t)
signal2 = amp2*np.sin(2*np.pi*fre2*t)
signal3 = amp3*np.sin(2*np.pi*fre3*t)

composite_signal = signal1+signal2+signal3

plt.figure(figsize=(20, 10))
plt.subplots_adjust(hspace=0.5)
plt.subplot(3,2,1)
plt.plot(t, signal1)
plt.title("Sine signal - 1")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.subplot(3,2,2)
plt.plot(t, signal2)
plt.title("Sine signal - 2")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.subplot(3,2,3)
plt.plot(t, signal3)
plt.title("Sine signal - 3")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.subplot(3,2,4)
plt.plot(t, composite_signal)
plt.title("Composite signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
