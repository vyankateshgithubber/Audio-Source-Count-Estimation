import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import size
from scipy.io import loadmat
from scipy.io import wavfile
from scipy.io.wavfile import write

mat1 = loadmat('45.mat')
h1 = mat1['impulse_response'][:,0]
fs1, s1 = wavfile.read('1_0a139a.wav')
convolved_speech_data1 = np.convolve(s1[:80000],h1[:80000])
print(convolved_speech_data1,size(s1),size(h1))
write('sound1.wav',fs1,convolved_speech_data1.astype(np.int16))
plt.subplot(3,1,1)
plt.plot(np.arange(0,80000),h1[:80000])
plt.subplot(3,1,2)
plt.plot(s1)
plt.subplot(3,1,3)
plt.plot(convolved_speech_data1)
plt.show()


mat2 = loadmat('135.mat')
fs2, s2 = wavfile.read('1_00bdad.wav')
h2 = mat2['impulse_response'][:,0]
convolved_speech_data2 = np.convolve(s2[:80000],h2[:80000])
print(convolved_speech_data2,size(s2),size(h2))
write('sound2.wav',fs2,convolved_speech_data2.astype(np.int16))
plt.subplot(3,1,1)
plt.plot(np.arange(0,80000),h2[:80000])
plt.subplot(3,1,2)
plt.plot(s2)
plt.subplot(3,1,3)
plt.plot(convolved_speech_data2)
plt.show()

mixed_signal = convolved_speech_data1+convolved_speech_data2
write('sound.wav',fs2,mixed_signal.astype(np.int16))
plt.plot(mixed_signal)
plt.show()