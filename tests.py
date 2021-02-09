import numpy as np
from scipy import linalg
from scipy import signal
import math 
import matplotlib.pyplot as plt
import matplotlib as mpl
from source.auxiliary.other_utilities import get_adjusted_path_string
from source.auxiliary.auxiliary_functionalities import parse_load_signal
import source.postprocess.plotter_utilities as pl_ut
f = [1,2,3,6,8]
b = [2,3,4,5,6]
k = np.arange(1,10)
k = np.reshape(k, (3,3))

#k = np.delete(k, np.arange(0,1),1)
print(k)



# for val in f:
#     print (b[f[0]:f[-1]+1])

#print (mpl.rcParams.keys())
# fs = 10e3
# N = 1e5
# amp = 20
# freq = 1234.0
# noise_power = 0.001 * fs / 2
# time = np.arange(N) / fs
# b, a = signal.butter(2, 0.25, 'low')
# x = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
# y = signal.lfilter(b, a, x)
# x += amp*np.sin(2*np.pi*freq*time)
# y += np.random.normal(scale=0.1*np.sqrt(noise_power), size=time.shape)
# f, Pxy = signal.csd(x, y, fs, nperseg=1024)
# k = np.abs(Pxy)
# plt.rcParams.update(pl_ut.parameters)
# plt.semilogy(f, np.abs(Pxy))
# plt.xlabel('frequency [Hz]')
# plt.ylabel('CSD [V**2/Hz]')

# plt.savefig('test_rcc.pdf')
# plt.show()

#
