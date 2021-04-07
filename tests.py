import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import signal
import json
import os
import source.ESWL.eswl_auxiliaries as auxiliary
from source.auxiliary.other_utilities import get_adjusted_path_string

def rms(signal):
    '''
    returns the root mean square of a signal
    '''
    return np.sqrt(np.mean(np.square(signal)))

def cov_custom_std(signal1, signal2):
    
    mean1 = np.mean(signal1)
    mean2 = np.mean(signal2)
    cov = 0.0
    for i in range(len(signal1)):
        cov += (signal1[i] - mean1)*(signal2[i] - mean2)
        
    return cov/len(signal1)

def cov_custom_rms(signal1, signal2):
    
    mean1 = np.mean(signal1)
    mean2 = np.mean(signal2)
    cov = 0.0
    for i in range(len(signal1)):
        cov += (signal1[i])*(signal2[i])
        
    return cov/len(signal1)

dynamic_load_file = "input\\force\\generic_building\\dynamic_force_4_nodes.npy"
load_signals_raw = np.load(dynamic_load_file)

load_signals = auxiliary.parse_load_signal(load_signals_raw, 6, discard_time = 1000)

x = np.arange(0,10*np.pi,0.1)

sin = np.sin(x)
cos = np.cos(x + np.pi/2)

fig, ax = plt.subplots()
ax.plot(x,sin, label = 'sin')
ax.plot(x,cos, label = 'cos')
plt.legend()
plt.grid()
plt.show()

print ('std sin, cos', np.std(sin),np.std(cos))
print ('rms sin, cos', rms(sin),rms(cos))
print()

cov_sc_np = np.cov(sin,cos)[0][1]
cor_sc_np = np.corrcoef(sin,cos)[0][1]
print('numpy cov sin cos', cov_sc_np)
print('numpy cor sin cos', cor_sc_np)

# cov custom all std
cov_sc_std = cov_custom_std(sin,cos)
cor_sc_std_std = cov_sc_std /(np.std(sin), np.std(cos))
print('std cov sin cos    ', cov_sc_std)
print('std std cor sin cos', cor_sc_std_std)

# cov custom all rms
cov_sc_rms = cov_custom_rms(sin,cos)
cor_sc_rms_rms = cov_sc_rms /(rms(sin), rms(cos))
print('rms cov sin cos    ', cov_sc_rms)
print('rms rms cor sin cos', cor_sc_rms_rms)

print()
s1, s2 = load_signals['y'][2], load_signals['y'][3]

cov_custom = cov_custom_std(s1,s2)
cov_np = np.cov(s1,s2)[0][1]
cov_rms = cov_custom_rms(s1,s2)

cor_np_rms = cov_np/ (rms(s1)*rms(s2))
cor_custom_rms_rms = cov_rms/ (rms(s1)*rms(s2))
cor_custom_std = cov_np/ (np.std(s1)*np.std(s2))
cor_numpy = np.corrcoef(s1,s2)[0][1]

print ('cov custom', cov_custom)
print ('cov numpy ', cov_np)
print ('cov rms   ', cov_rms)
print ('custom - numpy', cov_custom- cov_np, 'this is', round((cov_custom- cov_np)/cov_np * 100, 5), '%', 'of the numpy value')
print ()
print ('cor numpy rms', round(cor_np_rms,3))
print ('cor custom rms rms', round(cor_custom_rms_rms,3))
print ('cor custom std', round(cor_custom_std,3))
print ('cor numpy corr', round(cor_numpy,3))
