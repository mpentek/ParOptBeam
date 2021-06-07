import numpy as np
import matplotlib.pyplot as plt

def parse_load_signal(signal_raw, dofs_per_node, discard_time = None):
    '''
    sorts the load signals in a dictionary with load direction as keys: 
    x,y,z: nodal force 
    a,b,g: nodal moments 
    deletes first entries until discard_time
    '''
    if dofs_per_node != 6:
        raise Exception('load signal parsing only for 6 dofs per node - check dynamic load files')
    else:
        signal = {}
        for i, label in enumerate(['x', 'y', 'z', 'a', 'b', 'g']):
            signal[label] = signal_raw[i::dofs_per_node]
            
            if discard_time:                
                signal[label] = np.delete(signal[label], np.arange(0,discard_time), 1)
  
    return signal

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


def cov_xpsd (signal1, signal2):
    '''
    covariance is the integral over thr cross power spectral density 
    the rms is the root of the integral of the cross power spectral density if the signals are the same
    integral bounds are 0 to 1st natural frequency = 0.20 Hz here
    '''


dynamic_load_file = "input\\force\\generic_building\\dynamic_force_4_nodes.npy"
load_signals_raw = np.load(dynamic_load_file)

load_signals = parse_load_signal(load_signals_raw, 6, discard_time = 1000)

x = np.arange(0,10*np.pi,0.1)

# with this devinition a correlation coefficient of -1.0 is expected
sin = 1+np.sin(x)
cos = 1+np.cos(x + np.pi/2)

fig, ax = plt.subplots()
ax.plot(x,sin, label = 'sin')
ax.plot(x,cos, label = 'cos')
plt.legend()
plt.grid()
#plt.show()

# round digits
rd = 4

print ('----- tests using sin and cos: correlation coefficent should be -1.0 -----\n')
# std and rms 
print ('std sin, cos', round(np.std(sin), rd),round(np.std(cos), rd))
print ('rms sin, cos', round(rms(sin), rd),round(rms(cos), rd))
print()

# # covariance and correlation coefficient 
# using numpy
cov_sc_np = np.cov(sin,cos)[0][1]
cor_sc_np = np.corrcoef(sin,cos)[0][1]
print('cov numpy sin cos', round(cov_sc_np, rd))
print('cor numpy sin cos', round(cor_sc_np, rd))

# using custom defintion with std
cov_sc_std = cov_custom_std(sin,cos)
cor_sc_std_std = cov_sc_std /(np.std(sin)*np.std(cos))
print('cov std sin cos    ', round(cov_sc_std, rd))
print('cor std std sin cos', round(cor_sc_std_std, rd))

# using custom definition rms 
cov_sc_rms = cov_custom_rms(sin,cos)
cor_sc_rms_rms = cov_sc_rms /(rms(sin)*rms(cos))
print('cov rms sin cos    ', round(cov_sc_rms, rd))
print('cor rms rms sin cos', round(cor_sc_rms_rms, rd))

# correlation mixed with numpy and rms 
cor_sc_np_rms = cov_sc_np /(rms(sin) * rms(cos))
print ('cor numpy rms sin cos', round(cor_sc_np_rms,rd))

print()
# # everything with 2 load signals
print ('------- tests using two load signals Fy (along wind) at different nodes (tip and one below) ----------\n')
s1, s2 = load_signals['y'][2], load_signals['y'][3]

print ('std signal1, signal2', round(np.std(s1),rd), round(np.std(s2),rd))
print ('rms signal1, signal2', round(rms(s1),rd), round(rms(s2),rd))
print()

cov_np = np.cov(s1,s2)[0][1]
cor_np = np.corrcoef(s1,s2)[0][1]
print ('cov numpy ', round(cov_np,rd))
print ('cor numpy ', round(cor_np,rd))

cov_std = cov_custom_std(s1,s2)
cor_std_std = cov_std/ (np.std(s1)*np.std(s2))
print ('cov std', round(cov_std,rd))
print ('cor std', round(cor_std_std,rd))
print ('custom - numpy', cov_std- cov_np, 'this is', round((cov_std- cov_np)/cov_np * 100, 5), '%', 'of the numpy value')


cov_rms = cov_custom_rms(s1,s2)
cor_rms_rms = cov_rms/ (rms(s1)*rms(s2))
print ('cov rms   ', round(cov_rms,rd))
print ('cor rms rms', round(cor_rms_rms,rd))

cor_np_rms = cov_np/ (rms(s1)*rms(s2))
print ('cor numpy rms', round(cor_np_rms,rd))


