import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch, periodogram, butter, lfilter, filtfilt, csd
from scipy.signal.windows import hann, boxcar
from scipy.integrate import simps
from math import ceil, log2
#from source.ESWL.eswl_auxiliaries import parse_load_signal
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

###################################################
# user input
z_ref = 76
lux_ref = 210
z0_ref = 0.25
freq_cutoff_ref = 2
ramp_up_time_steps = int(1.5 * 3000)

flow_data = "PSD\\flow_data.dat"

dynamic_load_file = "input\\force\\generic_building\\dynamic_force_4_nodes.npy"
time_file = 'input\\array_time.npy'
load_signals_raw = np.load(dynamic_load_file)


###################################################
# # own function definition
# def nextpow2(x):
#     return int(ceil(log2(abs(x))))

def rms(x,digits=3):
    return np.around(np.sqrt(np.mean(np.square(x))),digits)

def rms_spectra(y,x,digits=3):
    '''
    2 different forms of integration
    '''
    return [np.around(np.sqrt(np.trapz(abs(y),x)),digits), np.around(np.sqrt(simps(abs(y),x)),digits)] 

def integrate_spectra(y,x,digits=3):
    '''
    2 different forms of integration - without the sqrt 
    taking the absoulute value of the imaginary numbers
    '''
    return [np.around(np.trapz(np.real(y),x),digits), np.around(simps(abs(y),x),digits)] 

# lowpass filter https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def sel_idx(x, sel_x):
    if sel_x > x.max():
        print('the selected frequency', sel_x, 'is not contained in the spectrum')
        return 0
    else:
        idx = np.where(x >= sel_x)[0][0]
        return idx

def plot_result_spectra(vx_spectra_res, freq_cutoff_ref, z_ref, vx_mean, selected_signal, show_plot = False):
    # plot res
    plt.figure(1)
    plt.suptitle('PSD for ' + selected_signal)

    plt.loglog(vx_spectra_res['n_w'], vx_spectra_res['psd_w'], 'r--', label='psd_w')
    plt.xlabel('f')
    plt.ylabel('S')
    plt.legend()
    plt.grid()

    plt.axvline(
        freq_cutoff_ref,
        linestyle='--',
        label='cutoff')

    #plt.savefig("psd_vu_comp.pdf")

    plt.figure(2)
    plt.suptitle('PSD for ' + selected_signal + ' - normalized')

    plt.loglog(vx_spectra_res['n_w_r'], vx_spectra_res['psd_w_r'], 'g-.', label='psd_w_r')
    plt.xlabel('f_r')
    plt.ylabel('S_r')
    plt.legend()
    plt.grid()
    plt.xticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])

    plt.axvline(
        freq_cutoff_ref * z_ref / vx_mean,
        linestyle='--',
        label='cutoff')

    #plt.savefig("psd_vu_comp_scaled.pdf")

    if show_plot:
        plt.show()

'''
Discussion related to PSD:
https://groups.google.com/forum/#!topic/comp.soft-sys.matlab/NYu1u-923vc

An extension Matlab implementation collection
https://github.com/rohillarohan/Power-Spectral-Density-Estimation/blob/master/spectral_estimation.m
'''

def get_velocity_spectra_methods(time_series, velocity_series, z, z0, lux, subtract_mean = True):
    
    '''
    Notes comments relevant for initial testing and setup

    # initial try with recommended hanning window
    # win_han = hann(len(time_series),False)
    # switching to boxcar as this seem to deliver qualitatively 
    # the closest results to version 1 in high frequency range
    # win_box = boxcar(len(time_series),False)

    # NOTE: without windwos (boxcar should mean this) the periodigram looks the
    # same as with window, the welch version seems to be wrong

    # try with prescribed signal length using nfft
    # n_w, psd_w = welch(ux_fluct,fs,nfft=nfft,detrend=False,return_onesided=True)
    # n_p, psd_p = periodogram(ux_fluct,fs,nfft=nfft,detrend=False,return_onesided=True)
                
    # try with detrending false which might be better suited for low frequency signals 
    # https://stackoverflow.com/questions/46775223/equivalence-scipy-signal-welch-to-matlab-pwelch
    # detrend can be left constant as mean is already subtracted
    # n_w, psd_w = welch(ux_fluct,fs,detrend=False,return_onesided=True)
    # n_p, psd_p = periodogram(ux_fluct,fs,detrend=False,return_onesided=True)

    # n_w, psd_w = welch(ux_fluct,fs,return_onesided=True)
    # n_p, psd_p = periodogram(ux_fluct,fs,return_onesided=True)

    # NOTE: chosen setup
    # very good match of periodigramm and welch with hanning
    # n_w, psd_w = welch(ux_fluct,fs,window=win_box,nfft=len(time_series),return_onesided=True)
    # n_p, psd_p = periodogram(ux_fluct,fs,window=win_box,nfft=len(time_series),return_onesided=True)
    
    # also very good match with fft -> from double to single -> to psd
    # x_fft = np.fft.fft(ux_fluct,n=nfft)[:int(n/2)+1]
    # f = f[:int(n/2)+1]
    # lx = (time_series[-1] - time_series[0]) # * ux_mean
    # t_total = time_series[-1] - time_series[0]
    # factor 2*t_total/n**2 -> see what is missing from DFT definition for numpy.fft
    # implementation details https://docs.scipy.org/doc/numpy/reference/routines.fft.html
    # and what is presented in the M. Andre diss A.3 (sqrt(2*t_total)/n)**2 == 2*t_total/n**2
    # (equations 2.1, 2.8, 2.11, 2.15, A.3, A.4)
    # M. Andre diss https://mediatum.ub.tum.de/doc/1426694/1426694.pdf
    # px = abs(x_fft)**2 * 2*t_total/n**2
    # NOTE: also Andreas W cannot say why factor 2 is used to multiply t_total
    # TODO: just use without factor 2, as it seems to be an error
    # NOTE: for estimation of PSD one should use some subsampling and averaging of FFT (for long signals)
    
    Seems to be slightly off, check necessity of using newtpow
    #n_f, psd_f = psd_fft(ux_fluct,fs,nfft=2**nextpow2(len(ux_fluct)),t_total=time_series[-1]-time_series[0])
    

    On the choice of proper scaling from FFT to PSD
	http://zone.ni.com/reference/en-XX/help/370709E-01/smtcviconcepts/guid-1f3ad50a-188d-4377-bb4d-1bf50a772126/
	https://dsp.stackexchange.com/questions/32187/what-should-be-the-correct-scaling-for-psd-calculation-using-tt-fft/328#328
	https://dsp.stackexchange.com/questions/11185/why-the-rms-of-a-psd-curve-is-the-root-of-the-area-below

    Parseval's theorem
	http://blog.prosig.com/2015/01/06/rms-from-time-history-and-fft-spectrum/
    '''

    def psd_welch(signal_fluct, fs, nfft, window_type='box', use_onesided=True):
        if window_type == 'hann':
            win = hann(len(time_series),False)
        elif window_type == 'box':
            win = boxcar(len(time_series),False)
        else:
            pass
        return welch(signal_fluct,fs,window=win,nfft=nfft,return_onesided=use_onesided)

    ux_mean = np.mean(velocity_series)
    n = len(velocity_series)
    if subtract_mean:
        ux_fluct = (velocity_series - ux_mean)
    else:
        ux_fluct = velocity_series

    utau = 0.41 * ux_mean / np.log(z / z0)

    fs = 1/(time_series[1]-time_series[0])
    nfft = len(time_series)

    n_w, psd_w = psd_welch(ux_fluct,fs,nfft=nfft)    

    freq_fctr = z/(ux_mean)


    results = {}

    # unscaled results
    results['n_w'] = n_w
    results['psd_w'] = psd_w
    
    # dimensionless (scaled) results
    # only necessary for plotting?!
    results['n_w_r'] = n_w * freq_fctr
    results['psd_w_r'] = psd_w * (n_w/utau**2)

    return results

def cross_spectral_density(x,y, time_series, nperseg, window_type,
                    subtract_mean, use_fs = True, use_nfft = True,):
    '''
    retrun a freq arry and the accroding spectral array
    '''
    # if subtract_mean:
    #     x = x - np.mean(x)
    #     y = y - np.mean(y)
    # -------- FS NFFT 
    if use_fs:
        fs = 1/(time_series[1] - time_series[0])
    else:
        fs = 1.0# default value
    if use_nfft:
        nfft = len(time_series)
    else:
        nfft = None # default 
    # ------- WINDOW
    if window_type == 'hann':
        # default is hann but without argument?
        win = hann(len(time_series),False)
        f, csd_raw = csd(x,y, fs=fs,nfft=nfft, window = win)
    elif window_type == 'box':
        win = boxcar(len(time_series),False)
        f, csd_raw = csd(x,y, fs=fs,nfft=nfft, window = win)
    elif window_type == None:
        f, csd_raw = csd(x,y, fs=fs,nfft=nfft)
    # ------- NPERSEG
    if nperseg:
        f, csd_raw = csd(x,y, fs=fs,nfft=nfft, nperseg=nperseg)
      

    # EVALUATE CSD AT A FREQUENCY
    f_j = 0.2
    f_round = np.round(f,2)
    
    f_id = np.where(f_round ==np.round(f_j,2))[0]
    # NOTE: sofar: if more then one rounded f exists then the one that is the closest to f_j is taken
    # TODO:  maybe find the two closest freqs and then interoplate the csd (np.interpl1d, cubic,...)
    if len(f_id) > 1:
        possible_fs = f[f_id[0]:f_id[-1]+1] 
        difs = abs(possible_fs - f_j)
        use_id = np.argmin(difs)
        f_id_closest = f_id[use_id]
    else:
        f_id_closest = f_id[0]
    
    # find interpolation freq
    if f_j < f[f_id_closest]:
        xp = [f[f_id_closest - 1], f[f_id_closest]]
        yp = [csd_raw[f_id_closest - 1], csd_raw[f_id_closest]]
    elif f_j > f[f_id_closest]:
        xp = [f[f_id_closest], f[f_id_closest+1]]
        yp = [csd_raw[f_id_closest], csd_raw[f_id_closest+1]]

    csd_f_j = np.interp(f_j, xp, yp)

    real = ' only real'
    csd_f_j_close = csd_raw[f_id_closest]

    return f, csd_raw, csd_f_j

def cov_custom(signal1, signal2):
    '''
    numpy cov always subtracts the mean value 
    '''
    m1 = np.mean(signal1)
    m2 = np.mean(signal2)

    cov = 0.0
    # for i in range(len(signal1)):
    #     cov += (signal1[i] - m1)*(signal2[i] - m2)
    for i in range(len(signal1)):
        cov += (signal1[i] )*(signal2[i])
        
    return cov/len(signal1)

def plot_csd(result_list, signal_labels, show_plot, param_list, param_label):
    f_j = 0.2
    
    plt.figure(3)
    plt.suptitle('abs(CSD) of ' + signal_labels[0] + ' & ' + signal_labels[1] + ' - ' + param_label + ' as parameter')
    lstyles = ['-','--',':']
    max_csd = []
    for param_i, result in enumerate(result_list):
        
        f = result[0]
        csd_val = result[1]
        max_csd.append(max(csd_val))
        csd_fj_p = result[2]
        label_txt = str(param_list[param_i]) + ' csd_fj: ' + str(round(csd_fj_p,3))

        #plt.semilogy(csd_result[0], np.real(csd_result[1]), label = 'csd')#, label = 'csd at f_j: ' + str(csd_f_j))
        plt.loglog(f, np.abs(csd_val), linestyle = lstyles[param_i] ,label = label_txt)#, label = 'csd at f_j: ' + str(csd_f_j))
        #plt.loglog(f, np.real(csd))#, label = 'csd at f_j: ' + str(csd_f_j))
        #plt.vlines(f[f_id_closest], 0, max(csd), label = 'estimated f', linestyles='-.', color = 'g')
    
    plt.vlines(f_j, 0, max(max_csd) + 1e+14, label = 'natural f', linestyles='--', color = 'grey')

    plt.xlabel('frequency')
    plt.ylabel('CSD ')
    plt.legend(loc= 'lower left')
    plt.grid()
    if show_plot:
        plt.show()

###################################################
# data read and process
# velocities
series_t = np.loadtxt(flow_data, skiprows=ramp_up_time_steps, usecols=(0,))
series_vx = np.loadtxt(flow_data, skiprows=ramp_up_time_steps, usecols=(2,))

# sin cos 
x = np.arange(0,10*np.pi,0.1)

# with this devinition a correlation coefficient of -1.0 is expected
period = 1/np.pi
sin = 1+np.sin(period * x)
cos = 1+np.cos(period * x + np.pi/2)

# load signal 4 nodes
time_array = np.load(time_file)
load_signals = parse_load_signal(load_signals_raw, 6)
shape_dif = time_array.shape[0] - load_signals_raw.shape[1]
direction = 'y'
node1, node2 = 2,3

data = {'vx':(series_t,series_vx), 'sin':(x,sin),'cos':(x,cos),
         'force1':(time_array[shape_dif:], load_signals[direction][node1]),
         'force2':(time_array[shape_dif:], load_signals[direction][node2])}

# # ------------ SETTINGS FOR WHICH DATA IS USED AND SOME PARAMETERS

selected_signal_1 = 'force1'
selected_signal_2 = 'force2'

subtract_mean = False
partial_freqs = False
show_plots = True

# # CSD SETTINGS
nperseg_avail = [1024, 2048, 4096]
window_type_avail = ['hann','box', None]

use_fs = True
use_nfft = True
nperseg = nperseg_avail[1]
window_type = window_type_avail[0]

param_label = 'nps' #'win'
if param_label == 'nps':
    params_list = nperseg_avail
elif param_label == 'win':
    params_list = window_type_avail

if subtract_mean:   
    print('\n------- Subtracting mean value of the signal -------')
else:   
    print('\n------- NOT Subtracting mean value of the signal -------')

time_series = data[selected_signal_1][0]
f_sample = 1/(time_series[1] - time_series[0])
s1 = data[selected_signal_1][1]
s2 = data[selected_signal_2][1]
signal_mean = np.mean(s1)
# CSD result
result_list = []
for param in params_list:
    if param_label == 'nps':
        nps = param
        win = None
    elif param_label== 'win':
        nps = None
        win = param
    
    f_csd, csd_raw, csd_fj = cross_spectral_density(s1, s2, time_series,
                                             nps, win,
                                             subtract_mean = subtract_mean,
                                             use_fs = True, use_nfft = True,
                                             )

    result_list.append((f_csd, csd_raw, csd_fj))

plot_csd(result_list, [selected_signal_1, selected_signal_2], show_plots, params_list, param_label)

print('\nFor the whole frequency content')
#http://blog.prosig.com/2015/01/06/rms-from-time-history-and-fft-spectrum/
rms_s1 = rms(s1)
rms_s2 = rms(s2)
std_s1 = rms(s1 - np.mean(s1)) 
std_s2 = rms(s2 - np.mean(s2))
print('STD s1 time series: ', std_s1)
print('RMS s1 time series: ', rms_s1)
if subtract_mean:
    sig = std_s1
    sig_label = 'STD'
else:
    sig = rms_s1
    sig_label = 'RMS'

f_sample = 1/(series_t[1] - series_t[0])



# round digits
digits = 3

print()
# print ('\n----- tests using sin and cos: correlation coefficent should be -1.0 -----\n')
# # std and rms 
print ('mean', selected_signal_1, ',', selected_signal_2, round(np.mean(s1), digits), ',', round(np.mean(s2), digits))
print ('std', selected_signal_1, ',', selected_signal_2, round(np.std(s1), digits), ',', round(np.std(s2), digits))
print ('rms' , selected_signal_1, ',', selected_signal_2, round(rms(s1), digits),  ',', round(rms(s2), digits))
if selected_signal_1 == selected_signal_2:
    rms_csd = rms_spectra(csd_raw, f_csd)
    # only using trapz here
    # if the signals are the same, this rms should be the rms from above per definition --> depending on how csd does its things
    print ('rms csd:', round(abs(rms_csd[0]),digits))

# # ---------------- covariances -----------------------------------

print()
if subtract_mean:
    s1 = s1 - np.mean(s1)
    s2 = s2 - np.mean(s2)
cov_xy_np = np.cov(s1,s2)[0][1]
cov_xy_custom = cov_custom(s1,s2) 
# cov_xy_csd_sqrt = rms_spectra(csd_result[1], csd_result[0])
# cov_xy_csd_no_sqrt = integrate_spectra(csd_result[1], csd_result[0])

print('cov of', selected_signal_1, '&', selected_signal_2, 'using...')
print('...numpy cov: ', round(cov_xy_np,digits), ' sqrt:', round(np.sqrt(abs(cov_xy_np)),digits))
print('...custom cov:', round(cov_xy_custom,digits), ' sqrt:', round(np.sqrt(abs(cov_xy_custom)), digits))
#print('\nwith different integration methods (trapz, simps)')
# print('...csd cov sqrt:   ', round(abs(cov_xy_csd_sqrt[0]),digits))#, round(abs(cov_xy_csd_sqrt[1]),digits))
# print('...csd cov NO sqrt:', round(abs(cov_xy_csd_no_sqrt[0]),digits))#, round(abs(cov_xy_csd_no_sqrt[1]),digits))
# print('with no sqrt this is', round(cov_xy_np/cov_xy_csd_no_sqrt[0] * 100,digits), '%', 'of the numpy cov (using trapz)')
# #print('with no sqrt this is', round(cov_xy_np/cov_xy_csd_no_sqrt[1] * 100,digits), '%', 'of the numpy cov (using simps)')

print()
# # # covariance and correlation coefficient 

cor_np = np.corrcoef(s1,s2)[0][1]
cor_cov_np_std = cov_xy_np /(std_s1 * std_s2)
cor_cov_np_rms = cov_xy_np /(rms_s1 * rms_s2)
#cor_cov_csd_std = cov_xy_csd_no_sqrt[0] /(std_s1 * std_s2)
print('numpy.corrcoeff:', round(cor_np, digits))
print('np cov / std:   ', round(cor_cov_np_std, digits))
print('np cov / rms:   ', round(cor_cov_np_rms, digits))
#print('csd no sqrt / std', round(cor_cov_csd_std, digits))


if subtract_mean:   
    print('\n------- Subtracted mean value of the signal -------')
else:   
    print('\n------- NOT Subtracted mean value of the signal -------')
