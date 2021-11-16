import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch, periodogram, butter, lfilter, filtfilt, csd
from scipy.signal.windows import hann, boxcar
from scipy.integrate import simps
from math import ceil, log2
from os.path import join as os_join
from os.path import sep as os_sep
import plot_settings as plot_settings

#from source.ESWL.eswl_auxiliaries import parse_load_signal

dest_latex = os_join(*['C:',os_sep,'Users','Johannes','LRZ Sync+Share','MasterThesis','Abgabe','Text','images'])
dest = os_join(*['plots', 'ESWL_plots','XPSD'])

plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]

width = plot_settings.cm2inch(7)
height = plot_settings.cm2inch(5)

plot_params = plot_settings.get_params(width =width, height=height, use_tex = True, minor_ticks=False)

plt.rcParams.update({'axes.formatter.limits':(-3,3)}) 
#plt.rcParams.update({'figure.figsize': (width, height)})
plt.rcParams.update(plot_params)

plot_options = {'show_plots':True,'savefig':True,'savefig_latex':True}

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

flow_data = os_join(*["PSD","flow_data.dat"])

dynamic_load_file = os_join(*["input","force","generic_building","dynamic_force_4_nodes.npy"])
time_file = os_join(*['input','array_time.npy'])
load_signals_raw = np.load(dynamic_load_file)


###################################################
# # own function definition
# def nextpow2(x):
#     return int(ceil(log2(abs(x))))

def rms(x,digits=3):
    return np.around(np.sqrt(np.mean(np.square(x))),digits)

def integrate_csd(y,x,digits=3):
    '''
    2 different forms of integration
    '''
    # np.around(np.sqrt(np.trapz(np.abs(y),x)),digits)
    return np.sqrt(np.trapz(np.abs(y),x))

def integrate_psd(y,x,digits=3):
    '''
    2 different forms of integration - without the sqrt 
    taking the absoulute value of the imaginary numbers
    '''
    # np.around(np.trapz(np.abs(y),x),digits)#, np.around(simps(abs(y),x),digits)] 
    return np.sqrt(np.trapz(np.abs(y),x))

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
                    subtract_mean, use_fs = True, use_nfft = True):
    '''
    retrun a freq arry and the accroding spectral array
    '''
    # if subtract_mean:
    #     x = x - np.mean(x)
    #     y = y - np.mean(y)
    # -------- NFFT 
    
    if use_nfft:
        nfft = len(time_series)
    else:
        nfft = None # default 
        
    # ------- WINDOW - if a window type is specified then no nperseg can be used
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

def csd_short(x,y,fs, nfft, window):
    
    csd_f, csd_raw = csd(x,y, fs=fs,nfft=nfft, window = window)

    return csd_f, csd_raw

def psd_welch(signal, fs, nfft, win):

    psd_f, psd_raw = welch(signal,fs,window=win,nfft=nfft,return_onesided=True) 

    return  psd_f, psd_raw  

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

def plot_csd(result_list, signal_labels, show_plot, param_list, param_label, norm_with_box_res = True,
            options = {'show_plots':True,'savefig':False,'savefig_latex':False}):
    f_j = 0.2
    
    plt.figure(3)
    # if not options['savefig_latex']:
    #     plt.suptitle('abs(CSD) of ' + signal_labels[0] + ' & ' + signal_labels[1] + ' - ' + param_label + ' as parameter')
    lstyles = ['-','--',':','-.']
    max_csd = []
    for param_i, result in enumerate(result_list):
        
        f = result[0]
        csd_val = result[1]
        max_csd.append(max(csd_val))
        csd_fj_p = result[2]
        if norm_with_box_res:
            if param_list[param_i] == 'box':
                norm = 1/ csd_fj_p
        else:
            norm = 1

        label_txt = 'window type ' + r'${}$'.format(str(param_list[param_i])) + r' $CSD(f_{j}):$ ' + str(round(csd_fj_p*norm,1))

        #plt.semilogy(csd_result[0], np.real(csd_result[1]), label = 'csd')#, label = 'csd at f_j: ' + str(csd_f_j))
        plt.loglog(f, np.abs(csd_val), linestyle = lstyles[param_i] ,label = label_txt)#, label = 'csd at f_j: ' + str(csd_f_j))
        #plt.loglog(f, np.real(csd))#, label = 'csd at f_j: ' + str(csd_f_j))
        #plt.vlines(f[f_id_closest], 0, max(csd), label = 'estimated f', linestyles='-.', color = 'g')0, max(max_csd) + 1e+14, 
    
    plt.axvline(f_j, ymin=10e-5, label = r'$f_{j}$', linestyle='--', color = 'grey')
    plt.ylim(bottom = 10e-5)
    plt.xlabel(r'$f [Hz]$')
    plt.ylabel(r'$|CSD|$ ')
    plt.legend(loc= 'lower left')
    plt.grid()

    save_title = 'csd_window_compare'


    if options['savefig']:
        plt.savefig(dest + os_sep + save_title)
        plt.savefig(dest + os_sep + save_title + '.svg')
        print ('\nsaved:',dest + os_sep + save_title)
        #plt.close()
    
    if options['savefig_latex']:
        plt.savefig(dest_latex + os_sep + save_title)
        print ('\nsaved:',dest_latex + os_sep + save_title)
    if options['show_plots']:
        plt.show()

def duration_check(signals_list, time_series):
    np_results, psd_results, csd_results = [],[],[]
    # NUMPY STD
    np_start = timeit.default_timer()
    for signal in signals_list:
        std_np = np.std(signal)
        np_results.append(std_np)
    np_end = timeit.default_timer()
    np_time = np_end - np_start
    # CSD
    fs = 1/(time_series[1] - time_series[0])
    nfft = len(time_series)
    win = boxcar(len(time_series),False)

    csd_start = timeit.default_timer()
    for signal in signals_list:
        f_csd, csd_raw = csd_short(signal, signal,fs, nfft, win)
        std_csd = integrate_csd(csd_raw, f_csd)
        csd_results.append(std_csd)
    csd_end = timeit.default_timer()
    csd_time = csd_end - csd_start
    # PSD
    psd_start = timeit.default_timer()
    for signal in signals_list:
        f_psd, psd_raw = psd_welch(signal, fs, nfft, win)
        std_psd = integrate_psd(psd_raw, f_psd)
        psd_results.append(std_psd)
    psd_end = timeit.default_timer()
    psd_time = psd_end - psd_start

    digits = 5
    plt.plot(np.arange(len(signals_list)), np.asarray(np_results), label = r'np.std execution time:' + str(round(np_time, digits)) + ' s', linestyle = '-')
    plt.plot(np.arange(len(signals_list)), np.asarray(psd_results), label = r'$\int_0^\infty$' + r'psd execution time:' + str(round(psd_time, digits))+ ' s', linestyle = '-.')
    plt.plot(np.arange(len(signals_list)), np.asarray(csd_results), label = r'$\int_0^\infty$' + r'csd execution time:' + str(round(csd_time, digits))+ ' s', linestyle = ':')
    plt.xlabel(r'$signal_{i}$')
    plt.ylabel(r'$std(signal_{i})$')
    plt.title('execution time of calculation of the standard deviation of ' + str(len(signals_list)) + ' signals, each with length ' + str(len(signals_list[0])))
    plt.legend()
    plt.grid()
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

#duration_check(signals_list, time_array[:-shape_dif])

data = {'vx':(series_t,series_vx), 'sin':(x,sin),'cos':(x,cos),
         'force1':(time_array[:-shape_dif], load_signals[direction][node1]),
         'force2':(time_array[:-shape_dif], load_signals[direction][node2])}

# # ------------ SETTINGS FOR WHICH DATA IS USED AND SOME PARAMETERS

selected_signal_1 = 'force1'
selected_signal_2 = 'force2'

subtract_mean = False
partial_freqs = False
show_plots = True

check_rms = True

# round digits
digits = 3

# # CSD SETTINGS
nperseg_avail = [2048, 4096, 8192, 16258]
window_type_avail = ['box','hann', None]

fs = 1/(time_array[1] - time_array[0])
print ('fs from time array:', fs)
Ts = time_array[1] - time_array[0]
use_nfft = True
nperseg = nperseg_avail[1]
window_type = window_type_avail[0]

param_label = 'win' #'win', 'fs','nps
if param_label == 'nps':
    params_list = nperseg_avail
elif param_label == 'win':
    params_list = window_type_avail
elif param_label == 'fs':
    params_list = [0.01,0.1,1.0, Ts]

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
    elif param_label == 'fs':
        fs = 1/param
        nps, win = None,None
    
    #win = 'box'
    f_csd, csd_raw, csd_fj = cross_spectral_density(s1, s2, time_series,
                                             nps, win,
                                             subtract_mean,
                                             use_fs = fs, use_nfft = True,
                                             )

    if check_rms:
        for s, signal in enumerate([s1,s2]):
            f_csd, csd_raw, csd_fj = cross_spectral_density(signal, signal, time_series,
                                             nps, win,
                                             subtract_mean,
                                             use_fs = fs, use_nfft = True,
                                             )
            std_np = np.std(signal)
            std_csd = integrate_csd(csd_raw, f_csd)
            print(param_label , param)
            print ('    std of signal', s, 'numpy:', round(std_np,digits))
            print ('    std of signal', s, 'CSD  :', round(std_csd,digits))


    result_list.append((f_csd, csd_raw, csd_fj))

plot_csd(result_list, [selected_signal_1, selected_signal_2], show_plots, params_list, param_label, options= plot_options)

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



print()
# print ('\n----- tests using sin and cos: correlation coefficent should be -1.0 -----\n')
# # std and rms 
print ('mean', selected_signal_1, ',', selected_signal_2, round(np.mean(s1), digits), ',', round(np.mean(s2), digits))
print ('std', selected_signal_1, ',', selected_signal_2, round(np.std(s1), digits), ',', round(np.std(s2), digits))
print ('rms' , selected_signal_1, ',', selected_signal_2, round(rms(s1), digits),  ',', round(rms(s2), digits))
if selected_signal_1 == selected_signal_2:
    rms_csd = integrate_csd(csd_raw, f_csd)
    # only using trapz here
    # if the signals are the same, this rms should be the rms from above per definition --> depending on how csd does its things
    print ('rms csd:', round(abs(rms_csd),digits))

# # ---------------- covariances -----------------------------------

print()
if subtract_mean:
    s1 = s1 - np.mean(s1)
    s2 = s2 - np.mean(s2)
cov_xy_np = np.cov(s1,s2)[0][1]
cov_xy_custom = cov_custom(s1,s2) 
# cov_xy_csd_sqrt = integrate_csd(csd_result[1], csd_result[0])

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
