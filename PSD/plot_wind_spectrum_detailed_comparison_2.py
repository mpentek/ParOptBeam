import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch, periodogram, butter, lfilter, filtfilt
from scipy.signal.windows import hann, boxcar
from scipy.integrate import simps
from math import ceil, log2


###################################################
# user input
z_ref = 76
lux_ref = 210
z0_ref = 0.25
freq_cutoff_ref = 2
ramp_up_time_steps = int(1.5 * 3000)

flow_data = "PSD/flow_data.dat" 


###################################################
# # own function definition
# def nextpow2(x):
#     return int(ceil(log2(abs(x))))

def rms(x,digits=3):
    return np.around(np.sqrt(np.mean(np.square(x))),digits)

def rms_spectra(y,x,digits=3):
    return [np.around(np.sqrt(np.trapz(y,x)),digits), np.around(np.sqrt(simps(y,x)),digits)] 

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

# # bandpass filter https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
# def butter_bandpass(lowcut, highcut, fs, order=5):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(order, [low, high], btype='band')
#     return b, a


# def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
#     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#     y = lfilter(b, a, data)
#     return y

def sel_idx(x, sel_x):
    return np.where(x >= sel_x)[0][0]

'''
Discussion related to PSD:
https://groups.google.com/forum/#!topic/comp.soft-sys.matlab/NYu1u-923vc

An extension Matlab implementation collection
https://github.com/rohillarohan/Power-Spectral-Density-Estimation/blob/master/spectral_estimation.m
'''

def get_velocity_spectra_methods(time_series, velocity_series, z, z0, lux):
    
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
    
    def psd_periodogram(signal_fluct, fs, nfft, window_type='box', use_onesided=True):
        if window_type == 'hann':
            win = hann(len(time_series),False)
        elif window_type == 'box':
            win = boxcar(len(time_series),False)
        else:
            pass
        return periodogram(signal_fluct,fs,window=win,nfft=nfft,return_onesided=use_onesided)

    def psd_fft(signal_fluct, fs, nfft, t_total):
        f = np.linspace(0,fs,nfft,endpoint=True)
        # FFT and take onesided results
        x_fft = np.fft.fft(signal_fluct,n=nfft)[:int(n/2)+1]
        # onsided frequency content
        f = f[:int(n/2)+1]
        psd = abs(x_fft)**2 * t_total/n**2
        # NOTE: factor 2 seems to be needed
        # TODO: check why       
        return f, psd*2

    def psd_red_kaimal(n_r):
        # generating the one sided dimensionless spectrum from dimensionless frequency 
        psd = 105./2. * n_r / (1. + 33. * n_r)**(5. / 3.)

        # NOTE: seems that the energy content is only correct when *2
        # TODO: check why
        return psd*2

    ux_mean = np.mean(velocity_series)
    n = len(velocity_series)
    ux_fluct = (velocity_series - ux_mean)
    utau = 0.41 * ux_mean / np.log(z / z0)

    fs = 1/(time_series[1]-time_series[0])
    nfft = len(time_series)

    n_w, psd_w = psd_welch(ux_fluct,fs,nfft=nfft)    
    n_p, psd_p = psd_periodogram(ux_fluct,fs,nfft=nfft)
    n_f, psd_f = psd_fft(ux_fluct,fs,nfft=nfft,t_total=time_series[-1]-time_series[0])

    freq_fctr = z/(ux_mean)

    n_k_r = n_f * freq_fctr
    psd_k_r = psd_red_kaimal(n_k_r)

    results = {}

    # unscaled results
    results['n_f'] = n_f
    results['n_w'] = n_w
    results['n_p'] = n_p
    results['n_k'] = n_k_r / freq_fctr
    
    results['psd_f'] = psd_f
    results['psd_w'] = psd_w
    results['psd_p'] = psd_p
    
    # avoid division by 0
    # TODO: do not check with zero but tolerance
    if n_k_r[0] == 0:
        results['psd_k'] = np.insert(psd_k_r[1:] / (n_k_r[1:] / freq_fctr/utau**2),0,0)
    else:
        results['psd_k'] = psd_k_r / (n_k_r / freq_fctr/utau**2)

    # dimensionless (scaled) results
    
    results['n_f_r'] = n_f * freq_fctr
    results['n_w_r'] = n_w * freq_fctr
    results['n_p_r'] = n_p * freq_fctr
    results['n_k_r'] = n_k_r

    results['psd_f_r'] = psd_f * (n_f/utau**2)
    results['psd_w_r'] = psd_w * (n_w/utau**2)
    results['psd_p_r'] = psd_p * (n_p/utau**2)
    results['psd_k_r'] = psd_k_r

    return results


###################################################
# data read and process
series_t = np.loadtxt(flow_data, skiprows=ramp_up_time_steps, usecols=(0,))
series_vx = np.loadtxt(flow_data, skiprows=ramp_up_time_steps, usecols=(2,))

vx_mean = np.mean(series_vx)
vx_spectra_res = get_velocity_spectra_methods(series_t, series_vx, z=z_ref, z0=z0_ref, lux=lux_ref)

'''
RMS(time_signal)=RMS(complex_components_of_FFT)=Integral(psd)
http://blog.prosig.com/2015/01/06/rms-from-time-history-and-fft-spectrum/
https://dsp.stackexchange.com/questions/11185/why-the-rms-of-a-psd-curve-is-the-root-of-the-area-below
'''

print('For the whole frequency content')
#http://blog.prosig.com/2015/01/06/rms-from-time-history-and-fft-spectrum/
print('RMS time series: ', str(rms((series_vx - vx_mean))))
# https://dsp.stackexchange.com/questions/11185/why-the-rms-of-a-psd-curve-is-the-root-of-the-area-below
print('RMS psd (as integration with trapz and simps): ')
for case in ['w', 'p', 'f','k']:
    print('   psd_'+ case +': ' + ", ".join([str(val) for val in rms_spectra(vx_spectra_res['psd_'+ case +''],vx_spectra_res['n_'+ case +''])]))

# Consider content in ranges:
fs = 1/(series_t[1] - series_t[0])

for sel_freq in [0.025,0.25,2.5,5,10,25]:
    print('\nFor frequency content up to ' + str(sel_freq)+ ' Hz: ')
    print('RMS ', str(rms(butter_lowpass_filter(series_vx - vx_mean, sel_freq, fs))))
    print('RMS psd content (as integration with trapz and simps): ')
    for case in ['w', 'p', 'f','k']:
        idx = sel_idx(vx_spectra_res['n_w'],sel_freq)
        print('   psd_'+ case +': ' + ", ".join([str(val) for val in rms_spectra(vx_spectra_res['psd_'+ case +''][:idx],vx_spectra_res['n_'+ case +''][:idx])]))

# plot res
plt.figure(1)
plt.suptitle('Wind PSD for Vx')

plt.loglog(vx_spectra_res['n_f'], vx_spectra_res['psd_f'], 'b:', label='psd_f')
plt.loglog(vx_spectra_res['n_p'], vx_spectra_res['psd_p'], 'g-.', label='psd_p')
plt.loglog(vx_spectra_res['n_w'], vx_spectra_res['psd_w'], 'r--', label='psd_w')
plt.loglog(vx_spectra_res['n_k'], vx_spectra_res['psd_k'], 'k-', label='psd_k')
plt.xlabel('f')
plt.ylabel('S')
plt.legend()
plt.grid()

plt.axvline(
    freq_cutoff_ref,
    linestyle='--',
    label='cutoff')

plt.savefig("psd_vu_comp.pdf")

plt.figure(2)
plt.suptitle('Wind PSD for Vx - normalized')

plt.loglog(vx_spectra_res['n_f_r'], vx_spectra_res['psd_f_r'], 'b:', label='psd_f_r')
plt.loglog(vx_spectra_res['n_p_r'], vx_spectra_res['psd_p_r'], 'g-.', label='psd_p_r')
plt.loglog(vx_spectra_res['n_w_r'], vx_spectra_res['psd_w_r'], 'r--', label='psd_w_r')
plt.loglog(vx_spectra_res['n_k_r'], vx_spectra_res['psd_k_r'], 'k-', label='psd_k_r')
plt.xlabel('f_r')
plt.ylabel('S_r')
plt.legend()
plt.grid()
plt.xticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])

plt.axvline(
    freq_cutoff_ref * z_ref / vx_mean,
    linestyle='--',
    label='cutoff')

plt.savefig("psd_vu_comp_scaled.pdf")


plt.show()
