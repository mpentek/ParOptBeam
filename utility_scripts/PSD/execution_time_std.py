import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch, csd
from scipy.signal.windows import boxcar
import timeit
from os.path import join as os_join
from os.path import sep as os_sep
import plot_settings 

###################################################


dynamic_load_file = os_join(*["input","force","generic_building","dynamic_force_4_nodes.npy"])
dynamic_load_file_60 = os_join(*["input","force","generic_building","dynamic_force_61_nodes.npy"])
time_file = os_join(*["input","array_time.npy"])
load_signals_raw_4 = np.load(dynamic_load_file)
load_signals_raw_60 = np.load(dynamic_load_file_60)

# # things to use
signals_list =load_signals_raw_60[:-5,:]

time_array = np.load(time_file)
shape_dif = time_array.shape[0] - load_signals_raw_4.shape[1]

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

###################################################

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

def csd_short(x,y,fs, nfft, window):
    
    csd_f, csd_raw = csd(x,y, fs=fs,nfft=nfft, window = window)

    return csd_f, csd_raw

def psd_welch(signal, fs, nfft, win):

    psd_f, psd_raw = welch(signal,fs,window=win,nfft=nfft,return_onesided=True) 

    return  psd_f, psd_raw  

def duration_check(signals_list, time_series, n_executions = 10, n_exe_check = False,
            options = {'show_plots':True,'savefig':False,'savefig_latex':False}):
    
    times = {'np.std':[], 'psd':[], 'csd':[]}
    for execution in range (n_executions):
        np_results, psd_results, csd_results = [],[],[]

        fs = 1/(time_series[1] - time_series[0])
        nfft = len(time_series)
        win = boxcar(len(time_series),False)

        # NUMPY STD
        np_start = timeit.default_timer()
        for signal in signals_list:
            std_np = np.std(signal)
            np_results.append(std_np)
        np_end = timeit.default_timer()
        np_time = np_end - np_start

        times['np.std'].append(np_time)
        # CSD

        csd_start = timeit.default_timer()
        for signal in signals_list:
            f_csd, csd_raw = csd_short(signal, signal,fs, nfft, win)
            std_csd = integrate_csd(csd_raw, f_csd)
            csd_results.append(std_csd)
        csd_end = timeit.default_timer()
        csd_time = csd_end - csd_start

        times['csd'].append(csd_time)

        # PSD
        psd_start = timeit.default_timer()
        for signal in signals_list:
            f_psd, psd_raw = psd_welch(signal, fs, nfft, win)
            std_psd = integrate_psd(psd_raw, f_psd)
            psd_results.append(std_psd)
        psd_end = timeit.default_timer()
        psd_time = psd_end - psd_start
        times['psd'].append(psd_time)

        if execution == n_executions-1:
            digits = 5
            plt.figure(1)
            plt.plot(np.arange(len(signals_list)), np.asarray(np_results),
                     label = r'$std_{time}$' + ' runtime: ' + r'${} s$'.format(str(round(np_time, digits))), linestyle = '-')
            plt.plot(np.arange(len(signals_list)), np.asarray(psd_results), 
                     label = r'$\int_0^\infty$' + r'$PSD(X_{i})$' + ' runtime: ' + r'${} s$'.format(str(round(psd_time, digits))),  linestyle = '-.')
            plt.plot(np.arange(len(signals_list)), np.asarray(csd_results), 
                     label = r'$\int_0^\infty$' + r'$CSD(X_{i}, X_{i}):$'+ ' runtime: ' + r'${} s$'.format(str(round(csd_time, digits))),  linestyle = ':')
            plt.xlabel(r'$X_{i}$')
            plt.ylabel(r'$std(X_{i})$')
            #plt.title('execution time of calculation of the standard deviation of ' + str(len(signals_list)) + ' signals, each with length ' + str(len(signals_list[0])))
            plt.legend()#fontsize= 'xx-large')
            plt.grid()
            plt.xlim(left=0)
            save_title = 'runtime_compare'
            if options['savefig']:
                plt.savefig(dest + os_sep + save_title)
                #plt.savefig(dest + os_sep + save_title + '.svg')
                print ('\nsaved:',dest + os_sep + save_title)
                #plt.close()
            
            if options['savefig_latex']:
                plt.savefig(dest_latex + os_sep + save_title)
                print ('\nsaved:',dest_latex + os_sep + save_title)
            if options['show_plots']:
                plt.show()
            plt.show()
    
    if n_exe_check:
        x = np.arange(n_executions)
        plt.figure(2)
        for method in times:
            plt.plot(x, times[method], label = method)

        plt.xlabel('run_i')
        plt.ylabel('time (run_i) [s]')
        plt.title('execution times of different runs')
        plt.legend()
        plt.show()

    
###################################################


duration_check(signals_list, time_array[:-shape_dif], n_executions=1, options=plot_options)

