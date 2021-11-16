#===============================================================================
'''
sample various statistical functions. some are taken form Wind Course
'''
#===============================================================================
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt 
from matplotlib.offsetbox import AnchoredText
import numpy as np
import scipy.stats as sps
from scipy.stats import norm
from scipy.stats.distributions import gumbel_r as gumbel
from scipy.stats import gaussian_kde
import timeit


def rms(signal):
    return np.sqrt(np.mean(np.square(signal)))

def evaluation_limits(given_series, min_fac = 0.6 , max_fac = 1.4):
    min_series = min(given_series)
    if min_series < 0:
        x_min = 1.2 * min_series
    else:
        x_min = min_fac * min_series

    x_max = max_fac* max(given_series)

    return x_min, x_max

def round05(x, prec=2, base=.5):
    '''
    return either the number as x.0 or x.5
    '''
    return round(base * round(float(x)/base),prec)

#===============================================================================

# VARIOUS PDFS 

def get_pdf_kde(given_series, pdf_x = None):
    '''
    The function get_pdf_kde evaluates the probability distribution function (pdf)
    of the samples by using a non-parametric estimation technique called Kernal Desnity 
    Estimation (KDE). More details can be found at 
    https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.gaussian_kde.html.
    '''

    series_max = np.max(given_series)
    series_min = np.min(given_series)
    
    kde = gaussian_kde(given_series)
    if not pdf_x.any():
        xmin, xmax = evaluation_limits(given_series)
        pdf_x = np.linspace(xmin, xmax, 1000)

    pdf_y = kde(pdf_x)
    return pdf_x, pdf_y

def get_pdf_normal(given_series):
    '''
    The function get_pdf_normal estimates the normal pdf of the signal from the mean 
    and standard deviation of the samples. Recall the fact that a Normal distribution 
    can be entirely defined by two parameters, namely the mean and standard deviation. 
    More details about the function mlab.normpdf can be found at 
    https://matplotlib.org/api/mlab_api.html. 
    '''    

    series_max = np.max(given_series)
    series_std = np.std(given_series)
    series_m = np.mean(given_series)
    series_min = np.min(given_series)
    series_step = (series_max - series_min)/ 1000
    # series_pdf = mlab.normpdf(np.arange(series_min, 
    #                                         series_max + series_step, 
    #                                         series_step), 
    #                               series_m, 
    #                               series_std)
    #series_pdf = norm.pdf(given_series)
    
    pdf_x = np.arange(series_min, series_max + series_step, series_step)
    series_pdf = (np.exp(-(pdf_x**2)/2))/np.sqrt(2*np.pi)
    pdf_y = series_pdf
    return pdf_x, pdf_y

def plot_pdf_fitted_normal (given_series, signal_label):
    x_norm, y_norm = get_pdf_normal_scipy(given_series)
    x_gamma, y_gamma = get_pdf_gamma_scipy(given_series)
    x_gev, y_gev = get_pdf_gev_scipy(given_series)
    skew = sps.skew(given_series)
    mean = np.mean(given_series)
    std = np.std(given_series)
    stats = get_title(given_series)

    fig_norm = plt.figure('normal distribution ' + signal_label)
    ax_norm = fig_norm.add_subplot(111)
    ax_norm.set_title('PDF of normal distribution of ' + signal_label + '\n' + stats)
    ax_norm.set_xlabel(signal_label + ' [MNm]')

    ax_norm.plot(x_norm, y_norm, label = 'normal')
    ax_norm.plot(x_gamma, y_gamma, label ='gamma')
    ax_norm.plot(x_gev, y_gev , label ='gev')
    ax_norm.vlines(mean + 3*std, 0, max(y_norm), linestyle = '--', color = 'red')
    ax_norm.vlines(mean - 3*std, 0, max(y_norm), linestyle = '--', color = 'red', label = r'mean +/- 3$\sigma$')
    ax_norm.hist(given_series, bins = 200, density = True, color = 'lightgray')

    ax_norm.legend()

    plt.show()

def get_pdf_normal_scipy(given_series):
    min_series = min(given_series)
    if min_series < 0:
        x_min = 1.2 * min_series
    else:
        x_min = 0.4 * min_series

    x_max = 1.2* max(given_series)

    x = np.linspace(x_min, x_max, 1000)
    params = norm.fit(given_series)
    
    y = norm.pdf(x, *params)

    return x, y

def get_pdf_gamma_scipy(given_series):
    min_series = min(given_series)
    if min_series < 0:
        x_min = 1.2 * min_series
    else:
        x_min = 0.4 * min_series

    x_max = 1.2* max(given_series)

    x = np.linspace(x_min, x_max, 1000)
    params = sps.gamma.fit(given_series)
    
    y = sps.gamma.pdf(x, *params)

    return x, y

def plot_pdf_fitted_gamma (given_series, signal_label):
    x_gamma, y_gamma = get_pdf_gamma_scipy(given_series)
    skew = sps.skew(given_series)
    mean = np.mean(given_series)
    std = np.std(given_series)
    stats = get_title(given_series)

    fig_gamma = plt.figure('gamma distribution ' + signal_label)
    ax_gamma = fig_gamma.add_subplot(111)
    ax_gamma.set_title('PDF of gamma distribution of ' + signal_label + '\n' + stats)
    ax_gamma.set_xlabel(signal_label + ' [MNm]')

    ax_gamma.plot(x_gamma, y_gamma)
    ax_gamma.vlines(mean + 3*std, 0, max(y_gamma), linestyle = '--', color = 'red')
    ax_gamma.vlines(mean - 3*std, 0, max(y_gamma), linestyle = '--', color = 'red', label = r'mean +/- 3$\sigma$')
    ax_gamma.hist(given_series, bins = 200, density = True, color = 'lightgray')

    ax_gamma.legend()

    plt.show()

def get_pdf_skewnorm(given_series):
    min_series = min(given_series)
    if min_series < 0:
        x_min = 1.2 * min_series
    else:
        x_min = 0.4 * min_series

    x_max = 1.2* max(given_series)

    x = np.linspace(x_min, x_max, 1000)
    params = sps.skewnorm.fit(given_series)
    pdf_y = sps.skewnorm.pdf(x, *params)

    return x, pdf_y

def get_pdf_const(given_series):
    '''
    The function get_pdf_const mimcs the pdf of a constant signal which is a vertical
    line extending to infinity and the cdf is a vertical line to unity. 
    As 'KDE' and 'Normal' are unable to deliver this behavious use this function for 
    constant series   
    '''    

    series_std = np.std(given_series)
    series_m = np.mean(given_series)
    series_step = (2 * series_m - 0)/ 1000
    if series_std ==0.0:
        pdf_x = np.arange(0, 2 * series_m , series_step)
        pdf_y = np.zeros(len(pdf_x))
        pdf_y[int(len(pdf_x)/2)] = len(given_series)
    else : 
        raise Exception("The given series is not a Constant signal, use 'KDE' or 'Normal'")
    return pdf_x, pdf_y

def get_cdf_gev(x, shape_param, loc_param, scale_param,  gev_type = None):
    '''
    Coles eq. 3.2, Bonn meterological institute
    -> these are the cumulative distributions 
    '''
    if gev_type == 'I':
        return np.exp(-np.exp(-((x-loc_param)/scale_param)))
    else:
        # https://stackoverflow.com/questions/45384602/numpy-runtimewarning-invalid-value-encountered-in-power 
        a = -(1+shape_param*((x-loc_param)/scale_param))
        res = np.sign(a) * np.abs(a) **(-1/shape_param)
        # np.exp(-(1+shape_param*((x-loc_param)/scale_param))**(-1/shape_param))
        return np.exp(res)

def get_pdf_gev(x, shape_param, loc_param, scale_param,  gev_type = None):
    '''
    wikipedia PDF of GEV
    https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution 
    '''
    if gev_type == 'I':
        t = np.exp(-(x-loc_param)/scale_param)
        shape_param = 0
    else:
        a = (1+shape_param* ((x-loc_param)/scale_param))
        # https://stackoverflow.com/questions/45384602/numpy-runtimewarning-invalid-value-encountered-in-power 
        t = np.sign(a) * (np.abs(a)) ** (-1/shape_param)
    pdf_y = 1/scale_param * np.sign(t) * np.abs(t) ** (shape_param +1) * np.exp(-t)

    return pdf_y

def get_pdf_gev_scipy(given_series):
    min_series = min(given_series)
    if min_series < 0:
        x_min = 1.2 * min_series
    else:
        x_min = 0.4 * min_series

    x_max = 1.2* max(given_series)

    x = np.linspace(x_min, x_max, 1000)
    params = sps.genextreme.fit(given_series)
    pdf_y = sps.genextreme.pdf(x, *params)

    return x, pdf_y

def get_pdf(given_series, case='KDE'):
    if case == 'KDE':
        return get_pdf_kde(given_series)
    elif case == 'Normal':
        return get_pdf_normal(given_series)
    elif case == 'Constant':
        return get_pdf_const(given_series)
    else:
        raise NotImplementedError("PDF type not implemented, choose either KDE, Normal or Constant")
    
def get_fft(given_series, sampling_freq):
    '''
    The function get_fft estimates the Fast Fourier transform of the given signal 
    '''

    signal_length=len(given_series)

    freq_half =  np.arange(0, 
                           sampling_freq/2 - sampling_freq/signal_length + sampling_freq/signal_length, 
                           sampling_freq/signal_length)

    # single sided fourier
    series_fft = np.fft.fft(given_series)
    series_fft = np.abs(series_fft[0:int(np.floor(signal_length/2))])/np.floor(signal_length/2)  
    
    max_length = len(freq_half)
    if max_length < len(series_fft):
        max_length = len(series_fft)
    
    freq_half = freq_half[:max_length-1]
    series_fft = series_fft[:max_length-1]
    
    return freq_half, series_fft
    
def get_ecdf(series_pdf_x, series_pdf_y):
    '''
    The function get_ecdf computes the emperital CDF of the given PDF.
    '''
    
    # set up data
    dx = series_pdf_x[1] - series_pdf_x[0]    
    Y = series_pdf_y
    # normalize data
    Y /= (dx * series_pdf_y).sum()
    # compute ecdf
    CY = np.cumsum(Y * dx)
    
    return CY

# ==============================================================================
# PEAK OVER THRESHOLD

def plot_pot(given_series, pot_index, stats_series,threshold_param):
    plt.figure(num='POT', figsize=(8, 2.5))
    mean = stats_series['mean']
    std = stats_series['std']
    thres_val = mean + threshold_param * std
    
    time_series = np.arange(len(given_series))
    plt.plot(time_series, given_series, label = 'signal')

    # plotting the extracted pot - as a scatter plot with round red markers
    plt.scatter(time_series[pot_index], given_series[pot_index], marker ='s', color = 'r', label = 'POT')
    plt.axhline(y=thres_val, color='y', linestyle='--')
    plt.axhline(y=-thres_val, color='y', linestyle='--')
    plt.ylabel('Amplitude')
    plt.title('Peak Over Threshold\n mean: ' + str(round(mean,2)) + ' std: ' + str(round(std,2)))
    plt.show()

def get_pot(given_series, threshold_value):
    '''
    The function get_pot computes the Peak Over Threshold for a given threshold value.
    '''
    pot_index = []
    pot_extreme = []

    for i in range(len(given_series)): 
         if  threshold_value < np.abs(given_series[i]): 
             pot_index.append(i)
             pot_extreme.append(np.abs(given_series[i]))
   
    return pot_index, pot_extreme

def get_pot_runtime(given_series, threshold_param):
    '''
    The function get_pot_runtime does a runtime evaluation of the Peak Over Threshold 
    for a given threshold parameter
    '''
    tic = timeit.default_timer()    
    
    # renaming
    values = given_series
    all_values = len(given_series)
    
    #setting up necessary variables
    meannew = values[0]
    meanold = meannew
    rmsnew = np.abs(values[0])
    rmsold = rmsnew
    part1 = values[0]*values[0]
    part2 = values[0]

    arraysorted = np.zeros(all_values) #preallocate with zeros
    arraysorted[0] = values[0] # create sorted array with new values
    
    pot_index = []
    pot_extreme = []
    
    res_m = np.zeros(all_values)
    res_m[0] = meannew
    res_rms = np.zeros(all_values)
    res_rms[0] = rmsnew

    res_std = np.zeros(all_values)
    
    res_med = np.zeros(all_values)
    res_med[0] = values[0]    
    
    res_skew = np.zeros(all_values)

    res_thres = np.zeros(all_values)
    res_thres[0] = meannew + threshold_param * 0
    
    
    # starting calculation loop
    for i in range(1,all_values): 
        # calculate mean
        meannew = (meanold * (i-1) + values[i])/i
        meanold = meannew
        
        # calculate rms
        rmsnew = np.sqrt(((i-1)*rmsold*rmsold + values[i]*values[i])/i)
        rmsold = rmsnew
        
        # calculate standard deviation, initally summing up parts
        part1 = part1 + values[i] * values[i]
        part2 = part2 + values[i]
        standarddev = np.sqrt((part1 - 2* meannew * part2 + meannew*meannew* i)/(i-1))
        
        
        # calculate median 
        if values[i]>= arraysorted[i-1]:
          arraysorted[i] = values[i]
        elif values[i]<= arraysorted[0]:
          arraysorted[1:i+1] = arraysorted[0:i]
          arraysorted[0] = values[i]
        else:
          j = i-1
          push_here = j
          while values[i] < arraysorted[j]:
            j = j - 1
            push_here = j
    
          arraysorted[push_here+1:i+1] = arraysorted[push_here:i]
          arraysorted[push_here+1] = values[i] 
    
        if (i % 2 == 1): # check if it matches matlab indexing
            medianv = (arraysorted[i//2] + arraysorted[i //2+1])/2                 
        else:
            medianv = arraysorted[i//2]
            
        # calculate skewness
        skewness = (meannew - medianv)/standarddev

        # threshold 6 sigma criterium
        threshold = (meannew + threshold_param * standarddev)
        if  threshold < np.abs(values[i]): 
             pot_index.append(i)
             pot_extreme.append(np.abs(values[i]))
                        
        # append results
        res_m[i] = meannew
        res_rms[i] = rmsnew
        res_std[i] = standarddev
        res_med[i] = medianv
        res_skew[i] = skewness
        res_thres[i] = threshold
        
    toc = timeit.default_timer()
    
    print('Elapsed time for get_pot_runtime function evaluation: ', toc-tic ,'s\n')

    return res_m, res_rms, res_std, res_med, res_skew, res_thres, pot_index, pot_extreme

def fit_pot_vals_to_gpd(pot_extremes, threshold_value):
    '''
    using scipy to fit the extracted values. Setting initial values as in swe_ws2021_1_4
    '''
    params = sps.genpareto.fit(pot_extremes, 0, loc = threshold_value , scale = 1)
    print (params)
    xmin, xmax = evaluation_limits(pot_extremes, min_fac = 0.9, max_fac = 1.1)
    pdf_x = np.linspace(xmin,xmax,100)
    pdf_y = sps.genpareto.pdf(pdf_x, *params)

    return pdf_x, pdf_y

def level_of_exceedance_pot(given_series, threshold_value, P1, dur_ratio = 1):
    '''
    Coles Eq. 4.14
    shape = xi
    loc = mu
    scale = sigma
    '''

    pot_index, pot_extremes = get_pot(given_series, threshold_value)
    shape, loc, scale = sps.genpareto.fit(pot_extremes, 0, loc = threshold_value , scale = 1)

    m = len(given_series) * dur_ratio
    zeta_est = (len(pot_extremes) / m)* dur_ratio # coles eq.: 4.14 ff
    m_ex = m*10

    m_own = 1/(1-P1)

    if shape == 0:
        x_m = threshold_value + scale*np.log(m*dur_ratio * zeta_est) #coles eq: 4.14
    else:
        x_m = threshold_value + scale/shape * ((m*zeta_est)**shape - 1) # coles eq: 4.13

    if x_m < threshold_value:
        raise ('x_m too small -> select larger m')
    else:
        return x_m

def pot_holmes_return_period(given_series, threshold_value, P1, dur_ratio = 1):
    '''
    Holmes: eq. 2.17
    but estimating shape and scale with Pareto
    '''
    pot_index, pot_extremes = get_pot(given_series, threshold_value)
    total_sample_len = len(given_series)

    dur = total_sample_len*dur_ratio
    
    lam =  len(pot_extremes) * dur_ratio #number of crossings per period dur 

    shape, loc, scale = sps.genpareto.fit(pot_extremes, 0, loc = threshold_value , scale = 1)

    return_period = 1/(1-P1)
    if shape == 0:
        Z_R = threshold_value + scale * np.log(lam*return_period)
    else:
        Z_R = threshold_value + scale/shape * (1-(lam*return_period)**-shape)

    return Z_R

def gpd_model_parameter_check(given_series, stats_series, param0 = 0, param1 = None, use_parametric = True, num_thresholds = 50):

    glob_max = max(given_series)

    if not param1:
        param1 = (glob_max - stats_series['mean'])/stats_series['std']
    param_max = (glob_max - stats_series['mean'])/stats_series['std']
    m = int(stats_series['mean'])
    gm = int(glob_max)

    thresholds = np.arange(m, gm, 1.0)
    threshold_params = np.linspace(param0, param1, num_thresholds, endpoint=False)
    #threshold_params = np.arange(param0, upper_limit, 0.09)#, endpoint=False)
    param_ticks = np.arange(param0, round05(param1)+0.5, 0.5)

    params = {'scale':[], 'scale*':[], 'shape':[], 'max/glob max':[]}

    lim25 = stats_series['mean'] + 2.5 * stats_series['std']
    lim30 = stats_series['mean'] + 3.0 * stats_series['std']

    if use_parametric:
        for threshold_param in threshold_params:
            threshold = stats_series['mean'] + threshold_param * stats_series['std']
            pot_index, pot_extremes = get_pot(given_series, threshold)
            shape, loc, scale = sps.genpareto.fit(pot_extremes, 0, loc = threshold , scale = 1)
            params['scale'].append(scale)
            params['scale*'].append(scale - shape*threshold)
            params['shape'].append(shape)
            pred_max = pot_holmes_return_period(given_series, threshold, P1 = 0.98, dur_ratio = 1)
            params['max/glob max'].append(pred_max / glob_max)
        
        lim25, lim30, upper = 2.5, 3.0, param_max
        x = threshold_params

    # thresholds directly as threshold values
    else:
        for threshold in thresholds:
            pot_index, pot_extremes = get_pot(given_series, threshold)
            shape, loc, scale = sps.genpareto.fit(pot_extremes, 0, loc = threshold , scale = 1)
            params['scale'].append(scale)
            params['scale*'].append(scale - shape*threshold)
            params['shape'].append(shape)

        lim25 = stats_series['mean'] + 2.5 * stats_series['std']
        lim30 = stats_series['mean'] + 3.0 * stats_series['std']
        upper = glob_max
        x = thresholds

    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(threshold_params[0], threshold_params[-1] + 0.5, 1)
    minor_ticks = np.arange(threshold_params[0], threshold_params[-1] + 0.5, 0.5)

    

    fig, axes = plt.subplots(len(params),1)
    for i, param in enumerate(params.items()):
        
        axes[i].plot(x, param[1])
        axes[i].axvline(lim25, color='y', linestyle = '--', label = r'$\mu + 2.5\sigma$')
        axes[i].axvline(lim30, color='y', linestyle = '--', label = r'$\mu + 3.0\sigma$')
        axes[i].axvline(upper, color='r', linestyle = '--', label = r'$max_{global}$')
        axes[-1].set_xlabel('threshold parameter')
        axes[i].set_ylabel(param[0])
        axes[i].set_xticks(param_ticks)
        #axes[i].set_xticks(minor_ticks, minor=True)
        axes[i].legend(loc= 'upper right')
        axes[i].grid(which='both')
    plt.show()


# ==============================================================================
# BLOCK MAXIMA  

def get_bm(given_series, initial_block_size):
    '''
    The function get_bm computes the Block Maxima of the signal for a given block size
    '''

    # for even block sizes 
    if len(given_series) % 2 !=0:
        given_series = given_series[1:]
        block_size = initial_block_size - 1
    else:
        block_size = initial_block_size
    # normal quantities
    mean = np.mean(given_series)
    std = np.std(given_series)
    upper_tail = mean + 3*std
    lower_tail = mean - 3*std

    block_max = np.abs(given_series[0])
    
    bm_index = []
    bm_extreme = [] 

    for i in range(1,len(given_series)):
        if block_max < np.abs(given_series[i]):
            block_max = np.abs(given_series[i])
            block_max_idx = i

        if (i+1) % block_size == 0: 
            # check if the found extreme lays withing the tails -> distinguish between + and - for non zero mean +/-fluctuating signals
            # if block_max < upper_tail:
            #     print ('block size', block_size, ' is to small')
            #     get_bm(given_series, block_size + 100)
            bm_index.append(block_max_idx)
            bm_extreme.append(block_max)
            
            block_max = 0

    return bm_index, bm_extreme, block_size

def get_block_stats(given_series, abs_block_size, block_index ,significance_level = 5.):
    '''
    returns for each block the statistics of the data of this block each stat in one list 
    0: mean
    1: std
    2: skewness
    3: Anderson - Darling Test (Normal) (Last entry is the AD for the whole series)
        significance level in % is the level to which the Null Hypothesis should be fullfilled
        possible values: 15.0, 10.0, 5.0, 2.5, 1.0 [%]
    '''
    
    m_i, s_i, sk_i, anderson = [], [], [], []
    #block_index = [0 + n*abs_block_size for n in np.ar]
    for i in range(len(block_index)-1):
        id1 = block_index[i]
        id2 = block_index[i+1]
        sub = given_series[id1:id2]

        stats, critical_vals, sig_level = sps.anderson(sub, dist='norm')
        idx = list(sig_level).index(significance_level)

        if stats > critical_vals[idx]:
            test = 'No'
        else:
            test = 'Yes'

        anderson.append(test)

        m_i.append(np.mean(sub))
        s_i.append(np.std(sub))
        sk_i.append(sps.skew(sub))
    
    stats, critical_vals, sig_level = sps.anderson(given_series, dist='norm')
    if stats > critical_vals[-1]:
        test = 'Given series: AD rejected'
    else:
        test = 'Given series: AD accepted'

    anderson.append(test)

    return m_i, s_i, sk_i, anderson

def plot_block_stats_eval(block_stats, block_sizes, label, normalize_with_total = True):
    '''
    scatter plots of statistical moments around the ones of the total series
    '''
    
    if normalize_with_total:
        n_m, n_s, n_sk = 1/block_stats['mean_tot'], 1/block_stats['std_tot'], 1/block_stats['skew_tot']
    else:
        n_m, n_s, n_sk = 1,1,1

    fig_be, (ax_m, ax_s, ax_sk) = plt.subplots(nrows=3, sharex=True, num = 'block stats')

    fig_be.suptitle(label)
    
    scales = [10/s for s in block_sizes]

    b_0 = 0
    x_tick_loc = []
    #try:
    for b_i, size in enumerate(block_sizes):
        x = np.arange(1, len(np.asarray(block_stats['mean'][b_i])) + 1) * scales[b_i] + b_0 + 3
        dx = x[-1] - b_0
        b_0 = x[-1]
        x_tick_loc.append((b_0+0.5) - dx/2)

        ax_m.vlines(b_0 + 0.6, min(block_stats['mean'][-1])*n_m, max(block_stats['mean'][-1])*n_m, color='y', linestyle = '--')
        ax_s.vlines(b_0 + 0.6, min(block_stats['std'][-1])*n_s, max(block_stats['std'][-1])*n_s, color='y', linestyle = '--')
        ax_sk.vlines(b_0 + 0.6, min(block_stats['skew'][-1])*n_sk, max(block_stats['skew'][-1])*n_sk, color='y', linestyle = '--')

        ax_m.scatter(x, np.asarray(block_stats['mean'][b_i])*n_m)
        ax_s.scatter(x, np.asarray(block_stats['std'][b_i])*n_s)
        ax_sk.scatter(x, np.asarray(block_stats['skew'][b_i])*n_sk)

    ax_m.hlines(block_stats['mean_tot']*n_m, 0, b_0, label = 'mean_total ' + str(round(block_stats['mean_tot'],3)))
    ax_s.hlines(block_stats['std_tot']*n_s, 0, b_0, label = 'std_total ' + str(round(block_stats['std_tot'],3)))
    ax_sk.hlines(block_stats['skew_tot']*n_sk, 0, b_0, label = 'skew_total ' + str(round(block_stats['skew_tot'],3)))

    ax_m.legend(loc = 'upper left')
    ax_s.legend(loc = 'upper left')
    ax_sk.legend(loc = 'upper left')
    ax_m.set_ylabel('mean/mean_tot')
    ax_s.set_ylabel('std/std_tot')
    ax_sk.set_ylabel('skewness/skew_tot')
    ax_sk.set_xlabel('number of blocks')
    plt.xticks(x_tick_loc, [str(block_size) for block_size in block_sizes])

    plt.show()
    
    # except ValueError:
    #     print ('\nplotting block stats evaluation not possible for these block sizes\n')

def probplot_eval(given_series, block_sizes):
    pass 


def create_gev_distributed_series(shape, loc, scale, size):
    rv = sps.genextreme.rvs(shape, loc = loc, scale = scale, size= size)

    return rv

def get_bar_width(block_sizes):
    bw = []

    for b in block_sizes:
        if b <= 1000:
            bw.append(0.05)
        elif b <= 10000:
            bw.append(0.1)
        elif b <= 100000:
            bw.append(1.0)   
        elif b <= 1000000:
            bw.append(10)

    return bw

# =============================================================================
# MAXIMUM LIKELIHOOD
def maxest_mle(given_series, n_blocks, p1, dur_ratio):
    '''
    estimates the GEV parameters shape, loc, scale using MLE (build in python)
    for p1 of non excedance computes the maximum value
    '''
    # using the same methodology as BLUE implementation
    t = len(given_series)
    n = n_blocks
    given_series_max = np.zeros([n,1])
    given_series_min = np.zeros([n,1])

    # blocking
    r = np.fmod(t,n)
    if r == 0:
        for i in np.arange(0,n):
            a = given_series[int(i*t/n):int((i+1)*t/n)]
            given_series_max[i] = a.max()
            given_series_min[i] = a.min()

    elif r > n/2:
        q = int(np.fix(t/n)+1)
        for i in np.arange(0,n-1):
            a = given_series[i*q:(i+1)*q]
            given_series_max[i] = a.max()
            given_series_min[i] = a.min()

    else:
        q = int(np.fix(t/n))
        for i in np.arange(0,n-1):
            a = given_series[i*q:(i+1)*q]
            given_series_max[i] = a.max()
            given_series_min[i] = a.min()

    if (dur_ratio == 0):
        dur = n
    else:
        dur = dur_ratio * n

    #params_max = sps.genextreme.fit(given_series_max)

    gev_shape, gev_loc, gev_scale = sps.genextreme.fit(given_series_max)
    # Coles Eq. 3.4
    predicted_max = gev_loc - (gev_scale/gev_shape) * ( 1- (-np.log (p1) )**(-gev_shape)) # for duration of 1 epoch
    predicted_max_dur = gev_loc - (gev_scale/gev_shape) * ( 1- (-np.log (p1) * dur)**(-gev_shape)) # for longer duration

    return predicted_max, predicted_max_dur

# ===============================================================================
# BLUE METHOD

#from bluecoefficients import bluecoeff

def blue4pressure(series, n_blocks, P1 = 0.80, P2 = 0.5704, dur_ratio = None):
    '''
    From a time series, blue4pressure estimates
    extremes of positive and negative values based on Lieblein's BLUE 
    (Best Linear Unbiased Estimate) method applied to n_blocks epochs. 
    
    Extremes are estimated for the duration of the record and for a ratio of it for probabilities of non-exceedance 
    P1 and P2 of the Gumbel distribution fitted to the epochal peaks.
    n = integer, dur need NOT be an integer.
    Written by Dat Duthinh 8_25_2015, 2_2_2016, 2_6_2017

    Reference: 
    1) Julius Lieblein "Efficient Methods of Extreme-Value
       Methodology" NBSIR 74-602 OCT 1974 for n = 4:16
    2) Nicholas John Cook "The designer's guide to wind loading of
       building structures" part 1, British Research Establishment 1985 Table C3
       pp. 321-323 for n = 17:24. Extension to n=100 by Adam Pintar Feb 12 2016.
    3) INTERNATIONAL STANDARD, ISO 4354 (2009-06-01), 2nd edition, “Wind 
       actions on structures,” Annex D (informative) “Aerodynamic pressure and 
       force coefficients,” Geneva, Switzerland, p. 22
    
    INPUT 
    series = vector of time history of pressure coefficients
    n = number of epochs (integer)of series data, 4 <= n <= 100
    dur = number of epochs for estimation of extremes. Default dur = n
          dur need not be an integer
    NOTE:
    replaced dur by dur_ratio to have the same as maxminest
    P1, P2 = probabilities of non-exceedance of extremes in EV1 (Gumbel).  
    P1 defaults to 0.80 (ISO) and P2 to 0.5704 (mean).

    OUTPUT 
    suffix max for + peaks, min for - peaks of pressure coeff.

    NOTE:
    changed: default returning extremes for the duration of the record.
             if dur_ratio is NOT given this is the return value.
             if dur_ratio is given the extremes for the duration of dur = dur_ratio * len(series) is returned 
             e.g. series is 2h long with dur_ratio = 0.5 the extreme within a period of 1h can be calculated. 
             => like this it is anlogous to NIST maxminest/qnt

    these are the returned values:

    p1_rmax (p1_rmin)= extreme value of positive (negative) peaks with probability of non-exceedance P1 for duration of series
    p2_rmax (p2_rmin)= extreme value of positive (negative) peaks with probability of non-exceedance P2 for for dur duration of series
        
    Computed but not returned are:

    p1_max (p1_min)= extreme value of positive (negative) peaks with probability of non-exceedance P1 for duration of 1 epoch
    p2_max (p2_min)= extreme value of positive (negative) peaks withprobability of exceedance P2 for duration of 1 epoch

    series_max (series_min)= vector of n positive (negative) epochal peaks
    u_max, b_max (u_min, b_min) = location and scale parameters of EV1
    (Gumbel) for positive (negative) peaks
    '''
    
    import numpy as np

    # Size of series array
    t = len(series)

    # Initialize variables
    series_max = np.zeros([n_blocks,1])
    series_min = np.zeros([n_blocks,1])

    # Find the peaks for each of the n user-defined epochs
    # and store in seriesmax and seriesmin arrays

    # Separate cases if n evenly divides t or not
    # NOTE this here is actually the block generation
    r = np.fmod(t,n_blocks)
    if r == 0:
        for i in np.arange(0,n_blocks):
            a = series[int(i*t/n_blocks):int((i+1)*t/n_blocks)]
            series_max[i] = a.max()
            series_min[i] = a.min()

    elif r > n_blocks/2:
        q = int(np.fix(t/n_blocks)+1)
        for i in np.arange(0,n_blocks-1):
            a = series[i*q:(i+1)*q]
            series_max[i] = a.max()
            series_min[i] = a.min()

    else:
        q = int(np.fix(t/n_blocks))
        for i in np.arange(0,n_blocks-1):
            a = series[i*q:(i+1)*q]
            series_max[i] = a.max()
            series_min[i] = a.min()
        
    # Coefficients for all n
    [ai,bi]= bluecoeff(n_blocks)

    # Organize values in ascending or descending order
    x_max = np.sort(series_max,axis=0)
    x_min = np.sort(series_min,axis=0)
    x_min = x_min[::-1]

    # defaults    
    if not dur_ratio:
        dur = n_blocks
    else:
        dur = dur_ratio * n_blocks
    # ************************** MAX CASE PEAK ***************************
    u = 0 # location parameter
    b = 0 # scale parameter

    # Calculate parameters of location and scale
    # Lieblein eq. 4
    for j in np.arange(0,n_blocks):
        u = u + ai[j]*x_max[j]
        b = b + bi[j]*x_max[j]
    
    # NOTE difference of one epoch and more
    p1_max = u - b*np.log(-np.log(P1)) # for 1 epoch
    p1_rmax = p1_max + b*np.log(dur) # for longer duration
    p2_max = u - b*np.log(-np.log(P2)) # for 1 epoch
    p2_rmax = p2_max + b*np.log(dur) # for longer duration
    u_max = u
    b_max = b
    # ************************** MIN CASE PEAK ***************************
    u = 0
    b = 0

    # Calculate parameters of location and scale
    for j in np.arange(0,n_blocks):
        u = u + ai[j]*x_min[j]
        b = b + bi[j]*x_min[j]

    # NOTE what does it mean for one epoch or longer
    p1_min = u - b*np.log(-np.log(P1))  # for 1 epoch
    p1_rmin = p1_min + b*np.log(dur)  # for longer duration
    p2_min = u - b*np.log(-np.log(P2))  # for 1 epoch
    p2_rmin = p2_min + b*np.log(dur)  # for longer duration
    u_min = u
    b_min = abs(b)
    
    #print(p1_max, p2_max, p1_rmax, p2_rmax, u_max, b_max, series_max, p1_min, p2_min, p1_rmin, p2_rmin, u_min, b_min, series_min)
    return p1_rmax, p1_rmin, p2_rmax, p2_rmin

# MAX MIN ESTIMATION - NIST 

import scipy.special as special
import scipy.interpolate as interpolate
import math
import bisect

def stdgaminv(p,gam):
    
    abs_tol = 10**-3
    rel_tol = 10**-3
    
    if gam<0.1 or gam>150:
        raise ValueError('The shape parameter gamma must be between 0.1 and 150')

    p = np.array(p)
    
    x_max = 10**np.polyval([-0.009486738 ,0.03376901, 0.1151316, 0.2358172, 1.139717],math.log10(gam))
    
    max_iter = 200

    current_iter = 0
   
    while special.gammainc(gam,x_max)<max(p):
        

        current_iter+=1
        
        if current_iter>max_iter:
            raise ValueError('Maximum specified probability is too high:{}'.format(max(p)))
        else:
            x_max *=1.5
            
    x_min = 10**np.polyval([-0.0854665, 0.866249, -3.25511, 5.14328, -0.90924, -8.09135, 12.3393, -5.89628],math.log10(gam))
    current_iter = 0
    
    while special.gammainc(gam,x_min)>min(p):
        current_iter +=1
        
        if current_iter>max_iter:
            raise ValueError('Minimum specified probability is too low:{}'.format(min(p)))
        else:
            x_min *=0.1
    
    
    n_check = 1000
    x_check = np.linspace(x_min,x_max,n_check)
    

    p_check = special.gammainc(gam,x_check)
    

    p_check, ind_u = np.unique(p_check,return_index = True)
    
    
    x_check = x_check[ind_u]

    f = interpolate.interp1d(p_check,x_check,fill_value='extrapolate')

    x_est = f(p)
    
    
    max_iter = 15
    current_iter= 0
    x_step = np.ones(x_est.shape)
    
    while any(abs(x_step)>abs_tol) and any(abs(np.divide(x_step,x_est)>rel_tol)):
        current_iter+=1
        
        if current_iter>max_iter:
            break
        
        p_est =special.gammainc(gam,x_est)

        p_check = np.append(p_check,p_est)
        x_check = np.append(x_check,x_est)
        p_check, ind_u = np.unique(p_check,return_index = True)

        x_check = x_check[ind_u]

        f = interpolate.interp1d(p_check,x_check,fill_value='extrapolate')
        
        x_interp = f(p)

        x_step = x_interp-x_est

        x_est = x_interp
        
    x = x_est.reshape(p.size)

    return x

def stdnorminv(p):
    x = -1*np.sqrt(2)*special.erfcinv(2*p)

    return x

def stdnormcdf(x):
    p = 0.5 *special.erfc(-x/math.sqrt(2))
    return p

def maxmin_qnt_est (record, cdf_p_max = 0.975, cdf_p_min = 0.025, cdf_qnt = 0.995, dur_ratio = 1):
    '''
    The function computes estimated quantile values of a given time series.
    And gives a quantile value for probability of exceedance: cdf_qnt
    The function computes expected maximum and minimum values of time series by estimating probability distributions for the peaks.
    INPUT.
        record: is a time series for which the peaks should be estimated

        dur_ratio(optional) = allows peaks to be estimated for a duration that differs from the duration of the record itself (y_pk calculation):
                              dur_ratio = [duration for peak estimation]/[duration of record]
                              (If unspecified, a value of 1 is used.)
        cdf_p_max, cdf_p_min -> integration limits?!
        cdf_qnt quantile value (probability of non-exceedance) for which an extreme should be returned single value between 0 and 1 !
    '''

    n_cdf_pk = 1000
    
    cdf_pk_min = cdf_p_min
    cdf_pk_max = cdf_p_max

    cdf_pk = np.linspace(cdf_pk_min,cdf_pk_max,n_cdf_pk)

    if cdf_qnt not in cdf_pk:
        c = list(cdf_pk)
        bisect.insort(c, cdf_qnt)
        cdf_pk = np.asarray(c)
        id_qnt = c.index(cdf_qnt)
    else:
        id_qnt = int(np.where(cdf_pk == cdf_qnt)[0])

    rsize = np.array(record).shape
    
    if len(rsize) == 1:
        rec_size = 1
    else:
        rec_size = rsize[0]
    
    max_est = np.zeros((rec_size,1))
    min_est = np.zeros((rec_size,1))
    max_std = np.zeros((rec_size,1))
    min_std = np.zeros((rec_size,1))

    max_qnt = np.zeros(rec_size)
    min_qnt = np.zeros(rec_size)

    # NOTE rec_size is the number of time hisories given to the funciton! not the sample size of each record
    for i in np.arange(rec_size):
        if rec_size == 1:
            x = record
        else:
            x = record[:,i]
        
        n = x.size
        
        mean_x = np.mean(x)

        # std_x = np.std(x,ddof = 1) # Original
        std_x = np.std(x)

        skew_x = np.sum(np.power(x-mean_x,3))/(n*std_x**3)

        # sign conversion if negative skewes record
        X = x*np.sign(skew_x)
        
        sort_X = np.sort(X)

        mean_X = mean_x*np.sign(skew_x)
        std_X = std_x
        CDF_X = np.divide(np.arange(1,n+1),n+1)

        n_coarse = min([n,1000])

        CDF_coarse = np.linspace(1/(n_coarse+1),n_coarse/(n_coarse+1),n_coarse)
        
        f = interpolate.interp1d(CDF_X,sort_X)
        X_coarse = f(CDF_coarse)
        
        mean_X_coarse = np.mean(X_coarse)

        # std_X_coarse = np.std(X_coarse) # Original
        std_X_coarse = np.std(X_coarse, ddof=1)

        gamma_min = 1
        gamma_max = 125
        n_gamma = 19
        n_start = 7

        gamma_list = np.logspace(math.log10(gamma_min),math.log10(gamma_max),n_gamma)

        gam_PPCC_list = np.zeros(gamma_list.shape)
        count = 0
        beta_coarse_list = np.zeros((125,1))
        mu_coarse_list =np.zeros((125,1))
        
        for j in np.arange(n_start,-1,-1):
          
            count+=1
            
            s_gam_j = stdgaminv(CDF_coarse,gamma_list[j])
            
            mean_s_gam_j = np.mean(s_gam_j)

            # linear regression:
            
            beta_coarse_list[j] = (np.sum(np.multiply(s_gam_j,X_coarse))-(n_coarse*mean_s_gam_j*mean_X_coarse))/(np.sum(np.power(s_gam_j,2))-(n_coarse*mean_s_gam_j**2))
            
            mu_coarse_list[j]=(mean_X_coarse - beta_coarse_list[j]*mean_s_gam_j)

            #Probability Plot Correlation Coefficient:
           
            # gam_PPCC_list[j] = (beta_coarse_list[j]*np.std(s_gam_j)/std_X_coarse) # Original
            gam_PPCC_list[j] = (beta_coarse_list[j]*np.std(s_gam_j,ddof=1)/std_X_coarse)

            X_coarse_fit_j = mu_coarse_list[j] + beta_coarse_list[j]*s_gam_j

            if gam_PPCC_list[j] == max(gam_PPCC_list):
                gam = gamma_list[j]
                gam_PPCC_max = gam_PPCC_list[j]
            else:
                break
        
        if gam_PPCC_list[n_start-1] < gam_PPCC_list[n_start]:
            # if the PPCC decreased with decreasing gamda, try increasing gamma: 
            for j in np.arange(n_start+1,n_gamma):
                count += 1
                # Obtain the Gamma Distribution Parameters for current gamma:
                
                s_gam_j = stdgaminv(CDF_coarse,gamma_list[j])   # standard variate
                mean_s_gam_j = np.mean(s_gam_j)
                # linear regression:
                beta_coarse_list[j] = (np.sum(np.multiply(s_gam_j,X_coarse))-(n_coarse*mean_s_gam_j*mean_X_coarse))/(np.sum(np.power(s_gam_j,2))-(n_coarse*mean_s_gam_j**2))
                
                mu_coarse_list[j] = mean_X_coarse - beta_coarse_list[j]*mean_s_gam_j
                #Probability Plot Correlation Coefficient:
                # gam_PPCC_list[j] = beta_coarse_list[j]* np.std(s_gam_j)/std_X_coarse # Original
                gam_PPCC_list[j] = beta_coarse_list[j]* np.std(s_gam_j, ddof=1)/std_X_coarse
                X_coarse_fit_j = mu_coarse_list[j] + beta_coarse_list[j]*s_gam_j

                ##
                # BLOCK needs extra indent not to break out prematurely
                ##
                if gam_PPCC_list[j] == max(gam_PPCC_list):
                    gam = gamma_list[j]
                    gam_PPCC_max = gam_PPCC_list[j]
                else:
                    break
                ##
                # BLOCK needs extra indent
                ##
                
            ##
            # ORIGINAL
            ##
            # if gam_PPCC_list[j] == max(gam_PPCC_list):
            #     gam = gamma_list[j]
            #     gam_PPCC_max = gam_PPCC_list[j]
            # else:
            #     break
            ##
            # ORIGINAL
            ##

        s_gam = stdgaminv(CDF_X,gam)
        mean_s_gam = np.mean(s_gam)

        beta = (np.sum(np.multiply(s_gam,sort_X))-n*mean_s_gam*mean_X)/(np.sum(np.power(s_gam,2))-n*mean_s_gam**2) #0.12
        mu = mean_X - beta*mean_s_gam
        # gam_PPCC = beta*np.std(s_gam)/std_X # Original
        gam_PPCC = beta*np.std(s_gam, ddof=1)/std_X
        
        x_fit = mu +beta*s_gam

        # Obtain the Normal Distribution Parameters for lower portion of CDF

        CDF_split = 0.25
        f = interpolate.interp1d(CDF_X,sort_X)
        X_split = f(CDF_split)

        ind_low = np.where(sort_X<X_split)
        
        X_low = sort_X[ind_low]
        n_low = len(X_low)
        CDF_low = CDF_X[ind_low]

        s_norm_low = stdnorminv(CDF_low)
        mean_s_norm_low = np.mean(s_norm_low)
        mean_X_low = np.mean(X_low)

        # linear regression:
        
        sigma_low = (np.sum(np.multiply(s_norm_low,X_low))-n_low*mean_s_norm_low*mean_X_low)/(np.sum(np.power(s_norm_low,2))-n_low*mean_s_norm_low**2)
        
        mu_low=mean_X_low - sigma_low*mean_s_norm_low
        X_low_fit = mu_low +sigma_low*s_norm_low

        # Probability Plot Correlation Coefficient:

        # norm_PPCC = sigma_low*np.std(s_norm_low)/np.std(X_low) # Original
        norm_PPCC = sigma_low*np.std(s_norm_low, ddof=1)/np.std(X_low, ddof=1)

        X_u = np.mean(sort_X[np.where(abs(CDF_X-0.5) == min(abs(CDF_X-0.5)))])

        front = np.where(X[1:]>=X_u)
        back = np.where(X[0:-1]<X_u)
        
        Nupcross = len(set(front[0]) & set(back[0]))
        
        if Nupcross<100:
            print('The number of median upcrossings is low {}'.format(Nupcross))
            print('The record may be too short for accurate peak estimation. Aminimum of 100 is recommended.')
        
        # everything performed on the Gaussian process y(t)
        y_pk = np.sqrt(2.0*np.log(np.divide(-dur_ratio*Nupcross,np.log(cdf_pk))))

        CDF_y = stdnormcdf(y_pk)
        
        # Perform the mapping procedure to compute the CDF of largest peak for X(t) from y(t)
        # theoretical extremes in non - Gaussian space?
        X_max = stdgaminv(CDF_y,gam) * beta 
        X_max+= + mu

        X_min = np.multiply(stdnorminv(1-CDF_y),sigma_low)
        
        X_min+=mu_low
        # probability distribution function of the extreme values in non-Gaussian space
        pdf_pk = np.multiply(np.multiply(-y_pk,cdf_pk),np.log(cdf_pk))
        
        # Compute the Mean of the Peaks for process X(t)

        if np.sign(skew_x)>0:
            max_est[i] = np.trapz((np.multiply(pdf_pk,X_max)),y_pk)
            min_est[i] = np.trapz((np.multiply(pdf_pk,X_min)),y_pk)
            max_std[i] = np.trapz((np.multiply(np.power((X_max-max_est[i]),2),pdf_pk)),y_pk)
            min_std[i] = np.trapz((np.multiply(np.power((X_min-min_est[i]),2),pdf_pk)),y_pk)

            max_qnt[i] = X_max[id_qnt] # fill the ith row x_max contains the 
            min_qnt[i] = X_min[id_qnt]
            
        else:
            ##
            # ORIGINAL
            ##
            # max_est[i] = np.trapz((np.multiply(pdf_pk,X_max)),y_pk)
            # min_est[i] = np.trapz((np.multiply(pdf_pk,X_min)),y_pk)
            ##

            # UPDATE according to initial MATLAB -> seems to be able to robustly handle
            # normal random as well
            ##
            max_est[i] = -np.trapz((np.multiply(pdf_pk,X_min)),y_pk)
            min_est[i] = -np.trapz((np.multiply(pdf_pk,X_max)),y_pk)

            max_std[i] = np.trapz((np.multiply(np.power((-X_min-max_est[i]),2),pdf_pk)),y_pk)
            min_std[i] = np.trapz((np.multiply(np.power((-X_max-min_est[i]),2),pdf_pk)),y_pk)

            max_qnt[i] = -X_min[id_qnt]
            min_qnt[i] = -X_max[id_qnt]

    return max_qnt, min_qnt, max_est, min_est, max_std, min_std, Nupcross

def maxminqnt (record, dur_ratio, cdf_qnt):
    '''
    NOTE: ---------------- THIS IS INCLUDED NOW IN MAXMIN_QNT_EST ----------------

    The function maxminqnt computes quantiles (i.e., values corresponding to specified probabilities of non-exceedance)
    of the maximum and minimum values of the input time series.
    INPUT.
        record: is a time series for which the peaks should be estimated (each row in the record can be a time series)

        dur_ratio(optional) = allows peaks to be estimated for a duration that differs from the duration of the record itself (y_pk calculation):
        dur_ratio = [duration for peak estimation]/[duration of record]
        (If unspecified, a value of 1 is used -> peak for a duration of the given series is returned.)

        cdf_qnt: is a vector/list of quantiles
    NOTE: basically this is the exact same function as maxminest. Maxminest has 1 quantile value set originally.
          Just that est integrates something at the end and returns like this the mean expected extreme values.    
    '''
    
    if max(cdf_qnt) >= 1 or min(cdf_qnt) <= 0:
        raise ValueError('values of cdf_qnt must be between 0 and 1')

    rsize = np.array(record).shape
    
    if len(rsize) == 1:
        rec_size = 1
    else:
        rec_size = rsize[0]

    max_qnt = np.zeros((rec_size,len(cdf_qnt)))
    min_qnt = np.zeros((rec_size,len(cdf_qnt)))

    # NOTE rec_size is the number of time hisories given to the funciton! not the sample size of each record
    for i in np.arange(rec_size):
        if rec_size == 1:
            x = record
        else:
            x = record[:,i]
        
        n = x.size
        
        mean_x = np.mean(x)

        # std_x = np.std(x,ddof = 1) # Original
        std_x = np.std(x)

        skew_x = np.sum(np.power(x-mean_x,3))/(n*std_x**3)

        # sign conversion if negative skewes record
        X = x*np.sign(skew_x)
        
        sort_X = np.sort(X)

        mean_X = mean_x*np.sign(skew_x)
        std_X = std_x
        CDF_X = np.divide(np.arange(1,n+1),n+1)

        n_coarse = min([n,1000])

        CDF_coarse = np.linspace(1/(n_coarse+1),n_coarse/(n_coarse+1),n_coarse)
        
        f = interpolate.interp1d(CDF_X,sort_X)
        X_coarse = f(CDF_coarse)
        
        mean_X_coarse = np.mean(X_coarse)

        # std_X_coarse = np.std(X_coarse) # Original
        std_X_coarse = np.std(X_coarse, ddof=1)

        gamma_min = 1
        gamma_max = 125
        n_gamma = 19
        n_start = 7

        gamma_list = np.logspace(math.log10(gamma_min),math.log10(gamma_max),n_gamma)

        gam_PPCC_list = np.zeros(gamma_list.shape)
        count = 0
        beta_coarse_list = np.zeros((125,1))
        mu_coarse_list =np.zeros((125,1))
        
        for j in np.arange(n_start,-1,-1):
          
            count+=1
            
            
            s_gam_j = stdgaminv(CDF_coarse,gamma_list[j])
            
            mean_s_gam_j = np.mean(s_gam_j)

            # linear regression:
            
            beta_coarse_list[j] = (np.sum(np.multiply(s_gam_j,X_coarse))-(n_coarse*mean_s_gam_j*mean_X_coarse))/(np.sum(np.power(s_gam_j,2))-(n_coarse*mean_s_gam_j**2))
            
            mu_coarse_list[j]=(mean_X_coarse - beta_coarse_list[j]*mean_s_gam_j)

            #Probability Plot Correlation Coefficient:
           
            # gam_PPCC_list[j] = (beta_coarse_list[j]*np.std(s_gam_j)/std_X_coarse) # Original
            gam_PPCC_list[j] = (beta_coarse_list[j]*np.std(s_gam_j,ddof=1)/std_X_coarse)

            X_coarse_fit_j = mu_coarse_list[j] + beta_coarse_list[j]*s_gam_j

            if gam_PPCC_list[j] == max(gam_PPCC_list):
                gam = gamma_list[j]
                gam_PPCC_max = gam_PPCC_list[j]
            else:
                break
        
        if gam_PPCC_list[n_start-1] < gam_PPCC_list[n_start]:
            # if the PPCC decreased with decreasing gamda, try increasing gamma: 
            for j in np.arange(n_start+1,n_gamma):
                count += 1
                # Obtain the Gamma Distribution Parameters for current gamma:
                
                s_gam_j = stdgaminv(CDF_coarse,gamma_list[j])   # standard variate
                mean_s_gam_j = np.mean(s_gam_j)
                # linear regression:
                beta_coarse_list[j] = (np.sum(np.multiply(s_gam_j,X_coarse))-(n_coarse*mean_s_gam_j*mean_X_coarse))/(np.sum(np.power(s_gam_j,2))-(n_coarse*mean_s_gam_j**2))
                
                mu_coarse_list[j] = mean_X_coarse - beta_coarse_list[j]*mean_s_gam_j
                #Probability Plot Correlation Coefficient:
                # gam_PPCC_list[j] = beta_coarse_list[j]* np.std(s_gam_j)/std_X_coarse # Original
                gam_PPCC_list[j] = beta_coarse_list[j]* np.std(s_gam_j, ddof=1)/std_X_coarse
                X_coarse_fit_j = mu_coarse_list[j] + beta_coarse_list[j]*s_gam_j

                if gam_PPCC_list[j] == max(gam_PPCC_list):
                    gam = gamma_list[j]
                    gam_PPCC_max = gam_PPCC_list[j]
                else:
                    break

        s_gam = stdgaminv(CDF_X,gam)
        mean_s_gam = np.mean(s_gam)

        beta = (np.sum(np.multiply(s_gam,sort_X))-n*mean_s_gam*mean_X)/(np.sum(np.power(s_gam,2))-n*mean_s_gam**2) #0.12
        mu = mean_X - beta*mean_s_gam
        # gam_PPCC = beta*np.std(s_gam)/std_X # Original
        gam_PPCC = beta*np.std(s_gam, ddof=1)/std_X
        
        x_fit = mu +beta*s_gam

        # Obtain the Normal Distribution Parameters for lower portion of CDF

        CDF_split = 0.25
        f = interpolate.interp1d(CDF_X,sort_X)
        X_split = f(CDF_split)

        ind_low = np.where(sort_X<X_split)
        
        X_low = sort_X[ind_low]
        n_low = len(X_low)
        CDF_low = CDF_X[ind_low]

        s_norm_low = stdnorminv(CDF_low)
        mean_s_norm_low = np.mean(s_norm_low)
        mean_X_low = np.mean(X_low)

        # linear regression:
        
        sigma_low = (np.sum(np.multiply(s_norm_low,X_low))-n_low*mean_s_norm_low*mean_X_low)/(np.sum(np.power(s_norm_low,2))-n_low*mean_s_norm_low**2)
        
        mu_low=mean_X_low - sigma_low*mean_s_norm_low
        X_low_fit = mu_low +sigma_low*s_norm_low

        # Probability Plot Correlation Coefficient:

        # norm_PPCC = sigma_low*np.std(s_norm_low)/np.std(X_low) # Original
        norm_PPCC = sigma_low*np.std(s_norm_low, ddof=1)/np.std(X_low, ddof=1)

        X_u = np.mean(sort_X[np.where(abs(CDF_X-0.5) == min(abs(CDF_X-0.5)))])

        front = np.where(X[1:]>=X_u)
        back = np.where(X[0:-1]<X_u)
        
        Nupcross = len(set(front[0]) & set(back[0]))
        
        if Nupcross<100:
            print('The number of median upcrossings is low {}'.format(Nupcross))
            print('The record may be too short for accurate peak estimation.')
        
        y_pk = np.sqrt(2.0*np.log(np.divide(-dur_ratio*Nupcross,np.log(cdf_qnt))))
        
        CDF_y = stdnormcdf(y_pk)
        
        # Perform the mapping procedure to compute the CDF of largest peak for X(t) from y(t)

        X_max = stdgaminv(CDF_y,gam) * beta + mu
        #X_max+= + mu
            
        X_min = np.multiply(stdnorminv(1-CDF_y),sigma_low) + mu_low
        
        #X_min+=mu_low
        pdf_pk = np.multiply(np.multiply(-y_pk,cdf_qnt),np.log(cdf_qnt))

        if np.sign(skew_x)>0:
            max_qnt[i,:] = X_max # fill the ith row x_max contains the 
            min_qnt[i,:] = X_min
        else:
            max_qnt[i,:] = -X_min
            min_qnt[i,:] = -X_max

    return max_qnt, min_qnt

# =================================
# Predicted extreme of a gumbel type I distribution 

def predicted_extreme_gumbel(loc_param, scale_param, return_period):
    '''
    Coles eq. 3.4
        1-1/return_period = probability of non exceedance 
        (e.g. 1 - 1/50 = 98%)
    '''
    return loc_param - scale_param * np.log(-np.log(1-1/return_period))

# =================================
# model checking

def get_probplot(x, distribution_type, params, method_i, method_label, block_size):
    '''
    uses Filiben estimates for theoretical quantiles
    params = (shape, loc, scale) shape not provided if gumbel
    '''
    
    prob_fig = plt.figure('propability plots with block size ' + str(block_size))
    prob_fig.suptitle('propability plots with block size '+ str(block_size))
    ax_prob = prob_fig.add_subplot(2,2,method_i+1) 

    res = sps.probplot(x, sparams=params, dist = distribution_type, plot = ax_prob)

    ax_prob.set_title(method_label)
    txt = method_label + '\n'
    for param in params:
        txt += str(round(param,3)) + '\n'

    txt_box = AnchoredText(txt, loc = 2)
    ax_prob.add_artist(txt_box)

def get_title(series):

    m = np.mean(series)
    s = np.std(series)
    sk = sps.skew(series)

    t = ' mean: ' + str(round(m,2)) + ' std: ' + str(round(s,2)) + ' skew: ' + str(round(sk,2))
    return t

def plot_series_raw(s0, s1, s2, bins = 100):
    '''
    s0, s1 normal series
    s2 skewed
    '''
    fig = plt.figure(figsize=(7.5,1.5))
    ax1 = plt.subplot(1,3,1)
    ax2 = plt.subplot(1,3,2)
    ax3 = plt.subplot(1,3,3)
    #ax4 = plt.subplot(2,2,4)

    x,y = get_pdf_normal_scipy(s0)
    ax1.plot(x,y)
    ax1.set_title(get_title(s0))
    ax1.hist(s0, bins = bins, density = True, color = 'lightgray')

    x,y = get_pdf_normal_scipy(s1)
    ax2.plot(x,y)
    ax2.hist(s1, bins = bins, density = True, color = 'lightgray')
    ax2.set_title(get_title(s1))
    
    x,y = get_pdf_skewnorm(s2)
    ax3.plot(x,y)
    ax3.hist(s2, bins = bins, density = True, color = 'lightgray')
    ax3.set_title(get_title(s2))

    # gev
    # x,y = get_pdf_gev_scipy(s3)
    # ax4.plot(x,y)
    # ax4.hist(s3, bins = bins, density = True, color = 'lightgray')
    # ax4.set_title(get_title(s3))

    plt.show()
