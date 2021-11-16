import numpy as np
from scipy import signal
from scipy.optimize import minimize, minimize_scalar
from functools import partial
import datetime
import matplotlib.pyplot as plt


def calculate_length_scale_criaciv_unipd(x, fs, u_mean, LL0, nperseg, freq_range, test_mode=True):
    '''
    x - array like time series: here of velocity
    fs - sampling frequency
    u_mean - mean velocity -> Ref global or of point?!
    LL0 - tentative value for L_u for a first guess of the optimizer
    nperseg - a tentativ value for windo size -> choose array length?!
    ver - as a switch between 1 and 2 to test implementations
    '''
    t_start = datetime.datetime.now()

    # PSD estimation via Welch's method
    def pwelch():
        n, S = signal.welch(x,
                            fs,
                            # deprecated
                            #window=signal.hanning(nperseg, True),
                            window=signal.windows.hann(nperseg, True),
                            scaling='density',
                            nperseg=nperseg,
                            noverlap=nperseg/2,
                            nfft=nperseg,
                            detrend='linear',
                            return_onesided=True
                            )

        # adimensional spectrum
        Sad = n * S / np.var(x)

        return(Sad, n)

    Sad, n = pwelch()

    id_min = np.where(n <= freq_range[0])[0][-1]
    id_max = np.where(n > freq_range[1])[0][0]

    Sad = Sad[id_min:id_max]
    n = n[id_min:id_max]

    # plot dimensionless spectrum - which serves as target
    fig = plt.figure()
    plt.title('Target spectrum for optimization')
    plt.loglog(n, Sad)
    # plt.show()

    # NOTE ver 1 and ver 2 seem to deliver almost identical results

    # if ver == 1:
    #     # original version from GF

    #     # NOTE: valid if assuming a von Karman spectrum
    #     # this has in the definition the turbulence length scale LL
    #     def my_func(Sad, n, u_mean, LL):
    #         # Von Karman spectrum
    #         my_func = sum((Sad - (4 * n * LL / u_mean) *
    #                  ((1 + 70.8 * ((n * LL / u_mean) ** (2)))**(-5 / 6)))**2)
    #         return my_func

    #     def opt_fun(LL): return my_func(Sad, n, u_mean, LL)
    #     res = minimize(fun=opt_fun, x0=LL0, tol=1e-6,)

    #     # scalar optimization so use ...[0]
    #     L = (res.x)[0]
    # elif ver == 2:
    # modified verion  by MP

    def my_func(Sad, n, u_mean, LL):
        # Von Karman spectrum
        target_series = Sad
        current_series = (4 * n * LL / u_mean) / \
            ((1 + 70.8 * ((n * LL / u_mean) ** 2))**(5/6))

        # fig = plt.figure()
        # plt.loglog(n, current_series)
        # plt.show()

        return np.sum(np.power(100 * np.subtract(target_series, current_series), 2))

    # using partial to fix some parameters for the
    optimizable_function = partial(my_func, Sad, n, u_mean)

    minimization_result = minimize_scalar(optimizable_function,
                                          tol=1e-6,
                                          method='Bounded',
                                          bounds=(10, 500),
                                          options={'disp': True})

    # returning only one value!
    L = minimization_result.x

    if test_mode:
        ll_series = np.linspace(1, 500, num=500)
        func_eval = []
        for ll in np.nditer(ll_series):
            func_eval.append(my_func(Sad, n, u_mean, ll))

        # plot objective function which is optimized
        # one can see that it is ill conditioned as it has an asymptote...
        fig = plt.figure()
        plt.title('Objective function for optimization')
        plt.plot(ll_series, func_eval)
        # plt.show()
        print()

    # return(L, n, Sad)
    t_end = datetime.datetime.now()
    t_calc = (t_end-t_start)

    return {'autocorr_full': None, 'autocorr_trunc': None, 'turb_len': L, 'calc_time': t_calc.microseconds}


# def calculate_length_scale_statik(time_series, velocity_series, target_lux=[80.0, 100.0, 120.0]):
def calculate_length_scale_statik(time_series, velocity_series):
    '''
    Spectral length for target autocorrelation
    specified by default
    '''
    t_start = datetime.datetime.now()

    t = time_series
    dt = t[1] - t[0]
    # time shift to start from 0 the autocorrelation results
    t = t - t[0]

    u = velocity_series

    umean = np.mean(u)
    u = u - umean
    nx = len(u)
    ur = np.array([u[i] for i in range(nx - 1, -1, -1)])
    u = np.hstack((u, u[:-1]))
    # it is the correlation of velocity with itself - so autocorrelation
    r_uu = np.convolve(ur, u, mode='valid') / nx
    # r/r[0] represents the normalized autocorrelation of velocity
    r_uu = r_uu / r_uu[0]

    loc = np.argmax(r_uu < 0)

    r_uu_trunc = r_uu[:loc]

    r_uu_integral = 0
    for val in r_uu_trunc:
        r_uu_integral += val * dt

    # TODO: should this be u_mean for point
    # or u_ref_mean = 40 m/s for all points?
    L = r_uu_integral * umean

    # results['target'] = {}

    # for tl in target_lux:
    #     f1 = np.exp(-0.822 * (umean * t / tl)**0.77)
    #     target_autocorr = 0.5 * (f1 + f1**2)

    #     results['target']["Luu={:5.1f}".format(tl)] = target_autocorr

    t_end = datetime.datetime.now()
    t_calc = (t_end-t_start)

    return {'autocorr_total': r_uu, 'autocorr_trunc': r_uu_trunc, 'turb_len': L, 'calc_time': t_calc.microseconds}


def calculate_length_scale_uwo(u, uav, dt):
    """
    Calculates the length scale of a velocity time history given.

    u - times series of velocity
    uav - reference velocity
    dt - time step

    """
    t_start = datetime.datetime.now()

    u = u - np.mean(u)

    corr = signal.correlate(u, u, mode='full', method='auto')

    u_std = np.std(u)

    corr = corr[int(len(corr)/2):]/(u_std**2*len(u))

    loc = np.argmax(corr < 0)

    corr_trunc = corr[:loc]

    # TODO: should this be uav = u_mean for point
    # or uav = u_ref_mean = 40 m/s for all points?
    L = uav*np.trapz(corr_trunc, dx=dt)

    t_end = datetime.datetime.now()
    t_calc = (t_end-t_start)

    return {'autocorr_total': corr, 'autocorr_trunc': corr_trunc, 'turb_len': L, 'calc_time': t_calc.microseconds}


def compare_lux_methods(t_series, vel_series, v_ref, lux_target, freq_range):

    u_mean = np.mean(vel_series)
    u_fluct = vel_series - u_mean

    dt = t_series[1] - t_series[0]
    fs = 1 / dt

    # introducing a factor to see the effect of window length
    win_len_fctr = 1.0
    # needs an int convert so that welch can use it
    win_len = int(len(vel_series) / win_len_fctr)

    results = {}
    results['statik'] = calculate_length_scale_statik(t_series, vel_series)

    results['uwo'] = calculate_length_scale_uwo(vel_series, u_mean, dt)

    results['criacivunipd'] = calculate_length_scale_criaciv_unipd(
        u_fluct, fs, u_mean, lux_target, win_len, freq_range)

    return results
