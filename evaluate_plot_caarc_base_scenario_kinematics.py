import os
from os.path import join as os_join
import json

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec



# ===============================================================================

# custom plot definitions
# return cm from inch


def cm2inch(value):
    return value / 2.54


# paper size for a4 landscape
height_9 = cm2inch(9)
height_21 = cm2inch(21)
width_16 = cm2inch(16)

# custom rectangle size for figure layout
cust_rect = [0.05, 0.05, 0.95, 0.95]

# custom set up for the lines in plots 
LINE_TYPE_SETUP = {"color":          ["grey", "black", "red", "green", "blue", "magenta"],
                   "linestyle":      ["solid",    "dashed",  "dashdot",    "dotted",   "(offset,on-off-dash-seq)",   ":"],
                   "marker":         ["o",    "s",  "^",    "p",   "x", "*"],
                   "markeredgecolor": ["grey", "black", "red", "green", "blue", "magenta"],
                   "markerfacecolor": ["grey", "black", "red", "green", "blue", "magenta"],
                   "markersize":     [4,      4,    4,      4,    4,    4]}
# direct input
plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]

# options
# for customizing check https://matplotlib.org/users/customizing.html
plot_settings =  {
            'no_turb': {
                'disp': {'x_lim': [-0.1,0.35],
                         'y_lim': [-0.3,0.3]},
                'acc': {'x_lim': [-0.35,0.35],
                        'y_lim': [-0.35,0.35]},
                'spec_disp': {'x_lim': [0,0],
                         'y_lim': [10**-6,0.9]},
                'spec_acc': {'x_lim': [0,0],
                        'y_lim': [10**-9,0.9]}
            },
            'turb': {
                'disp': {'x_lim': [-0.1, 0.35],
                         'y_lim': [-2.5, 2.5]},
                'acc': {'x_lim': [-0.4, 0.4],
                        'y_lim': [-4, 4]},
                'spec_disp': {'x_lim': [0,0],
                         'y_lim': [10**-6,3]},
                'spec_acc': {'x_lim': [0,0],
                        'y_lim': [10**-9,3]}
            }
        }

params = {
    'text.usetex': True,
    'font.size': 6,
    'font.family': 'lmodern',
    # 'text.latex.unicode': True,
    'figure.titlesize': 8,
    'figure.figsize': (width_16, height_9),
    'figure.dpi': 300,
    # 'figure.constrained_layout.use': True,
    # USE with the suplot_tool() to check which settings work the best
    'figure.subplot.left': 0.1,
    'figure.subplot.bottom': 0.15,
    'figure.subplot.right': 0.9,
    'figure.subplot.top': 0.9,
    'figure.subplot.wspace': 0.20,
    'figure.subplot.hspace': 0.30,
    #
    'axes.titlesize': 8,
    'axes.titlepad': 6,
    'axes.labelsize': 6,
    'axes.labelpad': 4,
    'axes.grid': 'True',
    'axes.grid.which': 'both',
    'axes.xmargin': 0.1,
    'axes.ymargin': 0.1,
    'lines.linewidth': 0.5,
    'lines.markersize': 5,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'ytick.minor.visible': False,
    'xtick.minor.visible': False,
    'grid.linestyle': '-',
    'grid.linewidth': 0.25,
    'grid.alpha': 0.5,
    'legend.fontsize': 6,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight'
}
plt.rcParams.update(params)

offset_factor = 1.05
pad_factor = 0.15


# ===============================================================================
def get_fft(given_series, sampling_freq):
    '''
    The function get_fft estimates the Fast Fourier transform of the given signal 
    '''

    signal_length = len(given_series)

    freq_half = np.arange(0,
                          sampling_freq/2 - sampling_freq/signal_length + sampling_freq/signal_length,
                          sampling_freq/signal_length)

    # single sided fourier
    series_fft = np.fft.fft(given_series-np.mean(given_series))
    series_fft = np.abs(
        series_fft[0:int(np.floor(signal_length/2))])/np.floor(signal_length/2)

    max_length = len(freq_half)
    if max_length < len(series_fft):
        max_length = len(series_fft)

    freq_half = freq_half[:max_length-1]
    series_fft = series_fft[:max_length-1]

    return freq_half, series_fft


def get_ramp_up_index(times_series, ramp_up_time):
    return np.where(times_series >= ramp_up_time + ramp_up_time / 5)[0][0]


def get_rms(given_series):
    return np.sqrt(np.mean(given_series**2))

# Parametric run - base scenario 
case_pre = 'CaarcBeam'
case_mid = ['Cont']
case_suf = {'no_turb': ['90'],
            'turb': ['90']}
subplot_title = {'Cont' :'Continuous beam ',
                'Int' :'With intervals ',
                'IntOut' :'With outrigger '}
considered_cases = [60]
damping_to_plot = ['0.000','0.025']
ramp_up_time = 0 #30 * 1.5

# result directory 
if not os.path.isdir('output/Caarc/consolidated_plots'):
    os.mkdir('output/Caarc/consolidated_plots')
if not os.path.isdir('output/Caarc/consolidated_plots/base_scenario'):
    os.mkdir('output/Caarc/consolidated_plots/base_scenario')
output_dir = 'output/Caarc/consolidated_plots/base_scenario'


for key, value in case_suf.items():
    # acceleration and displacement 
    fig_kin = plt.figure(figsize= (width_16,height_21))
    fig_kin.suptitle('Top floor responses - mgnitude with time (in plan)')
    n_row = len(case_mid)
    n_col = len(value)
    gs = gridspec.GridSpec(4, 2)
    ax_kin = [[fig_kin.add_subplot(gs[i, j]) for i in range(4)]
                for j in range(2)]
    
    # acceleration and displacement time histories for damping 0.025 ( base scenario) 
    fig_time_hist = plt.figure(figsize= (width_16,height_9))
    fig_time_hist.suptitle('Top floor responses with time')
    gs = gridspec.GridSpec(2, 2)
    ax_time_hist = [[fig_time_hist.add_subplot(gs[i, j]) for i in range(2)]
                for j in range(2)]
    
     # force and  moment for damping 0.025 ( base scenario) 
    fig_time_hist_force = plt.figure(figsize=(width_16,height_9))
    fig_time_hist_force.suptitle('Reaction forces at bottom with time')
    gs = gridspec.GridSpec(2, 2)
    ax_time_hist_force = [[fig_time_hist_force.add_subplot(gs[i, j]) for i in range(2)]
                for j in range(2)]
    # displacement and spectra plotting
    fig_spectra = plt.figure(figsize=(width_16,height_9))
    fig_spectra.suptitle('Spectra of top floor responses')
    gs = gridspec.GridSpec(2, 3)
    ax_spec = [[fig_spectra.add_subplot(gs[i, j]) for i in range(2)]
                    for j in range(3)]

    # plotting and processing the data 

    for i_damping_ratio, damping_ratio in enumerate(['0.000', '0.01', '0.025', '0.05']):

        working_folder = os_join(*['output',
                                    'Caarc',
                                    'spatial',
                                    key,
                                    damping_ratio.replace('.', '_')])
        for i in range(n_row):
            for j in range(n_col):
                # kinetic_ energy 
                kin_en = []
                    # Kinematics disp-acc-xy plot
                max_val = {'disp_x':[], 'disp_y':[],'disp_magn':[],
                    'acc_x':[], 'acc_y':[],'acc_magn':[]}
                rms_val = {'disp_x':[], 'disp_y':[],'disp_magn':[],
                    'acc_x':[], 'acc_y':[],'acc_magn':[]}
                for cc in considered_cases:

                    file_name = case_pre + case_mid[i] + value[j]
                   
                    # read in time
                    time_series = np.loadtxt(os_join(*[working_folder,file_name, 
                                                       str(cc), 'dynamic_analysis_result_displacement_for_dof_-4.dat']), usecols=(0,))
                    # read in displacement
                    disp_y = np.loadtxt(os_join(*[working_folder,file_name, 
                                                  str(cc), 'dynamic_analysis_result_displacement_for_dof_-4.dat']), usecols=(1,))
                    disp_x = np.loadtxt(os_join(*[working_folder,file_name, 
                                                  str(cc), 'dynamic_analysis_result_displacement_for_dof_-5.dat']), usecols=(1,))
                    # read in acceleration
                    acc_y = np.loadtxt(os_join(*[working_folder,file_name, 
                                                 str(cc), 'dynamic_analysis_result_acceleration_for_dof_-4.dat']), usecols=(1,))
                    acc_x = np.loadtxt(os_join(*[working_folder,file_name, 
                                                 str(cc), 'dynamic_analysis_result_acceleration_for_dof_-5.dat']), usecols=(1,))
                    # read in reaction force
                    force_x = np.loadtxt(os_join(*[working_folder,file_name, 
                                                  str(cc), 'dynamic_analysis_result_reaction_for_dof_1.dat']), usecols=(1,))
                    force_y = np.loadtxt(os_join(*[working_folder,file_name, 
                                                  str(cc), 'dynamic_analysis_result_reaction_for_dof_2.dat']), usecols=(1,))
                    # read in reaction moments
                    moment_x = np.loadtxt(os_join(*[working_folder,file_name, 
                                                 str(cc), 'dynamic_analysis_result_reaction_for_dof_4.dat']), usecols=(1,))
                    moment_y = np.loadtxt(os_join(*[working_folder,file_name, 
                                                 str(cc), 'dynamic_analysis_result_reaction_for_dof_5.dat']), usecols=(1,))                                             
                    # evaluate displacement
                    for ii in range(len(time_series)):
                        if ii == 0:
                            disp_max = [disp_x[ii], disp_y[ii], time_series[ii]]
                            val = (disp_x[ii]**2 + disp_y[ii]**2)**0.5
                        else:
                            if (disp_x[ii]**2 + disp_y[ii]**2)**0.5 > val:
                                disp_max = [disp_x[ii],
                                            disp_y[ii], time_series[ii]]
                                val = (disp_x[ii]**2 + disp_y[ii]**2)**0.5
                    max_val['disp_magn'].append(val)
                    max_val['disp_x'].append(max(disp_x))
                    max_val['disp_y'].append(max(disp_y))
                    # displacement spectra
                    fft_x, fft_y = get_fft(disp_y[get_ramp_up_index(
                        time_series, ramp_up_time):], 1/(time_series[1]-time_series[0]))
                    rms_val['disp_y'].append(get_rms(fft_y))

                    if damping_ratio in damping_to_plot: 
                        ax_spec[1][0].set_title('RMS of ' + '{:.4f} '.format(
                            get_rms(fft_y)) + 'dis-across flow')
                        ax_spec[1][0].loglog(fft_x, fft_y,
                            color=LINE_TYPE_SETUP["color"][i_damping_ratio],
                            linestyle=LINE_TYPE_SETUP["linestyle"][i_damping_ratio])
                        ax_spec[1][0].set_ylim(plot_settings[str(key)]['spec_disp']['y_lim'])
                        ax_spec[1][0].set_yticklabels([])
                        ax_spec[1][0].grid(b=True,which='minor')
                        ax_spec[1][0].tick_params(axis="x",direction="in")
                        ax_spec[1][0].tick_params(axis="y",direction="in")
                        #                     
                    fft_x, fft_y = get_fft(disp_x[get_ramp_up_index(
                        time_series, ramp_up_time):], 1/(time_series[1]-time_series[0]))
                    rms_val['disp_x'].append(get_rms(fft_y))

                    if damping_ratio in damping_to_plot: 
                        ax_spec[0][0].set_title('RMS of ' + '{:.4f} '.format(
                            get_rms(fft_y)) + 'dis-along flow')
                        ax_spec[0][0].loglog(fft_x, fft_y,
                            color=LINE_TYPE_SETUP["color"][i_damping_ratio],
                            linestyle=LINE_TYPE_SETUP["linestyle"][i_damping_ratio])
                        ax_spec[0][0].set_ylim(plot_settings[str(key)]['spec_disp']['y_lim'])
                        ax_spec[0][0].grid(b=True,which='minor')
                        ax_spec[0][0].tick_params(axis="x",direction="in")
                        ax_spec[0][0].tick_params(axis="y",direction="in")

                    disp_magn = [(x**2 + y**2)**0.5 for x,
                                 y in zip(disp_x, disp_y)]
                    fft_x, fft_y = get_fft(disp_magn[get_ramp_up_index(
                        time_series, ramp_up_time):], 1/(time_series[1]-time_series[0]))
                    rms_val['disp_magn'].append(get_rms(fft_y))
                    if damping_ratio in damping_to_plot: 
                        ax_spec[2][0].set_title('RMS of ' + '{:.4f} '.format(
                            get_rms(fft_y)) + 'dis-mag')
                        ax_spec[2][0].loglog(fft_x, fft_y,
                            color=LINE_TYPE_SETUP["color"][i_damping_ratio],
                            linestyle=LINE_TYPE_SETUP["linestyle"][i_damping_ratio],label=damping_ratio)
                        ax_spec[2][0].legend()
                        ax_spec[2][0].set_ylim(plot_settings[str(key)]['spec_disp']['y_lim'])
                        ax_spec[2][0].set_yticklabels([])
                        ax_spec[2][0].grid(b=True,which='minor')
                        ax_spec[2][0].tick_params(axis="x",direction="in")
                        ax_spec[2][0].tick_params(axis="y",direction="in")

                    
                    # evaluate acceleration
                    for ii in range(len(time_series)):
                        if ii == 0:
                            acc_max = [acc_x[ii], acc_y[ii], time_series[ii]]
                            val = (acc_x[ii]**2 + acc_y[ii]**2)**0.5
                        else:
                            if (acc_x[ii]**2 + acc_y[ii]**2)**0.5 > val:
                                acc_max = [acc_x[ii], acc_y[ii], time_series[ii]]
                                val = (acc_x[ii]**2 + acc_y[ii]**2)**0.5
                    max_val['acc_magn'].append(val)
                    max_val['acc_x'].append(max(acc_x))
                    max_val['acc_y'].append(max(acc_y))

                    # acceleration spectra
                    fft_x, fft_y = get_fft(acc_y[get_ramp_up_index(
                        time_series, ramp_up_time):], 1/(time_series[1]-time_series[0]))
                    rms_val['acc_y'].append(get_rms(fft_y))

                    if damping_ratio in damping_to_plot: 
                        ax_spec[1][1].set_title('RMS of ' + '{:.4f} '.format(
                            get_rms(fft_y)) + 'acc-across flow')
                        ax_spec[1][1].loglog(fft_x, fft_y,
                            color=LINE_TYPE_SETUP["color"][i_damping_ratio],
                            linestyle=LINE_TYPE_SETUP["linestyle"][i_damping_ratio])
                        ax_spec[1][1].set_ylim(plot_settings[str(key)]['spec_acc']['y_lim'])
                        ax_spec[1][1].set_yticklabels([])
                        ax_spec[1][1].grid(b=True,which='minor')
                        ax_spec[1][1].tick_params(axis="x",direction="in")
                        ax_spec[1][1].tick_params(axis="y",direction="in")

                    fft_x, fft_y = get_fft(acc_x[get_ramp_up_index(
                        time_series, ramp_up_time):], 1/(time_series[1]-time_series[0]))
                    rms_val['acc_x'].append(get_rms(fft_y))

                    if damping_ratio in damping_to_plot: 
                        ax_spec[0][1].set_title('RMS of ' + '{:.4f} '.format(
                            get_rms(fft_y)) + 'acc-along flow')
                        ax_spec[0][1].loglog(fft_x, fft_y,
                            color=LINE_TYPE_SETUP["color"][i_damping_ratio],
                            linestyle=LINE_TYPE_SETUP["linestyle"][i_damping_ratio])
                        ax_spec[0][1].set_ylim(plot_settings[str(key)]['spec_acc']['y_lim'])
                        ax_spec[0][1].grid(b=True,which='minor')
                        ax_spec[0][1].tick_params(axis="x",direction="in")
                        ax_spec[0][1].tick_params(axis="y",direction="in")
                       
                    # calculate the rms of acceleration 
                    acc_magn = [(x**2 + y**2)**0.5 for x,
                                y in zip(acc_x, acc_y)]
                    fft_x, fft_y = get_fft(acc_magn[get_ramp_up_index(
                        time_series, ramp_up_time):], 1/(time_series[1]-time_series[0]))
                    rms_val['acc_magn'].append(get_rms(fft_y))
                    if damping_ratio in damping_to_plot: 
                        ax_spec[2][1].set_title('RMS of ' + '{:.4f} '.format(
                            get_rms(fft_y)) + 'acc-mag')
                        ax_spec[2][1].loglog(fft_x, fft_y,
                            color=LINE_TYPE_SETUP["color"][i_damping_ratio],
                            linestyle=LINE_TYPE_SETUP["linestyle"][i_damping_ratio])
                        ax_spec[2][1].set_ylim(plot_settings[str(key)]['spec_acc']['y_lim'])
                        ax_spec[2][1].set_yticklabels([])
                        ax_spec[2][1].grid(b=True,which='minor')
                        ax_spec[2][1].tick_params(axis="x",direction="in")
                        ax_spec[2][1].tick_params(axis="y",direction="in")

                # plotting the displacement values
                ax_kin[0][i_damping_ratio].set_title('Max value of ' + '{:.3f}'.format(
                        max_val['disp_magn'][0]) + ' [m] at ' + '{:.3f}'.format(disp_max[-1]) + ' [s]')
                ax_kin[0][i_damping_ratio].plot(disp_x[get_ramp_up_index(
                        time_series, ramp_up_time):], disp_y[get_ramp_up_index(
                        time_series, ramp_up_time):],label=damping_ratio,
                color=LINE_TYPE_SETUP["color"][i_damping_ratio],
                linestyle=LINE_TYPE_SETUP["linestyle"][i_damping_ratio])
                ax_kin[0][i_damping_ratio].tick_params(axis="x",direction="in")
                ax_kin[0][i_damping_ratio].tick_params(axis="y",direction="in")
                ax_kin[0][i_damping_ratio].set_ylabel('Disp [m]')
                ax_kin[0][i_damping_ratio].set_xlim(plot_settings[str(key)]['disp']['x_lim'])
                ax_kin[0][i_damping_ratio].set_ylim(plot_settings[str(key)]['disp']['y_lim'])
                if i_damping_ratio == 3 : 
                    ax_kin[0][i_damping_ratio].set_xlabel('Disp [m]')
                else: 
                    ax_kin[0][i_damping_ratio].set_xticklabels([])
                
                # plottign the acceleration values
                ax_kin[1][i_damping_ratio].set_title('Max value of ' + '{:.3f}'.format(
                        max_val['acc_magn'][0]) + ' [m/s2] at ' + '{:.3f}'.format(acc_max[-1]) + ' [s]')
                ax_kin[1][i_damping_ratio].plot(acc_x[get_ramp_up_index(
                        time_series, ramp_up_time):], acc_y[get_ramp_up_index(
                        time_series, ramp_up_time):],label=damping_ratio,
                color=LINE_TYPE_SETUP["color"][i_damping_ratio],
                linestyle=LINE_TYPE_SETUP["linestyle"][i_damping_ratio])
                ax_kin[1][i_damping_ratio].tick_params(axis="x",direction="in")
                ax_kin[1][i_damping_ratio].tick_params(axis="y",direction="in")
                ax_kin[1][i_damping_ratio].set_ylabel('Acc [m/s2]')
                ax_kin[1][i_damping_ratio].set_xlim(plot_settings[str(key)]['acc']['x_lim'])
                ax_kin[1][i_damping_ratio].set_ylim(plot_settings[str(key)]['acc']['y_lim'])
                #ax_kin[1][i_damping_ratio].axis('equal')
                ax_kin[1][i_damping_ratio].legend()
                if i_damping_ratio == 3 : 
                    ax_kin[1][i_damping_ratio].set_xlabel('Acc [m/s2]')
                else: 
                    ax_kin[1][i_damping_ratio].set_xticklabels([])
                
                if damping_ratio in damping_to_plot: 
                    # plotting the displacement timehistories 
                    ax_time_hist[0][0].set_title('Displacement in along the flow direction')
                    ax_time_hist[0][0].plot(time_series[get_ramp_up_index(
                            time_series, ramp_up_time):], disp_x[get_ramp_up_index(
                            time_series, ramp_up_time):],label=damping_ratio,
                    color=LINE_TYPE_SETUP["color"][i_damping_ratio],
                    linestyle=LINE_TYPE_SETUP["linestyle"][i_damping_ratio])
                    ax_time_hist[0][0].tick_params(axis="x",direction="in")
                    ax_time_hist[0][0].tick_params(axis="y",direction="in")
                    ax_time_hist[0][0].set_ylabel('Disp [m]')
                    ax_time_hist[0][0].set_xticklabels([])
                    
                    ax_time_hist[0][1].set_title('Displacement in across the flow direction')
                    ax_time_hist[0][1].plot(time_series[get_ramp_up_index(
                            time_series, ramp_up_time):], disp_y[get_ramp_up_index(
                            time_series, ramp_up_time):],label=damping_ratio,
                    color=LINE_TYPE_SETUP["color"][i_damping_ratio],
                    linestyle=LINE_TYPE_SETUP["linestyle"][i_damping_ratio])
                    ax_time_hist[0][1].tick_params(axis="x",direction="in")
                    ax_time_hist[0][1].tick_params(axis="y",direction="in")
                    ax_time_hist[0][1].set_ylabel('Disp [m]')
                    ax_time_hist[0][1].set_xlabel('Time [s]')
                    
                    # plotting the acceleration timehistories 
                    ax_time_hist[1][0].set_title('Acceleration in along the flow direction')
                    ax_time_hist[1][0].plot(time_series[get_ramp_up_index(
                            time_series, ramp_up_time):], acc_x[get_ramp_up_index(
                            time_series, ramp_up_time):],label=damping_ratio,
                    color=LINE_TYPE_SETUP["color"][i_damping_ratio],
                    linestyle=LINE_TYPE_SETUP["linestyle"][i_damping_ratio])
                    ax_time_hist[1][0].tick_params(axis="x",direction="in")
                    ax_time_hist[1][0].tick_params(axis="y",direction="in")
                    ax_time_hist[1][0].set_ylabel('Acc [m/s2]')
                    ax_time_hist[1][0].set_xticklabels([])
                    ax_time_hist[1][0].legend()
                    
                    ax_time_hist[1][1].set_title('Acceleration in across the flow direction')
                    ax_time_hist[1][1].plot(time_series[get_ramp_up_index(
                            time_series, ramp_up_time):], acc_y[get_ramp_up_index(
                            time_series, ramp_up_time):],label=damping_ratio,
                    color=LINE_TYPE_SETUP["color"][i_damping_ratio],
                    linestyle=LINE_TYPE_SETUP["linestyle"][i_damping_ratio])
                    ax_time_hist[1][1].tick_params(axis="x",direction="in")
                    ax_time_hist[1][1].tick_params(axis="y",direction="in")
                    ax_time_hist[1][1].set_ylabel('Acc [m/s2]')
                    ax_time_hist[1][1].set_xlabel('Time [s]')

                    # plotting the reaction force timehistories 
                    ax_time_hist_force[0][0].set_title('Reaction force in along the flow direction')
                    ax_time_hist_force[0][0].plot(time_series[get_ramp_up_index(
                            time_series, ramp_up_time):], force_x[get_ramp_up_index(
                            time_series, ramp_up_time):] / 10**6 ,label=damping_ratio,
                    color=LINE_TYPE_SETUP["color"][i_damping_ratio],
                    linestyle=LINE_TYPE_SETUP["linestyle"][i_damping_ratio])
                    ax_time_hist_force[0][0].tick_params(axis="x",direction="in")
                    ax_time_hist_force[0][0].tick_params(axis="y",direction="in")
                    ax_time_hist_force[0][0].set_ylabel('Fx [x 10**6 N]')
                    ax_time_hist_force[0][0].set_xticklabels([])
                    
                    ax_time_hist_force[0][1].set_title('Reaction force in across the flow direction')
                    ax_time_hist_force[0][1].plot(time_series[get_ramp_up_index(
                            time_series, ramp_up_time):], force_y[get_ramp_up_index(
                            time_series, ramp_up_time):] / 10**6 ,label=damping_ratio,
                    color=LINE_TYPE_SETUP["color"][i_damping_ratio],
                    linestyle=LINE_TYPE_SETUP["linestyle"][i_damping_ratio])
                    ax_time_hist_force[0][1].tick_params(axis="x",direction="in")
                    ax_time_hist_force[0][1].tick_params(axis="y",direction="in")
                    ax_time_hist_force[0][1].set_ylabel('Fy [x 10**6 N]')
                    ax_time_hist_force[0][1].set_xlabel('Time [s]')

                    # plotting the reaction moment timehistories 
                    ax_time_hist_force[1][0].set_title('Base moment  in along the flow direction')
                    ax_time_hist_force[1][0].plot(time_series[get_ramp_up_index(
                            time_series, ramp_up_time):], moment_x[get_ramp_up_index(
                            time_series, ramp_up_time):] / 10**9 ,label=damping_ratio,
                    color=LINE_TYPE_SETUP["color"][i_damping_ratio],
                    linestyle=LINE_TYPE_SETUP["linestyle"][i_damping_ratio])
                    ax_time_hist_force[1][0].tick_params(axis="x",direction="in")
                    ax_time_hist_force[1][0].tick_params(axis="y",direction="in")
                    ax_time_hist_force[1][0].set_ylabel('Mx [x 10**9 Nm]')
                    ax_time_hist_force[1][0].set_xticklabels([])
                    ax_time_hist_force[1][0].legend()
                    
                    ax_time_hist_force[1][1].set_title('Base moment in across the flow direction')
                    ax_time_hist_force[1][1].plot(time_series[get_ramp_up_index(
                            time_series, ramp_up_time):], moment_y[get_ramp_up_index(
                            time_series, ramp_up_time):]/ 10**9,label=damping_ratio,
                    color=LINE_TYPE_SETUP["color"][i_damping_ratio],
                    linestyle=LINE_TYPE_SETUP["linestyle"][i_damping_ratio])
                    ax_time_hist_force[1][1].tick_params(axis="x",direction="in")
                    ax_time_hist_force[1][1].tick_params(axis="y",direction="in")
                    ax_time_hist_force[1][1].set_ylabel('My [x 10**9 Nm]')
                    ax_time_hist_force[1][1].set_xlabel('Time [s]')

    fig_kin.savefig(
        os_join(output_dir, 'kinematic_summary_consolidated_spatial_'+str(key)+'.png'))
    plt.close(fig=fig_kin)
    print('kinematic_summary_consolidated_spatial_'+str(key)+'.png'+ 'file created')
    
    fig_time_hist.savefig(
        os_join(output_dir, 'time_history_kinematic_summary_consolidated_spatial_'+str(key)+'.png'))
    plt.close(fig=fig_time_hist)
    print('time_history_kinematic_summary_consolidated_spatial_'+str(key)+'.png'+ 'file created')


    fig_time_hist_force.savefig(
        os_join(output_dir, 'time_history_reaction_summary_consolidated_spatial_'+str(key)+'.png'))
    plt.close(fig=fig_time_hist_force)
    print('time_history_reaction_summary_consolidated_spatial_'+str(key)+'.png'+ 'file created')

    fig_spectra.savefig(
        os_join(output_dir, 'spectra_summary_consolidated_spatial_'+str(key)+'.png'))
    plt.close(fig=fig_spectra)
    print('spectra_summary_consolidated_spatial_'+str(key)+'.png'+ 'file created')


print('plotting Caarc parametric base results consolidated finished!')


