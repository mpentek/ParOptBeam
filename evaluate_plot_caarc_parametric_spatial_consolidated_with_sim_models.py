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
height = cm2inch(8)
width = cm2inch(14)

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
params = {
    'text.usetex': True,
    'font.size': 6,
    'font.family': 'lmodern',
    # 'text.latex.unicode': True,
    'figure.titlesize': 8,
    'figure.figsize': (width, height),
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

# Parametric run
case_pre = 'CaarcBeam'
case_mid = ['Cont', 'Int', 'IntOut']
case_suf = {'no_turb': ['45', '90'],
            'turb': ['0', '45', '90']}
subplot_title = {'Cont' :'Continuous beam ',
                'Int' :'With intervals ',
                'IntOut' :'With outrigger '}
considered_cases = [ 'sone', 'stwo', 1, 2, 3, 15, 30, 60]
considered_cases_xlabel = [1.25, 2.25, 1, 2, 3, 4, 5, 6]
xticks_labels = ['I', 'II', '1', '2', '3', '15', '30', '60']
considered_cases_simplified = ['sone','stwo']
cases_not_to_plot = [30,60]
no_no_plot = len(cases_not_to_plot)
index_sim_case = len(considered_cases_simplified)
ramp_up_time = 30 * 1.5

# result directory 
if not os.path.isdir('output/Caarc/consolidated_plots'):
    os.mkdir('output/Caarc/consolidated_plots')
if not os.path.isdir('output/Caarc/consolidated_plots/spatial_with_sim_model'):
    os.mkdir('output/Caarc/consolidated_plots/spatial_with_sim_model')
output_dir = 'output/Caarc/consolidated_plots/spatial_with_sim_model'


for key, value in case_suf.items():
    # kinetic energy
    fig_kin_energy = plt.figure()
    fig_kin_energy.suptitle('Kinetic energy with spatial reduction')
    n_row = len(case_mid)
    n_col = len(value)
    gs = gridspec.GridSpec(n_row, n_col)
    ax_kin_en = [[fig_kin_energy.add_subplot(gs[i, j]) for i in range(n_row)]
                for j in range(n_col)]
    # max displacement 
    fig_max_dis_mag = plt.figure()
    fig_max_dis_mag.suptitle('Maximum top floor displacement(magnitude) with spatial reduction')
    ax_dis_max = [[fig_max_dis_mag.add_subplot(gs[i, j]) for i in range(n_row)]
                for j in range(n_col)]
    # rms displacement 
    fig_rms_dis_mag = plt.figure()
    fig_rms_dis_mag.suptitle('RMS of top floor displacement(magnitude) with spatial reduction')
    ax_dis_rms = [[fig_rms_dis_mag.add_subplot(gs[i, j]) for i in range(n_row)]
                for j in range(n_col)]
    # max acceleration 
    fig_max_acc_mag = plt.figure()
    fig_max_acc_mag.suptitle('Maximum top floor acceleration(magnitude) with spatial reduction')
    ax_acc_max = [[fig_max_acc_mag.add_subplot(gs[i, j]) for i in range(n_row)]
                for j in range(n_col)]
    # rms acceleration 
    fig_rms_acc_mag = plt.figure()
    fig_rms_acc_mag.suptitle('RMS of top floor acceleration(magnitude) with spatial reduction')
    ax_acc_rms = [[fig_rms_acc_mag.add_subplot(gs[i, j]) for i in range(n_row)]
                for j in range(n_col)]
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
                    subplot_title_i = subplot_title.get(case_mid[i]) + value[j] + '°'
                    # kinetic energy 
                    with open(os_join(*[working_folder, 
                                        file_name, 
                                        str(cc), 
                                        'kinetic_energy.dat'])) as input_data:
                        input_data.seek(0)
                        my_value = float(input_data.readline().split()[-2])
                        kin_en.append(my_value)
                    
                    # needs special treatment
                    if cc == 'sone': 
                        # read in time
                        # time_series = np.loadtxt(os_join(*[working_folder,file_name, 
                        #                                 str(cc), 'dynamic_analysis_result_displacement_for_dof_-4.dat']), usecols=(0,))
                        time_series = np.loadtxt(os_join(*[working_folder,file_name, 
                                                        str(cc), 'dynamic_analysis_result_displacement_for_dof_displacement_xz_plane_top.dat']), usecols=(0,))

                        # read in displacement
                        # disp_y = np.loadtxt(os_join(*[working_folder,file_name, 
                        #                             str(cc), 'dynamic_analysis_result_displacement_for_dof_-4.dat']), usecols=(1,))
                        disp_y = np.loadtxt(os_join(*[working_folder,file_name, 
                                                    str(cc), 'dynamic_analysis_result_displacement_for_dof_displacement_xz_plane_top.dat']), usecols=(1,))
                        # disp_x = np.loadtxt(os_join(*[working_folder,file_name, 
                        #                             str(cc), 'dynamic_analysis_result_displacement_for_dof_-5.dat']), usecols=(1,))
                        disp_x = np.loadtxt(os_join(*[working_folder,file_name, 
                                                    str(cc), 'dynamic_analysis_result_displacement_for_dof_displacement_xy_plane_top.dat']), usecols=(1,))

                        # read in acceleration
                        # acc_y = np.loadtxt(os_join(*[working_folder,file_name, 
                        #                             str(cc), 'dynamic_analysis_result_acceleration_for_dof_-4.dat']), usecols=(1,))
                        acc_y = np.loadtxt(os_join(*[working_folder,file_name, 
                                                    str(cc), 'dynamic_analysis_result_acceleration_for_dof_acceleration_xz_plane_top.dat']), usecols=(1,))
                        # acc_x = np.loadtxt(os_join(*[working_folder,file_name, 
                        #                             str(cc), 'dynamic_analysis_result_acceleration_for_dof_-5.dat']), usecols=(1,))
                        acc_x = np.loadtxt(os_join(*[working_folder,file_name, 
                                                    str(cc), 'dynamic_analysis_result_acceleration_for_dof_acceleration_xy_plane_top.dat']), usecols=(1,))
                                            
                        # for all other cases naming is matching
                    else:
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
                    # calculate thr rms of displacement 
                    disp_magn = [(x**2 + y**2)**0.5 for x,
                                 y in zip(disp_x, disp_y)]
                    fft_x, fft_y = get_fft(disp_magn[get_ramp_up_index(
                        time_series, ramp_up_time):], 1/(time_series[1]-time_series[0]))

                    rms_val['disp_magn'].append(get_rms(fft_y))
                    # evaluate acceleration
                    for ii in range(len(time_series)):
                        if ii == 0:
                            acc_max = [acc_x[ii], acc_y[ii], time_series[ii]]
                            val = (acc_x[ii]**2 + acc_y[ii]**2)**0.5
                        else:
                            if (acc_x[ii]**2 + acc_y[ii]**2)**0.5 > val:
                                acc_max = [acc_x[ii], acc_y[ii], time_series[ii]]
                                val = (acc_x[ii]**2 + acc_y[ii]**2)**0.5
                    
                    # calculate the rms of acceleration 
                    acc_magn = [(x**2 + y**2)**0.5 for x,
                                y in zip(acc_x, acc_y)]
                    fft_x, fft_y = get_fft(acc_magn[get_ramp_up_index(
                        time_series, ramp_up_time):], 1/(time_series[1]-time_series[0]))
                    rms_val['acc_magn'].append(get_rms(fft_y))
                    max_val['acc_magn'].append(val)
                    max_val['acc_x'].append(max(acc_x))
                    max_val['acc_y'].append(max(acc_y))
                # plottign the kinetic energy values 
                
                xticks_values = considered_cases_xlabel# list(range(1, len(considered_cases) + 1))

                ax_kin_en[j][i].set_title(subplot_title_i)

                # sone and stwo
                ax_kin_en[j][i].plot(xticks_values[:index_sim_case], [x/kin_en[-1] for x in kin_en[:index_sim_case]],
                    color=LINE_TYPE_SETUP["color"][i_damping_ratio],
                    linestyle=LINE_TYPE_SETUP["linestyle"][i_damping_ratio],
                    marker=LINE_TYPE_SETUP["marker"][i_damping_ratio],
                    markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][i_damping_ratio],
                    markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][i_damping_ratio],
                    markersize=LINE_TYPE_SETUP["markersize"][i_damping_ratio])
       
                ax_kin_en[j][i].plot(xticks_values[index_sim_case:-no_no_plot], [x/kin_en[-1] for x in kin_en[index_sim_case:-no_no_plot]],label=damping_ratio,
                    color=LINE_TYPE_SETUP["color"][i_damping_ratio],
                    linestyle=LINE_TYPE_SETUP["linestyle"][i_damping_ratio],
                    marker=LINE_TYPE_SETUP["marker"][i_damping_ratio],
                    markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][i_damping_ratio],
                    markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][i_damping_ratio],
                    markersize=LINE_TYPE_SETUP["markersize"][i_damping_ratio])
                ax_kin_en[j][i].tick_params(axis="x",direction="in")
                ax_kin_en[j][i].tick_params(axis="y",direction="in")
                ax_kin_en[j][i].set_xticks(xticks_values[:-no_no_plot])                
                if i == n_row-1 : 
                    ax_kin_en[j][i].set_xticklabels(xticks_labels)
                    ax_kin_en[j][i].set_xlabel("No. of elements")
                else: 
                    ax_kin_en[j][i].set_xticklabels([])
                if i ==0 and j == n_col-1: 
                    ax_kin_en[j][i].legend()
                
                # plotting the displacement maximum values 
                ax_dis_max[j][i].set_title(subplot_title_i)
                
                # sone and stwo
                ax_dis_max[j][i].plot(xticks_values[:index_sim_case], [x/max_val['disp_magn'][-1] for x in max_val['disp_magn'][:index_sim_case]],
                    color=LINE_TYPE_SETUP["color"][i_damping_ratio],
                    linestyle=LINE_TYPE_SETUP["linestyle"][i_damping_ratio],
                    marker=LINE_TYPE_SETUP["marker"][i_damping_ratio],
                    markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][i_damping_ratio],
                    markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][i_damping_ratio],
                    markersize=LINE_TYPE_SETUP["markersize"][i_damping_ratio])
                
                ax_dis_max[j][i].plot(xticks_values[index_sim_case:-no_no_plot], [x/max_val['disp_magn'][-1] for x in max_val['disp_magn'][index_sim_case:-no_no_plot]],label=damping_ratio,
                    color=LINE_TYPE_SETUP["color"][i_damping_ratio],
                    linestyle=LINE_TYPE_SETUP["linestyle"][i_damping_ratio],
                    marker=LINE_TYPE_SETUP["marker"][i_damping_ratio],
                    markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][i_damping_ratio],
                    markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][i_damping_ratio],
                    markersize=LINE_TYPE_SETUP["markersize"][i_damping_ratio])
                ax_dis_max[j][i].tick_params(axis="x",direction="in")
                ax_dis_max[j][i].tick_params(axis="y",direction="in")
                
                ax_dis_max[j][i].set_xticks(xticks_values[:-no_no_plot])
                
                if i == n_row-1 : 
                    ax_dis_max[j][i].set_xticklabels(xticks_labels)
                    ax_dis_max[j][i].set_xlabel("No. of elements")
                else: 
                    ax_dis_max[j][i].set_xticklabels([])
                if i ==0 and j == n_col-1: 
                    ax_dis_max[j][i].legend()
                
                # plotting the displacement RMS values 
                
                ax_dis_rms[j][i].set_title(subplot_title_i)
                
                # sone and stwo
                ax_dis_rms[j][i].plot(xticks_values[:index_sim_case], [x/rms_val['disp_magn'][-1] for x in rms_val['disp_magn'][:index_sim_case]],
                    color=LINE_TYPE_SETUP["color"][i_damping_ratio],
                    linestyle=LINE_TYPE_SETUP["linestyle"][i_damping_ratio],
                    marker=LINE_TYPE_SETUP["marker"][i_damping_ratio],
                    markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][i_damping_ratio],
                    markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][i_damping_ratio],
                    markersize=LINE_TYPE_SETUP["markersize"][i_damping_ratio])
                
                ax_dis_rms[j][i].plot(xticks_values[index_sim_case:-no_no_plot], [x/rms_val['disp_magn'][-1] for x in rms_val['disp_magn'][index_sim_case:-no_no_plot]],label=damping_ratio,
                    color=LINE_TYPE_SETUP["color"][i_damping_ratio],
                    linestyle=LINE_TYPE_SETUP["linestyle"][i_damping_ratio],
                    marker=LINE_TYPE_SETUP["marker"][i_damping_ratio],
                    markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][i_damping_ratio],
                    markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][i_damping_ratio],
                    markersize=LINE_TYPE_SETUP["markersize"][i_damping_ratio])
                ax_dis_rms[j][i].tick_params(axis="x",direction="in")
                ax_dis_rms[j][i].tick_params(axis="y",direction="in")
                ax_dis_rms[j][i].set_xticks(xticks_values[:-no_no_plot])
                if i == n_row-1 : 
                    ax_dis_rms[j][i].set_xticklabels(xticks_labels)
                    ax_dis_rms[j][i].set_xlabel("No. of elements")
                else: 
                    ax_dis_rms[j][i].set_xticklabels([])
                if i ==0 and j == n_col-1: 
                    ax_dis_rms[j][i].legend()
                
                # plotting the acceleration maximum values 
                
                ax_acc_max[j][i].set_title(subplot_title_i)
                # sone and stwo
                ax_acc_max[j][i].plot(xticks_values[:index_sim_case], [x/max_val['acc_magn'][-1] for x in max_val['acc_magn'][:index_sim_case]],
                    color=LINE_TYPE_SETUP["color"][i_damping_ratio],
                    linestyle=LINE_TYPE_SETUP["linestyle"][i_damping_ratio],
                    marker=LINE_TYPE_SETUP["marker"][i_damping_ratio],
                    markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][i_damping_ratio],
                    markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][i_damping_ratio],
                    markersize=LINE_TYPE_SETUP["markersize"][i_damping_ratio])
                # rest until 15 el
                ax_acc_max[j][i].plot(xticks_values[index_sim_case:-no_no_plot], [x/max_val['acc_magn'][-1] for x in max_val['acc_magn'][index_sim_case:-no_no_plot]],label=damping_ratio,
                    color=LINE_TYPE_SETUP["color"][i_damping_ratio],
                    linestyle=LINE_TYPE_SETUP["linestyle"][i_damping_ratio],
                    marker=LINE_TYPE_SETUP["marker"][i_damping_ratio],
                    markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][i_damping_ratio],
                    markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][i_damping_ratio],
                    markersize=LINE_TYPE_SETUP["markersize"][i_damping_ratio])
                ax_acc_max[j][i].tick_params(axis="x",direction="in")
                ax_acc_max[j][i].tick_params(axis="y",direction="in")
                ax_acc_max[j][i].set_xticks(xticks_values[:-no_no_plot])
                if i == n_row-1 : 
                    ax_acc_max[j][i].set_xticklabels(xticks_labels)
                    ax_acc_max[j][i].set_xlabel("No. of elements")
                else: 
                    ax_acc_max[j][i].set_xticklabels([])
                if i ==0 and j == n_col-1: 
                    ax_acc_max[j][i].legend()
                
                # plotting the acceleration rms values 
                ax_acc_rms[j][i].set_title(subplot_title_i)
                
                # sone and stwo
                ax_acc_rms[j][i].plot(xticks_values[:index_sim_case], [x/rms_val['acc_magn'][-1] for x in rms_val['acc_magn'][:index_sim_case]],
                    color=LINE_TYPE_SETUP["color"][i_damping_ratio],
                    linestyle=LINE_TYPE_SETUP["linestyle"][i_damping_ratio],
                    marker=LINE_TYPE_SETUP["marker"][i_damping_ratio],
                    markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][i_damping_ratio],
                    markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][i_damping_ratio],
                    markersize=LINE_TYPE_SETUP["markersize"][i_damping_ratio])
                
                # rest until 15 el
                ax_acc_rms[j][i].plot(xticks_values[index_sim_case:-no_no_plot], [x/rms_val['acc_magn'][-1] for x in rms_val['acc_magn'][index_sim_case:-no_no_plot]],label=damping_ratio,
                    color=LINE_TYPE_SETUP["color"][i_damping_ratio],
                    linestyle=LINE_TYPE_SETUP["linestyle"][i_damping_ratio],
                    marker=LINE_TYPE_SETUP["marker"][i_damping_ratio],
                    markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][i_damping_ratio],
                    markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][i_damping_ratio],
                    markersize=LINE_TYPE_SETUP["markersize"][i_damping_ratio])
                ax_acc_rms[j][i].tick_params(axis="x",direction="in")
                ax_acc_rms[j][i].tick_params(axis="y",direction="in")
                
                ax_acc_rms[j][i].set_xticks(xticks_values[:-no_no_plot])
                if i == n_row-1 : 
                    ax_acc_rms[j][i].set_xticklabels(xticks_labels)
                    ax_acc_rms[j][i].set_xlabel("No. of elements")
                else: 
                    ax_acc_rms[j][i].set_xticklabels([])
                if i ==0 and j == n_col-1: 
                    ax_acc_rms[j][i].legend()

    fig_kin_energy.savefig(
        os_join(output_dir, 'kin_en_summary_consolidated_spatial_'+str(key)+'.png'))
    plt.close(fig=fig_kin_energy)
    print('kin_en_summary_consolidated_spatial_'+str(key)+'.png'+ 'file created')

    fig_max_dis_mag.savefig(
        os_join(output_dir, 'max_dis_magnitude_summary_consolidated_spatial_'+str(key)+'.png'))
    plt.close(fig=fig_max_dis_mag)
    print('max_dis_magnitude_summary_consolidated_spatial_'+str(key)+'.png'+ 'file created')

    fig_rms_dis_mag.savefig(
        os_join(output_dir, 'rms_dis_magnitude_summary_consolidated_spatial_'+str(key)+'.png'))
    plt.close(fig=fig_rms_dis_mag)
    print('rms_dis_magnitude_summary_consolidated_spatial_'+str(key)+'.png'+ 'file created')

    fig_max_acc_mag.savefig(
        os_join(output_dir, 'max_acc_magnitude_summary_consolidated_spatial_'+str(key)+'.png'))
    plt.close(fig=fig_max_acc_mag)
    print('max_acc_magnitude_summary_consolidated_spatial_'+str(key)+'.png'+ 'file created')

    fig_rms_acc_mag.savefig(
        os_join(output_dir, 'rms_acc_magnitude_summary_consolidated_spatial_'+str(key)+'.png'))
    plt.close(fig=fig_rms_acc_mag)
    print('rms_acc_magnitude_summary_consolidated_spatial_'+str(key)+'.png'+ 'file created')
print('plotting Caarc parametric spatial results consolidated finished!')


