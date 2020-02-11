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
width = cm2inch(29.7)
height = cm2inch(21)

# custom rectangle size for figure layout
cust_rect = [0.05, 0.05, 0.95, 0.95]

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
    series_fft = np.fft.fft(given_series)
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

# ===============================================================================


# ==============================================
# Model choice

# NOTE: all currently available files
parametric_runs = {
    'input/force/caarc/0_turb/force_dynamic_0_turb': {
        'plot_settings': {
            '0_000': {
                'disp': {'x_lim': [-2.5, 4.0],
                         'y_lim': [-3.25, 3.25]},
                'acc': {'x_lim': [-4.0, 4.0],
                        'y_lim': [-4.0, 4.0]},
            },
            '0_025': {
                'disp': {'x_lim': [-0.4, 0.8],
                         'y_lim': [-0.6, 0.6]},
                'acc': {'x_lim': [-0.65, 0.65],
                        'y_lim': [-0.65, 0.65]},
            },
        },
        'output_folder_prefix': 'turb',
        'project_params':
            ['ProjectParameters3DCaarcBeamCont0.json',
             'ProjectParameters3DCaarcBeamInt0.json',
             'ProjectParameters3DCaarcBeamIntOut0.json']},
    'input/force/caarc/45_turb/force_dynamic_45_turb': {
        'plot_settings': {
            '0_000': {
                'disp': {'x_lim': [-2.5, 4.0],
                         'y_lim': [-3.25, 3.25]},
                'acc': {'x_lim': [-4.0, 4.0],
                        'y_lim': [-4.0, 4.0]},
            },
            '0_025': {
                'disp': {'x_lim': [-0.4, 0.8],
                         'y_lim': [-0.6, 0.6]},
                'acc': {'x_lim': [-0.65, 0.65],
                        'y_lim': [-0.65, 0.65]},
            },
        },
        'output_folder_prefix': 'turb',
        'project_params':
            ['ProjectParameters3DCaarcBeamCont45.json',
             'ProjectParameters3DCaarcBeamInt45.json',
             'ProjectParameters3DCaarcBeamIntOut45.json']},
    'input/force/caarc/90_turb/force_dynamic_90_turb': {
        'plot_settings': {
            '0_000': {
                'disp': {'x_lim': [-2.5, 4.0],
                         'y_lim': [-3.25, 3.25]},
                'acc': {'x_lim': [-4.0, 4.0],
                        'y_lim': [-4.0, 4.0]},
            },
            '0_025': {
                'disp': {'x_lim': [-0.4, 0.8],
                         'y_lim': [-0.6, 0.6]},
                'acc': {'x_lim': [-0.65, 0.65],
                        'y_lim': [-0.65, 0.65]},
            },
        },
        'output_folder_prefix': 'turb',
        'project_params':
            ['ProjectParameters3DCaarcBeamCont90.json',
             'ProjectParameters3DCaarcBeamInt90.json',
             'ProjectParameters3DCaarcBeamIntOut90.json']},
    'input/force/caarc/90_no_turb/force_dynamic_90_no_turb': {
        'plot_settings': {
            '0_000': {
                'disp': {'x_lim': [-0.2, 0.6],
                         'y_lim': [-0.4, 0.4]},
                'acc': {'x_lim': [-0.5, 0.5],
                        'y_lim': [-0.5, 0.5]},
            },
            '0_025': {
                'disp': {'x_lim': [-0.1, 0.3],
                         'y_lim': [-0.2, 0.2]},
                'acc': {'x_lim': [-0.2, 0.2],
                        'y_lim': [-0.2, 0.2]},
            },
        },
        'output_folder_prefix': 'no_turb',
        'project_params':
            ['ProjectParameters3DCaarcBeamCont90.json',
             'ProjectParameters3DCaarcBeamInt90.json',
             'ProjectParameters3DCaarcBeamIntOut90.json']}}

# ==============================================
# Parametric run

considered_cases = [[1, 2, 3],
                    [15, 30, 60]]

ramp_up_time = 30 * 1.5

for damping_ratio in ['0.000', '0.025']:
    for load_file, parametric_run in parametric_runs.items():

        for available_model in parametric_run['project_params']:

            #wait = input("check...")

            # ==============================================
            # Parameter read
            with open(os_join(*['input', 'parameters', 'caarc', available_model]), 'r') as parameter_file:
                parameters = json.loads(parameter_file.read())

            working_folder = os_join(*['output',
                                       'Caarc',
                                       parametric_run['output_folder_prefix'],
                                       damping_ratio.replace('.', '_'),
                                       parameters['model_parameters']['name']])

            # ==============================================
            # Kinematics disp-acc-xy plot

            for i_cc, cc in enumerate(considered_cases):

                fig = plt.figure(i_cc)
                n_row = len(cc)
                n_col = 2
                gs = gridspec.GridSpec(n_row, n_col)
                ax = [[fig.add_subplot(gs[i, j]) for i in range(n_row)]
                      for j in range(n_col)]

                for n_row, n_el in enumerate(cc):

                    # read in time
                    time_series = np.loadtxt(os_join(*[working_folder,
                                                       str(n_el), 'dynamic_analysis_result_displacement_for_dof_-4.dat']), usecols=(0,))

                    # read in displacement
                    disp_y = np.loadtxt(os_join(*[working_folder,
                                                  str(n_el), 'dynamic_analysis_result_displacement_for_dof_-4.dat']), usecols=(1,))
                    disp_x = np.loadtxt(os_join(*[working_folder,
                                                  str(n_el), 'dynamic_analysis_result_displacement_for_dof_-5.dat']), usecols=(1,))

                    # evaluate displacement
                    for i in range(len(time_series)):
                        if i == 0:
                            disp_max = [disp_x[i], disp_y[i], time_series[i]]
                            val = (disp_x[i]**2 + disp_y[i]**2)**0.5
                        else:
                            if (disp_x[i]**2 + disp_y[i]**2)**0.5 > val:
                                disp_max = [disp_x[i],
                                            disp_y[i], time_series[i]]
                                val = (disp_x[i]**2 + disp_y[i]**2)**0.5

                    # plot displacement
                    ax[0][n_row].set_title('El ' + str(n_el) + ': Max value of ' + '{:.3f}'.format(
                        val) + ' [m] at ' + '{:.3f}'.format(disp_max[-1]) + ' [s]')
                    ax[0][n_row].plot(disp_x, disp_y, 'b--')
                    ax[0][n_row].plot([0.0, disp_max[0]], [
                                      0.0, disp_max[1]], 'r-')
                    ax[0][n_row].set_xlabel('Disp [m]')
                    ax[0][n_row].set_ylabel('Disp [m]')
                    ax[0][n_row].set_xlim(
                        parametric_run['plot_settings'][damping_ratio.replace('.', '_')]['disp']['x_lim'])
                    ax[0][n_row].set_ylim(
                        parametric_run['plot_settings'][damping_ratio.replace('.', '_')]['disp']['y_lim'])

                    # ax[0][n_row].quiver(
                    #     0.0, 0.0, (0.0 + disp_max[0]), (0.0 + disp_max[1]))

                    # read in acceleration
                    acc_y = np.loadtxt(os_join(*[working_folder,
                                                 str(n_el), 'dynamic_analysis_result_acceleration_for_dof_-4.dat']), usecols=(1,))
                    acc_x = np.loadtxt(os_join(*[working_folder,
                                                 str(n_el), 'dynamic_analysis_result_acceleration_for_dof_-5.dat']), usecols=(1,))

                    # evaluate acceleration
                    for i in range(len(time_series)):
                        if i == 0:
                            acc_max = [acc_x[i], acc_y[i], time_series[i]]
                            val = (acc_x[i]**2 + acc_y[i]**2)**0.5
                        else:
                            if (acc_x[i]**2 + acc_y[i]**2)**0.5 > val:
                                acc_max = [acc_x[i], acc_y[i], time_series[i]]
                                val = (acc_x[i]**2 + acc_y[i]**2)**0.5

                    # plot acceleration
                    ax[1][n_row].set_title('El ' + str(n_el) + ': Max value of ' + '{:.3f}'.format(
                        val) + ' [m/s2] at ' + '{:.3f}'.format(acc_max[-1]) + ' [s]')
                    ax[1][n_row].plot(acc_x, acc_y, 'b--')
                    ax[1][n_row].plot([0.0, acc_max[0]], [
                                      0.0, acc_max[1]], 'r-')
                    ax[1][n_row].set_xlabel('Acc [m/s2]')
                    ax[1][n_row].set_ylabel('Acc [m/s2]')
                    ax[1][n_row].set_xlim(
                        parametric_run['plot_settings'][damping_ratio.replace('.', '_')]['acc']['x_lim'])
                    ax[1][n_row].set_ylim(
                        parametric_run['plot_settings'][damping_ratio.replace('.', '_')]['acc']['y_lim'])

                    # ax[1][n_row].quiver(
                    #     0.0, 0.0, (0.0 + acc_max[0]), (0.0 + acc_max[1]))

                # gs.tight_layout(fig, rect=cust_rect)

                plt.savefig(
                    os_join(working_folder, 'comparison_disp_acc_xy_' + str(i_cc) + '.png'))

                print(os_join(working_folder, 'comparison_disp_acc_xy_' +
                              str(i_cc) + '.png') + ' file created')

            # ==============================================
            # Kinematics disp-acc-spectra plot

            # displacement
            fig = plt.figure()
            n_row = len(cc)
            n_col = 3
            gs = gridspec.GridSpec(n_row, n_col)
            ax = [[fig.add_subplot(gs[i, j]) for i in range(n_row)]
                  for j in range(n_col)]

            for n_row, n_el in enumerate(cc):

                # read in time
                time_series = np.loadtxt(os_join(*[working_folder,
                                                   str(n_el), 'dynamic_analysis_result_displacement_for_dof_-4.dat']), usecols=(0,))

                # read in and plot displacement y
                disp_y = np.loadtxt(os_join(*[working_folder,
                                              str(n_el), 'dynamic_analysis_result_displacement_for_dof_-4.dat']), usecols=(1,))
                fft_x, fft_y = get_fft(disp_y[get_ramp_up_index(
                    time_series, ramp_up_time):], 1/(time_series[1]-time_series[0]))

                ax[0][n_row].set_title('El ' + str(n_el) + ': RMS of ' + '{:.6f}'.format(
                    get_rms(fft_y)) + ' for dispy')
                ax[0][n_row].loglog(fft_x, fft_y, 'k-')

                # read in and plot displacement x
                disp_x = np.loadtxt(os_join(*[working_folder,
                                              str(n_el), 'dynamic_analysis_result_displacement_for_dof_-5.dat']), usecols=(1,))
                fft_x, fft_y = get_fft(disp_x[get_ramp_up_index(
                    time_series, ramp_up_time):], 1/(time_series[1]-time_series[0]))

                ax[1][n_row].set_title('El ' + str(n_el) + ': RMS of ' + '{:.6f}'.format(
                    get_rms(fft_y)) + ' for dispx')
                ax[1][n_row].loglog(fft_x, fft_y, 'k-')

                # calculate and plot displacement mang
                disp_magn = [(x**2 + y**2)**0.5 for x,
                             y in zip(disp_x, disp_y)]
                fft_x, fft_y = get_fft(disp_magn[get_ramp_up_index(
                    time_series, ramp_up_time):], 1/(time_series[1]-time_series[0]))

                ax[2][n_row].set_title('El ' + str(n_el) + ': RMS of ' + '{:.6f}'.format(
                    get_rms(fft_y)) + ' for dispmagn')
                ax[2][n_row].loglog(fft_x, fft_y, 'k-')

            #gs.tight_layout(fig, rect=cust_rect)

            plt.savefig(
                os_join(working_folder, 'comparison_disp_spectra_' + str(i_cc) + '.png'))

            print(os_join(working_folder, 'comparison_disp_spectra_' +
                          str(i_cc) + '.png') + ' file created')

            # acceleration
            fig = plt.figure()
            n_row = len(cc)
            n_col = 3
            gs = gridspec.GridSpec(n_row, n_col)
            ax = [[fig.add_subplot(gs[i, j]) for i in range(n_row)]
                  for j in range(n_col)]

            for n_row, n_el in enumerate(cc):

                # read in time
                time_series = np.loadtxt(os_join(*[working_folder,
                                                   str(n_el), 'dynamic_analysis_result_acceleration_for_dof_-4.dat']), usecols=(0,))

                # read in and plot acceleration y
                acc_y = np.loadtxt(os_join(*[working_folder,
                                             str(n_el), 'dynamic_analysis_result_acceleration_for_dof_-4.dat']), usecols=(1,))
                fft_x, fft_y = get_fft(acc_y[get_ramp_up_index(
                    time_series, ramp_up_time):], 1/(time_series[1]-time_series[0]))

                ax[0][n_row].set_title('El ' + str(n_el) + ': RMS of ' + '{:.6f}'.format(
                    get_rms(fft_y)) + ' for accy')
                ax[0][n_row].loglog(fft_x, fft_y, 'k-')

                # read in and plot acceleration x
                acc_x = np.loadtxt(os_join(*[working_folder,
                                             str(n_el), 'dynamic_analysis_result_acceleration_for_dof_-5.dat']), usecols=(1,))
                fft_x, fft_y = get_fft(acc_x[get_ramp_up_index(
                    time_series, ramp_up_time):], 1/(time_series[1]-time_series[0]))

                ax[1][n_row].set_title('El ' + str(n_el) + ': RMS of ' + '{:.6f}'.format(
                    get_rms(fft_y)) + ' for accx')
                ax[1][n_row].loglog(fft_x, fft_y, 'k-')

                # calculate and plot acceleration mang
                acc_magn = [(x**2 + y**2)**0.5 for x, y in zip(acc_x, acc_y)]
                fft_x, fft_y = get_fft(acc_magn[get_ramp_up_index(
                    time_series, ramp_up_time):], 1/(time_series[1]-time_series[0]))

                ax[2][n_row].set_title('El ' + str(n_el) + ': RMS of ' + '{:.6f}'.format(
                    get_rms(fft_y)) + ' for accmagn')
                ax[2][n_row].loglog(fft_x, fft_y, 'k-')

            #gs.tight_layout(fig, rect=cust_rect)
            plt.savefig(
                os_join(working_folder, 'comparison_acc_spectra_' + str(i_cc) + '.png'))

            print(os_join(working_folder, 'comparison_acc_spectra_' +
                          str(i_cc) + '.png') + ' file created')

print('Evaluate Caarc parametric results finished!')
