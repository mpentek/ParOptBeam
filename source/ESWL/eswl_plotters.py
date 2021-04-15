import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter

import source.auxiliary.global_definitions as GD

def cm2inch(value):
    return value/2.54

LINE_TYPE_SETUP = {"color":          ["grey", "black", "red", "green", "blue", "magenta", 'orange','gold','purple'],
                   "linestyle":      ["--",    "-",  "-",    "-",   "-",   "-"],
                   "marker":         ["o",    "s",  "^",    "p",   "x", "*"],
                   "markeredgecolor": ["grey", "black", "red", "green", "blue", "magenta"],
                   "markerfacecolor": ["grey", "black", "red", "green", "blue", "magenta"],
                   "markersize":     [4,      4,    4,      4,    4,    4]}
LINESTYLE = ["--",    "-.",  ":",    "-",   "-",   "-"]

# SAVE DESTINATION 
destination = 'plots\\ESWL_plots'

# global PLOT PARAMS
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
width = cm2inch(16)
height = cm2inch(11.0)

params = {
        # FIGURE
        'text.usetex': True,
        'font.size': 6,
        #'font.family': 'lmodern', -> not supported when usetex?!
        #'text.latex.unicode': True,
        'figure.titlesize': 8,
        'figure.figsize': (width, height),
        'figure.dpi': 300,
        'figure.constrained_layout.use': True,
        # SUBPLOTS
        # USE with the suplot_tool() to check which settings work the best
        'figure.subplot.left': 0.1,
        'figure.subplot.bottom': 0.15,
        'figure.subplot.right': 0.9,
        'figure.subplot.top': 0.8,
        'figure.subplot.wspace': 0.20,
        'figure.subplot.hspace': 0.30,
        # AXES
        'axes.titlesize': 8,
        'axes.titlepad': 6,
        'axes.labelsize': 6,
        'axes.labelpad': 4,
        'axes.grid': True,
        'axes.grid.which': 'both',
        'axes.xmargin': 0.1,
        'axes.ymargin': 0.1,
        # FORMATTER
        'axes.formatter.limits':(-3,3), #limit der tausender ab wann sie in potenzen dargestellt werden 
        #'axes.formatter.min_exponent': 4,
        # TICKS
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'ytick.minor.visible': False,
        'xtick.minor.visible': False,
        'grid.linestyle': '-',
        'grid.linewidth': 0.25,
        'grid.alpha': 0.5,
        'legend.fontsize': 4,
        # LINES
        'lines.linewidth': 0.5,
        'lines.markersize': 1,
        'lines.markerfacecolor': 'darkgrey',
        # TEXT
        'mathtext.default': 'regular',
        # SAVING
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        # ERRORBARS
        'errorbar.capsize': 2.0,
        
    }

#plt.rcParams.update(params)

def convert_for_latex(string):
    return '\_'.join(string.split('_'))

def plot_directional_load_components(eswl_total, nodal_coordinates ,load_components, response_label):
    '''
    gets the eswl dictionary and the structure nodal coordiantes dictionary 
    plots the total ESWL for each directional component (y,z,...)
    '''
    fig, ax = plt.subplots()

    for component in load_components:
        eswl = eswl_total[response_label][component]
        F_M = GD.RESPONSE_DIRECTION_MAP
        ax.plot(eswl, nodal_coordinates['x0'], label = component)

    ax.plot(nodal_coordinates['y0'], nodal_coordinates['x0'], 
                label = 'structure', 
                marker = 'o', 
                color = 'grey', 
                linestyle = '--')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0e'))
    ax.set_xlabel('load')
    ax.set_ylabel('height [m]')

    plt.title('ESWL for ' + response_label)
    plt.legend()
    plt.grid()
    plt.show()

def plot_eswl_components(eswl_components, nodal_coordinates, load_directions, response_label, textstr, influences ,components_to_plot = ['all']):
  
    
    # plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
    # plt.rcParams.update(params2)
    
    fig, ax = plt.subplots(1, len(load_directions), sharey=True)
    fig.canvas.set_window_title('for_'+response_label)

    if components_to_plot[0] == 'all':
        # taking the keys from the first direction
        components = eswl_components[response_label]['y'].keys()
    else:
        components = components_to_plot
    for i, direction in enumerate(load_directions):
        # for labels
        if direction in ['x','y','z']:
            unit = '[N]'
        else:
            unit = '[Nm]'
        # sturcutre and nodes
        ax[i].plot(nodal_coordinates['y0'], nodal_coordinates['x0'], 
                label = r'$structure$', 
                marker = 'o', 
                color = 'grey', 
                linestyle = '--')
        ax[i].set_title(r'nodal force ' + r'${}$'.format(GD.DIRECTION_LOAD_MAP[direction]))
        # plot each component
        for j, component in enumerate(components):
            eswl = eswl_components[response_label][direction][component]
            if component == 'total':
                line = '-'
                color = LINE_TYPE_SETUP['color'][i+2]
            else:
                line = LINESTYLE[j]
                color = LINE_TYPE_SETUP['color'][i+2]

            component_label = convert_for_latex(component)
            #my_label = r'${}_{{{}}}$'.format(GD.DIRECTION_LOAD_MAP[direction], component_label)
            my_label = r'${}$'.format(component_label)

            ax[i].plot(eswl, 
                        nodal_coordinates['x0'], 
                        label = my_label,
                        linestyle = line,
                        color = color)

        # calculate R for each direction -> to check if the signs are sensible
        R_i = sum(np.multiply(influences[response_label][direction], eswl_components[response_label][direction]['total']))
        ax[i].plot(0,0, 
                label = r'$R_i=$' + r'${:.2e}$'.format(round(R_i)),
                linestyle = ' ',
                color = 'k')

        # settings
        ax[i].locator_params(axis='x', nbins = 4)
        #ax[i].xaxis.set_major_formatter(FormatStrFormatter('%.0e'))
        ax[i].ticklabel_format(style='sci', axis='x', scilimits=(-3,3))
        # ax[i].xmargin = 0.5
        # ax[i].ymargin = 0.5
        ax[i].set_xlabel('load '+unit)
        ax[i].legend(fontsize = 12)
        ax[i].grid()
    ax[0].set_ylabel('height [m]')

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax[-1].text(-0.3, 0.12, textstr, transform=ax[-1].transAxes, fontsize=10,
            verticalalignment='top', bbox= props)

    fig.suptitle('ESWL for base ' + response_label, fontsize = 14)

    # saving of figure
    dest = destination
    component_folder_dict = {'mean':'mean_parts',
                             'background':'background_parts', 'lrc':'background_parts',
                             'resonant':'resonant_parts', 'resonant_m_lumped':'resonant_parts','resonant_m':'resonant_parts',
                             'all':'all_parts',
                             'mixed':'mixed_parts'}
    
    fname = 'for_'+ response_label
    if len(components_to_plot) == 1:
        key = components_to_plot[0]
    elif set(components_to_plot).issubset(['background', 'lrc']):
        key = components_to_plot[0]
        for comp in components_to_plot:
            fname += '_' + comp[0]
    elif set(components_to_plot).issubset(['resonant', 'resonant_m', 'resonant_m_lumped']):
        key = components_to_plot[0]
        fname += '_r'
        for comp in components_to_plot:
            s = comp.split('_')
            if len(s) == 2:
                fname += '_m'
            elif len(s) == 3:
                fname += '_l'
    else:
        key = 'mixed'
        for comp in components_to_plot:
            fname += '_' + comp[0]
        
    full_dest = dest + os.path.sep + component_folder_dict[key] + os.path.sep + fname
    #plt.savefig(full_dest)

    #fig.tight_layout()
    plt.show()

def plot_rho(list_of_rhos, response):
    fig, ax = plt.subplots(2,3, sharey=True)
    nodes = np.arange(len(list_of_rhos['y'][next(iter(list_of_rhos['y']))]))
    
    for d, direction in enumerate(list_of_rhos):
        rho_labels = []
        for i, rho in enumerate(list_of_rhos[direction]):
            rho_labels.append(rho)
            if direction in ['y','z']:
                row, col = 0, d
            else:
                row, col = 1, d-2
            if rho == 'rho_Rps_corrcoef': 
                l_st = '--'
            else: l_st = '-'
                 
            ax[row][col].plot(list_of_rhos[direction][rho], nodes, linestyle=l_st)
            ax[row][col].set_title('lrc coeff p'+ direction + ' - ' + response)
            ax[row][col].grid(True)

    ax[0][0].set_ylabel('node')
    ax[1][0].set_ylabel('node')
    for i in range(3):
        ax[1][i].set_xlabel('rho')

    for i in range(len(rho_labels)):
        ax[0][2].plot(0,0)    

    ax[0][2].legend(rho_labels)
    
    plt.show()

def plot_inluences(eswl_object):
    moment_load = {'y':'Mz', 'z':'My', 'a':'Mx','b':'My', 'g':'Mz'}
    shear_load = {'y':'Qy', 'z':'Qz'}
    fig, ax = plt.subplots(1, len(eswl_object.influences[eswl_object.response])-1,sharey=True)
    fig.canvas.set_window_title('for_' +  eswl_object.response)

    nodal_coordinates = eswl_object.structure_model.nodal_coordinates
    influences = eswl_object.influences[eswl_object.response]
    if eswl_object.response in ['Mx','My','Mz']:
        unit = '[Nm]'
    elif eswl_object.response in ['Qx','Qy','Qz']:
        unit = '[N]'
    for i, direction in enumerate(eswl_object.load_directions):

        ax[i].plot(nodal_coordinates['y0'], nodal_coordinates['x0'], 
                label = 'structure', 
                marker = 'o', 
                color = 'grey', 
                linestyle = '--')
        ax[i].plot(influences[direction], nodal_coordinates['x0'],
                    label = GD.DIRECTION_LOAD_MAP[direction] + '_Static_analysis',
                    color = LINE_TYPE_SETUP['color'][i+2])
        # expected simple influences
        if direction in ['y','z']:
            
            if moment_load[direction] == eswl_object.response:
                if direction == 'z' and eswl_object.response == 'My':
                    ax[i].plot(-nodal_coordinates['x0'], nodal_coordinates['x0'],
                            label = 'expected',
                            color = 'black',
                            linestyle= '--') 
                else:
                    ax[i].plot(nodal_coordinates['x0'], nodal_coordinates['x0'],
                                label = 'expected',
                                color = 'black',
                            linestyle= '--') 
            elif shear_load[direction] == eswl_object.response:
                ax[i].vlines(1.0, 0, nodal_coordinates['x0'][-1],
                            label = 'expected',
                            color = 'black',
                            linestyle= '--')
                ax[i].set_xlim(-0.1, 1.1)
            else:
                ax[i].plot(nodal_coordinates['x0'][0], 0,
                            label = 'expected 0',
                            color = 'black')        
        else:
            
            if moment_load[direction] == eswl_object.response:
                ax[i].vlines(1.0, 0, nodal_coordinates['x0'][-1],
                            label = 'expected',
                            color = 'black',
                            linestyle= '--')
                ax[i].set_xlim(-0.1, 1.1)
            else:
                ax[i].plot(nodal_coordinates['x0'][0], 0,
                            label = 'expected 0',
                            color = 'black',
                            linestyle= '--') 
        ax[i].locator_params(axis='x', nbins = 5)
        ax[i].set_xlabel('response '+ unit)
        ax[i].legend(fontsize = 8 )
        ax[i].grid()
        #ax[i].set_title(GD.DIRECTION_LOAD_MAP[direction] + ' on '+ eswl_object.response)
    ax[0].set_ylabel('height [m]')

    fig.suptitle('influence functions for ' + eswl_object.response + ' due to unit load at the respective nodes')
    
    plt.show()

# LOAD SIGNALS

def plot_load_time_histories(load_signals, nodal_coordinates):

    fig, ax = plt.subplots(1, len(load_signals))

    for i, component in enumerate(load_signals):
        if component == 'sample_freq':
            break
        # structure
        ax[i].set_title('load signal direction ' + component)
        ax[i].plot(nodal_coordinates['y0'],
                    nodal_coordinates['x0'], 
                    label = 'structure', 
                    marker = 'o', 
                    color = 'grey', 
                    linestyle = '--')
        for node in range(len(nodal_coordinates['x0'])):
            shift_scale = 1e+05
            if component == 'a':
                shift_scale *= 5
            ax[i].plot(np.arange(0,len(load_signals[component][node])),
                        load_signals[component][node] + nodal_coordinates['x0'][node]*shift_scale)
    
        #ax[i].legend()
        ax[i].grid()
    plt.show()

def plot_load_time_histories_node_wise(load_signals, n_nodes, discard_time, load_signal_labels = None):
    '''
    for each node it plots the time history of the loading 
    parameter: load_signal_labels 
                if not set all signals in the load_signal dict are plotted 
                can be specified with a list of labels to plot
    '''
    ax_i = list(range(n_nodes-1, -1, -1))
    if load_signal_labels:
        components = load_signal_labels
    else:
        load_signals.pop('sample_freq')
        components = load_signals.keys()
    for component in components:
        # if component == 'sample_freq':
        #     break
        fig, ax = plt.subplots(n_nodes, 1)
        fig.canvas.set_window_title('signals of ' +  GD.DIRECTION_LOAD_MAP[component] )
        fig.suptitle('signal of ' + GD.DIRECTION_LOAD_MAP[component])

        for node in range(n_nodes):
            # show also mean 
            mean = np.mean(load_signals[component][node])
            ax[ax_i[node]].hlines(mean, 0, len(load_signals[component][node]), 
                                  color = 'r', 
                                  linestyle = '--', 
                                  label = 'mean: ' + '{:.2e}'.format(mean))

            ax[ax_i[node]].plot(np.arange(0,len(load_signals[component][node])),
                                load_signals[component][node],
                                label = 'node_'+str(node))

            ax[ax_i[node]].vlines(discard_time, min(load_signals[component][node]), max(load_signals[component][node]), 
                                  color = 'grey', 
                                  linestyle = '--',
                                  label = 'discard')

            #ax[ax_i[node]].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
            ax[ax_i[node]].locator_params(axis='y', nbins = 5)
            ax[ax_i[node]].yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
            ax[ax_i[node]].grid()
            ax[ax_i[node]].legend()
        plt.show()


def plot_n_mode_shapes(mode_shapes_sorted, charact_length ,n = 3):

    fig, ax = plt.subplots(1, n)
    fig.canvas.set_window_title('mode shapes first 3 modes')
    fig.suptitle('mode shapes of first 3 modes - rotations multiplied with characteristic length')

    for mode in range(n):
        for d, dof_label in enumerate(mode_shapes_sorted):
            multiplier = 1.0
            if dof_label in ['a','b','g']:
                multiplier = charact_length
            ax[mode].set_title('mode ' + str(mode+1))

            ax[mode].plot(mode_shapes_sorted[dof_label][:,mode] * multiplier,
                            np.arange(len(mode_shapes_sorted[dof_label][:,mode])),
                            label = dof_label,
                            color = LINE_TYPE_SETUP['color'][2+d])

        ax[mode].legend()
        ax[mode].set_xlabel('deformation[m]')
        ax[mode].set_ylabel('node [-]')
        ax[mode].locator_params(axis='y', nbins = 3)
        ax[mode].xaxis.set_major_formatter(FormatStrFormatter('%.1e'))
        ax[mode].grid()
    
    plt.show()


