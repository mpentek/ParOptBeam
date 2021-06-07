import numpy as np
from math import ceil
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.ticker import FormatStrFormatter
from os.path import join as os_join

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from source.auxiliary import global_definitions as GD
from source.auxiliary.other_utilities import get_signed_maximum
from source.auxiliary.auxiliary_functionalities import get_fitted_array
from source.auxiliary.auxiliary_functionalities import cm2inch


'''
Everything should boil down to 2 main visualizaiton types:
(1) "Static" (so not animated) image
(2) "Dynamic" (so animated) image

-> for both of these use a gray dashed line for the undeformed configurations
-> use with continuous black lines the deformed configuration
-> use dot marker for nodes
-> have an adjustable scaling factor for deformation
-> have an adjustable scaling factor for the given external forces
-> add the base reaction with the same factor as for forces
-> for forces use quiver plot


use version (1) for plotting 
    last static solve results 
    chosen eigenform and frequency
    a chosen time step for the dynamic simulation

use version (2) for animating 
    chosen eigenform and frequency
    the total duration of a dynamic simulation

'''

# only a maximum number of line plots is available
# these are formatted for : undeformed, (primary) deformed 0, (other) deformed 1, 2, 3
LINE_TYPE_SETUP = {"color":          ["grey", "black", "red", "green", "blue", "magenta"],
                   "linestyle":      ["--",    "-",  "-",    "-",   "-",   "-"],
                   "marker":         ["o",    "s",  "^",    "p",   "x", "*"],
                   "markeredgecolor": ["grey", "black", "red", "green", "blue", "magenta"],
                   "markerfacecolor": ["grey", "black", "red", "green", "blue", "magenta"],
                   "markersize":     [4,      4,    4,      4,    4,    4]}
LINESTYLE = ["--",    "-.",  ":",    "-",   "-",   "-"]

LEGEND_SETUP = {"fontsize": [16, 14, 12, 10, 8]}

width = cm2inch(5)
height = cm2inch(3)
parameters = {'text.usetex': False,

          'font.size': 10,

          'font.family': 'sans-serif',

          'text.latex.unicode': False,

          'figure.titlesize': 10,

          'figure.figsize': (width, height),

          'figure.dpi': 300,

          'axes.titlesize': 10,

          'axes.labelsize': 10,

          'axes.grid': 'True',

          'axes.grid.which': 'both',

          'axes.xmargin': 0.05,

          'axes.ymargin': 0.05,

          'lines.linewidth': 0.5,

          'lines.linestyle': '--',

          'lines.markersize': 3,

          'xtick.labelsize': 10,

          'ytick.labelsize': 10,

          'ytick.minor.visible': 'true',

          'xtick.minor.visible': 'true',

          'grid.linestyle': '-',

          'grid.linewidth': 0.5,

          'grid.alpha': 0.3,

          'legend.fontsize': 10,

          'savefig.dpi': 300,

          'savefig.format': 'pdf',

          'savefig.bbox': 'tight'

          }


'''
geometry = {"undeformed":...,
            "deformation":...,
            "deformed": None}

where deformed = undeformed + deformation is done with these utilities 
taking scaling into consideration

geometry needs to contain the additional nodal information(s) when passed 
to the plot function

force = {"external":...,
         "base_reaction":...}
scaling_factor = {"deformation":...
                  "force":...}

plot_limits = {"x":[... , ...]
               "y":[... , ...]}
# defined based upon geometry["deformed"]
'''


def plot_properties(pdf_report, display_plot, plot_title, struct_property_data, plot_legend, plot_style):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(plot_title)

    for idx in range(len(struct_property_data)):
        ax.plot(struct_property_data[idx]['x'], struct_property_data[idx]
                ['y'], plot_style[idx], label=plot_legend[idx])
    ax.legend()

    if pdf_report is not None:
        pdf_report.savefig()
        plt.close(fig)

    if display_plot:
        plt.show()

def plot_result_2D(pdf_report, display_plot, plot_title, geometry, force, scaling):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    geometry["deformed"] = [geometry["deformation"][0] * scaling["deformation"] + geometry["undeformed"][0], # x
                            geometry["deformation"][1] * scaling["deformation"] + geometry["undeformed"][1], # y
                            geometry["deformation"][2] * scaling["deformation"] + geometry["undeformed"][2], # z
                            geometry["deformation"][3] * scaling["deformation"] + geometry["undeformed"][1]] # visualize twist a in 2D plane

      # set title
    deflections, undeformed, min_deformation, max_deformation = [], [], [], []
    for dof, deformation in enumerate(geometry['deformed'][1:]): # without axial deformation
        min_deformation.append(min(deformation))
        max_deformation.append(max(deformation))
        for value in deformation:
            if value != 0.0:
                deflections.append(dof+1) # contains 1,2 or 3 if coupled motion multiple 
                break 

    # plot undeformed
    ax.plot(geometry["undeformed"][1], # y
        geometry["undeformed"][0], # x
        label='undeformed',
        color=LINE_TYPE_SETUP["color"][0],
        linestyle=LINE_TYPE_SETUP["linestyle"][0],
        marker=LINE_TYPE_SETUP["marker"][0],
        markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][0],
        markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][0],
        markersize=LINE_TYPE_SETUP["markersize"][0])
    # deformed
    for i in range(1,4):
        maximum_deformation = max(abs(geometry['deformed'][i]))
        minimum_deformation = min(geometry['deformed'][i])
        # if not np.any(geometry['deformed'][i]):
        #     maximum_deformation = ' is 0'
        label_dof = GD.DOF_LABELS['3D'][i]
        if label_dof == 'a':
                label_dof = '\u03B1'
        ax.plot(geometry["deformed"][i], 
                geometry["deformed"][0], # x
                label= 'deformation in ' + label_dof + ' |max|: ' +  '{0:.2e}'.format(maximum_deformation),
                color=LINE_TYPE_SETUP["color"][2+i],
                linestyle=LINE_TYPE_SETUP["linestyle"][1],
                marker=LINE_TYPE_SETUP["marker"][1],
                markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][1],
                markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][1],
                markersize=LINE_TYPE_SETUP["markersize"][1]) 


    ax.set_ylabel('height x - coord')
    ax.set_xlabel('deflection')
    offset =  max(max_deformation)/50
    ax.set_xlim(min(min_deformation)-offset, max(max_deformation)+offset)

    ax.legend(loc = 'lower right')
    plt.grid()
    plt.title(plot_title)  
    plt.legend()
    if display_plot:
        plt.show()
        display_plot = False
    
def plot_result_2D_multiple_modes(display_plot, plot_titles, geometry, scaling, number_of_modes):
    fig, axes = plt.subplots(1, number_of_modes)

    geometry["deformed"] = [geometry["deformation"][0] * scaling["deformation"] + geometry["undeformed"][0][:, np.newaxis], # x
                            geometry["deformation"][1] * scaling["deformation"] + geometry["undeformed"][1][:, np.newaxis], # y
                            geometry["deformation"][2] * scaling["deformation"] + geometry["undeformed"][2][:, np.newaxis], # z
                            geometry["deformation"][3] * scaling["deformation"] + geometry["undeformed"][1][:, np.newaxis]] # visualize twist a in 2D plane

    for i in range(number_of_modes):
        # plot undeformed
        axes[i].plot(geometry["undeformed"][1], # y
                    geometry["undeformed"][0], # x
                    label='undeformed',
                    color=LINE_TYPE_SETUP["color"][0],
                    linestyle=LINE_TYPE_SETUP["linestyle"][0],
                    marker=LINE_TYPE_SETUP["marker"][0],
                    markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][0],
                    markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][0],
                    markersize=LINE_TYPE_SETUP["markersize"][0])
        max_deformations, min_deformations = [], []
        for j in range(1,4): #plot first three dofs
            max_deformations.append(max(geometry['deformed'][j][:, i]))
            min_deformations.append(min(geometry['deformed'][j][:, i]))
            absolute_maximum = max(abs(geometry['deformed'][j][:, i]))

            print ('mode ', j, 'direction ', GD.DOF_LABELS['3D'][j], geometry["deformed"][j][:, i])

            label_dof = GD.DOF_LABELS['3D'][j]
            if label_dof == 'a':
                label_dof = '\u03B1'

            axes[i].plot(geometry["deformed"][j][:, i], 
                        geometry["deformed"][0][:, i], # x
                        label= label_dof + ' |max|: ' +  '{0:.2e}'.format(absolute_maximum),
                        color=LINE_TYPE_SETUP["color"][2+j],
                        linestyle=LINE_TYPE_SETUP["linestyle"][1],
                        marker=LINE_TYPE_SETUP["marker"][1],
                        markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][1],
                        markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][1],
                        markersize=LINE_TYPE_SETUP["markersize"][1]) 
            axes[i].set_title(plot_titles[i])
        print('')
        axes[i].set_xlim(min(min_deformations)-5e-5, max(max_deformations)+5e-5)
        axes[i].xaxis.set_major_formatter(FormatStrFormatter('%.0e'))
        axes[i].legend(loc = 'lower right', fontsize = LEGEND_SETUP['fontsize'][3])
        axes[i].grid()

    if display_plot:
        plt.show()
        display_plot = False

def plot_result(pdf_report, display_plot, plot_title, geometry, force, scaling, n_data):

    # default parameter
    # if reaction_force is None:
    #     reaction_force = np.zeros(1)
    # if force is None:
    #     force = np.zeros((len(displacement), n_data))
    #     force_scaling_factor=0.0

    # TEST GEOMETRY
    # print('TESTING GEOMETRY:')
    # print(geometry)
    # Set up figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')  # fig.add_subplot(111)

    # Handover data
    # x_undef = np.zeros(len(undeformed_geometry))
    # y_undef = undeformed_geometry
    # x_def = displacement*displacement_scaling_factor
    # x_force = force*force_scaling_factor

    # TODO: avoid try and except
    # use matrices straightforward
    try:
        # single / static case
        geometry["deformed"] = [geometry["deformation"][0] * scaling["deformation"] + geometry["undeformed"][0],
                                geometry["deformation"][1] * scaling["deformation"] + geometry["undeformed"][1],
                                geometry["deformation"][2] * scaling["deformation"] + geometry["undeformed"][2]]
    except:
        # NOTE: now donw for 0-x, 1-y, 2-z DoFs
        # TODO: make consistent and generic
        geometry["deformed"] = [geometry["deformation"][0] * scaling["deformation"] + geometry["undeformed"][0][:, np.newaxis],
                                geometry["deformation"][1] * scaling["deformation"] + geometry["undeformed"][1][:, np.newaxis],
                                geometry["deformation"][2] * scaling["deformation"] + geometry["undeformed"][2][:, np.newaxis]]
        pass
    # Axis, Grid and Label

    plot_limits = get_plot_limits(geometry["deformed"])
    ##

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # set axes, grid
    #ax.set_xlim(plot_limits["x"][0], plot_limits["x"][1])

    # PMT NOTE: manually overwriding:
    #ax.set_xlim(-0.00035, 0.00035)

    #ax.set_ylim(plot_limits["y"][0], plot_limits["y"][1])
    # ax.set_xticks(np.arange(xmin, xmax, 1))
    # ax.set_yticks(np.arange(ymin, ymax, 1))

    # Plot figure
    plt.grid()
    plt.title(plot_title)    # set title
    # plot undeformed
    ax.plot(geometry["undeformed"][0],
            geometry["undeformed"][1],
            geometry["undeformed"][2],
            label='undeformed',
            color=LINE_TYPE_SETUP["color"][0],
            linestyle=LINE_TYPE_SETUP["linestyle"][0],
            marker=LINE_TYPE_SETUP["marker"][0],
            markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][0],
            markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][0],
            markersize=LINE_TYPE_SETUP["markersize"][0])

    if(n_data == 1):
        # TODO not using current formatting yet, needs update
        ax.plot(geometry["deformed"][0],
                geometry["deformed"][1],
                geometry["deformed"][2],
                label="deformed",
                color=LINE_TYPE_SETUP["color"][1],
                linestyle=LINE_TYPE_SETUP["linestyle"][1],
                marker=LINE_TYPE_SETUP["marker"][1],
                markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][1],
                markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][1],
                markersize=LINE_TYPE_SETUP["markersize"][1])

        try:
            # TODO: syntax neesds to be updated for quiver and forces
            ax.quiver(geometry["undeformed"][0],
                      geometry["undeformed"][1],
                      geometry["undeformed"][2],
                      force["external"][0],
                      force["external"][1],
                      force["external"][2],
                      color="red")

            ax.quiver(geometry["undeformed"][0],
                      geometry["undeformed"][1],
                      geometry["undeformed"][2],
                      force["reaction"][0],
                      force["reaction"][1],
                      force["reaction"][2],
                      color="green")

        except:
            # forces are None
            pass

    # TODO: make generic and compatible with all possible DoFs
    # multiple func in one plot
    elif (n_data < 5):
        for i in range(n_data):
            # TODO not using current formatting yet, needs update
            ax.plot(geometry["deformed"][0][:, i],
                    geometry["deformed"][1][:, i],
                    geometry["deformed"][2][:, i],
                    label="mode " + str(i+1),
                    color=LINE_TYPE_SETUP["color"][i+1],
                    linestyle=LINE_TYPE_SETUP["linestyle"][i+1],
                    marker=LINE_TYPE_SETUP["marker"][i+1],
                    markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][i+1],
                    markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][i+1],
                    markersize=LINE_TYPE_SETUP["markersize"][i+1])

    else:
        raise Exception(" Plot format not supported for the request " +
                        str(n_data) + ", maximum 5 allowed.")

    # we do not have legend -> uncomnneted line ax.legend() to avoid waring: No labelleb objects found
    ax.legend()
    geometry = {"deformed": None}

    if pdf_report is not None:
        pdf_report.savefig()
        plt.close(fig)

    if display_plot:
        plt.show()
        display_plot = False

def plot_CAARC_ParOpt_eigenmodes_normed(CAARC_eigenmodes, geometry, plot_titles, scaling, display_plot, max_normed, dof_ids = [1,2,3],  number_of_modes = 3):
    fig, axes = plt.subplots(1, number_of_modes)

    suptitle = 'eigenmodes of CAARC and ParOpt'
    if max_normed:
        suptitle += ' maximum normed'
    fig.suptitle(suptitle, fontsize = 16)

    geometry["deformed"] = [geometry["deformation"][0] * scaling["deformation"] + geometry["undeformed"][0][:, np.newaxis], # x
                            geometry["deformation"][1] * scaling["deformation"] + geometry["undeformed"][1][:, np.newaxis], # y
                            geometry["deformation"][2] * scaling["deformation"] + geometry["undeformed"][2][:, np.newaxis], # z
                            geometry["deformation"][3] * scaling["deformation"] + geometry["undeformed"][1][:, np.newaxis]] # visualize twist a in 2D plane

    for i in range(number_of_modes):
        # plot undeformed
        mode_id = i+1
        axes[i].plot(geometry["undeformed"][1], # y
                    geometry["undeformed"][0], # x
                    label='undeformed',
                    color=LINE_TYPE_SETUP["color"][0],
                    linestyle=LINE_TYPE_SETUP["linestyle"][0],
                    marker=LINE_TYPE_SETUP["marker"][0],
                    markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][0],
                    markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][0],
                    markersize=LINE_TYPE_SETUP["markersize"][0])

        max_deformations, min_deformations = [], []
        for j in range(dof_ids[0],dof_ids[-1]+1): #plot first three dofs
            label_dof = GD.DOF_LABELS['3D'][j]
            if label_dof == 'a':
                label_dof = '\u03B1'
            # # CAARC PART # #

            max_deformations.append(max(CAARC_eigenmodes['shape'][mode_id][GD.DOF_LABELS['3D'][j]]))
            min_deformations.append(min(CAARC_eigenmodes['shape'][mode_id][GD.DOF_LABELS['3D'][j]]))
           
        
            signed_maximum = get_signed_maximum(CAARC_eigenmodes['shape'][mode_id][GD.DOF_LABELS['3D'][j]])
            y_caarc = CAARC_eigenmodes['shape'][mode_id][GD.DOF_LABELS['3D'][j]]
            if max_normed:
                y_caarc /= signed_maximum

            axes[i].plot(y_caarc, 
                        CAARC_eigenmodes['storey_level'], # x
                        label= label_dof + '_caarc' + ' |max|: ' +  '{0:.2e}'.format(signed_maximum),
                        color=LINE_TYPE_SETUP["color"][2+j],
                        linestyle=LINE_TYPE_SETUP["linestyle"][1])

            # # PAROPT PART # # 

            max_deformations.append(max(geometry['deformed'][j][:, i]))
            min_deformations.append(min(geometry['deformed'][j][:, i]))

            signed_maximum = get_signed_maximum(geometry['deformed'][j][:, i])

            y_paropt = geometry["deformed"][j][:, i]
            if max_normed:
                y_paropt /= signed_maximum
            
            axes[i].plot(y_paropt, 
                        geometry["deformed"][0][:, i], # x
                        label= label_dof + '_parOpt' + ' |max|: ' +  '{0:.2e}'.format(signed_maximum),
                        color=LINE_TYPE_SETUP["color"][2+j],
                        linestyle=LINE_TYPE_SETUP["linestyle"][1],
                        marker=LINE_TYPE_SETUP["marker"][1],
                        markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][1],
                        markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][1],
                        markersize=LINE_TYPE_SETUP["markersize"][1]) 
            
            # set title
            axes[i].set_title(plot_titles[i])

        #axes[i].set_xlim(min(min_deformations)+min(min_deformations)/2, max(max_deformations)+max(max_deformations)/2)
        axes[i].xaxis.set_major_formatter(FormatStrFormatter('%.0e'))
        axes[i].legend(loc = 'upper left', fontsize = LEGEND_SETUP['fontsize'][3])
        axes[i].grid()
    
    
    print()
    if display_plot:
        plt.show()    
    
def plot_CAARC_ParOpt_eigenmodes_y_alpha(CAARC_eigenmodes, geometry, plot_titles, scaling, display_plot, use_fitted_CAARC , max_normed, dof_ids = [1,2,3], number_of_modes = 3):
    y_a_fitted = False
    fig, axes = plt.subplots(1, number_of_modes)
    suptitle = 'eigenmodes - y/\u03B1'
    if max_normed:
        suptitle += ' max normed'
    if use_fitted_CAARC:
        suptitle += ' - y & \u03B1 fitted'
    if y_a_fitted:
        suptitle += ' - y/\u03B1 fitted'
    fig.suptitle(suptitle, fontsize = LEGEND_SETUP["fontsize"][0])

    geometry["deformed"] = [geometry["deformation"][0] * scaling["deformation"] + geometry["undeformed"][0][:, np.newaxis], # x
                            geometry["deformation"][1] * scaling["deformation"] + geometry["undeformed"][1][:, np.newaxis], # y
                            geometry["deformation"][2] * scaling["deformation"] + geometry["undeformed"][2][:, np.newaxis], # z
                            geometry["deformation"][3] * scaling["deformation"] + geometry["undeformed"][1][:, np.newaxis]] # visualize twist a in 2D plane

    for i in range(number_of_modes):
        # plot undeformed
        mode_id = i+1
        axes[i].plot(geometry["undeformed"][1], # y
                    geometry["undeformed"][0], # x
                    label='undeformed',
                    color=LINE_TYPE_SETUP["color"][0],
                    linestyle=LINE_TYPE_SETUP["linestyle"][0],
                    marker=LINE_TYPE_SETUP["marker"][0],
                    markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][0],
                    markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][0],
                    markersize=LINE_TYPE_SETUP["markersize"][0])

        max_deformations, min_deformations = [], []
        for j in range(dof_ids[0], dof_ids[1]):#,dof_ids[-1]+1): #plot first three dofs
            label_dof = GD.DOF_LABELS['3D'][j]
            if label_dof == 'a':
                label_dof = '\u03B1'
            # # CAARC PART # #
            
            signed_maximum = get_signed_maximum(CAARC_eigenmodes['shape'][mode_id][GD.DOF_LABELS['3D'][j]])
            y = CAARC_eigenmodes['shape'][mode_id][GD.DOF_LABELS['3D'][j]]
            if max_normed:
                y /= signed_maximum

            max_deformations.append(max(CAARC_eigenmodes['shape'][mode_id][GD.DOF_LABELS['3D'][j]]/signed_maximum))
            min_deformations.append(min(CAARC_eigenmodes['shape'][mode_id][GD.DOF_LABELS['3D'][j]]/signed_maximum))

            signed_maximum_a = get_signed_maximum(CAARC_eigenmodes['shape'][mode_id][GD.DOF_LABELS['3D'][3]])
            alpha= CAARC_eigenmodes['shape'][mode_id][GD.DOF_LABELS['3D'][3]]
            if max_normed:
                alpha /= signed_maximum_a

            y_a_caarc = np.zeros(len(alpha))
            for v, val in enumerate(y):
                if alpha[v] != 0.0:
                    y_a_caarc[v] += val/alpha[v]
            
            if y_a_fitted:
                y_a_caarc = get_fitted_array(CAARC_eigenmodes['storey_level'], y_a_caarc, 3) 
        
            axes[i].plot(y_a_caarc, 
                        CAARC_eigenmodes['storey_level'], # x
                        label= label_dof + '/\u03B1_caarc',# + ' |max|: ' +  '{0:.2e}'.format(signed_maximum),
                        color=LINE_TYPE_SETUP["color"][j],
                        linestyle=LINE_TYPE_SETUP["linestyle"][1])
            
            axes[i].plot(y, 
                        CAARC_eigenmodes['storey_level'], # x
                        label=  'y_caarc',# + ' |max|: ' +  '{0:.2e}'.format(signed_maximum),
                        color=LINE_TYPE_SETUP["color"][j+2],
                        linestyle=LINE_TYPE_SETUP["linestyle"][1])
            
            axes[i].plot(alpha, 
                        CAARC_eigenmodes['storey_level'], # x
                        label= '\u03B1_caarc',# + ' |max|: ' +  '{0:.2e}'.format(signed_maximum),
                        color=LINE_TYPE_SETUP["color"][j+3],
                        linestyle=LINE_TYPE_SETUP["linestyle"][1])

            # # PAROPT PART # # 
            
            
            max_deformations.append(max(geometry['deformed'][j][:, i]/signed_maximum))
            min_deformations.append(min(geometry['deformed'][j][:, i]/signed_maximum))
            signed_maximum = get_signed_maximum(geometry['deformed'][j][:, i])
            y = geometry["deformed"][j][:, i]
            if max_normed:
                y /= signed_maximum

            signed_maximum_a = get_signed_maximum(geometry['deformed'][3][:, i])
            alpha = geometry['deformed'][3][:, i]
            if max_normed:
                alpha /= signed_maximum_a

            y_a_paropt = np.zeros(len(alpha))
            for v, val in enumerate(y):
                if alpha[v] != 0.0:
                    y_a_paropt[v] += val/alpha[v]

            axes[i].plot(y_a_paropt, 
                        geometry["deformed"][0][:, i], # x
                        label= label_dof + '/\u03B1_parOpt',# + ' |max|: ' +  '{0:.2e}'.format(signed_maximum),
                        color=LINE_TYPE_SETUP["color"][j+1],
                        linestyle=LINE_TYPE_SETUP["linestyle"][1],
                        marker=LINE_TYPE_SETUP["marker"][1],
                        markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][1],
                        markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][1],
                        markersize=LINE_TYPE_SETUP["markersize"][1]) 
            
            #difference = y_a_caarc - y_a_paropt

            # set title
            axes[i].set_title(plot_titles[i])

        #print('')
        #axes[i].set_xlim(min(min_deformations)+min(min_deformations)/2, max(max_deformations)+max(max_deformations)/2)
        axes[i].xaxis.set_major_formatter(FormatStrFormatter('%.0e'))
        axes[i].legend(loc = 'upper left', fontsize = LEGEND_SETUP['fontsize'][3])
        axes[i].grid()
    print()
    if display_plot:
        plt.show()

def plot_CAARC_eigenmodes(CAARC_eigenmodes, display_plot, max_normed, suptitle = 'CAARC',number_of_modes = 3):
        fig, axes = plt.subplots(1, number_of_modes)

        fig.suptitle(suptitle)
        scaling = {"deformation": 1,
                   "force": 1}

        frequencies_CAARC = [0.231,0.429,0.536]

        plot_titles = []
        for selected_mode in range(number_of_modes):
            plot_titles.append("Eigenmode " + str(selected_mode + 1) + "\n" + "  Frequency: " + str(np.round(
                frequencies_CAARC[selected_mode], 2)) + "  Period: " + str(np.round(
                1/frequencies_CAARC[selected_mode], 2)) + "\n")

        for i in range(number_of_modes):
            # plot undeformed
            axes[i].axvline(0.0, CAARC_eigenmodes['storey_level'][0], CAARC_eigenmodes['storey_level'][-1], # y
                        #CAARC_eigenmodes['storey_level'], # x
                        label='undeformed',
                        color=LINE_TYPE_SETUP["color"][0],
                        linestyle=LINE_TYPE_SETUP["linestyle"][0])
            max_deformations, min_deformations = [], []
            mode_id = i +1
            for j, sway in enumerate(['y', 'z', 'a']): #plot first three dofs
                label_dof = GD.DOF_LABELS['3D'][j+1]
                if label_dof == 'a':
                    label_dof = '\u03B1'
                max_deformations.append(max(CAARC_eigenmodes['shape'][mode_id][sway]))
                min_deformations.append(min(CAARC_eigenmodes['shape'][mode_id][sway]))
                absolute_maximum = max(abs(CAARC_eigenmodes['shape'][mode_id][sway]))
                if max_normed:
                    signed_maximum = get_signed_maximum(CAARC_eigenmodes['shape'][mode_id][sway])
                    y = CAARC_eigenmodes['shape'][mode_id][sway]/signed_maximum
                else:
                    y = CAARC_eigenmodes['shape'][mode_id][sway]

                axes[i].plot(y, 
                            CAARC_eigenmodes['storey_level'], # x
                            label= label_dof + ' |max|: ' +  '{0:.2e}'.format(absolute_maximum),
                            color=LINE_TYPE_SETUP["color"][2+j],
                            linestyle=LINE_TYPE_SETUP["linestyle"][1]) 
                axes[i].set_title(plot_titles[i])

            #axes[i].set_xlim(min(min_deformations)+min(min_deformations)/2, max(max_deformations)+max(max_deformations)/2)
            #axes[i].set_ylim(0.0, CAARC_eigenmodes['storey_level'][-1]+10.0)
            axes[i].xaxis.set_major_formatter(FormatStrFormatter('%.0e'))
            axes[i].legend(loc = 'upper left', fontsize = LEGEND_SETUP['fontsize'][3])
            axes[i].grid()

        if display_plot:
            plt.show()

def plot_optimizable_function(optimizable_function, label, x = np.arange(0.01,10,0.1)):
        # x = np.arange(0.01,10,0.1)
        #x = np.logspace(-1.5,1,100)
        y= []
        for i in x:
            y.append(optimizable_function(i))
            #print(optimizable_function(x[i]))
        np.asarray(y)
        fig, ax1 = plt.subplots()
        ax1.set_title(label + ' objective function')
        ax1.set_xlabel('multiplier factors')
        ax1.set_ylabel('obtimizable function: norm(target - current ' + label + ')')

        ## plot simulation values
        ax1.plot(x, y)
        #plt.vlines(opt_fctr, 0, 0.2, label = str(opt_fctr))
        #plt.legend()
        plt.grid()
        plt.show()

def animate_result(title, array_time, geometry, force, scaling):

    # just for testing, where to get the undeformed structure?
    #
    # First set up the figure, the axis, and the plot element we want to
    # animate

    # TODO: animation needs a step and the number of frames in animate needs to take this into consideration
    # make more generic and robust
    step = 5

    fig = plt.figure()
    ax = Axes3D(fig)  # fig.gca(projection='3d')

    # TODO: animate needs scaling as well
    # set min and max values
    #xmin = displacement_time_history.min()
    #xmax = displacement_time_history.max()
    #xmin = xmin - ceil((xmax-xmin)/10)
    #xmax = xmax + ceil((xmax-xmin)/10)

    # TODO extend and use plot limits

    geometry["deformed"] = [geometry["deformation"][0] * scaling["deformation"] + geometry["undeformed"][0][:, np.newaxis],
                            geometry["deformation"][1] * scaling["deformation"] + geometry["undeformed"][1][:, np.newaxis],
                            geometry["deformation"][2] * scaling["deformation"] + geometry["undeformed"][2][:, np.newaxis]]

    xmin = np.min(geometry["deformed"][0])
    xmax = np.max(geometry["deformed"][0])
    # xmin = xmin - ceil((xmax-xmin)/30)
    # xmax = xmax + ceil((xmax-xmin)/30)

    ymin = np.min(geometry["deformed"][1])
    ymax = np.max(geometry["deformed"][1])
    # ymin = ymin - ceil((ymax-ymin)/30)
    # ymax = ymax + ceil((ymax-ymin)/30)

    zmin = np.min(geometry["deformed"][2])
    zmax = np.max(geometry["deformed"][2])
    # zmin = zmin - ceil((zmax-zmin)/30)
    # zmax = zmax + ceil((zmax-zmin)/30)

    ax.set_xlim3d(xmin, xmax)
    ax.set_ylim3d(ymin, ymax)
    ax.set_zlim3d(zmax, zmin)

    # text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
    # text_mode = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.grid(True)

    # ax.set_ylim(-0.0005,0.0005)
    # #ax.set_zlim(0,1.1)

    # TODO: clear if this part or init is at all neded or both together redundant
    # NOTE: Adding the comma un-packs the length one list into
    undeformed_line, = ax.plot(geometry["undeformed"][0],
                               geometry["undeformed"][1],
                               geometry["undeformed"][2],
                               label='undeformed',
                               color=LINE_TYPE_SETUP["color"][0],
                               linestyle=LINE_TYPE_SETUP["linestyle"][0],
                               marker=LINE_TYPE_SETUP["marker"][0],
                               markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][0],
                               markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][0],
                               markersize=LINE_TYPE_SETUP["markersize"][0])

    # NOTE: Can't pass empty arrays into 3d version of plot()
    # NOTE: Adding the comma un-packs the length one list into
    deformed_line, = ax.plot([], [], [],
                             color=LINE_TYPE_SETUP["color"][1],
                             linestyle=LINE_TYPE_SETUP["linestyle"][1],
                             marker=LINE_TYPE_SETUP["marker"][1],
                             markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][1],
                             markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][1],
                             markersize=LINE_TYPE_SETUP["markersize"][1])

    text = ax.text(10, 0, 0, 'init', color='red')

    # initialization function: plot the background of each frame
    def init():
        undeformed_line.set_data([], [])
        deformed_line.set_data([], [])
        # NOTE: there is no .set_data() for 3 dim data...
        undeformed_line.set_3d_properties([])
        deformed_line.set_3d_properties([])

        text.set_text('')

        return undeformed_line, deformed_line, text

    # animation function.  This is called sequentially

    def animate(i):
        undeformed_line.set_data(geometry["undeformed"][0],
                                 geometry["undeformed"][1])

        deformed_line.set_data(geometry["deformed"][0][:, step * i],
                               geometry["deformed"][1][:, step * i])

        # NOTE: there is no .set_data() for 3 dim data...
        undeformed_line.set_3d_properties(geometry["undeformed"][2])
        deformed_line.set_3d_properties(geometry["deformed"][2][:, step * i])

        text.set_text('{0:.2f}'.format(array_time[i]) + "[s]")

        return undeformed_line, deformed_line, text

    # call the animator.  blit=True means only re-draw the parts that have
    # changed.
    # frames = number of columns in result
    animation.FuncAnimation(fig, animate, init_func=init,
                            frames=len(array_time[::step])-1, interval=20, blit=True)

    plt.show()


def get_plot_limits(deformed_geometry, offset_factor=10.):
    try:
        # case of dynamic / multi plot
        x_min = np.matrix.min(deformed_geometry[0])
        x_max = np.matrix.max(deformed_geometry[0])
        y_min = np.matrix.min(deformed_geometry[1])
        y_max = np.matrix.max(deformed_geometry[1])
    except:
        # case of static / single plot
        x_min = np.amin(deformed_geometry[0])
        x_max = np.amax(deformed_geometry[0])
        y_min = np.amin(deformed_geometry[1])
        y_max = np.amax(deformed_geometry[1])

    plot_limits = {"x": [x_min - (x_max - x_min) / offset_factor,
                         x_max + (x_max - x_min) / offset_factor],
                   "y": [y_min - (y_max - y_min) / offset_factor,
                         y_max + (y_max - y_min) / offset_factor]}

    return plot_limits


def plot_dynamic_result(pdf_report, display_plot, plot_title, result_data, array_time):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(plot_title)
    plt.grid()
    plt.title(plot_title + ' Vs Time')    # set title
    # plot undeformed
    plt.plot(array_time,
             result_data,
             color=LINE_TYPE_SETUP["color"][1],
             linestyle=LINE_TYPE_SETUP["linestyle"][1],
             marker= "None",#LINE_TYPE_SETUP["marker"][1],
             markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][1],
             markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][1],
             markersize=LINE_TYPE_SETUP["markersize"][1])
    #ax.legend()

    if pdf_report is not None:
        pdf_report.savefig()
        plt.close(fig)

    if display_plot:
        plt.show()


def plot_table(pdf_report, display_plot, plot_title, table_data, row_labels, column_labels):

    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    plt.title(plot_title)

    # Add a table at the bottom of the axes
    ax.table(cellText=table_data,
             rowLabels=row_labels,
             colLabels=column_labels,
             loc='center')

    fig.tight_layout()

    if pdf_report is not None:
        pdf_report.savefig()
        plt.close(fig)

    if display_plot:
        plt.show()

