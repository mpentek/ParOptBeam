import numpy as np
from math import ceil
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib import animation

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from source.auxiliary import global_definitions as GD


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
                            geometry["deformation"][3] * scaling["deformation"] + geometry["undeformed"][1]] # visualize twist a

    plt.grid()
    plt.title(plot_title)    # set title
    deflections, def_min, def_max = [], [], []
    for dof, deformation in enumerate(geometry['deformed'][1:]):
        def_min.append(min(deformation))
        def_max.append(max(deformation))
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
    for i, deflection in enumerate(deflections):
        ax.plot(geometry["deformed"][deflection], # a
                geometry["deformed"][0], # x
                label= 'deformed_' + GD.DOF_LABELS['3D'][deflection],
                color=LINE_TYPE_SETUP["color"][2+i],
                linestyle=LINE_TYPE_SETUP["linestyle"][1],
                marker=LINE_TYPE_SETUP["marker"][1],
                markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][1],
                markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][1],
                markersize=LINE_TYPE_SETUP["markersize"][1]) 
        

    ax.set_ylabel('height x - coord')
    ax.set_xlabel('deflection')
    ax.set_xlim(min(def_min)-5e-5, max(def_max)+5e-5)
    # if (n_data == 2):
    #        ax.plot(geometry["deformed"][2], # y
    #             geometry["deformed"][0], # x
    #             label="deformed z",
    #             color=LINE_TYPE_SETUP["color"][2],
    #             linestyle=LINE_TYPE_SETUP["linestyle"][1],
    #             marker=LINE_TYPE_SETUP["marker"][1],
    #             markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][1],
    #             markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][1],
    #             markersize=LINE_TYPE_SETUP["markersize"][1]) 

    if display_plot:
        plt.legend()
        plt.show()
    

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
    ax.legend()

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
