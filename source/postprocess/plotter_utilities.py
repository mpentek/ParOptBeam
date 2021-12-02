import numpy as np
from math import ceil
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.ticker import FormatStrFormatter
from os.path import join as os_join

#import matplotlib as mpl
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
LINESTYLE = ["--",    "-.",  ":",    "--",   "-",   "-"]

LEGEND_SETUP = {"fontsize": [16, 14, 12, 10, 8]}

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

# ESWL PLOTTERS

def plot_eswl_components(eswl_components, response_label, parameters, display_plot):

    if parameters['load_directions_to_plot'] == 'automatic':
        load_directions = GD.LOAD_DIRECTIONS_RESPONSES_UNCOUPLED[response_label]
    else:
        load_directions = parameters['load_directions_to_plot']
    
    if len(load_directions) == 1:
        fig, ax = plt.subplots(num='for_'+response_label)
        axes = [ax]
    else:
        fig, axes = plt.subplots(ncols = len(load_directions), sharey=True, num='for_'+response_label)
    
    fig.suptitle('ESWL for base ' + (response_label), fontsize = 10)
    
    x_coords = eswl_components['x_coords']
    # to present it as a line load
    dx = x_coords[1]
    l_i = np.full(len(x_coords), dx)
    l_i[0] /= 2
    l_i[-1] /= 2

    if parameters['components_to_plot'][0] == 'all':
        # taking the keys from the first direction
        components = eswl_components[response_label]['y'].keys()
    else:
        components = parameters['components_to_plot']
    
    for i, direction in enumerate(load_directions):
        # for labels
        unit_label = GD.UNITS_LOAD_DIRECTION[direction]
        force_label = '{}'.format(GD.DIRECTION_LOAD_MAP[direction],lower=True)
       
        axes[i].plot(np.zeros(len(x_coords)), x_coords, 
                            label = r'structure', 
                            marker = 'o', 
                            color = 'grey', 
                            linestyle = '--')

        axes[i].set_title('{}'.format(GD.DIRECTION_LOAD_MAP[direction], lower = True))
        # plot each component
        for j, component in enumerate(components):
            if component not in eswl_components[response_label][direction].keys():
                #print ('\nskipped:', component, 'as it was not computed')
                continue
            eswl = eswl_components[response_label][direction][component] / l_i 
            if component == 'total':
                line = '-'
                color = LINE_TYPE_SETUP['color'][i+2]
            else:
                line = LINESTYLE[j]
                color = LINE_TYPE_SETUP['color'][i+2]

            ax_eswl = axes[i].plot(eswl, 
                                x_coords, 
                                label = component,
                                linestyle = line,
                                color = color)

        eswl_leg = axes[i].legend(fontsize = 9)        
        
        # settings
        axes[i].locator_params(axis='x', nbins = 4)
        axes[i].locator_params(axis='y', nbins = 4)
        axes[i].set_ylim(bottom=0,top=200)
        axes[i].add_artist(eswl_leg)
        
        axes[i].set_xlabel('{}'.format(GD.DIRECTION_LOAD_MAP[direction], lower = True) + r' ${}$'.format(unit_label))
        
        axes[i].grid()

    axes[0].set_ylabel('height [m]')
    
    # if pdf_report is not None:
    #     pdf_report.savefig()
    #     plt.close(fig)
    #     # plt.savefig(os_join(destination,fname))
    #     # print ('\nsaved:',os_join(destination,fname))

    #if display_plot:
    plt.show()

def plot_eswl_dyn_compare(result_dict, compare = ['dyn_est','eswl'], unit='N',
                         options = None):

    if options['update_plot_params']:
        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.rcParams.update(options['update_plot_params'])

    dest = os_join(*['plots', 'ESWL_plots', 'dyn_eswl_compare'])

    directions_naming = {'Mz':'alongwind', 'My':'crosswind','Mx':'torsion'}

    factor = 0.5
    colors = ['tab:blue', 'darkgray', 'gray',  'lightgray']
    fig, ax = plt.subplots(num='res compare')
    x = np.arange(0,len(result_dict['labels']),1)* factor
    width = 0.15
    x_labels = result_dict['labels']
    if result_dict['norm_by'] == 'glob_max':
        norm_by = 1/np.array(result_dict['dyn_max'])
    if result_dict['norm_by'] == 'ref_vel':
        if result_dict['labels'] == ['Mz','Qy','My','Qz','Mx']:
            ref_areas = np.array([45**2 *180, 45*180, 30**2*180, 30*180, 30*45*180])
        else:
            raise Exception('coefficient calculation not correct for this order of responses')
        norm_by = 1/(0.5 * 1.225 * result_dict['u_mean']**2 * ref_areas)
        x_labels = ['C' + var for var in result_dict['labels']]

    #for i, val in enumerate(compare):
    res_dyn = np.array(result_dict[compare[0]])* GD.UNIT_SCALE[unit]
    res_eswl = np.array(result_dict[compare[1]])* GD.UNIT_SCALE[unit]
    #split = 2
    col = 3
    split = [-width/2, width/2]
    if len(compare) == 3:
        #split =3
        width = 0.1
        col += 1
        split = [0, width,  -width]
        res_dyn_1 = np.array(result_dict[compare[2]])* GD.UNIT_SCALE[unit]
        rects_dyn_1 = ax.bar(x + split[2],  res_dyn_1 * norm_by, 
                        width, color = colors[0], alpha = 0.5, label=convert_for_latex(compare[2]))


    rects_dyn = ax.bar(x + split[0],  res_dyn * norm_by, 
                    width, color = colors[0], label=convert_for_latex(compare[0]))
    rects_eswl = ax.bar(x + split[1], res_eswl * norm_by, 
                    width, color = colors[1], label=convert_for_latex(compare[1]))
    dx = (x[2] - x[1])/2
    ax.axvline(x[1]+dx, linestyle = '--', color = 'grey')
    ax.axvline(x[3]+dx, linestyle = '--', color = 'grey')
    ax.axhline(1,linestyle = '--', color = 'k',label= convert_for_latex('glob_max'))

    for r_i, r in enumerate(['Mz','My','Mx']):
        ax.text(x[result_dict['labels'].index(r)],1.9, directions_naming[r])
    
    ax.legend(loc= 'upper center', bbox_to_anchor=(0.5, -0.15), ncol= col)
    ax.set_ylabel(r'$R_{i}/glob_{max}(R_{i})$')
    ax.set_xticks(x)
    ax.set_xticklabels([convert_for_latex(x) for x in x_labels])

    save_title = 'reaction_compare_' + compare[0]

    if len(compare) == 3:
        save_title += '_' + compare[-1]

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
    
def plot_component_rate(eswl_obj, display_plot, components_to_compare = ['total','mean','lrc', 'resonant_m_lumped'], 
                        options = None):
    ''' 
    eswl_1,_2: already directional components of the eswl
    ''' 

    dest = os_join(*['plots', 'ESWL_plots', 'all_parts'])

    eswl_components = eswl_obj.eswl_components
    del eswl_components['x_coords']
    naming = {'mean':'mean', 'gle':'beswl', 'resonant':'reswl', 'resonant_m':'reswl','resonant_m_lumped':'reswl','total':'total','lrc':'beswl'}
    directions_naming = {'Mz':'alongwind', 'My':'crosswind','Mx':'torsion'}
    fig, ax = plt.subplots(num='R_i compare')

    colors = ['tab:blue', 'dimgray', 'darkgray', 'lightgray']
    width = 0.2
    factor = 1

    components = ['total', 'mean', eswl_obj.settings['beswl_to_combine'], eswl_obj.settings['reswl_to_combine']]
    x = np.arange(len(eswl_components))*factor
    d_i = factor/(len(components)+1)
    width = d_i * len(components)
    
    for e_i, response in enumerate(eswl_components):
        R_tot = eswl_obj.static_response[response][eswl_obj.settings['at_height']]
        influences = eswl_obj.influences[response]
        #direction
        eswl = eswl_components[response] 
        for c_i, component in enumerate(components):
            R_i = 0
            for direction in eswl:
                # computing the part of the response caused by the specific component
                R_i += sum(np.multiply(influences[direction], eswl[direction][component]))

            # for the legend
            if e_i == 0:
                label = component
            else:
                label = None

            rect_ri = ax.bar(x[e_i] - width + c_i * d_i, R_i/R_tot, width = d_i, color = colors[c_i], label = label)
            # if response in ['My','Mz']:

        ax.text(x[e_i]-width/2, 1.02, response)#directions_naming[response]
            
        if e_i != len(eswl_components)-1:
            ax.axvline(x[e_i], linestyle = '--', color = 'grey')

    ax.axhline(1, linestyle = '--', color = 'grey')
    ax.legend(bbox_to_anchor = (0.5, -0.02), loc ='upper center', ncol = 4)
    
    ax.set_ylabel('rate of '+ r'$R_{total}$')
    ax.set_xticks([])

    save_title = 'component_compare'

    # if options['savefig']:
    #     plt.savefig(dest + os_sep + save_title)
    #     #plt.savefig(dest + os_sep + save_title + '.svg')
    #     print ('\nsaved:',dest + os_sep + save_title)
    #     #plt.close()

    #if display_plot:
    plt.show()


def plot_influences(eswl_object):
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


def plot_n_mode_shapes(mode_shapes, charact_length ,n = 3, options = None, save_suffix=''):
    ''' 
    mode_shapes_sorted: mode shapres as a dictionary with dofs 
    charact_length: scale for the rotaional dofs
    ''' 
    if not isinstance(mode_shapes, dict):
        utilities.sort_row_vectors_dof_wise(mode_shapes)

    fig, ax = plt.subplots(1, n, sharey=True)
    fig.canvas.set_window_title('mode shapes first 3 modes')
    fig.suptitle('mode shapes of first 3 modes - rotations multiplied with characteristic length')
    if save_suffix == '_caarc_A':
        dof_labels = list(mode_shapes_sorted[0].keys())
        dof_labels = ['y', 'a', 'z']
    else:
        dof_lables = mode_shapes_sorted
    for mode in range(n):
        for d, dof_label in enumerate(dof_labels):
            multiplier = 1.0
            if dof_label in ['a','b','g']:
                multiplier = charact_length
            ax[mode].set_title('mode ' + str(mode+1))
            if save_suffix == '_caarc_A':
                mode_s = np.asarray(mode_shapes_sorted[mode][dof_label])
            else:
                mode_s = mode_shapes_sorted[dof_label][:,mode]
            ax[mode].plot(mode_s * multiplier,
                            np.arange(len(mode_s)),
                            label = dof_label,
                            color = LINE_TYPE_SETUP['color'][2+d])

        ax[mode].legend()
        ax[mode].set_xlabel('defl. [m]')
        ax[mode].set_ylabel('height [m]')
        ax[mode].locator_params(axis='y', nbins = 3)
        
        ax[mode].xaxis.set_major_formatter(FormatStrFormatter('%.1e'))
        ax[mode].locator_params(axis='x', nbins = 5)
        ax[mode].grid()
    
    save_title = 'eigenmodes' + save_suffix

    if options['savefig']:
        plt.savefig(dest + os_sep + save_title)
        plt.savefig(dest + os_sep + save_title + '.pdf')
        print ('\nsaved:',dest + os_sep + save_title)
        #plt.close()
    
    if options['savefig_latex']:
        plt.savefig(dest_latex + os_sep + save_title)
        print ('\nsaved:',dest_latex + os_sep + save_title)

    if options['show_plots']:
        plt.show()


def plot_load_time_histories(load_signals, nodal_coordinates):

    fig, ax = plt.subplots(1, len(load_signals))

    try:
        if nodal_coordinates['x0']:
            nodal_coordinates = nodal_coordinates['x0']
    except KeyError:
        nodal_coordinates = nodal_coordinates

    for i, component in enumerate(load_signals):
        if component == 'sample_freq':
            break
        # structure
        ax[i].set_title('load signal direction ' + component)
        ax[i].plot(np.zeros(len(nodal_coordinates)),
                    nodal_coordinates, 
                    label = 'structure', 
                    marker = 'o', 
                    color = 'grey', 
                    linestyle = '--')
        for node in range(len(nodal_coordinates)):
            shift_scale = 1e+05
            if component == 'a':
                shift_scale *= 5
            ax[i].plot(np.arange(0,len(load_signals[component][node])),
                        load_signals[component][node] + nodal_coordinates['x0'][node]*shift_scale)
    
        #ax[i].legend()
        ax[i].grid()
    plt.show()

def plot_load_time_histories_node_wise(load_signals, n_nodes, discard_time, load_signal_labels = None,
                                        options = {'show_plots':True,'savefig':False,'savefig_latex':False, 'update_plot_params':None}):
    '''
    for each node it plots the time history of the loading 
    parameter: load_signal_labels 
                if not set all signals in the load_signal dict are plotted 
                can be specified with a list of labels to plot
    '''

    dest = os_join(*['plots', 'ESWL_plots','load_signals_4_nodes'])

    mpl.rcParams.update(mpl.rcParamsDefault)

    if options['update_plot_params']:
        plt.rcParams.update(options['update_plot_params'])

    ax_i = list(range(n_nodes-1, -1, -1))
    if load_signal_labels:
        components = load_signal_labels
    else:
        load_signals.pop('sample_freq')
        components = load_signals.keys()
    for component in components:
        # if component == 'sample_freq':
        #     break
        fig, ax = plt.subplots(n_nodes, 1, num = 'signals of ' +  GD.DIRECTION_LOAD_MAP[component])
        fig.suptitle('signal of ' + GD.DIRECTION_LOAD_MAP[component])

        for node in range(n_nodes):
            # show also mean 
            mean = np.mean(load_signals[component][node])
            ax[ax_i[node]].hlines(mean, 0, len(load_signals[component][node]), 
                                  color = 'dimgray', 
                                  linestyle = '--', 
                                  label = 'mean: ' + '{:.2e}'.format(mean))

            ax[ax_i[node]].plot(np.arange(len(load_signals[component][node])),
                                load_signals[component][node],
                                label = 'node_'+str(node))

            # ax[ax_i[node]].vlines(discard_time, min(load_signals[component][node]), max(load_signals[component][node]), 
            #                       color = 'grey', 
            #                       linestyle = '--',
            #                       label = 'discard')

            ax[ax_i[node]].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
            ax[ax_i[node]].locator_params(axis='y', nbins = 5)
            ax[ax_i[node]].yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
            ax[ax_i[node]].grid()
            ax[ax_i[node]].set_ylabel(r'$x_{}(t)$'.format(node))

            ax[ax_i[node]].set_xlim(0, len(load_signals[component][node]))
            
        ax.legend()

        ax[ax_i][0].set_xlabel(r'$t [s]$')

        save_title = 'dyn_load_' + str(n_nodes) + '_' + component

        if options['savefig']:
            plt.savefig(dest + os_sep + save_title)
            print ('\nsaved:',dest + os_sep + save_title)
            #plt.close()

        if options['savefig_latex']:
            plt.savefig(dest_latex + os_sep + save_title)
            print ('\nsaved:',dest_latex + os_sep + save_title)

            #plt.close()

        if options['show_plots']:
            plt.show()

        plt.show()