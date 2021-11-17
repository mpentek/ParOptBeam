import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from os.path import join as os_join
from os.path import sep as os_sep
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.offsetbox import AnchoredText


import source.auxiliary.global_definitions as GD
import source.ESWL.eswl_auxiliaries as utilities
import source.auxiliary.other_utilities as other_utils

'''
Legenden Außerhalb
unten 
axes[0,d_i].legend(bbox_to_anchor = (0.5, -0.4), loc ='upper center', ncol = 4)
rechts
axes[0,-1].legend(bbox_to_anchor = (1.04, 1), loc ='upper left')
'''

dest_latex = os_join(*['C:',os_sep,'Users','Johannes','LRZ Sync+Share','MasterThesis','Abgabe','Text','images'])

default_plot_options = {'show_plots':True,'savefig':False,'savefig_latex':False, 'update_plot_params':False, 'present':False}

def cm2inch(value):
    return value/2.54

LINE_TYPE_SETUP = {"color":          ["grey", "black", "tab:blue", "tab:orange", "tab:green", "tab:red",'tab:purple', 'tab:brown','tab:pink'],
                   "linestyle":      ["--",    "-",  "-",    "-",   "-",   "-"],
                   "marker":         ["o",    "s",  "^",    "p",   "x", "*"],
                   "markeredgecolor": ["grey", "black", "red", "green", "blue", "magenta"],
                   "markerfacecolor": ["grey", "black", "red", "green", "blue", "magenta"],
                   "markersize":     [4,      4,    4,      4,    4,    4]}
LINESTYLE = ["--",    "-.",  ":",  "-",   "-",   "-"]

COLORS = ['tab:blue','tab:orange','tab:green']

# SAVE DESTINATION 
destination = os_join('plots','ESWL_plots')


def convert_for_latex(string):
    l = list(string.split('_'))
    label = r'${}$'.format(l[0])
    for i in range(1,len(l)):
        label += r'$_{{{}}}$'.format(l[i])

    return label

def convert_load_for_latex(load_label,lower=False):
    ''' 
    use lower = True if line load and False if point load
    ''' 
    if lower:
        load = list(load_label)[0].lower()
    else:
        load = list(load_label)[0]
    direction = list(load_label)[1]

    return r'${}$'.format(load) + r'$_{{{}}}$'.format(GD.GREEK[direction])


# RESWL 
def plot_eswl_damping_compare(eswl_to_plot, damping_labels, nodal_coordinates, eswl_component, load_directions, unit = 'N',
                    options = default_plot_options):
    ''' 
    load_vector: list of selected eswl loads -> here with different damping cases underlying
    response_label: List of response for which the eswl should be shown -  as list a new supplot is shown for each one
    nodal_coordinates: passed once
    damping labels: hard coded damping coefficnets of the damping cases 
    eswl_component: list of compoents to show in the plot (mean, background, ...)
    load_directions: the load direction associated with the 'response_lable'-  as list the same thinf as reponse_lable holdes

    '''
    dest = os_join(*['plots', 'ESWL_plots','resonant_parts'])
    mpl.rcParams.update(mpl.rcParamsDefault)

    plt.rcParams.update(options['update_plot_params'])

    if len(eswl_to_plot) == 1:
        fig, ax = plt.subplots(num='eswl_damping_compare')
        axes = [ax]
    else:
        fig, axes = plt.subplots(ncols = len(eswl_to_plot), sharey=True, num='eswl_damping_compare')

    for ax in axes:
        ax.plot(np.zeros(len(nodal_coordinates)), nodal_coordinates, 
                    label = r'structure', 
                    marker = 'o', 
                    color = 'grey', 
                    linestyle = '--')
    
    #plt.show()

    dx = nodal_coordinates[1]
    l_i = np.full(len(nodal_coordinates), dx)
    l_i[0] /= 2
    l_i[-1] /= 2


    for response_i, eswl_i in enumerate(eswl_to_plot.items()):
        # make it force per unit height
        load_direction = load_directions[response_i]

        for i, eswl in enumerate(eswl_i[1]):
            resp_label = eswl_i[0]
            unit_label = GD.UNITS_LOAD_DIRECTION[load_direction]
            unit_label = unit_label.replace(unit_label[1], unit)
            force_label = '{}'.format(convert_load_for_latex(GD.DIRECTION_LOAD_MAP[load_direction],lower=True))

            damping_label = r'case ' + r'$\xi=$' + r'${}$'.format(damping_labels[i]) + r'$\%$'
            # for compoents to plot in one subplot
            if response_i == 0 and i == 0:
                axes[-1].plot(0,0, linestyle = '-', color = COLORS[i], label = damping_label)
            if response_i == 1 and i == 1:
                axes[-1].plot(0,0, linestyle = '-', color = COLORS[i], label = damping_label)
            
            load_vector_r = abs(np.divide(eswl[load_direction][eswl_component[0]], l_i)) * GD.UNIT_SCALE[unit]
            label_r = r'reswl ' #+ convert_load_for_latex(load_label) #+' ' + r'$\xi=$' + r'${}$'.format(damping_labels[i]) + r'$\%$'

            load_vector_b = [None]
            save_suffix = ''
            if len(eswl_component) > 1:
                load_vector_b = abs(np.divide(eswl[load_direction][eswl_component[1]], l_i)) * GD.UNIT_SCALE[unit]
                save_suffix = '_inc_beswl'
                label_b = r'beswl '# + convert_load_for_latex(load_label)

            

            load_r = axes[response_i].plot(load_vector_r,
                            nodal_coordinates,
                            label = label_r)

            color = load_r[0].get_color()

            if load_vector_b[0] != None:
                axes[response_i].plot(load_vector_b,
                                        nodal_coordinates,
                                        label = label_b,
                                        color = color,
                                        linestyle = '--')

        
        axes[response_i].set_title('ESWL for '+ convert_load_for_latex(resp_label))
        axes[response_i].set_ylim(bottom = 0)
        axes[response_i].yaxis.set_major_locator(MultipleLocator(50))
        axes[response_i].xaxis.set_major_locator(MultipleLocator(100))
     
        #axes[response_i].locator_params(axis='x', nbins=5)
        axes[response_i].set_xlabel(force_label + r'${}$'.format(unit_label))
        axes[0].set_ylabel(r'height $[m]$')
        axes[response_i].grid(True)

    # LEGEND PLACING
    # # below the plot:
    # ax.legend(bbox_to_anchor = (0.5, -0.1), loc ='upper center', ncol = 2)
    # # right upper corner
    
    axes[-1].legend(bbox_to_anchor = (1.04, 1), loc ='upper left')
    
    responses = '_'.join(eswl_to_plot.keys())
    save_title = 'reswl_damping_compare_for_' + responses + save_suffix

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
    
# ESWL TOTAL
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


def plot_eswl_components(eswl_components, nodal_coordinates, load_directions_to_include, response_label, 
                         textstr, influences ,components_to_plot = ['all'], gb_label = '', go_final = False, R_total = None, unit= 'N', save_suffix = '',
                         options = default_plot_options):
    
    if options['update_plot_params']:
        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.rcParams.update(options['update_plot_params'])

    dest = os_join(*['plots', 'ESWL_plots'])
    
    plt.rcParams.update({'axes.formatter.limits':(-3,3)}) 
    if load_directions_to_include == 'all':
        load_directions = GD.LOAD_DIRECTION_MAP['all']
    else:
        load_directions = load_directions_to_include
    
    if len(load_directions) == 1:
        fig, ax = plt.subplots(num='for_'+response_label)
        axes = [ax]
    else:
        fig, axes = plt.subplots(ncols = len(load_directions), sharey=True, num='for_'+response_label)
    
    fig.suptitle('ESWL for base ' + convert_load_for_latex(response_label), fontsize = 10)
    
    dx = nodal_coordinates['x0'][1]
    l_i = np.full(len(nodal_coordinates['x0']), dx)
    l_i[0] /= 2
    l_i[-1] /= 2

    #fig, ax = plt.subplots(1, len(load_directions), sharey=True, num=)

    if components_to_plot[0] == 'all':
        # taking the keys from the first direction
        components = eswl_components[response_label]['y'].keys()
    else:
        components = components_to_plot
    
    if go_final:
        naming = {'mean':'mean', 'gle':'beswl', 'res_base_distr':'reswl', 'res_mod_cons':'reswl','res_mod_lumped':'reswl','total':'total','lrc':'beswl'}

    for i, direction in enumerate(load_directions):
        # for labels
        unit_label = GD.UNITS_LOAD_DIRECTION[direction]
        unit_label = unit_label.replace(unit_label[1], unit)
        force_label = '{}'.format(convert_load_for_latex(GD.DIRECTION_LOAD_MAP[direction],lower=True))
       
        axes[i].plot(nodal_coordinates['y0'], nodal_coordinates['x0'], 
                label = r'structure', 
                marker = 'o', 
                color = 'grey', 
                linestyle = '--')

        if not go_final:
            axes[i].set_title('{}'.format(convert_load_for_latex(GD.DIRECTION_LOAD_MAP[direction], lower = True)))
        # plot each component
        for j, component in enumerate(components):
            if component not in eswl_components[response_label][direction].keys():
                #print ('\nskipped:', component, 'as it was not computed')
                continue
            eswl = eswl_components[response_label][direction][component] / l_i * GD.UNIT_SCALE[unit]
            if component == 'total':
                line = '-'
                color = LINE_TYPE_SETUP['color'][i+2]
            else:
                line = LINESTYLE[j]
                color = LINE_TYPE_SETUP['color'][i+2]

            if go_final:
                component_label = naming[component]
            else:
                component_label = component#r'${}$'.format(component)#convert_for_latex(component)

            ax_eswl = axes[i].plot(eswl, 
                        nodal_coordinates['x0'], 
                        label = component_label,
                        linestyle = line,
                        color = color)

        
        eswl_leg = axes[i].legend(fontsize = 9)
        # calculate R for each direction -> to check if the signs are sensible
        
        #R_tot = sum(np.multiply(influences[response_label][direction], eswl_components[response_label][direction]['total']))
        R_i = sum(np.multiply(influences[response_label][direction], eswl_components[response_label][direction]['total']))
        if R_total:
            if response_label != 'Mx':
                R_i = R_i/R_total
                label_R_i = '{}'.format(convert_load_for_latex(response_label)) +\
                                        r'$_i$' +r'$($' +'{}'.format(force_label) +r'$)$' + r'$={}$'.format(round(R_i,2)) + ' of total    ' #+ r'$\quad$'

                r_i_ax =axes[i].plot(0,0, 
                                    label = label_R_i,
                                    linestyle = '')

                r_i_legend = axes[i].legend(loc= 'upper center', handles= r_i_ax, fontsize = 9)

        else:
            axes[i].plot(0,0, 
                label = r'${}$'.format(response_label) + r'$_i = $ ' + r'${:.2e}$'.format(round(R_i)),
                linestyle = ' ',
                color = 'k')

        # settings
        axes[i].locator_params(axis='x', nbins = 4)
        axes[i].locator_params(axis='y', nbins = 4)
        #axes[i].set_yticks([])
        #axes[i].set_yticklabels([])
        axes[i].set_ylim(bottom=0,top=200)
        if R_total:
            axes[i].add_artist(eswl_leg)
        if R_total:
            if response_label != 'Mx':
                axes[i].add_artist(r_i_legend)

            # props = dict(facecolor='lightgray')
            # r_i_text = AnchoredText(label_R_i,  loc='upper center')#, bbox_to_anchor=(0.5, -0.02))#, fontsize= 10),boxstyle='round', 
            # r_i_text.patch.set_boxstyle('round', prop= props)#, edgecolor = )
            # axes[i].add_artist(r_i_text)

        axes[i].set_xlabel('{}'.format(convert_load_for_latex(GD.DIRECTION_LOAD_MAP[direction], lower = True)) + r' ${}$'.format(unit_label))
        
        axes[i].grid()
    axes[0].set_ylabel('height [m]')

    if textstr:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        result_text = AnchoredText(textstr,  loc='lower right')#, fontsize= 10)bbox_to_anchor=(1.05, 0.8),
        axes[-1].add_artist(result_text)


    # saving of figure
    component_folder_dict = {'mean':'mean_parts',
                             'gle':'background_parts', 'lrc':'background_parts',
                             'resonant':'resonant_parts', 'resonant_m_lumped':'resonant_parts','resonant_m':'resonant_parts',
                             'all':'all_parts',
                             'mixed':'mixed_parts'}
    
    fname = 'for_'+ response_label
    if len(components_to_plot) == 1:
        key = components_to_plot[0]
    elif set(components_to_plot).issubset(['gle', 'lrc']):
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
    
    fname += '_' + gb_label + save_suffix

    full_dest = dest + os.path.sep + component_folder_dict[key] + os.path.sep + fname
    
    if options['savefig']:
        plt.savefig(full_dest)
        #plt.savefig(full_dest + '.svg')
        print ('\nsaved:',full_dest)
        #plt.close()
    
    if options['savefig_latex']:
        plt.savefig(dest_latex + os_sep + fname)
        print ('\nsaved:',dest_latex + os_sep + fname)

    if options['show_plots']:
        plt.show()

def plot_eswl_dyn_compare(result_dict, compare = ['dyn_est','eswl'], unit='N',
                         options = default_plot_options):

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
    
def plot_component_rate(eswl_dict, components_to_compare = ['total','mean','lrc', 'resonant_m_lumped'], 
                        options = default_plot_options):
    ''' 
    eswl_1,_2: already directional components of the eswl
    ''' 

    if options['update_plot_params']:
        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.rcParams.update(options['update_plot_params'])

    dest = os_join(*['plots', 'ESWL_plots', 'all_parts'])

    if components_to_compare[0] == 'all':
        # taking the keys from the first direction
        components = eswl_components[response_label]['y'].keys()
    else:
        components = components_to_compare
    naming = {'mean':'mean', 'gle':'beswl', 'resonant':'reswl', 'resonant_m':'reswl','resonant_m_lumped':'reswl','total':'total','lrc':'beswl'}
    directions_naming = {'Mz':'alongwind', 'My':'crosswind','Mx':'torsion'}
    fig, ax = plt.subplots(num='R_i compare')

    colors = ['tab:blue', 'dimgray', 'darkgray', 'lightgray']
    width = 0.2
    x = np.arange(len(eswl_dict))*0.3
    
    d_i = width/len(components)
    for e_i, eswl_resp in enumerate(eswl_dict):
        R_tot = eswl_dict[eswl_resp][1]
        direction = eswl_dict[eswl_resp][2][0]
        eswl = eswl_dict[eswl_resp][0][eswl_resp][direction]
        influences = eswl_dict[eswl_resp][3][eswl_resp]
        R_i = []
        for c_i, component in enumerate(components):
            R_i = sum(np.multiply(influences[direction], eswl[component])) / R_tot
            if e_i == 0:
                label = naming[component]
            else:
                label = None
            rect_ri = ax.bar(x[e_i] - width + c_i * d_i, R_i, width = d_i, color = colors[c_i], label = label)
            if eswl_resp in ['My','Mz']:
                if c_i == 0:
                    ax.text(x[e_i] - width + c_i * d_i, 1.02, directions_naming[eswl_resp])
            else:
                if c_i == 1:
                    ax.text(x[e_i] - width + c_i * d_i, 1.02, directions_naming[eswl_resp])
        if e_i != len(eswl_dict)-1:
            ax.axvline(x[e_i], linestyle = '--', color = 'grey')

    if options['present']:
        ax.legend(bbox_to_anchor = (1.01, 1), loc ='upper left')
    else:
        ax.legend(bbox_to_anchor = (0.5, -0.02), loc ='upper center', ncol = 4)
    
    ax.set_ylabel('rate of '+ r'$R_{total}$')
    ax.set_xticks([])

    save_title = 'component_compare'

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

# PEAK FACTOR

def plot_gb_eval(res_arrs, resp, direction,
                options = default_plot_options):

    dest = os_join(*['plots', 'ESWL_plots','gb_eval'])

    gb = np.arange(1, 5.1, 0.1)
    fig = plt.figure(num='gb_eval')

    for i, res_arr in enumerate(res_arrs):
        plt.plot(gb, res_arr/res_arr[0], label = convert_load_for_latex(resp[i]))

    plt.axvline(3, linestyle = '--', color= 'gray')
    plt.axvline(5, linestyle = '--', color= 'gray', label = 'typical range')
    plt.xlabel(r'$g_{b}$')
    plt.ylabel(r'$M_{i}/M_{i}(g_{b}=1.0)$')
    plt.legend()
    plt.locator_params(axis='y', nbins=6)
    plt.grid()


    save_title = 'gb_' + resp[0] + '_' + resp[1]# + '_' + direction

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

def plot_gb_eval_eswl(result_dict, nodal_coordinates, component = 'total', direction = 'automatic', unit = 'N',
                      options = default_plot_options):
    ''' 
    result_dict: key1: response, key2: gb_i
    ''' 
    
    if options['update_plot_params']:
        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.rcParams.update(options['update_plot_params'])

    dest = os_join(*['plots', 'ESWL_plots','gb_eval'])

    fig, axes = plt.subplots(ncols=len(result_dict), num='gb eval eswl', sharey=True)
    colors = ['tab:blue','tab:orange', 'tab:green']
    lines = [':', '--', '-']

    for r_i, response in enumerate(result_dict):

        axes[r_i].plot(nodal_coordinates['y0'], nodal_coordinates['x0'], 
                        label = r'structure', 
                        marker = 'o', 
                        color = 'grey', 
                        linestyle = '--')

        if direction == 'automatic':
            load_direction = GD.LOAD_DIRECTIONS_RESPONSES_UNCOUPLED[response][0]
        else:
            load_direction = direction

        unit_label = GD.UNITS_LOAD_DIRECTION[load_direction]
        unit_label = unit_label.replace(unit_label[1], unit)
        force_label = '{}'.format(convert_load_for_latex(GD.DIRECTION_LOAD_MAP[load_direction],lower=True))

        for gb_i, gb in enumerate(result_dict[response]):

            eswl = abs(result_dict[response][gb][load_direction][component] * GD.UNIT_SCALE[unit])

            if gb == 'default':
                gb = '3.5'

            axes[r_i].plot(eswl, nodal_coordinates['x0'],
                            label = gb,
                            color= colors[r_i],
                            linestyle = lines[gb_i])
                            
        axes[r_i].locator_params(axis='y', nbins = 4)
        axes[r_i].set_ylim(bottom=0,top=200)
        axes[r_i].set_xlabel('{}'.format(force_label) + r' ${}$'.format(unit_label))
        axes[r_i].grid()
        axes[r_i].legend()
        axes[r_i].set_title('ESWL for '+ convert_load_for_latex(response))
    
    axes[0].set_ylabel('height [m]')

    save_title = 'eswl_gb_eval_ ' + '_'.join(result_dict.keys())

    if options['savefig']:
        plt.savefig(dest + os_sep + save_title)
        plt.savefig(dest + os_sep + save_title + '.svg')
        print ('\nsaved:',dest + os_sep + save_title)
        #plt.close()
    
    if options['savefig_latex']:
        plt.savefig(dest_latex + os_sep + save_title)
        print ('\nsaved:',dest_latex + os_sep + save_title)

    if options['show_plots']:
        plt.show()

def plot_gb_absolute(result_dict, 
                     options = default_plot_options):

    if options['update_plot_params']:
        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.rcParams.update(options['update_plot_params'])

    dest = os_join(*['plots', 'ESWL_plots', 'gb_eval'])

    directions_naming = {'Mz':'alongwind', 'My':'crosswind','Mx':'torsion'}

    factor = 0.25
    colors = ['tab:blue', 'darkgray', 'gray',  'lightgray']
    fig, ax = plt.subplots(num='gb compare')
    x = np.arange(0,len(result_dict['labels'])/2,1) 
    width = 0.3
    x_labels = result_dict['labels'][::2]

    ax.axhline(3.5,linestyle = '--', color = colors[0],label= r'$g_{b}$' + ' assumed')

    est = result_dict['gb_val'][::2]
    qnt = result_dict['gb_val'][1::2]

    x1 = x + width/2
    x2 = x - width/2

    rects_dyn = ax.bar(x1[:4],  qnt, 
                    width, color = colors[0], label=r'$g_{b}$' + ' adjusted for quantile')
    
    rects_dyn = ax.bar(x2[:4],  est, 
                    width, color = colors[0], alpha=0.5, label=r'$g_{b}$' + ' adjusted for estimate')

    for i, g in enumerate(result_dict['gb_val'][::2]):
        ax.text(x[i] - width, g+0.1, r'${}$'.format(round(g,2)))
    for i, g in enumerate(result_dict['gb_val'][1::2]):
        ax.text(x[i] , g+0.1, r'${}$'.format(round(g,2)))
  
    dx = (x[2] - x[1])/2
    ax.axvline(x[1]+dx, linestyle = '--', color = 'grey')
    #ax.axvline(x[3]+dx, linestyle = '--', color = 'grey')
    

    # for r_i, r in enumerate(['Mz','My']):#,'Mx'
    #     if r == 'Mz':
    #         ax.text(x[result_dict['labels'].index(r)],4.7, directions_naming[r]) #1.9 for estimate, 4.5 for qnt
    #     elif r == 'My':
    #         ax.text(x[result_dict['labels'].index(r)],2, directions_naming[r]) #1.9 for estimate, 4.5 for qnt

    ax.text(x1[0] , 5.5, 'alongwind')#4.7
    ax.text(x1[2] , 5.5, 'crosswind')#2.5
    
    ax.legend(loc= 'upper right')#, bbox_to_anchor=(0.5, -0.15), ncol= 3)
    ax.set_ylabel(r'$g_{b}$')
    ax.set_xticks(x)
    ax.set_xticklabels([convert_for_latex(x) for x in x_labels])

    save_title = 'gb_abs_compare' + '_est_qnt'

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

# BACKGROUND

def plot_background_eval(result_dict, nodal_coordinates, directions = 'automatic', unit='N',
                         options= default_plot_options):
    
    if options['update_plot_params']:
        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.rcParams.update(options['update_plot_params'])

    dest = os_join(*['plots', 'ESWL_plots','background_parts'])

    dx = nodal_coordinates['x0'][1]
    l_i = np.full(len(nodal_coordinates['x0']), dx)
    l_i[0] /= 2
    l_i[-1] /= 2

    fig, axes = plt.subplots(ncols=len(result_dict), num='background eval eswl', sharey=True)
    colors = ['tab:blue','tab:orange', 'tab:green']
    lines = ['--', '-',':']
    h_nodes = {'base':0, '2/3H':7}
    for r_i, response in enumerate(result_dict):
        axes[r_i].plot(nodal_coordinates['y0'], nodal_coordinates['x0'], 
                        label = r'structure', 
                        marker = 'o', 
                        color = 'grey', 
                        linestyle = '--')

        if directions == 'automatic':
            load_direction = GD.LOAD_DIRECTIONS_RESPONSES_UNCOUPLED[response][0]
        else:
            load_direction = directions

        unit_label = GD.UNITS_LOAD_DIRECTION[load_direction]
        unit_label = unit_label.replace(unit_label[1], unit)
        force_label = '{}'.format(convert_load_for_latex(GD.DIRECTION_LOAD_MAP[load_direction],lower=True))
        shift_right = dict(My=0.5, Mz=0.55,Qy=0.55, Qz=0.5)

        #handles = ['structure']
        for h_i, h in enumerate(result_dict[response]):
            shift = 0
            if h == 'base':
                shift = 0.2

            box = dict(boxstyle="Round, pad=0.12",facecolor=colors[h_i], edgecolor=colors[h_i], alpha=0.2)
            axes[r_i].plot(0, nodal_coordinates['x0'][h_nodes[h]]+shift, marker='_', c = 'k', markersize=10)
            axes[r_i].text(shift_right[response], nodal_coordinates['x0'][h_nodes[h]]+3, h, bbox=box, fontsize = 9)

            
            for c_i, component in enumerate(['gle','lrc']):

                eswl = abs(result_dict[response][h].eswl_components[response][load_direction][component] / l_i * GD.UNIT_SCALE[unit])
                label = component + r'$_{{{}}}$'.format(h)
                line, = axes[r_i].plot(eswl, nodal_coordinates['x0'],
                                label = label,
                                color= colors[h_i],
                                linestyle = lines[c_i])

                            
        axes[r_i].locator_params(axis='y', nbins = 4)
        axes[r_i].set_ylim(bottom=0,top=200)
        axes[r_i].set_xlabel('{}'.format(force_label) + r' ${}$'.format(unit_label))
        axes[r_i].grid()
        axes[r_i].set_title('ESWL for '+ convert_load_for_latex(response))

    handles, labels = axes[0].get_legend_handles_labels()

    lgd = fig.legend(handles, labels,loc ='lower center', ncol = 5,bbox_to_anchor = (0.5, -0.02))
    #lgd = fig.legend(handles, labels,bbox_to_anchor = (1.0, 0.8), loc ='upper left', fontsize=8)
    axes[0].set_ylabel('height [m]')
    # legend appears over the labels, thus adjusting the bottom of the plot box 
    fig.subplots_adjust(bottom=0.2)

    
    save_title = 'background_eval_' + '_'.join(result_dict.keys())

    if options['savefig']:
        plt.savefig(dest + os_sep + save_title)#,bbox_extra_artists=(lgd,), bbox_inches='tight')
        #plt.savefig(dest + os_sep + save_title + '.svg')
        print ('\nsaved:',dest + os_sep + save_title)
        #plt.close()
    
    if options['savefig_latex']:
        plt.savefig(dest_latex + os_sep + save_title)
        print ('\nsaved:',dest_latex + os_sep + save_title)

    if options['show_plots']:
        plt.show()

def plot_quantify_background_methods(result_dict, nodal_coordinates, directions = 'automatic', unit='N', quantify = False, influences = None,
                         options= default_plot_options):
    
    if options['update_plot_params']:
        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.rcParams.update(options['update_plot_params'])

    dest = os_join(*['plots', 'ESWL_plots','background_parts'])

    R_i_abs = {}
    for response in result_dict:
        if directions == 'automatic':
            load_direction = GD.LOAD_DIRECTIONS_RESPONSES_UNCOUPLED[response][0]
        else:
            load_direction = directions

        unit_label = GD.UNITS_LOAD_DIRECTION[load_direction]
        unit_label = unit_label.replace(unit_label[1], unit)
        force_label = '{}'.format(convert_load_for_latex(GD.DIRECTION_LOAD_MAP[load_direction],lower=True))

        R_i_abs[response] = {'lrc':[],'gle':[]}

        fig, ax = plt.subplots(num='background abs'+response)
        for h_i, h in enumerate(result_dict[response]):

            for c_i, component in enumerate(['gle','lrc']):
                eswl_nodal = abs(result_dict[response][h].eswl_components[response][load_direction][component] * GD.UNIT_SCALE[unit])

                current_influence = abs(influences[response][h][load_direction])
                current_R =sum(np.multiply(current_influence, eswl_nodal))#/ GD.UNIT_SCALE[unit]
                print (response, h, component,':', current_R)
                print ('influences:',current_influence)
                print ('eswl:', eswl_nodal)#/ GD.UNIT_SCALE[unit]
                R_i_abs[response][component].append(current_R)
                
        factor = 1
        colors = ['tab:blue', 'darkgray', 'gray',  'lightgray']

        x = np.arange(0,2,1)* factor
        width = 0.15
        x_labels = result_dict[response].keys() # H
    
        res_lrc = abs(np.array(R_i_abs[response]['lrc']))
        res_gle = abs(np.array(R_i_abs[response]['gle']))

        print ('\nFor', response, 'differences are:')
        print ('at base', res_lrc[0] - res_gle[0])
        print ('at H', res_lrc[1] - res_gle[1])

        rects_lrc = ax.bar( x - width/2,res_lrc,  
                        width, color = 'w', hatch = '//' ,edgecolor=colors[0], label='lrc')
        rects_gle = ax.bar( x + width/2, res_gle,
                        width, color = 'w', hatch = 'O' ,edgecolor=colors[0], label='gle')
        dx = (x[1] - x[0])/2
        #ax.axhline(x[0]+dx, linestyle = '--', color = 'grey')
        #ax.axhline(x[3]+dx, linestyle = '--', color = 'grey')
        #ax.axhline(1,linestyle = '--', color = 'k',label= convert_for_latex('glob_max'))
        ax.legend(loc='upper right')
        ax.set_xlabel('absoulte reaction')
        ax.set_yticks(x)
        #ax.set_yticklabels(x_labels)

    save_title = 'background_eval_' + '_'.join(result_dict.keys()) + '_quantify'


    if options['savefig']:
        plt.savefig(dest + os_sep + save_title)#,bbox_extra_artists=(lgd,), bbox_inches='tight')
        #plt.savefig(dest + os_sep + save_title + '.svg')
        print ('\nsaved:',dest + os_sep + save_title)
        #plt.close()
    
    if options['savefig_latex']:
        plt.savefig(dest_latex + os_sep + save_title)
        print ('\nsaved:',dest_latex + os_sep + save_title)

    if options['show_plots']:
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

# INFLUENCES

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

# LOAD SIGNALS

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

def plot_load_time_histories_node_wise_figure(load_signals, n_nodes, discard_time, load_signal_labels = None, eigenmodes = None,
                                        options = default_plot_options):
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

    mode_1 = np.array([3.44e-05, 5.44e-05, 0.0001753, 0.000317]) *7000000
    mode_2 = np.array([3.44e-05, 5.44e-05, 0.0001753, 0.000317]) *-7000000

    ax_i = list(range(n_nodes-1, -1, -1))
    if load_signal_labels:
        components = load_signal_labels
    else:
        load_signals.pop('sample_freq')
        components = load_signals.keys()
    for component in components:
        # if component == 'sample_freq':
        #     break
        fig, ax = plt.subplots(num = 'signals of ' +  GD.DIRECTION_LOAD_MAP[component])
        fig.patch.set_visible(False)
        if not options['savefig_latex']:
            fig.suptitle('signal of ' + GD.DIRECTION_LOAD_MAP[component])

        max_previous = 0
        structure = []
        shift_right = 20
        cut_off = 10000
        for node in range(n_nodes):
            # show also mean 
            mean = np.mean(load_signals[component][node])

            extra = 0 
            if node==1:
                extra = 1000000

            if node == 0:
                label = 'mean'
                label1 = r'$X_{}(t)$'.format('z')
            else:
                label,label1 = None, None

            ax.hlines(mean + max_previous +extra, 
                        shift_right, len(load_signals[component][node]),
                        color = 'dimgray', 
                        linestyle = '--', 
                        label = label)#: ' + '{:.2e}'.format(mean))

            ax.plot(shift_right + np.arange(len(load_signals[component][node][:cut_off])),
                                load_signals[component][node][:cut_off] + max_previous +extra,
                                label = label1,
                                color= 'tab:blue')
            structure.append(mean + max_previous +extra)
            max_previous += max(load_signals[component][node][:cut_off])
            
            ax.grid()

            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            ax.set_xlim(right =len(load_signals[component][node][:cut_off]))
            

        dy = structure[1] - structure[0]

        base = [structure[0] - dy + 1000000, structure[0]]
        bottom = [-shift_right-20, 2*shift_right+20]
        ax.plot(np.full(2,shift_right), base, linestyle= '-', color = 'k')#, marker='o', markersize=30)


        ax.plot(np.full(n_nodes, shift_right), structure, linestyle= '-', color = 'k', marker='o', markersize=10)
        if eigenmodes:
            ax.plot([0,mode_1[1]], base, linestyle= '-', color = 'k')#, marker='o', markersize=30)
            #ax.plot([0,mode_2[1]], base, linestyle= '-', color = 'k')#, marker='o', markersize=30)
            ax.plot(mode_1, structure,  linestyle= '-', color = 'k', marker='o', markersize=10, alpha = 0.8)
            #ax.plot(mode_2, structure,  linestyle= '-', color = 'k', marker='o', markersize=10, alpha = 0.8)
        ax.set_ylim(bottom =base[0])

        ax.plot(bottom, np.full(2,base[0]), linestyle= '-', color = 'k', linewidth= 5)
        if not eigenmodes:
            ax.legend(loc= 'lower right')

        ax.set_xlabel(r'$t [s]$')

        ax.axis('off')

        save_title = 'dyn_load_' + str(n_nodes) + '_' + component + '_demo'

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

# MODE SHAPES

def plot_n_mode_shapes(mode_shapes_sorted, charact_length ,n = 3, options = default_plot_options, save_suffix=''):
    ''' 
    mode_shapes_sorted: mode shapres as a dictionary with dofs 
    charact_length: scale for the rotaional dofs
    ''' 

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

# # DYNAMIC ANALYSIS

def plot_dynamic_results(dynamic_analysis, dof_labels, node_id, result_variable, save_suffix = '', add_fft = False, include_extreme_value = None,
                         include_fft = False, log = True, unit = 'N',
                        options = default_plot_options):
    ''' 
    dynamic_analyis: analysis object
    dof_label: label of dof to plot
    node_id: node at whcih results to plot (starting from 0)
    result_variable: 'displacement','acceleration','reaction'
    ''' 
    mpl.rcParams.update(mpl.rcParamsDefault)

    plt.rcParams.update(options['update_plot_params'])

    dest = os_join(*['plots', 'ESWL_plots','dynamic_results'])

    try:
        if dynamic_analysis.name == 'dynamic_analysis':
            dynamic_analysis_solved = dynamic_analysis.solver 
    except AttributeError:
        dynamic_analysis_solved = dynamic_analysis

    cols = len(dof_labels)

    if not include_fft:
        fig, ax = plt.subplots(nrows = 1, ncols = cols, num='dyn_res')
        axes = [ax]
    else:
        fig, axes = plt.subplots(nrows = 2, ncols=cols, num='dyn_res')#, sharey='row')

    for d_i, dof_label in enumerate(dof_labels):
        if dof_label in GD.RESPONSE_DIRECTION_MAP.keys():
            response_label = dof_label
            dof_label = GD.RESPONSE_DIRECTION_MAP[dof_label]
        else:
            response_label = GD.RESPONSE_DIRECTION_MAP[dof_label]
        dof = GD.DOF_LABELS['3D'].index(dof_label) + (node_id * GD.DOFS_PER_NODE['3D'])
        dof_for_fft = dof_label

        if result_variable == 'displacement':
            result_data = dynamic_analysis_solved.displacement[dof, :] 
        elif result_variable == 'velocity':
            result_data = dynamic_analysis_solved.velocity[dof, :]
        elif result_variable == 'acceleration':
            result_data = dynamic_analysis_solved.acceleration[dof, :]
        elif result_variable == 'action':
            result_data = dynamic_analysis.force[dof, :]
        elif result_variable == 'reaction':
            if dof in dynamic_analysis_solved.structure_model.bc_dofs:# or dof in dynamic_analysis.structure_model.elastic_bc_dofs:
                result_data = dynamic_analysis_solved.dynamic_reaction[dof, :] * GD.UNIT_SCALE[unit]
            else:
                print ('\nReplacing the selected node by the ground node for reaction results')
                node_id = 0
                dof = GD.DOF_LABELS['3D'].index(dof_label)
                result_data = dynamic_analysis_solved.dynamic_reaction[dof, :] * GD.UNIT_SCALE[unit]
            #dof_label = response_label 

        digits = 2
        mean = round(np.mean(result_data), digits)
        std = round(np.std(result_data), digits)
        glob_max_id = np.argmax(result_data)
        glob_max = result_data[glob_max_id]   

        #print (dof_label, glob_max)     

        if include_extreme_value:
            extreme_value = utilities.extreme_value_analysis(dynamic_analysis_solved, dof_label) * GD.UNIT_SCALE[unit]
            if glob_max < 0:
                        extreme_value *= -1

        plot_title = result_variable.capitalize() + ' at node ' + str(node_id) + ' in ' + dof_label + ' direction'


        #fig, ax = plt.subplots(num=title)
        #ax = fig.add_subplot(111)
        #plt.title(plot_title + ' Vs Time ' + save_suffix)    # set title
        axes[0,d_i].set_xlabel(r'$t [s]$')
        axes[0,d_i].set_ylabel(convert_load_for_latex(response_label) + r' $[{}]$'.format(unit))

        axes[0,d_i].hlines(mean, dynamic_analysis_solved.array_time[0], dynamic_analysis_solved.array_time[-1], zorder=3, label=r'$\mu$', color = 'k')
        axes[0,d_i].hlines(mean + std, dynamic_analysis_solved.array_time[0], dynamic_analysis_solved.array_time[-1], zorder=3, label=r'$\mu +/- \sigma$', color = 'k', linestyle= '--')
        axes[0,d_i].hlines(mean - std, dynamic_analysis_solved.array_time[0], dynamic_analysis_solved.array_time[-1], zorder=3, color = 'k', linestyle= '--')

        axes[0,d_i].scatter(dynamic_analysis_solved.array_time[glob_max_id], glob_max, zorder=3, marker = 'o', color = 'r', s= 10, label = r'$max_{glob}$')
        axes[0,d_i].hlines(extreme_value, dynamic_analysis_solved.array_time[0], dynamic_analysis_solved.array_time[-1], color = 'r', linestyle= '--', label = r'$max_{est}$')
        
        label = dof_label
        axes[0,d_i].plot(dynamic_analysis_solved.array_time, result_data)#, label = label)

        axes[0,d_i].set_xlim(left =dynamic_analysis_solved.array_time[0], right=dynamic_analysis_solved.array_time[-1])
        axes[0,d_i].yaxis.set_tick_params(which='both', labelleft=True)
        axes[1,d_i].yaxis.set_tick_params(which='both', labelleft=True)
        
        
        #axes[0,d_i].legend()
        # unten 
        #axes[0,d_i].legend(bbox_to_anchor = (0.5, -0.4), loc ='upper center', ncol = 4)
        # rechts
        axes[0,-1].legend(bbox_to_anchor = (1.04, 1), loc ='upper left')
        axes[0,d_i].grid()

        if add_fft:
            plot_fft(dof_label=dof_for_fft, dynamic_analysis=[dynamic_analysis])  

        if include_fft:
            sway_naming = GD.MODE_CATEGORIZATION_REVERSE['3D'][dof_label]

            f_id = GD.CAARC_MODE_DIRECTIONS['0_deg'][sway_naming]
            eig_freqs = dynamic_analysis_solved.structure_model.eig_freqs
            natural_f = eig_freqs[f_id]
            sampling_freq = 1/dynamic_analysis_solved.dt
            dof_id = GD.DOF_LABELS['3D'].index(dof_label)
            time_series = dynamic_analysis_solved.dynamic_reaction[dof_id, :] * GD.UNIT_SCALE[unit]
            given_series = time_series
            is_type = 'reaction '

            #label = r'${}$'.format(response_label)

            freq_half, series_fft = utilities.get_fft(given_series, sampling_freq)

            if series_fft[0] == max(series_fft):
                freq_half, series_fft = freq_half[1:], series_fft[1:]

            axes[1,d_i].plot(freq_half, series_fft, linewidth = 0.5)
            if log:
                axes[1,d_i].set_xscale('log')
                axes[1,d_i].set_yscale('log')

            axes[1,d_i].set_xlim(right = 10)
            axes[1,d_i].axvline(natural_f, linestyle = '--', color = 'k', linewidth= 0.35,
                        label = r'$f$ ' + other_utils.prepare_string_for_latex(sway_naming) + r' = ${}$'.format(round(natural_f,2)) + r' $[Hz]$')
        
            if not log:
                axes[1,d_i].set_xlim(0.0,0.4)
                axes[1,d_i].set_ylim(bottom = 0)
            axes[1,d_i].set_ylabel(r'$S($' + convert_load_for_latex(response_label) + r'$)$')
            axes[1,d_i].set_xlabel(r'$f [Hz]$')

            axes[1,d_i].minorticks_off()
            
            if not log:
                axes[1,d_i].locator_params(axis='y', nbins=4)
                axes[1,d_i].xaxis.set_major_locator(MultipleLocator(0.1))
            else:
                axes[1,d_i].set_yticks([10e-5,10e-3,10e-1,10e1])
            axes[1,d_i].legend(loc= 'lower left', fontsize=8)

        axes[1,d_i].grid()
    
    if options['savefig'] or options['savefig_latex']:
        save_title = 'dyn_res_t_f_' + '_'.join(dof_labels)
    else:
        save_title = 'dynamic result of ' + '_'.join(dof_labels)

    if options['savefig']:
        plt.savefig(dest + os_sep + save_title)
        plt.savefig(dest + os_sep + save_title + '.svg')
        print ('\nsaved:',dest + os_sep + save_title)
        #plt.close()
    
    if options['savefig_latex']:
        plt.savefig(dest_latex + os_sep + save_title)
        print ('\nsaved:',dest_latex + os_sep + save_title)

    if options['show_plots']:
        plt.show()


def plot_fft(dof_label, dynamic_analysis = [None], damping_lables = [None], given_series = None, sampling_freq = None, natural_f = None, log =False,
            options = default_plot_options):
    ''' 
    either give it:
        - a list of dynamic analysis objects or 
        - directly a time series and the sample freqeuency
    
    dof_label: label of dof to plot. can be given as actual dof or as response from Qy,..., Mz
    ''' 
    dof_label = dof_label[1]
    mpl.rcParams.update(mpl.rcParamsDefault)

    plt.rcParams.update(options['update_plot_params'])

    sway_naming = GD.MODE_CATEGORIZATION_REVERSE['3D'][GD.RESPONSE_DIRECTION_MAP[dof_label]]

    f_id = GD.CAARC_MODE_DIRECTIONS['0_deg'][sway_naming]

    
    digits = 2

    if dof_label not in GD.DOF_LABELS['3D']:
        response_label = dof_label
        dof_label = GD.RESPONSE_DIRECTION_MAP[dof_label]

    
    dest = os_join(*['plots', 'ESWL_plots','dynamic_results'])
    fig, ax = plt.subplots(num='frequency_domain_result')
    is_type = 'action '
    max_list= []
    if dynamic_analysis[0]:
        for i, analysis in enumerate(dynamic_analysis):
            eig_freqs = analysis.structure_model.eig_freqs
            natural_f = eig_freqs[f_id]
            sampling_freq = 1/analysis.dt
            dof = GD.DOF_LABELS['3D'].index(dof_label)
            time_series = analysis.solver.dynamic_reaction[dof, :]
            given_series = time_series
            is_type = 'reaction '

            #label = r'${}$'.format(GD.DIRECTION_RESPONSE_MAP[dof_label])

            if len(dynamic_analysis) > 1:
                label = r'$\xi=$' + r'${}$'.format(damping_lables[i]) + r'$\%$'
            else:
                label = None

            freq_half, series_fft = utilities.get_fft(given_series, sampling_freq)

            if series_fft[0] == max(series_fft):
                freq_half, series_fft = freq_half[1:], series_fft[1:]

            max_val = max(series_fft)
            max_id = np.argmax(series_fft)
            current_line = ax.plot(freq_half, series_fft, label = label, linewidth = 0.35)
            clr = current_line[0].get_color()
            ax.plot(freq_half[max_id], max_val, marker = 'o', color = clr, markerfacecolor = clr, markersize= 3)
            max_list.append(max_val)

            if log:
                ax.set_xscale('log')
                ax.set_yscale('log')
            # idx = list(series_fft).index(max(series_fft))
            # ax.plot(0.2, max(series_fft), marker = 'o', label = 'at f='+ str(freq_half[idx]))

            if i == len(dynamic_analysis) -1 and natural_f:
                pos = {'Mz':natural_f+0.01,'My':0.01}
                sway_naming = sway_naming.replace('_',',')
                label = 'f_' + sway_naming
                label_f = convert_for_latex(label) + r'=${}$'.format(round(natural_f,2)) + r' $Hz$'
                ax.axvline(natural_f, linestyle = '--', color = 'darkgray', linewidth= 0.7)
                y = max_list[0] - max_list[1]
                ax.text(pos[response_label], y*1.5, label_f)

    # if given_series and sampling_freq:
    #     freq_half, series_fft = utilities.get_fft(given_series, sampling_freq)
    #     ax.plot(freq_half, series_fft, label = label)
    
    if not log:
        ax.set_xlim(0.0,0.4)
        ax.set_ylim(bottom = 0)
    ax.set_ylabel(r'$S($' + convert_load_for_latex(response_label) + r'$)$')
    ax.set_xlabel(r'$f [Hz]$')

    if not log:
        ax.locator_params(axis='y', nbins=5)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))

    plt.grid(True)
    plt.legend(bbox_to_anchor = (0.5, -0.4), loc ='upper center', ncol = 2)


    save_title = 'fft_' + GD.DIRECTION_RESPONSE_MAP[dof_label]
    if damping_lables:
        save_title += '_damping_compare'

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


# # OPTIMIZATION

def plot_objective_function_2D(objective_function, opt_res ,evaluation_space = [-5,5, 0.2],design_var_label = 'g_b',
                                options = default_plot_options):

    print ('\nEVALUATE AND PLOT OBJECTIVE FUNCTION\n')

    fig, ax = plt.subplots(figsize=(5,3), num='objective_func_1D')

    
    x = np.arange(evaluation_space[0], evaluation_space[1], evaluation_space[2])
    result = np.zeros(len(x))
    for i, val in enumerate(x):
        result[i] = objective_function(val)
    
    ax.plot(x, result)
    
    ax.plot(opt_res, objective_function(opt_res),linestyle = 'None', marker='o',mfc='r',mec='k', ms=4, label='optimized variable ' + str(round(opt_res,4)))

    # if optimization_object.optimized_design_params:
    #     ax.vlines(optimization_object.optimized_design_params, min(result), max(result), 
    #                 #label = 'optimized variable: ',# + str(round(optimization_object.final_design_variable, 2)), 
    #                 color = 'r', 
    #                 linestyle ='--')
    ax.set_title('optimizable function')
    ax.set_xlabel('values of ' + design_var_label)
    ax.set_ylabel(r'$ f  = (|tar| - |cur|) ^{2} / tar ^{2}$')
    ax.grid()
    ax.legend()
    # if self.savefig:
    #     plt.savefig(dest_mode_results)

    if options['show_plots']:
        plt.show()
    
    #plt.close()
    



