import numpy as np
from os.path import join as os_join
import matplotlib.pyplot as plt 

def get_CAARC_eigenmodes_version_old (src_file_name = 'CAARC_advanced_eigenmodes.txt'):
    '''
    here the modes are saved by the string of the mode id
    '''
    src = os_join(*['input', src_file_name])
    eigenmodes = {}
    eigenmodes['storey'] = np.flip(np.loadtxt(src, usecols = (0,))) # [-]
    eigenmodes['storey_level'] = np.flip(np.loadtxt(src, usecols = (1,))) # [m]
    eigenmodes['mass'] = 1231000.0
    for i in range (3):
        mode_id = str(i+1)
        eigenmodes[mode_id] = {}
        # from here use the PraOptBeam naming convention 
        eigenmodes[mode_id]['x'] = np.zeros(60) #include dummy defomration in longitudinal direction ParOpt deformed carries this 
        eigenmodes[mode_id]['y'] = np.flip(np.loadtxt(src, usecols = (3+3*i,))) # displacement in y direction 
        eigenmodes[mode_id]['z'] = np.flip(np.loadtxt(src, usecols = (2+3*i,)))# x axis CAARC definition gets z axis ParOpt
        eigenmodes[mode_id]['a'] = np.flip(np.loadtxt(src, usecols = (4+3*i,))) # torsional axis z in CAARC gets a = alpha in ParOpt

    return eigenmodes

def get_CAARC_eigenmodes(src_file_name = 'CAARC_advanced_eigenmodes.txt'):
    '''
    here the eigenmodes are appened to a list, like this they must not be accessed by theirs string
    a dummy is included such that id 1 is at position 1 in the list
    '''
    src = os_join(*['input', src_file_name])
    eigenmodes = {}
    eigenmodes['storey'] = np.flip(np.loadtxt(src, usecols = (0,))) # [-]
    eigenmodes['storey_level'] = np.flip(np.loadtxt(src, usecols = (1,))) # [m]
    eigenmodes['mass'] = 1231000.0
    eigenmodes['shape'] = []
    eigenmodes['shape'].append({'dummy': None}) # that the modes can be accesed by theirs ids 

    for i in range (3):
        eigenmodes['shape'].append({
        # from here use the PraOptBeam naming convention 
        'x': np.zeros(60), #include dummy defomration in longitudinal direction ParOpt deformed carries this 
        'y': np.flip(np.loadtxt(src, usecols = (3+3*i,))), # displacement in y direction 
        'z': np.flip(np.loadtxt(src, usecols = (2+3*i,))),# x axis CAARC definition gets z axis ParOpt
        'a': np.flip(np.loadtxt(src, usecols = (4+3*i,))) # torsional axis z in CAARC gets a = alpha in ParOpt
        })

    return eigenmodes

def get_CAARC_eigenform_polyfit (CAARC_eigenmodes, evaluate_at = None, degree = 5):
    '''
    retruns the values of a fitted caarc eigenmode.
    evaluate_at must be a list of x coordiantes at which the fitted curve should be evaluated.
    if it is not provided the fitted curve is evaluated at each storey level of caarc.
    '''
    eigenmodes_fitted = {} 
    #CAARC_eigenmodes = self.structure_model.CAARC_eigenmodes
    # returns the fitted polynomial and the discrete array of displacements
    if not evaluate_at:
        x = CAARC_eigenmodes['storey_level']
    else:
        x = evaluate_at 
    eigenmodes_fitted['storey_level'] = np.copy(x)
    eigenmodes_fitted['shape'] = []
    eigenmodes_fitted['shape'].append({'dummy for 0 mode': None})
    for mode_id in range(1,4):
        eigenmodes_fitted['shape'].append({})
        for dof_label in ['y', 'z', 'a']:
            y = CAARC_eigenmodes['shape'][mode_id][dof_label]
            current_polynomial = np.poly1d(np.polyfit(CAARC_eigenmodes['storey_level'],y,degree))
            values = []
            for x_i in x:# evaluate the fitted eigenmode at certain intervals
                values.append(current_polynomial(x_i))
            eigenmodes_fitted['shape'][mode_id][dof_label] = np.asarray(values)

    return eigenmodes_fitted

def get_m_eff(eigenmodes_dict, mode_id, main_direction_only, print_to_console):
    '''
    retruns the generalized mass and the participation factor of a mode 
    prints the effective mass that should be around 60% of the total mass (first modes)
    '''

    mass = eigenmodes_dict['mass'] # constant over height
    phi_y = eigenmodes_dict['shape'][mode_id]['y']
    phi_z = eigenmodes_dict['shape'][mode_id]['z']

    if main_direction_only:
        if mode_id == 1:
            participation_factor = (mass * sum(phi_y))**2 # mass not in the sum since it is constant
        elif mode_id == 2:
            participation_factor = (mass * sum(phi_z))**2
    else:
        participation_factor = (mass * sum(np.add(phi_y, phi_z)))**2


    if main_direction_only:
        if mode_id == 1:
            generalized_mass = mass * sum(np.square(phi_y))
        elif mode_id == 2:
            generalized_mass = mass * sum(np.square(phi_z))
    else:
        generalized_mass = mass * sum(np.add(np.square(phi_y), np.square(phi_z)))

    total_mass = 60*mass
    m_eff = participation_factor/generalized_mass 

    if print_to_console:
        print('m_eff of mode_id', mode_id,  ':', round(m_eff/total_mass, 4), 'of m_tot')  
        print ('generalized_mass:', round(generalized_mass, 2), 'should be 1 if mass normalized')     
        print ('participation_factor:', round(participation_factor,2)) 
        print () 

    return participation_factor, generalized_mass  

# eigenmodes = get_CAARC_eigenmodes_1() 
# print(eigenmodes['shape'][0]['y'])
# # get_m_eff(eigenmodes, 1)
