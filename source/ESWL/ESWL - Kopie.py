import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import source.postprocess.plotter_utilities as plotter_utilities
from source.auxiliary.auxiliary_functionalities import get_influence

class ESWL(object):

    def __init__(self, structure_model, eigenvalue_analysis, response, load_signals, load_directions = ['x', 'y', 'z', 'a', 'b', 'g']): # not sure if this can be done with a list 
        '''
        is initialize for each response
        [Qx, Qy, Qz, Mx, My, Mz]
        '''
        self.structure_model = structure_model
        self.eigenvalue_analysis = eigenvalue_analysis
        self.load_signals = load_signals
        self.load_directions = load_directions
        self.response = response

        # do this at creation since it is needed multiple times and pass it to R and B
        self.influences = {self.response: {}}
        for direction in self.load_directions:
            self.influences[self.response][direction] = np.zeros(self.structure_model.n_nodes)
            for node in range(1,self.structure_model.n_nodes):
                influence = get_influence(self.structure_model, direction, node, self.response)
                self.influences[self.response][direction][node] = influence
        #self.g_r = self._get_peak_factor('resonant', self.structure_model.eig_freqs[:3])

        self.eswl_total = {}
        self.mean_load = {}
        self.beswl_total = {}
        self.reswl_total = {}

        self.eswl_components = {}


    def calculate_total_ESWL(self):
        from source.ESWL.BESWL import BESWL
        from source.ESWL.RESWL import RESWL

        #########!!!! für jede direction wird jetzt ein neues objekt erzeugt und immer wieder initialisiert -> manche variablen sind aber Richtungs unabhängig 
        self.BESWL = BESWL(self.structure_model, self.influences, self.load_signals, self.load_directions, self.response)
        self.RESWL = RESWL(self.structure_model, self.influences, self.eigenvalue_analysis, self.load_signals, self.load_directions, self.response) # this is response specific 

        # initialized dictionary of total eswl -> 
        self.eswl_total[self.response] = {}
        self.mean_load[self.response] = {}
        self.beswl_total[self.response] = {}
        self.reswl_total[self.response] = {}

        self.eswl_components[self.response] = {}

        for direction in self.load_directions:
            #mean = 0.5 * self.structure_model.parameters["rho"] * mean_u**2 * self.structure_model.parameters["intervals"][i]["ly"] * C_d
            self.mean_load[self.response][direction] = np.zeros(self.structure_model.n_nodes)
            for i in range(1,len(self.mean_load)):
                self.mean_load[self.response][direction][i] = np.mean(self.load_signals[direction][i])
            
            R_max = self._get_maximum_response(self.response)

            # TODO: check if the multiplication with peak factor hear is correct-> must this be done twice: 
            # still not sure -> seems to be correct since g is also in R_max
            g_b = self._get_peak_factor('background')
            w_b = self.BESWL.weighting_factors_raw[self.response][direction] * g_b / R_max
            p_z_b = self.BESWL.spatial_distribution[self.response][direction] * g_b  
            self.beswl_total[self.response][direction] = w_b * p_z_b
            
            w_r = self.RESWL.weighting_factors_raw[self.response][direction] 
            p_z_r = self.RESWL.spatial_distribution[self.response][direction] 
            p_z_r_e = np.zeros(self.structure_model.n_nodes)
            for mode_id in range(3):
                g_r = self._get_peak_factor('resonant', self.structure_model.eig_freqs[mode_id])
                p_z_r_e += w_r[mode_id]* g_r / R_max * p_z_r[mode_id] * g_r  
            
            self.reswl_total[self.response][direction] = p_z_r_e

            self.eswl_total[self.response][direction] = self.mean_load[self.response][direction] + \
                                                        self.beswl_total[self.response][direction] + \
                                                        self.reswl_total[self.response][direction]

            self.eswl_components[self.response][direction] = {}
            self.eswl_components[self.response][direction]['mean'] = self.mean_load[self.response][direction] 
            self.eswl_components[self.response][direction]['background'] = self.beswl_total[self.response][direction]
            self.eswl_components[self.response][direction]['resonant'] = self.reswl_total[self.response][direction]
            self.eswl_components[self.response][direction]['total'] = self.eswl_total[self.response][direction]
                

            print ('\n added ESWL for direction', direction)

    def _get_peak_factor(self, response_type = None , frequency = None):
        if not response_type:
            return 3.5

        elif response_type == 'background':
            return 3.5

        elif response_type == 'resonant':
            #g_r = np.zeros(3)
            #for freq in frequencies:
            g_r = np.sqrt(2*np.log(600*frequency)) + 0.5772/np.sqrt(2*np.log(600*frequency))
            #print ('resonant peak factor calculated for a period of 10 min\n')
            return g_r

        elif response_type == 'extreme_value':
            pass
            # z.B. extreme value analysis or as Kaspersik

    def _get_maximum_response(self, response):
        
        # background
        g_b = self._get_peak_factor('background') 
        R_b = g_b**2 * self.BESWL.rms_background_response 

        # resonant 
        g_r = self._get_peak_factor('resonant', self.structure_model.eig_freqs[0])
        R_r = g_r**2 * self.RESWL.rms_resonant_response 

        return np.sqrt(R_b + R_r)

    def plot_eswl_load_components(self, components, response_label, total_only = True):
        
        plotter_utilities.plot_load_components(self.eswl_components, 
                                                self.structure_model.nodal_coordinates,
                                                components,
                                                response_label,
                                                total_only)


    def postprocess(self):
        pass
        #self.plot_load_components()

