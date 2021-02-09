import numpy as np
import sys 
from source.ESWL.ESWL import ESWL
from source.auxiliary.auxiliary_functionalities import get_influence


class BESWL(object):

    def __init__(self,strucutre_model, influence_function, load_signals, load_directions, response):
        '''
        wird für jede gesuchte Antwort Größe initialisiert 
        '''

        self.strucutre_model = strucutre_model
        self.influence_function = influence_function
        self.load_signals = load_signals
        self.response = response
        self.load_directions = load_directions # TODO: find which directions are available and sensible

        self.weighting_factors_raw = {self.response: {}} # first key is repsonse, 2nd direction
        self.spatial_distribution = {self.response: {}} #keys are directions values arrays of z

        self.get_rms_background_response()
        self.get_spatial_distribution()
        self.get_weighting_factor()

    
    def get_spatial_distribution(self):
        for direction in self.load_directions:
            self.spatial_distribution[self.response][direction] = np.zeros(self.strucutre_model.n_nodes)

            for i in range(1, self.strucutre_model.n_nodes):
                self.spatial_distribution[self.response][direction][i] = np.std(self.load_signals[direction][i])


    def get_weighting_factor(self):
        '''
        makes the spatial distribution response specific
        '''
        for direction in self.load_directions:

            w_b_s = 0.0
            for l in self.load_directions:
                static_load_response = self.get_static_load_response(l)
                B_sl = self.get_B_sl(direction, l)
                print ('B_sl for', self.response, 'of', str(direction), '&', str(l), ' ', B_sl)
                w_b_s += B_sl * static_load_response

            self.weighting_factors_raw[self.response][direction] = w_b_s

    def get_static_load_response(self, direction):
        '''
        current response du to unit load in direction s
        in the formula this is sig'_Rb
        '''
        sig_Rb = 0.0
        for i, z in enumerate(self.strucutre_model.nodal_coordinates["x0"][1:],1):
            sig_Rb += np.std(self.load_signals[direction][i]) * self.influence_function[self.response][direction][i]
        
        return sig_Rb
    

    def get_B_sl(self, s, l):
        '''
        factor representing loss of correlation of wind loads in different directions s & l
        '''
        B_sl_numerator = 0.0
        for i, z1 in enumerate(self.strucutre_model.nodal_coordinates["x0"][1:],1):
            for j, z2 in enumerate(self.strucutre_model.nodal_coordinates["x0"][1:],1): 
                mü_1 = self.influence_function[self.response][s][i]
                mü_2 = self.influence_function[self.response][l][j]
                covariance_matrix = np.cov(self.load_signals[s][i], self.load_signals[l][j]) # -> np.cov gives matrix.. 
                covariance = covariance_matrix[0][1]
                correlation_coefficient = covariance /(np.sqrt(covariance_matrix[0][0])*np.sqrt(covariance_matrix[1][1]))
                B_sl_numerator += mü_1 * mü_2 * covariance

        sig_Rb_s = self.get_static_load_response(s)
        sig_Rb_l = self.get_static_load_response(l)

        if sig_Rb_l == 0 or sig_Rb_s == 0:
            return 0.0

        else:
            return B_sl_numerator / (sig_Rb_s * sig_Rb_l)

    def get_rms_background_response(self):
        result = 0.0
        for s in self.load_directions:
            for l in self.load_directions:
                sig_Rb_s = self.get_static_load_response(s)
                sig_Rb_l = self.get_static_load_response(l)
                B_sl = self.get_B_sl(s , l)
                result += sig_Rb_s * sig_Rb_l * B_sl

        self.rms_background_response = result
        #return result #TODO: or the sqrt of the result ?
