import numpy as np
import sys 
import math
from source.ESWL.ESWL import ESWL
from source.ESWL.eswl_auxiliaries import get_influence, rms
import warnings
import source.auxiliary.global_definitions as GD

class BESWL(object):

    def __init__(self,strucutre_model, influence_function, load_signals, load_directions, response):
        '''
        wird für jede gesuchte Antwort Größe initialisiert 
        '''

        self.strucutre_model = strucutre_model
        self.influence_function = influence_function
        self.load_signals = load_signals
        self.response = response
        self.load_directions = load_directions 

        self.rho_collection_all = {}

        self.weighting_factors_raw = {self.response: {}} # first key is repsonse, 2nd direction
        self.spatial_distribution = {self.response: {}} # keys are directions, values arrays of z

        self.get_rms_background_response()
        self.get_spatial_distribution()
        self.get_weighting_factor()

    
    def get_spatial_distribution(self):
        '''
        if using Kareem this is the gust envelope = rms/std of load at each point
        Kareem eq. 25 / 32
        '''
        for direction in self.load_directions:
            self.spatial_distribution[self.response][direction] = np.zeros(self.strucutre_model.n_nodes)

            for i in range(self.strucutre_model.n_nodes):
                self.spatial_distribution[self.response][direction][i] = np.std(self.load_signals[direction][i])

    def get_weighting_factor(self):
        '''
        makes the spatial distribution response specific
        Kareem eq. 33
        '''
        self.intermediate_w_factors = {} #used to track it for some sign stuff
        for s in self.load_directions:
            self.intermediate_w_factors[s] = []
            w_b_s = 0.0
            for l in self.load_directions:
                static_load_response = self.get_static_load_response(l)
                B_sl = self.get_B_sl(s, l)
                w_b_s += B_sl * static_load_response

                if round(B_sl,3) != 0:
                    print ('B_sl for', self.response, 'of', str(s), '&', str(l), ' ', round(B_sl,3), 'static load response sig_'+l, round(static_load_response,0),
                        '    w_b_s_i:', round(B_sl * static_load_response, 2))
                    # track the sign ob w_b_s 
                    self.intermediate_w_factors[s].append(B_sl * static_load_response)

            self.weighting_factors_raw[self.response][s] = w_b_s

            # if self.intermediate_w_factors[s]:
            #     print ('    for direction', s, 'the sum of w_b_s used for w_b_s_total is composed by:', self.intermediate_w_factors[s], '\n')

    def get_static_load_response(self, direction):
        '''
        current response due to rms/std of load in direction s
        in the formula this is σ'_Rb
        Kareem eq. 22
        '''
        
        mean_z = np.array([np.mean(self.load_signals[direction][i]) for i in range(self.strucutre_model.n_nodes)])

        # TODO: is it necessary to take the sign here? guess yes since the coupled loads counteract otherwise
        # need to take signs since they matter in the coupled case b - z for My 
        signs = np.divide(mean_z, abs(mean_z))

        std_z = np.array([np.std(self.load_signals[direction][i]) for i in range(self.strucutre_model.n_nodes)])
        signed_std_z = np.multiply(signs, std_z)
        sig_Rb = sum(np.multiply(signed_std_z, self.influence_function[self.response][direction]))
        
        return sig_Rb

    def get_B_sl(self, s, l):
        '''
        factor representing loss of correlation of wind loads in different directions s & l and different heights
        Kareem eq. 21
        '''
        self.correlation_coefficient = []
        B_sl_numerator = 0.0
        for node1 in range(self.strucutre_model.n_nodes):
            for node2 in range(self.strucutre_model.n_nodes): 
                # influences
                mü_1 = self.influence_function[self.response][s][node1]
                mü_2 = self.influence_function[self.response][l][node2]
                # covatriance
                covariance_matrix = np.cov(self.load_signals[s][node1], self.load_signals[l][node2]) # -> np.cov gives matrix.. 
                covariance = covariance_matrix[0][1]
                self.correlation_coefficient.append(covariance /(np.sqrt(covariance_matrix[0][0])*np.sqrt(covariance_matrix[1][1])))
                # numerator
                B_sl_numerator += mü_1 * mü_2 * covariance

        sig_Rb_s = self.get_static_load_response(s)
        sig_Rb_l = self.get_static_load_response(l)

        if sig_Rb_l == 0 or sig_Rb_s == 0:
            return 0.0

        elif abs(sig_Rb_l) < 1e-5 and (sig_Rb_l) < 1e-5:
            # this seems to be numerically or so be possible but makes things wrong
            print ('static load response very small for:', s, '&',l)
        else:
            B_sl = B_sl_numerator / (sig_Rb_s * sig_Rb_l) 
            if round(B_sl,3) > 1.0:
                print ('Warning: B_sl of', s, 'at node_'+str(node1), l ,'at node_'+str(node2),'is larger than 1', B_sl)
                #raise Exception('B_sl is larger then 1:', B_sl)
            return B_sl

    def get_rms_background_response(self):
        '''
        gives the background part of the response.
        is called rms but is the rms of mean zero signal -> std
        Kareem eq. 20
        '''
        result = 0.0
        for s in self.load_directions:
            for l in self.load_directions:
                sig_Rb_s = self.get_static_load_response(s)
                sig_Rb_l = self.get_static_load_response(l)
                B_sl = self.get_B_sl(s , l)
                result += sig_Rb_s * sig_Rb_l * B_sl

        # here sqrt since sig*sig = sig² and rms/std is sig^1
        # NOTE -> returning the square, since this is used for the R-max calculation
        self.rms_background_response = result

    # # LRC coefficient according to Kaspersik 
    def get_columns_from_header(self, file):
        '''
        returns list with all column identifiers
        -> turns them all to lower cases 
        '''
        with open(file, 'r') as f:
            f.seek(0)
            first_line = f.readline()
            offset = len(first_line)
            f.seek(offset)
            second_line = f.readline()
        second_line = second_line.split()

        columns = []
        for word in second_line:
            columns.append(word)
        columns.pop(0)

        return columns

    def lrc_coeff(self, load_direction, print_infos = False):
        '''
        compute the LRC(load - response - correlation) coefficient according to KAsperski 1992.
        several (own) methods are tested and compared
        '''
        # also use the actual response time history 
        self.rho_collection_all[load_direction] = {}
        src_response = 'input\\force\\generic_building\\force_structure.dat'
        columns = self.get_columns_from_header(src_response)
        direction = GD.RESPONSE_DIRECTION_MAP[self.response]
        force_kratos = GD.DIRECTION_LOAD_MAP[direction]
        id_r = columns.index(force_kratos)
        response_time_hist = np.loadtxt(src_response, usecols=id_r+6)# +6 for body attached 

        shape_dif = response_time_hist.shape[0] - self.load_signals['x'][0].shape[0]

        # sig R_t: std of the response time history
        std_R_time_hist = np.std(response_time_hist[shape_dif:])

        # JZ: eq. 4.7, Kasperski: Eq. 3
        cov_Rps = np.zeros(self.strucutre_model.n_nodes) # covariance between reposne R and load p_s
        for node1 in range(self.strucutre_model.n_nodes):
            for node2 in range(self.strucutre_model.n_nodes):
                for l in self.load_directions: #l

                    I_pl_z = self.influence_function[self.response][l][node2]
                    cov_psl = np.cov(self.load_signals[load_direction][node1], self.load_signals[l][node2])[0][1]

                    cov_Rps[node1] += I_pl_z * cov_psl
        
        # standard deviation of the response using covariance method Kasperski (eq. 1)
        sig_R_2 = 0.0 # scalar value: just the rms/std of the response 
        for node1 in range(self.strucutre_model.n_nodes):
            for node2 in range(self.strucutre_model.n_nodes):
                for s in self.load_directions: #s
                    for l in self.load_directions: #l
                        I_ps_z = self.influence_function[self.response][s][node1]
                        I_pl_z = self.influence_function[self.response][l][node2]

                        cov_psl = np.cov(self.load_signals[s][node1], self.load_signals[l][node2])[0][1]

                        sig_R_2 += I_ps_z * I_pl_z * cov_psl

        std_R_cov_method = np.sqrt(abs(sig_R_2))
        rho_Rps_corrcoef = np.zeros(self.strucutre_model.n_nodes)

        for node in range(self.strucutre_model.n_nodes):
            load = self.load_signals[load_direction][node]
            rho_Rps_corrcoef[node] = np.corrcoef(response_time_hist[shape_dif:], load)[0][1]
    

        # TODO: here use std or rms 
        rms_ps = np.zeros(self.strucutre_model.n_nodes)
        std_ps = np.zeros(self.strucutre_model.n_nodes)
        cov_R_ps_np = np.zeros(self.strucutre_model.n_nodes)
        for i, signal in enumerate(self.load_signals[load_direction]):
            rms_ps[i] = rms(signal)
            std_ps[i] = np.std(signal)
            cov_R_ps_np[i] = np.cov(response_time_hist[shape_dif:], signal)[0][1]

        rho_Rps_cov_manual_R_covMethod = np.divide(cov_Rps, std_ps*std_R_cov_method)
        rho_Rps_cov_manual_R_timeHist = np.divide(cov_Rps, std_ps*std_R_time_hist) 
        rho_Rps_cov_timehist_R_timeHist = np.divide(cov_R_ps_np, std_ps* std_R_time_hist) 

        # # 1. differences between the methods when computing standard deviantion and covavriances
        if print_infos:
            print('\nstd of response', self.response)
            print('with np.std form time history:', round(std_R_time_hist))
            print('with covariance method:       ', round(std_R_cov_method))
            print('with B_sl:                    ', round(np.sqrt(self.rms_background_response)))
            dif = round(std_R_cov_method-std_R_time_hist)
            print('difference cov method - time hist:', dif,
             '; std of cov method is', str(round(std_R_cov_method/std_R_time_hist * 100, 1))+'%', 'of time history std')
            print()
            print ('covariance calculations cov_Rps -',self.response,'-',load_direction)
            print ('with Kasperski using influences: ', cov_Rps)
            print ('with np.cov using time histories:', cov_R_ps_np)
        
        # # 2. differences between the methods when computing the correlation coefficient 
        # using corrcoeff of numpy
        self.rho_collection_all[load_direction]['rho_Rps_corrcoef'] = rho_Rps_corrcoef
        # using eq. 3 for cov_Rps and covariance method for std of Response R
        self.rho_collection_all[load_direction]['rho_Rps_cov_manual_R_covMethod'] = rho_Rps_cov_manual_R_covMethod
        # using eq. 3 for cov_Rps and time history of R for std of Response R
        self.rho_collection_all[load_direction]['rho_Rps_cov_manual_R_timeHist'] = rho_Rps_cov_manual_R_timeHist
        # using time histories of R and p for cov_Rps and time history of R for std of Response R
        self.rho_collection_all[load_direction]['rho_Rps_cov_timehist_R_timeHist'] = rho_Rps_cov_timehist_R_timeHist

        # select one of them for further calculation
        rho_Rps = rho_Rps_cov_manual_R_covMethod

        return rho_Rps

    def get_beswl_LRC(self, load_direction):
        '''
        computes the LRC distribution of beswl for a given load direction
        this still needs to be multiplied with the peak factor g_b
        '''
        rho = self.lrc_coeff(load_direction)
        sig_ps = np.zeros(self.strucutre_model.n_nodes)
        for i, signal in enumerate(self.load_signals[load_direction]):
            sig_ps[i] = np.std(signal)

        return np.multiply(rho, sig_ps)

    