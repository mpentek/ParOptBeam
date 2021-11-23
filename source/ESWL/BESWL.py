import numpy as np
from source.ESWL.ESWL import ESWL
import source.auxiliary.global_definitions as GD

class BESWL(ESWL):

    def __init__(self,strucutre_model, influence_function, load_signals, response, options):
        '''
        Background Part of the ESWL
        Can be calculated with the GLE and the LRC approach respectivley (use options to set)
        - influence_function: influence of all load_directions on the current response
        - load_signals: time history of measured loads
        - load_directions: directions of the load to compute 
        - response: current response under consideration
        '''

        self.strucutre_model = strucutre_model
        self.influence_function = influence_function
        self.load_signals = load_signals
        self.response = response
        self.load_directions = GD.LOAD_DIRECTION_MAP['all']

        self.options = options     

        self.get_rms_background_response()

        if self.options['lrc']:
            self.get_weighting_factor_lrc()
            self.get_spatial_distribution_lrc()
            
        if self.options['gle']:
            self.get_spatial_distribution_gle()
            self.get_weighting_factor_gle()

    # ===========================================
    # # GLE Approach according to Chen and Kareem
    # ===========================================

    def get_spatial_distribution_gle(self):
        '''
        if using Kareem this is the gust envelope = rms/std of load at each point
        Kareem eq. 25 / 32
        '''
        self.spatial_distribution_gle = {self.response: {}} # keys are directions, values arrays of z
        for direction in self.load_directions:
            self.spatial_distribution_gle[self.response][direction] = np.zeros(self.strucutre_model.n_nodes)

            for i in range(self.strucutre_model.n_nodes):
                self.spatial_distribution_gle[self.response][direction][i] = np.std(self.load_signals[direction][i])

    def get_weighting_factor_gle(self):
        '''
        only if gle is used 
        makes the spatial distribution response specific
        Kareem eq. 33
        '''
        self.weighting_factors_raw_gle = {self.response: {}} 
        for s in self.load_directions:
            w_b_s = 0.0
            for l in self.load_directions:
                static_load_response = self.get_static_load_response(l)
                B_sl = self.get_B_sl(s, l)
                w_b_s += B_sl * static_load_response 
                if round(B_sl,3) != 0:
                    # print ('B_sl for', self.response, 'of', str(s), '&', str(l), ' ', round(B_sl,3), 'static load response sig_'+l, round(static_load_response,0),
                    #     '    w_b_s_i:', round(B_sl * static_load_response, 2))
                    pass

            self.weighting_factors_raw_gle[self.response][s] = w_b_s

    def get_static_load_response(self, direction):
        '''
        current response due to rms/std of load in direction s at z
        in the formula this is σ'_Rb
        Kareem eq. 22
        '''
        
        mean_z = np.array([np.mean(self.load_signals[direction][i]) for i in range(self.strucutre_model.n_nodes)])

        signs = np.divide(mean_z, abs(mean_z))

        std_z = np.array([np.std(self.load_signals[direction][i]) for i in range(self.strucutre_model.n_nodes)])
        signed_std_z = np.multiply(signs, std_z)

        func = np.multiply(std_z, self.influence_function[self.response][direction])

        sig_Rb = sum(func)
        
        return sig_Rb

    def get_B_sl(self, s, l):
        '''
        factor representing loss of correlation of wind loads in different directions s & l and different heights
        Kareem eq. 21
        TODO gets larger than 1 when usign e.g a 10 elements model -> should never get larger than 1
        '''
        self.correlation_coefficient = []
        B_sl_numerator = 0.0
        covariances = []
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
                covariances.append(covariance)
                B_sl_numerator += mü_1 * mü_2 * covariance #sum is correct here 

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
                print ('WARNING: B_sl of', s, l ,'is larger than 1:', B_sl)
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

    # ========================================
    # LRC COEFFICIENT according to Kaspersik
    # ========================================

    def get_spatial_distribution_lrc(self):
        '''
        computes the LRC distribution of beswl for a given load direction
        this still needs to be multiplied with the peak factor g_b
        '''
        self.spatial_distribution_lrc = {self.response: {}}
        for direction in self.load_directions:
            rho_Rps = self.get_lrc_coeff(direction)
            sig_ps = np.asarray([np.std(i) for i in self.load_signals[direction]])

            lrc = np.multiply(rho_Rps, sig_ps)

            if np.any(rho_Rps > 1):
                print ('\nWARNING lrc coefficient, rho_Rps is larger than 1: ', rho_Rps, 'for direction', direction, 'for response', self.response, 'BESWL LRC result might be wrong!')
            
            self.spatial_distribution_lrc[self.response][direction] = lrc

    def get_weighting_factor_lrc(self):
        '''
        this is the standard deviation of the response using covariance method Kasperski (eq. 1)
        NOTE from Kasperski extended it to a coupled case with wind loads in different directions causing the same reaction. 
        '''
        sig_R_2 = 0.0 # scalar value: just the rms/std of the response 
        for node1 in range(self.strucutre_model.n_nodes):
            for node2 in range(self.strucutre_model.n_nodes):
                for s in self.load_directions: 
                    for l in self.load_directions: 
                        I_ps_z = self.influence_function[self.response][s][node1]
                        I_pl_z = self.influence_function[self.response][l][node2]

                        cov_psl = np.cov(self.load_signals[s][node1], self.load_signals[l][node2])[0][1]

                        sig_R_2 += I_ps_z * I_pl_z * cov_psl  
        
        # saved as member since it is required multiple times
        self.weighting_factors_raw_lrc = sig_R_2

    def get_lrc_coeff(self, load_direction):
        '''
        compute the LRC(load - response - correlation) coefficient according to Kasperski 1992.
        '''
        std_ps = np.asarray([np.std(i) for i in self.load_signals[load_direction]])

        # COVARIANCE METHOD 
        # JZ: eq. 4.7, Kasperski: Eq. 3
        cov_Rps = np.zeros(self.strucutre_model.n_nodes) # covariance between reposne R and load p_s
        for node1 in range(self.strucutre_model.n_nodes):
            for node2 in range(self.strucutre_model.n_nodes):
                for l in self.load_directions: #l

                    I_pl_z = self.influence_function[self.response][l][node2]
                    cov_psl = np.cov(self.load_signals[load_direction][node1], self.load_signals[l][node2])[0][1]

                    cov_Rps[node1] += I_pl_z * cov_psl

        # standard deviation of the response using covariance method Kasperski (eq. 1)
        std_background_response = self.weighting_factors_raw_lrc
        std_R_cov_method = np.sqrt(abs(std_background_response))
        # lrc coefficient = correlation coefficient between ps and R
        rho_Rps = np.divide(cov_Rps, np.multiply(std_ps, std_R_cov_method))
            
        return rho_Rps
  
