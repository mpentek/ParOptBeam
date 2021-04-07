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
        self.spatial_distribution = {self.response: {}} #keys are directions, values arrays of z

        self.get_rms_background_response()
        self.get_spatial_distribution()
        self.get_weighting_factor()

    
    def get_spatial_distribution(self):
        '''
        if using Kareem this is the gust envelope = rms of load at each point
        '''
        for direction in self.load_directions:
            self.spatial_distribution[self.response][direction] = np.zeros(self.strucutre_model.n_nodes)

            for i in range(self.strucutre_model.n_nodes):
                self.spatial_distribution[self.response][direction][i] = np.std(self.load_signals[direction][i])

    def get_weighting_factor(self):
        '''
        makes the spatial distribution response specific
        '''
        self.intermediate_w_factors = {} #used to track it for some sign stuff
        for direction in self.load_directions:
            self.intermediate_w_factors[direction] = []
            w_b_s = 0.0
            for l in self.load_directions:
                static_load_response = self.get_static_load_response(l)
                B_sl = self.get_B_sl(direction, l)
                if round(B_sl,3) != 0:
                    print ('B_sl for', self.response, 'of', str(direction), '&', str(l), ' ', round(B_sl,3), 'static load response sig_'+l, round(static_load_response,0))
                w_b_s += B_sl * static_load_response
                # track the sign ob w_b_s 
                self.intermediate_w_factors[direction].append(w_b_s)

            self.weighting_factors_raw[self.response][direction] = w_b_s

    def get_static_load_response(self, direction):
        '''
        current response due to rms of load in direction s at z
        in the formula this is σ'_Rb
        '''
        
        array_rms = np.array([np.std(self.load_signals[direction][i]) for i in range(self.strucutre_model.n_nodes)])
        sig_Rb = sum(np.multiply(array_rms, self.influence_function[self.response][direction]))

        #sig_Rb_loop = 0.0
        # for i in range(self.strucutre_model.n_nodes):
        #     sig_Rb_loop += rms(self.load_signals[direction][i]) * self.influence_function[self.response][direction][i]
        
        return sig_Rb

    def get_B_sl(self, s, l):
        '''
        factor representing loss of correlation of wind loads in different directions s & l
        '''
        self.correlation_coefficient = []
        B_sl_numerator = 0.0
        for node1 in range(self.strucutre_model.n_nodes):
            for node2 in range(self.strucutre_model.n_nodes): 
                mü_1 = self.influence_function[self.response][s][node1]
                mü_2 = self.influence_function[self.response][l][node2]
                covariance_matrix = np.cov(self.load_signals[s][node1], self.load_signals[l][node2]) # -> np.cov gives matrix.. 
                covariance = covariance_matrix[0][1]
                self.correlation_coefficient.append(covariance /(np.sqrt(covariance_matrix[0][0])*np.sqrt(covariance_matrix[1][1])))
                B_sl_numerator += mü_1 * mü_2 * covariance

        sig_Rb_s = self.get_static_load_response(s)
        sig_Rb_l = self.get_static_load_response(l)

        if sig_Rb_l == 0 or sig_Rb_s == 0:
            return 0.0

        elif abs(sig_Rb_l) < 1e-5 and (sig_Rb_l) < 1e-5:
            print ('static load response very small for:', s, '&',l)
        else:
            B_sl = B_sl_numerator / (sig_Rb_s * sig_Rb_l)
            # if B_sl < 0:
            #    print ('Warning: B_sl of', s,'-',l ,'is negative', B_sl) 
            if round(B_sl,3) > 1.0:
                print ('Warning: B_sl of', s, 'at node_'+str(node1), l ,'at node_'+str(node2),'is larger than 1', B_sl)
                #raise Exception('B_sl is larger then 1:', B_sl)
            return B_sl

    def get_rms_background_response(self):
        result = 0.0
        for s in self.load_directions:
            for l in self.load_directions:
                sig_Rb_s = self.get_static_load_response(s)
                sig_Rb_l = self.get_static_load_response(l)
                B_sl = self.get_B_sl(s , l)
                result += sig_Rb_s * sig_Rb_l * B_sl

        # here sqrt since sig*sig = sig² and rms/std is sig^1
        self.rms_background_response = np.sqrt(result)
    
    # # VARIANTE 1 NOT USED AND ÜERARBEITET

    def load_response_correlation_coeff(self, load_direction):
        '''
        from Kasperskis LRC it gives directly the spatial distribution 
        '''
        #warnings.simplefilter('always')
        s = load_direction
        rho_r_ps = np.zeros(self.strucutre_model.n_nodes)
        numerator = 0.0
        denominator = 0.0
        total_count = 0
        count = 0
        negativ_covs = []
        errors = []
        #print ()

        for node1 in range(self.strucutre_model.n_nodes):
            for node2 in range(self.strucutre_model.n_nodes):
                #print ('correlations of node', node1, 'with node', node2)
                for l in self.load_directions:
                    mü_l = self.influence_function[self.response][l][node1]
                    rho_sl = np.corrcoef(self.load_signals[s][node1], self.load_signals[l][node2])[0][1]
                    sig_pl = np.std(self.load_signals[l][node1])
                    numerator += mü_l * rho_sl * sig_pl

                
                for s in self.load_directions:
                    for l in self.load_directions:
                        total_count +=1
                        mü_l = self.influence_function[self.response][l][node1]
                        mü_s = self.influence_function[self.response][s][node2]
                        rho_sl = np.corrcoef(self.load_signals[s][node1], self.load_signals[l][node2])[0][1]
                        #cov_rho = rho_sl * np.std(self.load_signals[s][node])* np.std(self.load_signals[l][node])
                        cov = np.cov(self.load_signals[s][node1], self.load_signals[l][node2])[0][1]
                        #cov = np.cov(abs(self.load_signals[s][node1]), abs(self.load_signals[l][node2]))[0][1] # is the same as correlation_coef/(sig_s * sig_p)?!
                        
                        if cov < 0:
                            pair = sorted((s,l))
                            if pair not in negativ_covs:
                                negativ_covs.append(pair)
                            #print ('covariance for', self.response, 'of', s, '&', l, cov)
                        # if round(cov - cov_rho, 0) != 0:
                        #     count += 1
                        #     dif_p = round((cov - cov_rho)/cov * 100,6)
                        #     print ('cov with rho for', self.response, 'of', s, '&', l, cov_rho)
                        #     print ('covariance for', self.response, 'of', s, '&', l, cov)
                        #     print ('difference cov - cov_rho:', round(cov - cov_rho, 0), 'this is', dif_p , '%')
                        #     if dif_p not in perc:
                        #         perc.append(dif_p)
                        #     print ()
                        sign = 1
                        if mü_l * mü_s * cov < 0:
                            # if l == 'g' and s == 'g':
                            #     print('gg')
                            sign = -1
                            errors.append((node1, node2, s,l))
                        #     #print ('here the irregualr sqrt: mü_'+ s, round(mü_s,2), 'mü_'+ l, round(mü_l,2), 'cov_'+s+l, round(cov,2) )
                        #     if l == 'z':
                        #         print ('mü_z', mü_l)
                        #         if mü_s < 0 or cov < 0:
                        #             print ('not mü_z cuasing problem')
                        #     elif s == 'z':
                        #         print ('mü_z', mü_s)
                        #         if mü_l < 0 or cov < 0:
                        #             print ('not mü_z cuasing problem')
                        # error = mü_l * mü_s * cov
                        #a = mü_l * mü_s * cov
                        denominator += np.sqrt(sign*mü_l * mü_s * cov)
                
            # print ('total differences:', count, 'of', total_count)
            # print ('following percentages are were collected:', perc)
                        # if denominator != 0:
                        #     if math.isnan(numerator / denominator):
                        #         print ('nan occurs: ', node1, node2, s, l)
                        #         print ('numerator:', numerator)
                        #         print ('denominator:', denominator)
            if denominator == 0:
                rho_r_ps[node1] = 0.0
            else:
                rho_r_ps[node1] = numerator / denominator 

        #print ('negative covarianves occur for these pairs:', negativ_covs)
        #print ('points at which the expression under sqrt is negative', errors)
        return rho_r_ps

    # # VARAINTE 2
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

    def lrc_coeff_1(self, load_direction):
        # cov(R,ps)
        #self.rho_collection = []
        self.rho_collection_all[load_direction] = []
        src = 'input\\force\\generic_building\\force_structure.dat'
        columns = self.get_columns_from_header(src)
        direct = GD.RESPONSE_DIRECTION_MAP[self.response]
        force_kr = GD.DIRECTION_LOAD_MAP[direct]
        id_r = columns.index(force_kr)
        response_time_hist = np.loadtxt(src, usecols=id_r+6)

        shape_dif = response_time_hist.shape[0] - self.load_signals['x'][0].shape[0]

        # sig R_t from the response time history
        sig_R_t = np.std(response_time_hist[shape_dif:])

        cov_Rps = np.zeros(self.strucutre_model.n_nodes)
        for node1 in range(self.strucutre_model.n_nodes):
            for node2 in range(self.strucutre_model.n_nodes):
                for direction in self.load_directions: #l

                    I_pl_z = self.influence_function[self.response][direction][node2]
                    cov_psl = np.cov(self.load_signals[load_direction][node1], self.load_signals[direction][node2])[0][1]

                    cov_Rps[node1] += I_pl_z * cov_psl
        
        sig_R_2 = 0.0 # scalar value: just the rms of the response 
        # TODO this could be taken form the time history of the response ?! -> see sig_R_t
        for node1 in range(self.strucutre_model.n_nodes):
            for node2 in range(self.strucutre_model.n_nodes):
                for s in self.load_directions: #s
                    for l in self.load_directions: #l
                        I_ps_z = self.influence_function[self.response][s][node1]
                        I_pl_z = self.influence_function[self.response][l][node2]

                        cov_psl = np.cov(self.load_signals[s][node1], self.load_signals[l][node2])[0][1]

                        sig_R_2 += I_ps_z * I_pl_z * cov_psl

        rho_r_ps_direct = np.zeros(self.strucutre_model.n_nodes)

        for node in range(self.strucutre_model.n_nodes):
            load = self.load_signals[load_direction][node]
            rho_r_ps_direct[node] = np.corrcoef(response_time_hist[shape_dif:], load)[0][1] 

        
        print('\nrms of response', self.response)
        print('rms from time history: ', round(sig_R_t))
        print('with covariance method:', round(np.sqrt(abs(sig_R_2))))
        print('std form time history  ', round(np.std(response_time_hist[shape_dif:])))
        print('with B_sl:             ', round(self.rms_background_response))
        dif = round(np.sqrt(abs(sig_R_2))-sig_R_t)
        print('difference time cov - time hist:', dif, 'that is', str(round(dif/sig_R_t * 100, 5))+'%', 'of time hist sigma')
        print()
        print ('std of response')
    

        # TODO: here use std or rms 
        rms_ps = np.zeros(self.strucutre_model.n_nodes)
        std_ps = np.zeros(self.strucutre_model.n_nodes)
        cov_R_ps_np = np.zeros(self.strucutre_model.n_nodes)
        for i, signal in enumerate(self.load_signals[load_direction]):
            rms_ps[i] = rms(signal)
            std_ps[i] = np.std(signal)
            cov_R_ps_np[i] = np.cov(response_time_hist[shape_dif:], signal)[0][1]

        #rho_Rps_manual_rms = np.divide(cov_Rps, rms_ps) / np.sqrt(abs(sig_R_2))
        rho_Rps_manual_std = np.divide(cov_Rps, std_ps) / np.sqrt(abs(sig_R_2))
        rho_Rps_t = np.divide(cov_Rps, std_ps) / sig_R_t
        rho_Rps_np = np.divide(cov_R_ps_np, std_ps) / sig_R_t
        
        # using corrcoeff of numpy
        self.rho_collection_all[load_direction].append(rho_r_ps_direct)
        # using covariance method and rms of load
        #self.rho_collection_all[load_direction].append(rho_Rps_manual_rms)
        # using covariance method and std of load
        self.rho_collection_all[load_direction].append(rho_Rps_manual_std)

        rho_Rps = rho_Rps_manual_std

        return rho_Rps


    def get_beswl_LRC_1(self, load_direction):
        rho = self.lrc_coeff_1(load_direction)
        sig_ps = np.zeros(self.strucutre_model.n_nodes)
        for i, signal in enumerate(self.load_signals[load_direction]):
            sig_ps[i] = np.std(signal)

        return np.multiply(rho, sig_ps)


    def get_beswl_LRC (self, load_direction):
        '''
        computes the LRC distribution of beswl for a given load direction
        this still needs to be multiplied with the peak factor g_b
        '''
        rho_r_ps = self.load_response_correlation_coeff(load_direction)

        return np.multiply(np.std(self.load_signals[load_direction]), rho_r_ps) 