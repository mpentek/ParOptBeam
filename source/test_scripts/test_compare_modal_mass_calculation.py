# --- External Imports ---
import numpy as np
from scipy import linalg

# --- Internal Imports ---
from source.test_utils.test_case import TestCase, TestMain

class test_compare_modal_mass_calculation(TestCase):

    def test_print_comparison(self):

        # compare with chapter 13.2.6 and 12.8 from Dynamic of Structures - Chopra
        m = 100
        comp_m = m*np.array([[1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1]])

        k = 31.54
        comp_k = k*np.array([[2, -1, 0, 0, 0],
                            [-1, 2, -1, 0, 0],
                            [0, -1, 2, -1, 0],
                            [0, 0, -1, 2, -1],
                            [0, 0, 0, -1, 1]])

        wn_analytic = np.array([0.285, 0.831, 1.310, 1.682, 1.919])*np.sqrt(k/m)
        Tn_analytic = 2.*np.pi/wn_analytic
        Ln_analytic = np.array([1.067, -0.336, 0.177, -0.099, 0.045])

        # eigenvectors are stored in columns
        phi_n = np.transpose(np.array([
            [0.334,     0.614,   0.895,   1.078,   1.173],
            [-0.895,    -1.173,  -0.641,  0.334,   1.078],
            [1.173,     0.334,   -1.078,  -0.641,  0.895],
            [-1.078,    0.895,   0.334,   -1.173,  0.641],
            [0.641,     -1.078,  1.173,   -0.895,  0.334]
        ]))

        # eigenvectors from linalg.eigh
        f_linalg, phi_n_linalg = linalg.eigh(comp_k,comp_m)

        phi_n = phi_n_linalg
        #self.assertMatrixAlmostEqual(phi_n, phi_n_linalg)

        eff_modal_mass_sum = np.zeros(5,)
        eff_modal_mass_sum = np.zeros(5,)
        rel_participation_sum = np.zeros(5,)
        for mode in range(5):
        # calculation of modal masses with summation floor wise
            total_mass = 0.
            eff_modal_numerator_sum = 0.
            eff_modal_denominator_sum = 0.

            for floor in range(5):
                storey_mass = m
                total_mass += storey_mass
                # numerator like this 100.0
                #eff_modal_numerator_sum += (storey_mass * phi_n[floor][mode]) **2

                eff_modal_numerator_sum += (storey_mass * phi_n[floor][mode])
                eff_modal_denominator_sum += storey_mass * phi_n[floor][mode] ** 2

            # denominator always 1.0

            # numerator like this NOT 100.0
            eff_modal_numerator_sum = eff_modal_numerator_sum **2

            eff_modal_mass_sum[mode] = eff_modal_numerator_sum / eff_modal_denominator_sum
            rel_participation_sum[mode] = eff_modal_mass_sum[mode] / total_mass



        # calculation of modal masses with multiplication
            eff_modal_numerator_multy = 0.
            eff_modal_denominator_multy = 0.
            eff_modal_mass_multy = np.zeros(5,)

            # numerator like this 100.0
            # V1
            #eff_modal_numerator_multy = np.matmul(np.matmul(np.transpose(phi_n[:,mode]),comp_m),np.matmul(comp_m,phi_n[:,mode]))
            # V2
            #eff_modal_numerator_multy = np.matmul(np.transpose(np.matmul(comp_m,phi_n[:,mode])),np.matmul(comp_m,phi_n[:,mode]))

            eff_modal_numerator_multy = (np.matmul(np.transpose(phi_n[:,mode]),np.matmul(comp_m,np.ones(5))))**2

            # denominator = Y'*m*Y
            # V1
            #eff_modal_denominator_multy = np.matmul(np.transpose(phi_n[:,mode]),np.matmul(comp_m,phi_n[:,mode]))
            # V2
            eff_modal_denominator_multy = np.matmul(np.matmul(np.transpose(phi_n[:,mode]),comp_m),phi_n[:,mode])
            # denominator always 1.0

            # # ignore rotational dofs
            # if label in ['a', 'b', 'g']:
            #     eff_modal_numerator = 0

            eff_modal_mass_multy[mode] = eff_modal_numerator_multy / eff_modal_denominator_multy
            # rel_participation[mode] = eff_modal_mass[label] / total_mass



            msg= '----- Mode ' + str(mode) + '------\n'
            msg += 'Analytic:\n'
            msg += '    Frequency = ' + str(wn_analytic[mode]) + '\n'
            msg += '    Period = ' + str(Tn_analytic[mode]) + '\n'
            msg += '    Contribution = ' + str(Ln_analytic[mode])+ '\n'
            msg += 'Summed:\n'
            msg += '    Contribution = ' +  str(eff_modal_mass_sum[mode]) + '\n'
            msg += '    Numerator = ' +str(eff_modal_numerator_sum) + '\n'
            msg += '    Denominator = ' +str(eff_modal_denominator_sum) + '\n'
            msg += '    rel. Participation = ' + str(rel_participation_sum[mode]) +'\n'
            msg += 'Multiplied:\n'
            msg += '    Contribution = ' + str(eff_modal_mass_multy[mode]) + '\n'
            msg += '    Numerator = ' +str(eff_modal_numerator_multy) + '\n'
            msg += '    Denominator = ' +str(eff_modal_denominator_multy) + '\n'
            print(msg)




if __name__ == "__main__":
    TestMain()