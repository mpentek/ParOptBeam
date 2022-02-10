import unittest
import unittest.mock as mock
import scipy
import numpy as np
from source.model.structure_model import StraightBeam


# TODO (mate kelemen)
class test_structure_model_eigenvalue_solve (unittest.TestCase):

    @mock.patch('scipy.linalg.eigh')
    @mock.patch('numpy.sqrt')
    @mock.patch('numpy.real')
    def test_function_calls (self,mock_np_real,mock_np_sqrt,mock_scipy_linalg_eigh):

        # mock self, numpy and scipy functions
        mock_self = mock.MagicMock()

        # define return values of mocked functions
        mock_np_real.return_value = mock_self.real_eig_values_raw
        mock_np_sqrt.return_value = mock_self.eig_values
        mock_scipy_linalg_eigh.return_value = [mock_self.eig_values_raw, mock_self.eigen_modes_raw]

        # run the tested method
        StraightBeam.eigenvalue_solve(mock_self)

        # assertions
        # assert scipy.linalg.eigh beeing called correctly
        scipy.linalg.eigh.assert_called_once_with(mock_self.comp_k, mock_self.comp_m)
        # assert np.sqrt and np.real beeing called correctly
        np.real.assert_called_once_with(mock_self.eig_values_raw)
        np.sqrt.assert_called_once_with(mock_self.real_eig_values_raw)

    def test_values (self):
        mock_self=mock.MagicMock()
        # example values
        mock_self.comp_k = [[ 6.72000000e+08,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  4.30133760e+04,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00, -5.37667200e+05],
        [ 0.00000000e+00,  0.00000000e+00,  1.72037376e+05,
        0.00000000e+00,  2.15046720e+06,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        2.36750769e+05,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  2.15046720e+06,
        0.00000000e+00,  3.58411200e+07,  0.00000000e+00],
        [ 0.00000000e+00, -5.37667200e+05,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  8.96112000e+06]]

        mock_self.comp_m = [[ 5.23333333e+03,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  5.83152906e+03,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00, -2.05597332e+04],
        [ 0.00000000e+00,  0.00000000e+00,  5.83183050e+03,
        0.00000000e+00,  2.05603612e+04,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        8.72265833e+01,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  2.05603612e+04,
        0.00000000e+00,  9.34802929e+04,  0.00000000e+00],
        [ 0.00000000e+00, -2.05597332e+04,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  9.34593596e+04]]

        # run the tested method
        StraightBeam.eigenvalue_solve(mock_self)

        # reference results obtained from matlab
        reference_eig_freqs = np.array([0.268649800092252,0.537253776812886,2.64654094768132,
        5.29034680480794,8.29165837006746,57.0316018041581])

        # assert equal
        rtol_lim = 1e-05
        self.assertIsNone(np.testing.assert_allclose(mock_self.eig_freqs,reference_eig_freqs,rtol=rtol_lim))

if __name__ == "__main__":
    unittest.main()