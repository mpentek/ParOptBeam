import unittest
import mock
import scipy
import numpy as np
from source.model.structure_model import StraightBeam


class test_structure_model_eigenvalue_solve (unittest.TestCase):
    # @mock.patch('scipy.linalg.eigh',return_value = ['eig_values_raw', 'eigen_modes_raw'])
    # @mock.patch('numpy.sqrt','eig_values')
    # @mock.patch('numpy.real','real_eig_values_raw')
    def test_eigenvalue_solve (self):
        # mock self, numpy and scipy functions
        mock_self=mock.MagicMock()
        scipy.linalg.eigh = mock.Mock(return_value=[mock_self.eig_values_raw, mock_self.eigen_modes_raw])
        np.sqrt = mock.Mock(return_value=mock_self.eig_values)
        np.real = mock.Mock(return_value=mock_self.real_eig_values_raw)
        
        # run the tested method
        StraightBeam.eigenvalue_solve(mock_self)

        # assertions
        # assert scipy.linalg.eigh beeing called correctly
        scipy.linalg.eigh.assert_called_once_with(mock_self.comp_k, mock_self.comp_m)
        # assert np.sqrt and np.real beeing called correctly 
        np.real.assert_called_once_with(mock_self.eig_values_raw)
        np.sqrt.assert_called_once_with(mock_self.real_eig_values_raw)


if __name__ == "__main__":
    unittest.main()     