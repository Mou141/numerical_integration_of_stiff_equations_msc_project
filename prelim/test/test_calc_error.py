# Test suite for ../calc_error.py

import sys # For sys.path
sys.path.append("../") # Add the parent directory to the python path (since this contains calc_error.py)

import pytest
import calc_error

import numpy as np

class ErrorTests:
    """Tests for absolute and relative errror calculations."""
    
    @pytest.mark.parametrize("y_num,y_exact,a_err", [(1.0, 1.0, 0.0), (1.0, -1.0, 0.0), (np.array([1.0, 1.0, 0.5]), np.array([1.0, -1.0, 2.5]), np.array([0.0, 0.0, 2.0]))]) # Test values and outputs to use (test one value that would be the same with and without abs, one that needs abs, and one array to check it works properly with arrays)
    def test_absolute_error(y_num, y_exact, a_err):
        """Tests calc_error.absolute_error by asserting that calc_error.absolute_error(y_num, y_exact) === a_err."""
        assert calc_error.absolute_error(y_num, y_exact) == a_err