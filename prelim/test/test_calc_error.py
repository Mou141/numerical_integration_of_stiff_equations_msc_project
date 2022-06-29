# Test suite for ../calc_error.py

from pathlib import Path
import sys # For sys.path
sys.path.append(Path(__file__).resolve().parent.parent) # Add the parent directory of the directory that this file to the python path (since this contains calc_error.py)

import pytest
import calc_error

import numpy as np

class ErrorTests:
    """Tests for absolute and relative errror calculations."""
    
    @pytest.mark.parametrize("y_num,y_exact,a_err", [(1.0, 1.0, 0.0), (1.0, -1.0, 0.0), (np.array([1.0, 1.0, 0.5]), np.array([1.0, -1.0, 2.5]), np.array([0.0, 0.0, 2.0]))]) # Test values and outputs to use (test one value that would be the same with and without abs, one that needs abs, and one array to check it works properly with arrays)
    def test_absolute_error(y_num, y_exact, a_err):
        """Tests calc_error.absolute_error by asserting that calc_error.absolute_error(y_num, y_exact) == a_err."""
        assert calc_error.absolute_error(y_num, y_exact) == a_err
    
    @pytest.mark.parametrize("y_num,a_err,f_err", [(5.0, 2.0, (2.0/5.0)), (-5.0, 2.0, (2.0/5.0)), (np.array([5.0, -5.0, 0.5]), np.array([2.0, 2.0, 0.1]), np.array([(2.0/5.0), (2.0/5.0)], (1.0/5.0)))])
    def test_fractional_error(y_num, a_err, f_err):
        """Tests calc_error.fractional_error by asserting that calc_error.fractional_error(a_err, y_num) == f_err."""
        assert calc_error.fractional_error(a_err, y_num) == f_err