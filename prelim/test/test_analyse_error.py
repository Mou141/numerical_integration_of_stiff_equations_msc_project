"""pytest tests for ../analyse_error.py."""

# Add the parent directory of this file's directory to the python path (because analyse_error.py is in that directory)
from pathlib import Path # For file path manipulations
import sys # For sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest # For testing functions

import numpy as np # For maths and arrays

import analyse_error # Module to test

class TestStatsFunctions:
    """Tests the functions which calculate statistics (i.e. l^2 norm, l^infinity norm, and mean)."""
    
    # Test parameters for test_2_norm
    l2_1 = (np.array([1.0, 1.0, 1.0, 1.0]), 2.0) # 1^2 = 1, sum is 4, sqrt(4) == 2
    l2_2 = (np.array([2.0, 2.0]), np.sqrt(8.0)) # 2^2 = 4, sum is 8
    l2_3 = (np.array([5.0, 1.5, 3.22, 5.6]), np.sqrt(68.9784)) # Sum is 68.9784
    l2_4 = (np.array([-2.0, 2.0]), np.sqrt(8.0)) # 2^2 == 4, (-2)^2 == 4, sum is 8 (i.e. check that negative numbers are squared correctly)

    @pytest.mark.parametrize("arr,l2", [l2_1, l2_2, l2_3, l2_4])
    def test_l_2_norm(self, arr, l2):
        """Tests that the l_2 norm of arr matches l2."""
        assert analyse_error.l_2_norm(arr) == pytest.approx(l2)
    
    # Test parameters for test_l_infinity_norm
    l_inf_1 = (np.array([2.0, 3.5, -1.0, 5.4]), 5.4) # 5.4 is the largest value so it should be returned
    l_inf_2 = (np.array([-6.5, 4.3, -3.0, 2.22]), 6.5) # 4.3 is the largest value, but -6.5 has the largest magnitude, so 6.5 should be returned

    @pytest.mark.parametrize("arr,l_inf", [l_inf_1, l_inf_2])
    def test_l_infinity_norm(self, arr, l_inf):
        """Tests that the l^infinity norm of arr matches l_inf."""
        assert analyse_error.l_infinity_norm(arr) == pytest.approx(l_inf)

    def test_l_infinity_norm_empty_array(self):
        """Tests that analyse_error.l_infinity_norm raises a ValueError when called with an empty array (np.max always raises a ValueError on an empty array)."""

        with pytest.raises(ValueError):
            analyse_error.l_infinity_norm(np.array([]))
    
    # Test parameters for test_calc_err_stats
    stats_1 = (np.array([2.0, -2.0]), (np.sqrt(8.0), 2.0, 2.0)) # l2 is sqrt(8), l_inf is 2, mean is 2
    stats_2 = (np.array([3.5, 12.3, -33.4]), (np.sqrt(1279.1), 33.4, -5.6667))

    @pytest.mark.parametrize("arr,stats", [stats_1, stats_2])
    def test_calc_err_stats(self, arr, stats):
        """Tests that analyse_error.test_calc_err_stats matches the contents of "stats"."""
        assert analyse_error.calc_err_stats(arr) == pytest.approx(stats)