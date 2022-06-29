# Test suite for ../calc_error.py

import sys # For sys.path
sys.path.append("../") # Add the parent directory to the python path (since this contains calc_error.py)

import pytest
import calc_error

class ErrorTests:
    """Tests for absolute and relative errror calculations."""
    
    def test_absolute_error(y_num, y_exact, a_err):
        """Tests calc_error.absolute_error by asserting that calc_error.absolute_error(y_num, y_exact) === a_err."""
        assert calc_error.absolute_error(y_num, y_exact) == a_err