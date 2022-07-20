"""Test suite for ../robertson_bulk_error.py for use with pytest."""
# Add the parent directory of this file's directory to the python path (because robertson_bulk_error.py is in that directory)
from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest

import robertson_bulk_error


class TestGenerateIVPs:
    """Contains tests for the robertson_bulk_error.generate_robertson_ivps function."""

    @pytest.mark.parametrize("n", [1, 10, 100, 1000000, 1024, 0])
    def test_returns_N_ivps(self, n):
        """Checks that robertson_bulk_error.generate_robertson_ivps(n) returns n values."""

        ivps = robertson_bulk_error.generate_robertson_ivps(n)

        # Count without storing all instances simultaneously (more memory efficient)
        assert sum(1 for i in ivps) == n

    @pytest.mark.parametrize("n", [0, -121])
    def test_returns_no_ivps(self, n):
        """Checks that robertson_bulk_error.generate_robertson_ivps(n) returns a 0 length iterator when executed with n."""

        # Don't bother with sum as above, since this should be 0 length
        ivps = list(robertson_bulk_error.generate_robertson_ivps(n))

        assert len(ivps) == 0


class TestCmdArgs:
    """Tests that robertson_bulk_error.parse_args parses command line arguments correctly."""

    # Default return value
    n_test_1 = ([], 10)
    # Short option
    n_test_2 = (["-n", "200"], 200)
    # Long option
    n_test_3 = (["--number", "200"], 200)

    @pytest.mark.parametrize("args,n", [n_test_1, n_test_2, n_test_3])
    def test_N_returned(self, args, n):
        """Tests that robertson_bulk_error.parse_args returns N for the specified command line arguments."""
        assert robertson_bulk_error.parse_args(args) == n

    # Incorrect option
    fail_test_1 = ["-dfgdg", "454"]
    # No number
    fail_test_2 = ["-n"]
    # Negative number
    fail_test_3 = ["-n", "-200"]
    # Zero
    fail_test_4 = ["-n", "0"]
    # Non-number
    fail_test_5 = ["-n", "abc"]

    @pytest.mark.parametrize(
        "args", [fail_test_1, fail_test_2, fail_test_3, fail_test_4, fail_test_5]
    )
    def test_fail_args(self, args):
        """Test that argparse attempts to exit the program when incorrect arguments are passed."""
        with pytest.raises(SystemExit):
            robertson_bulk_error.parse_args(args)


def all_nan(arr):
    """Returns True if all elements in array are nan."""
    return np.all(np.isnan(arr))


class TestCreateY0Array:
    """Tests the robertson_bulk_error.create_y0_array function."""

    @pytest.mark.parametrize("n", [0, 1, 10, 100])
    def test_array_shape(self, n):
        """Checks that an array of the correct shape is produced."""
        y0 = robertson_bulk_error.create_y0_array(n)

        assert y0.shape == (n, 3)

    def test_array_nan(self):
        """Checks that the array is initialised with np.nan."""
        y0 = robertson_bulk_error.create_y0_array(10)

        assert all_nan(y0)

    def test_array_type(self):
        """Checks that the array has type float."""
        y0 = robertson_bulk_error.create_y0_array(10)

        assert y0.dtype == np.dtype(float)


class TestRobertsonStatsTuple:
    """Tests the robertson_bulk_error.RobertsonStatsTuple class."""

    @pytest.mark.parametrize("n", [10, 100, 0])
    def test_tuple_lengths(self, n):
        """Checks that the length of the tuple is always 5 and that the lengths of the internal arrays match the value passed to robertson_bulk_error.RobertsonStatsTuple._create_tuple."""
        t = robertson_bulk_error.RobertsonStatsTuple._create_tuple(n)

        # Length of tuple (should never change)
        assert len(t) == 5

        # Lengths of arrays in tuple
        assert len(t.l2) == n
        assert len(t.l_inf) == n
        assert len(t.nfev) == n
        assert len(t.njev) == n
        assert len(t.nlu) == n

    def test_nan(self):
        """Checks that all arrays are initialised with np.nan."""
        t = robertson_bulk_error.RobertsonStatsTuple._create_tuple(10)

        assert all_nan(t.l2)
        assert all_nan(t.l_inf)
        assert all_nan(t.nfev)
        assert all_nan(t.njev)
        assert all_nan(t.nlu)

    def test_types(self):
        """Checks that the types of the arrays are correct."""
        # Types for comparison
        ftype = np.dtype(float)
        itype = np.dtype(int)

        t = robertson_bulk_error.RobertsonStatsTuple._create_tuple(10)

        assert t.l2.dtype == ftype
        assert t.l_inf.dtype == ftype
        assert t.nfev.dtype == itype
        assert t.njev.dtype == itype
        assert t.nlu.dtype == itype
