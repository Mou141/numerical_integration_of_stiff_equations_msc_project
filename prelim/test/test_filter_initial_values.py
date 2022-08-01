"""Test suites for ../filter_initial_values.py for use with pytest."""

from pathlib import Path
import sys

import pytest

# Add this file's parent directory to the python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

import robertson_bulk_error
import robertson_ivp

import filter_initial_values


class TestCmdArgs:
    """Tests that filter_initial_values.parse_args parses command line arguments correctly."""

    @pytest.mark.parametrize("args", [["file1.tsv", "file2.tsv"]])
    def test_successful_args(self, args):
        """Checks that the arguments passed to filter_initial_values.parse_args are returned correctly."""

        assert filter_initial_values.parse_args(args) == tuple(args)

    # Arguments that should fail
    # Empty
    fail_args_1 = []
    # Only 1 argument
    fail_args_2 = ["test1.tsv"]
    # Too many arguments
    fail_args_3 = ["test1.tsv", "test2.tsv", "test3.tsv"]

    @pytest.mark.parametrize("args", [fail_args_1, fail_args_2, fail_args_3])
    def test_fail_args(self, args):
        """Checks that argparse attempts to exit the program when the specified arguments are passed to filter_initial_values.parse_args."""

        with pytest.raises(SystemExit):
            filter_initial_values.parse_args(args)


class TestFilter:
    """Tests the filter_initial_values.filter_y0 function."""

    @staticmethod
    def _create_nan_array(n):
        """Creates an array of shape (n,) which contains floating point values and np.nan in random positions."""

        rng = np.random.default_rng()

        # Function for populating array
        # a is the index of the array
        # The function returns either 1.0, 0.5, 3.4e-6, or np.nan at random (i.e. approximately a quarter of the values will be np.nan)
        gen_func = lambda a: rng.choice([1.0, 0.5, 3.4e-6, np.nan])

        return np.fromfunction(gen_func, shape=(n,), dtype=float)

    @staticmethod
    def _create_y0_array(n):
        """Creates an array of shape (n, 3) which contains randomly chosen initial values."""

        y0 = robertson_bulk_error.create_y0_array(n)

        for i in range(0, n):
            y0[i] = robertson_ivp.get_random_y0()

        return y0

    @pytest.mark.parametrize("n", [1, 10, 100])
    def test_filter_y0_length(self, n):
        """Generates a set of initial values and an array to filter them with, performs the filtering, and then checks that the resulting array is less than or equal to the original in length."""

        y0 = self._create_y0_array(n)
        nan_array = self._create_nan_array(n)

        filtered_y0 = filter_initial_values.filter_y0(y0, nan_array)

        assert filtered_y0.shape[0] <= n

    @pytest.mark.parametrize("n", [1, 10, 100])
    def test_filter_y0_no_nan(self, n):
        """Generates a y0 array, filters it with an array that contains no nan values, and then checks the filtered array is the same length."""

        y0 = self._create_y0_array(n)
        nan_array = np.linspace(0.0, 1.0, n)

        filtered_y0 = filter_initial_values.filter_y0(y0, nan_array)

        assert filtered_y0.shape[0] == n
