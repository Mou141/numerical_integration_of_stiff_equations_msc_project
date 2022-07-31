"""Test suite for ../generate_safe_initial_values.py for use with pytest."""

from pathlib import Path
import sys

import pytest

# Add this file's parent directory to the python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import sys

import generate_safe_initial_values


class TestCmdArgs:
    """Tests the generate_safe_initial_values.parse_args function."""

    # Test arguments that should succeed, and the value of n they should return
    # default value (no arguments)
    success_args_1 = ([], 100)
    # short option
    success_args_2 = (["-n", "400"], 400)
    # long option
    success_args_3 = (["--number", "500"], 500)

    @pytest.mark.parametrize("args,n", [success_args_1, success_args_2, success_args_3])
    def test_success_args(self, args, n):
        """Executes generate_safe_initial_values.parse_args with the specified arguments and makes sure the specified n value is returned."""

        assert generate_safe_initial_values.parse_args(args) == n

    # Test arguments that should fail
    fail_args_1 = ["dsfs"]
    fail_args_2 = ["-n"]
    fail_args_3 = ["--number"]
    fail_args_4 = ["-n", "sdsd"]
    fail_args_5 = ["--number", "dfsdf"]
    fail_args_6 = ["-n", "0"]
    fail_args_7 = ["-n", "-45"]

    @pytest.mark.parametrize(
        "args",
        [
            fail_args_1,
            fail_args_2,
            fail_args_3,
            fail_args_4,
            fail_args_5,
            fail_args_6,
            fail_args_7,
        ],
    )
    def test_fail_args(self, args):
        """Checks that the specified (incorrect) arguments cause argparse to exit the program."""

        with pytest.raises(SystemExit):
            generate_safe_initial_values.parse_args(args)


class TestGenerateValues:
    """Tests the generate_safe_initial_values.generate_values function."""

    n_values = [100, 10, 1]

    @pytest.mark.parametrize("n", n_values)
    def test_not_nan(self, n):
        """Tests that the return value of generate_safe_initial_values.generate_values contains no nan values."""
        arr = generate_safe_initial_values.generate_values(n)

        assert np.all(np.logical_not(np.isnan(arr)))

    @pytest.mark.parametrize("n", n_values)
    def test_length(self, n):
        """Tests that the return value of generate_safe_initial_values.generate_values has shape (n, 3)."""
        arr = generate_safe_initial_values.generate_values(n)

        assert arr.shape == (n, 3)
