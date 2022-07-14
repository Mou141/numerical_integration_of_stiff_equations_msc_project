"""Contains tests for ../robertson_integration.py for use with pytest."""
from pathlib import Path
import sys

# Add the parent directory of this file's directory to the python path (because robertson_ivp.py is in that directory)
sys.path.append(str(Path(__file__).resolve().parent.parent))

import robertson_ivp
import robertson_integration

import pytest
import numpy as np


class TestCmdArgs:
    """Test parsing of command line arguments by robertson_integration.parse_args."""

    # short option
    pass_args_1 = ["-us"]
    # long option
    pass_args_2 = ["--use-static"]

    @pytest.mark.parametrize("args", [pass_args_1, pass_args_2])
    def test_parse_args_static(self, args):
        """Tests that the specified command line arguments cause robertson_integration.parse_args to return robertson_ivp.TEST_IVP."""

        assert robertson_integration.parse_args(args) is robertson_ivp.TEST_IVP

    def test_parse_args_empty(self):
        """Tests that empty command line arguments cause robertson_integration.parse_args to return a random set of initial values (i.e. that are not equal to those from robertson_ivp.TEST_IVP) but that the other members of the tuple are the same."""

        ivp = robertson_integration.parse_args([])

        assert np.any(ivp.y0 != robertson_ivp.TEST_IVP.y0)
        assert ivp.ODEFunction is robertson_ivp.initial_value_problem
        assert not ivp.has_analytical_solution
        assert ivp.t0 == pytest.approx(robertson_ivp.TEST_IVP.t0)

    # Random string
    fail_args_1 = ["fgfdgdfg"]
    # Value passed to "-us"
    fail_args_2 = ["-us", "fgdfg"]
    # Value passed to "--use-static"
    fail_args_3 = ["--use-static", "dfgdg"]

    @pytest.mark.parametrize("args", [fail_args_1, fail_args_2, fail_args_3])
    def test_parse_args_fail(self, args):
        """Tests that the specified command line arguments cause robertson_integration.parse_args to exit the program."""

        with pytest.raises(SystemExit):
            robertson_integration.parse_args(args)
