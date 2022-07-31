"""Contains tests for ../robertson_ivp.py for execution with pytest."""

from pathlib import Path
import sys
import itertools

# Add the parent directory of this file's directory to the python path (because robertson_ivp.py is in that directory)
sys.path.append(str(Path(__file__).resolve().parent.parent))

import robertson_ivp

import pytest
import numpy as np

# Test for static test value
def test_TEST_IVP():
    """Tests that sum(robertson_ivp.TEST_IVP.y0) == 1.0."""
    assert np.sum(robertson_ivp.TEST_IVP.y0) == pytest.approx(1.0)


class TestRandom:
    """Tests randomly generated y0 values for Robertson IVP."""

    @staticmethod
    def _fulfils_constraint(y0):
        """Returns True if sum(y0) == 1.0."""
        return np.sum(y0) == pytest.approx(1.0)

    # How many times to repeat each of the tests below
    test_count = 10

    # Contains tuples which numbers an execution of one of the functions [(1, f1), (1, f2), (2, f1), (2, f2)...]
    test_values = list(
        itertools.product(
            range(1, test_count + 1),
            [robertson_ivp.get_random_y0, robertson_ivp.get_safe_random_y0],
        )
    )

    # Repeat test test_count times, for each generator function marking the test with the value of execution_number
    @pytest.mark.parametrize("execution_number,gen_func", test_values)
    def test_get_random_y0(self, execution_number, gen_func):
        """Tests that the specified function returns a y0 where sum(y0) == 1.0."""

        y0 = gen_func()
        assert self._fulfils_constraint(y0)

    # Repeat as for previous test
    @pytest.mark.parametrize("execution_number,gen_func", test_values)
    def test_get_robertson_ivp(self, execution_number, gen_func):
        """Tests that robertson_ivp.get_robertson_ivp returns an IVPTuple where sum(y0) == 1.0 for the specified generator function."""

        ivp = robertson_ivp.get_robertson_ivp(gen_func=gen_func)
        assert self._fulfils_constraint(ivp.y0)
