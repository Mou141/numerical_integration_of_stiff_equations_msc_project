"""Tests the initial implementation of integrators.exp4.EXP4 with the same IVP as in initial_version_test.py
but the differential equation has an autonomous formulation (it depends only on y)."""

from initial_version_test import test_solution, TEST_T0, TEST_Y0, main
import numpy as np


def autonomous_test_function(t, y):
    """Autonomous version of initial_version_test.test_function."""
    return 2.0 * y


if __name__ == "__main__":
    main(
        TEST_T0,
        10.0,
        None,
        np.array([TEST_Y0]),
        autonomous_test_function,
        test_solution,
        autonomous=True,
    )
