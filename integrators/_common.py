"""Implements utility functions which are useful for multiple integrator implementations."""
# At present, only the exp4 integrator is to be implemented
# However some functions that the exp4 implementation requires are useful for other methods
# These functions are implemented here so that future integrator implementations can leverage them more easily

import numpy as np


def phi(z):
    """Returns: phi(z) = (e^z - 1)/z"""

    return (np.exp(z) - 1.0) / z


def phi_step_jacob(h, A, j=1.0):
    """Common function used by mathematical integrators.

        h: Size of the current step.
        A: Current value of the Jacobian.
        j, optional: Constant coefficient (defaults to 1.0)

    Returns: phi(j * h * A) (or phi(h * A) if j not specified)"""
    return phi(j * h * A)
