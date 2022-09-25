"""Implements utility functions which are useful for multiple integrator implementations."""
# At present, only the exp4 integrator is to be implemented
# However some functions that the exp4 implementation requires are useful for other methods
# These functions are implemented here so that future integrator implementations can leverage them more easily

import numpy as np
from warnings import warn


def phi(z):
    """Returns: phi(z) = (e^z - 1)/z"""

    return (np.exp(z) - 1.0) / z


def phi_step_jacob(h, A, j=1.0):
    """Common function used by mathematical integrators.

        h: Size of the current step.
        A: Current value of the Jacobian.
        j, optional: Constant coefficient (defaults to 1.0)

    Returns:
        phi(j * h * A) (or phi(h * A) if j not specified)"""

    return phi(j * h * A)


def phi_step_jacob_hA(hA, j):
    """Same as phi_step_jacob, but takes the product of h and A rather than separate values.
    For efficient computation of phi(j * h * A) where j changes but h and A don't.

    Returns:
        phi(j * hA)"""

    return phi(j * hA)


def warn_max_step_exceeded(h_abs, max_step):
    """Warns if the specified stepsize exceeds the maximum and returns the smaller of h_abs and max_step."""

    if h_abs > max_step:
        warn(
            "Initial stepsize {0} is greater than maximum {1}. {1} will be used.".format(
                h_abs, max_step
            )
        )
        return max_step

    else:
        return h_abs
