"""Implements utility functions which are useful for multiple integrator implementations."""
# At present, only the exp4 integrator is to be implemented
# However some functions that the exp4 implementation requires are useful for other methods
# These functions are implemented here so that future integrator implementations can leverage them more easily

import numpy as np
from scipy.linalg import expm
from numpy.linalg import inv


def phi(z):
    """Returns: phi(z) = (e^z - 1)/z"""

    z = np.asarray(z)

    if z.shape == (1, 1):
        # special case where matrix is 1 x 1 (and therefore cannot be inverted)
        return (expm(z) - 1.0) / z

    else:
        return np.matmul((expm(z) - 1.0), inv(z))


def phi_step_jacob(h, A, j=1.0):
    """Common function used by mathematical integrators.

        h: Size of the current step.
        A: Current Jacobian matrix.
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
