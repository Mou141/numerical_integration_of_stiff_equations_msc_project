"""Implements utility functions which are useful for multiple integrator implementations."""
# At present, only the exp4 integrator is to be implemented
# However some functions that the exp4 implementation requires are useful for other methods
# These functions are implemented here so that future integrator implementations can leverage them more easily

import numpy as np


def phi(z):
    """Common mathematical function for exponential integration methods.

    Returns: (e^z - 1)/z"""

    return (np.exp(z) - 1.0) / z
