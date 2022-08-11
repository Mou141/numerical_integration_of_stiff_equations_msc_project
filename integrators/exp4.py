"""A SciPy compatible implementation of the exp4 exponential integrator described in https://epubs.siam.org/doi/10.1137/S1064827595295337."""
from scipy.integrate import OdeSolver
import scipy.integrate._ivp.common as common


class EXP4(OdeSolver):
    """A subclass of the scipy.integrate.OdeSolver class which implements the exp4 numerical integration method."""
