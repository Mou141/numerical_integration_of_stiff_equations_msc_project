# Contains stiff problems for testing purposes (currently for preliminary investigation, may be incorporated into final package)
# Requires numpy
import numpy as np # For np.exp function
from collections import namedtuple

# There is an initial value problem which is stiff and also has an analytical solution: y'(t) = -15y(t), t >= 0, y(0) = 1. The solution being y(t) = exp(-15t)

# This is an implementation of the problem that will be accepted by scipy.integrate.solve_ivp
def initial_value_problem(t, y):
    """A stiff initial value problem with an analytical solution: y'(t) = -15y(t), t >= 0, y(0) = 1.
        
        returns: -15 * y
    """
    
    return -15.0 * (y[0])

# This is the solution to the above problem
def initial_value_solution(t):
    """The solution to the initial value problem y'(t) = -15y(t):
     
        returns: exp(-15t)"""
    
    return np.exp(-15.0 * t)

# Named tuple to contain IVP problem, solution, and initial value
IVPTuple = namedtuple("IVPTuple", ["ODEFunction", "SolutionFunction", "y0"])

# Named tuple that contains the problem, solution, and initial value for the stiff IVP implemented above
STIFF_IVP = IVPTuple(ODEFunction=initial_value_problem, SolutionFunction=initial_value_solution, y0=1.0)