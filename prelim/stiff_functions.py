# Contains stiff problems for testing purposes (currently for preliminary investigation, may be incorporated into final package)
import numpy as np # For array type and maths
from collections import namedtuple # For creating IVPTuple class

# Named tuple to contain IVP problem, solution, initial value and start position for integration (i.e. the value of t0 where y=y0)
class IVPTuple(namedtuple("IVPTuple", ["ODEFunction", "SolutionFunction", "y0", "t0"])):
    """Named tuple to contain an Initial Value Problem.
        
        ODEFunction: The ODE function which is integrated.
        SolutionFunction: A function providing the exact solution to the ODE.
        y0: The value(s) of the function when t=t0.
        t0: The value of t where y=y0."""
    
    # A property containing the number of dimensions of the IVP
    @property
    def ndim(self):
        """The number of dimensions of the Initial Value Problem."""
        return len(self.y0)

# There is a 1D initial value problem which is stiff and also has an analytical solution: y'(t) = -15y(t), t >= 0, y(0) = 1. The solution being y(t) = exp(-15t)

# This is an implementation of the problem that will be accepted by scipy.integrate.solve_ivp
def initial_value_problem(t, y):
    """A stiff initial value problem with an analytical solution: y'(t) = -15y(t), t >= 0, y(0) = 1.
        
        returns: -15 * y
    """
    
    return -15.0 * y

# This is the analytical solution to the above problem
def initial_value_solution(t):
    """The solution to the initial value problem y'(t) = -15y(t):
     
        returns: exp(-15t)"""
    
    return np.exp(-15.0 * t)

# Named tuple that contains the problem, solution, and initial value for the stiff IVP implemented above
STIFF_IVP = IVPTuple(ODEFunction=initial_value_problem, SolutionFunction=initial_value_solution, y0=np.array([1.0]), t0=0.0)

# Another stiff IVP with an exact solution is given in Numerical Analysis by Burden et. al.
def initial_value_problem2(t, y):
    """The 2D stiff initial value problem described in Numerical Analysis by Burden et. al.
        
        Arguments:
            y: A 1D array of two values: y1 and y2
            t: Function parameter."""
    
    y1_prime = (9.0 * y[0]) + (24.0 * y[1]) + (5.0 * np.cos(t)) - ((1.0/3.0) * np.sin(t))
    y2_prime = (-24.0 * y[0]) - (51.0 * y[1]) - (9.0 * np.cos(t)) + ((1.0/3.0) * np.sin(t))
    
    return np.array([y1_prime, y2_prime])

# The analytical solution to the above initial value problem
def initial_value_solution2(t):
    """The solution to the 2D stiff initial value problem described in Numerical Analysis by Burden et. al."""
    y1 = (2.0 * np.exp(-3.0 * t)) - np.exp(-39.0 * t) + ((1.0/3.0) * np.cos(t))
    y2 = (-np.exp(-3.0 * t)) + (2.0 * np.exp(-39.0 * t)) - ((1.0/3.0) * np.cos(t))
    
    return np.array([y1, y2])

# Named tuple that contains the problem, solution, and initial values for the stiff IVP above (the initial values being: y1(0) = 4/3, y2(0) = 2/3)
STIFF_IVP2 = IVPTuple(ODEFunction=initial_value_problem2, SolutionFunction=initial_value_solution2, y0=np.array([(4.0/3.0), (2.0/3.0)]), t0=0.0)