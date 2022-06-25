# Calculates the fractional error for numerical solutions to initial value problems at each point of the solution
import matplotlib.pyplot as plt # For plotting graphs
import numpy as np # For maths functions and arrays
from import test_ivp # For test_integrators function and INTEGRATORS global variable
import stiff_functions # For STIFF_IVP and STIFF_IVP2

def absolute_error(y_num, y_exact):
    """Calculates the absolute error in a numerical solution by comparing with the exact solution."""
    return np.abs(y_num - y_exact)
    
def fractional_error(y_num, y_exact):
    """Calculates the fractional error in a numerical solution by comparing with the exact solution."""
    return absolute_error(y_num, y_exact)/y_num