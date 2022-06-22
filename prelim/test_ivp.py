# Tries out SciPy's builtin stiff integrators
import scipy.integrate # For solve_ivp function
import numpy as np # For NumPy arrays
import matplotlib.pyplot as plt # For plotting graphs
import stiff_functions # See stiff_functions.py in same directory

# List of SciPy integrators to check
INTEGRATORS = ["Radau", "BDF"]

def main(start_t, end_t, integrators):
    """
    """
    pass
    
if __name__ == "__main__": # If python file being executed directly...
    main(0.0, 1000.0, INTEGRATORS)

# To test with other values, use interactive interpreter and load file as python module