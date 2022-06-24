# Tries out SciPy's builtin stiff integrators on a 2D Initial Value Problem
import matplotlib.pyplot as plt # For plotting graphs
import numpy as np # For maths and arrays
import stiff_functions # For the STIFF_IVP2 problem
from test_ivp # For test_integrators and INTEGRATORS

def main():
    end_t = 100.0 # End point for integration
    
    print("Testing 2D initial value problem for integrators: {0}...".format(", ".join(test_ivp.INTEGRATORS)))
    
    results = test_ivp.test_integrators(end_t, stiff_functions.STIFF_IVP2, integrators=test_ivp.INTEGRATORS) # Apply the integration methods to the IVP... (test_ivp.INTEGRATORS is passed explicity to guard against future code changes where it may no longer be the default)
    
    for method, solution in results.items():
        if solution.success: # If this integration method succeeded...
            print("Method '{0}' completed successfully.".format(method))
            
        else: # If this method failed...
            print("Method '{0}' failed.".format(method))