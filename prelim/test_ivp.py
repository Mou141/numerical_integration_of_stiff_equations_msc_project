# Tries out SciPy's builtin stiff integrators
import scipy.integrate # For solve_ivp function
import numpy as np # For NumPy arrays
import matplotlib.pyplot as plt # For plotting graphs
import stiff_functions # See stiff_functions.py in same directory

# List of SciPy integrators to check
INTEGRATORS = ["Radau", "BDF"]

def main(start_t, end_t, ivp=stiff_functions.STIFF_IVP, integrators=INTEGRATORS):
    """Takes an initial value problem, integrates it with the specified SciPy compatible integrators between the specified values,
    and produces a graph of the resulting integrations and the analytical solutoon for comparison.
    
    arguments:
        start_t: The start value to integrate from.
        end_t: The end value to integrate to.
        ivp, optional: A tuple containing the ode to integrate, the analytical solution to the IVP, and the given initial value (in that order). Defaults to the stiff IVP specified in stiff_functions.py.
        integrators, optional: A list of SciPy integrators to use to integrate this problem. Defaults to list specified in the INTEGRATORS global variable in this file."""
    
    plt.axes(xlabel="t", ylabel="y") # Label the axes
    
    t = np.linspace(start_t, end_t, 10000) # Create a linearly spaced array of values between start_t and end_t to plot the exact solution between
    y_exact = (ivp[1])(t) # Apply the solution function over the specified range
    plt.plot(t, y_exact, label="Exact") # Plot the exact solution with matplotlib.pyplot
    
    print("Integrating between t={0} and t={1}, y(0) = {2} with integrators: {3}.".format(start_t, end_t, ivp[2], ", ".join(integrators)))
    
    for integrator in integrators: # Iterate over each specified integrator...
        print("Using {0}...".format(integrator))
        
        solution = scipy.integrate.solve_ivp(ivp[0], t_span=(start_t, end_t), y0=ivp[2], method=integrator) # Perform the integration
        
        if solution.success: # If integration was successful...
            print("Integrated Successfully (SciPy message: '{0}').".format(solution.message))
            plt.plot(solution.t, solution.y, label=integrator) # Plot the result
        
        else: # If integration was successful...
            print("Integration failed (SciPy message: {0}).".format(solution.message))
    
    plt.legend(loc="best") # Add a legend to the graph
    plt.show() # Display the graph

if __name__ == "__main__": # If python file being executed directly...
    main(0.0, 1000.0)

# To test with other values, use interactive interpreter and load file as python module