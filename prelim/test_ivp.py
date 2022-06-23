# Tries out SciPy's builtin stiff integrators
import scipy.integrate # For solve_ivp function
import numpy as np # For NumPy arrays
import matplotlib.pyplot as plt # For plotting graphs
import stiff_functions # See stiff_functions.py in same directory

# List of SciPy integrators to check
INTEGRATORS = ["Radau", "BDF"]

def test_integrators(end_t, ivp, integrators=INTEGRATORS, dense_output=False):
    """For each of the specified stiff integrators available with SciPy, integrate the initial value problem specified from t0 to end_t.
    
    Arguments:
        end_t: End of range over which to integrate (must be > t_0 specified in ivp)
        ivp: A tuple specifying the initial value problem to integrate (must contain the same information in the same order as stiff_functions.IVPTuple, but need not be an instance of this class).
        integrators, optional: A list of methods with which to attempt to solve the initial value problem (defaults to the contents of the INTEGRATORS global variable).
        dense_output, optional: Whether to ask SciPy to compute a continuous solution (defaults to False).
        
    Returns:
        A dictionary of the integrators and the solution objects produced by scipy.integrate.solve_ivp"""
    
    ivp = stiff_functions.IVPTuple._make(ivp) # Convert ivp to an IVPTuple instance if it's another kind of iterable
    
    out = {} # Empty dictionary to contain solutions
    
    for method in integrators: # For each method specified...
        solution = scipy.integrate.solve_ivp(fun=ivp.ODEFunction, t_span=(ivp.t0, end_t), y0=np.array([ivp.y0]), method=method, dense_output=dense_output) # Attempt to find a solution
        out[method] = solution # Store that solution in the output dictionary
        
    return out

def main():
    end_t = 100.0 # End point of integration
    
    print("Testing stiff initial value problem for integrators: {0}...".format(", ".join(INTEGRATORS)))
    
    results = test_integrators(end_t, stiff_functions.STIFF_IVP) # Attempt to integrate between 0 and 100 for the stiff IVP
    
    for method, solution in results.items(): # Iterate over the methods and their solution objects...
        if solution.success: # If integration was successful...
            print("Method '{0}' completed successfully.".format(method))
            plt.plot(solution.t, solution.y[0], label=method) # Plot the solution
            
        else: # If integration failed...
            print("Method '{0}' failed.".format(method))
    
    # Plot the exact solution
    t = np.linspace(ivp.t0, end_t, 10000)
    y_exact = ivp.SolutionFunction(t)
    plt.plot(t, y_exact, label="Exact Solution")
    
    plt.axes(xlabel="t", ylabel="y") # Label graph axes
    plt.legend(loc="best") # Add a legend to the graph
    plt.show() # Display the graph

if __name__ == "__main__":
    main()