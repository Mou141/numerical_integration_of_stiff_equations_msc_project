"""Generates "continuous" numerical solutions and compares them to the analytical solution for a specified IVP."""
import numpy as np # For arrays and maths
import test_ivp # For test_integrators and INTEGRATORS
import stiff_functions # For IVPTuple
import calc_error # For linear_error and fractional_error

def get_continous_solutions(t_cont, results):
    """Calculates the continuous numerical solutions for each method over the integration range."""

    out = {} # Empty dict

    # For each method and associated solution...
    for method, solution in results.items():
        if solution.success: # If integration was successful for this method...
            out[method] = solution.sol(t_cont) # Calculate the continuous solution over the specified range

        else:
            print("Integration failed for method '{0}' ('{1}').".format(method, solution.message))
    
    return out

def get_errors(y_analytical, cont_solutions):
    """Calculates the linear and fractional errors of the specified continuous solutions.
        
        Arguments:
            y_analytical: The analytical solution.
            cont_solutions: A dictionary containing the continuous solution for each method."""

    out = {} # Empty dict

    for method, y_num in cont_solutions.items(): # Iterate over method names and associated continuous, analytical solution...
        lin_err = calc_error.linear_error(y_num, y_analytical) # Calculate linear error of continuous, analytical solution
        frac_err = calc_error.fractional_error(lin_err, y_num) # Calculate fractional error of continuous, analytical solution

        out[method] = (lin_err, frac_err)
    
    return out

def save_analytical(t_cont, y_analytical):
    """Save the analytical solution to file."""

    paths = ["t_continuous.tsv", "y_analytical.tsv"] # Files to save data to
    data = [t_cont, y_analytical.T] # Data to save to each file (y is transposed into columns but t doesn't need to be as it's 1D)

    for fname, arr in zip(paths, data): # Iterate over paths and data in parallel...
        np.savetxt(fname, arr, delimiter="\t") # Save each array to specified files

def save_results(results):
    """Saves the (discrete) numerical solutions for each (successful) method to file."""

    for method, solution in results.items(): # For each method and associated solution...
        if solution.success: # If the integration completed successfully
            paths = ["t_numerical_{0}.tsv".format(method), "y_numerical_{0}.tsv".format(method)] # Paths to save the data to
            data = [solution.t, solution.y.T] # Data to save to file (transposing y into columns)

            for fname, arr in zip(paths, data): # Iterate over paths and data in parallel...
                np.savetxt(fname, arr, delimiter="\t") # Save each array to specified files

def save_continuous_and_error(cont_solutions, cont_errors):
    """Saves the (continuous) numerical solutions and associated linear and fractional errors for each (successful) method to file."""

    methods = list(cont_solutions.keys()) # Get the names of the methods that were successful

    # Iterate over each method and its associated continuous numerical solution and linear and fractional errors
    for method, y_cont, errs in zip(methods, map(cont_solutions.get, methods), map(cont_errors.get, methods)):
        paths = ["y_continuous_{0}.tsv".format(method), "linear_error_continuous_{0}.tsv".format(method), "fractional_error_continuous_{0}.tsv".format(method)] # Paths to save data to
        data = [y_cont.T, (errs[0]).T, (errs[1]).T] # Data to save to file (transposing into columns)

        for fname, arr in zip(paths, data): # Iterate over file names and arrays in parallel...
            np.savetxt(fname, arr, delimiter="\t") # Save each array to specified files

def save_data(t_cont, y_analytical, results, cont_solutions, cont_errors):
    """Saves all the data generated to file."""

    save_analytical(t_cont, y_analytical) # Save the analytical solution to file
    save_results(results) # Save the (discrete) numerical solutions to file
    save_continuous_and_error(cont_solutions, cont_errors) # Save the (continous) numerical solutions to file and the associated linear and fractional errors

def test_integrators_continuous(end_t, ivp, N, integrators=test_ivp.INTEGRATORS, atol=1.0e-06, rtol=1.0e-03):
    """Tests the specified integrators on the specified IVP over the specified integration range, generating continuous, interpolated solutions which are then tests with the specified number of samples.

        Arguments:
            end_t: The end point of the integration.
            ivp: A tuple containing the same information as an IVPTuple instance, in the same order.
            N: Number of samples to plot the continuous solution for.
            integrators, optional: A list of the integration methods to use (defaults to the contents of test_ivp.INTEGRATORS).
            atol, optional: The absolute error tolerance of the methods (see SciPy documentation). Defaults to 1.0e-06.
            rtol, optional: The relative error tolerance of the methods (see SciPy documentation). Defaults to 1.0e-03."""

    ivp = stiff_functions.IVPTuple._make(ivp) # Convert ivp to an IVP Tuple if it isn't one

    t_cont = np.linspace(ivp.y0, end_t, N) # Create an array of t points of length N, linearly spaced between y0 and end_t
    y_analytical = ivp.SolutionFunction(t_cont) # Plot the analytical solution

    # Perform the integrations and get the numerical solutions...
    results = test_ivp.test_integrators(end_t, ivp, integrators=integrators, dense_output=True, atol=atol, rtol=rtol)
    cont_solutions = get_continous_solutions(t_cont, results)
    cont_errors = get_errors(y_analytical, cont_solutions)

    save_data(t_cont, y_analytical, results, cont_solutions, cont_errors) # Save data to file

if __name__ == "__main__":
    test_integrators_continuous(1.5, stiff_functions.STIFF_IVP, 1000000)