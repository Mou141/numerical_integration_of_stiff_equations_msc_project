# Calculates the fractional error for numerical solutions to initial value problems at each point of the solution
import matplotlib.pyplot as plt  # For plotting graphs
import numpy as np  # For maths functions and arrays
import test_ivp  # For test_integrators function and INTEGRATORS global variable
import stiff_functions  # For STIFF_IVP and STIFF_IVP2


def linear_error(y_num, y_exact):
    """Calculates the linear error in a numerical solution by comparing with the exact solution (i.e. y_num - y_exact)."""
    return y_num - y_exact


def fractional_error(lin_err, y_num):
    """Calculates the fractional error in a numerical solution from the exact solution and the absolute value of the linear error."""
    return np.abs(lin_err) / np.abs(y_num)


def save_error(method, t, lin_err, frac_err, y):
    """Saves the t data, linear error, fractional error, and y of the solution to separate files.

    Arguments:
        method: Name of the integration method used (used to name the files).
        t: Array containing t data.
        lin_err: Array containing linear error data.
        frac_err: Array containing fractional error data.
        y: Array containing dependent variables of solution.

    Returns:
        List of file names containing t, abs_err, frac_err, and y (in that order)."""

    # File paths for the t, error, and y files
    paths = [
        "t_{0}.tsv".format(method),
        "lin_err_{0}.tsv".format(method),
        "frac_err_{0}.tsv".format(method),
        "y_{0}.tsv".format(method),
    ]

    for data, path in zip(
        [t, lin_err.T, frac_err.T, y.T], paths
    ):  # For each dataset and path (transposing errors and y so that they are in column format)...
        np.savetxt(path, data, delimiter="\t")  # Save it to file

    return paths


def find_integration_errors(results, ivp, output=False):
    """Takes a dictionary that maps method names to results from scipy.integrate.solve_ivp and an Initial Value Problem, and plots the relative errors in that solution.

    Arguments:
        results: Dictionary of results from scipy.integrate.solve_ivp
        ivp: A tuple that contains the same information as stiff_functions.IVPTuple, in the same order.
        output, optional: If true, error values are written to .tsv files, 3 for each method. One contains the t values, one the error values, and the other the y-values."""

    ivp = stiff_functions.IVPTuple._make(
        ivp
    )  # If ivp isn't an IVPTuple, convert it to one
    methods = list(
        results.keys()
    )  # Get the keys in the dictionary and put them into a list (need to do this as need to keep order same for iteration AND legend labelling)

    figure, ax = plt.subplots(
        ivp.ndim, 1, sharex=True
    )  # Create a plot for each dimension of the IVP with a shared x-axis

    if ivp.ndim == 1:  # If there is only one plot...
        ax = [ax]  # Put ax in a list so that it can be subscripted

    # Label the x-axis
    ax[0].set_xlabel("t")

    if ivp.ndim == 1:  # If ivp is only 1 dimensional...
        ax[0].set_ylabel("err(y)")

    else:  # If ivp has more than 1 dimension...
        for i in range(0, ivp.ndim):  # For each dimension...
            ax[i].set_ylabel("err(y{0})".format((i + 1)))

    for method, solution in zip(
        methods, map(results.get, methods)
    ):  # For each method in the list, get the method and the solution...
        if solution.success:  # If integration completed successfully for this method...
            print("Integration succeeded for method '{0}'.".format(method))

            y_exact = ivp.SolutionFunction(
                solution.t
            )  # Calculate the analytical solution for the t values of the numerical solution
            lin_err = linear_error(solution.y, y_exact)  # Calculate the linear error
            frac_err = fractional_error(
                lin_err, solution.y
            )  # Calculate the fractional error

            # If output is true, save the error data to file
            if output:
                save_error(method, solution.t, lin_err, frac_err, solution.y)

            for i in range(0, ivp.ndim):  # For each dimension of the IVP...
                ax[i].plot(solution.t, frac_err[i])  # Plot the error in that dimension

        else:  # If integration failed...
            print("Integration failed for method '{0}'.".format(method))

            for i in range(0, ivp.ndim):  # For each dimension of the IVP...
                ax[i].plot(
                    [], []
                )  # Plot an empty graph (to keep legend ordering correct)

    figure.legend(
        methods, loc="upper right"
    )  # Add the methods to the legend and place the legend in the upper right
    plt.show()  # Display the graphs


def test_integrator_errors(
    end_t,
    ivp,
    integrators=test_ivp.INTEGRATORS,
    output=False,
    atol=1.0e-06,
    rtol=1.0e-03,
):
    """Applies a series of integration methods supported by scipy.integrate.solve_ivp to an initial value problem and plots the fractional errors of the results.

    Arguments:
        end_t: The end point of the integration.
        ivp: A tuple containing the same information as an IVPTuple instance, in the same order.
        integrators, optional: A list of the integration methods to use (defaults to the contents of test_ivp.INTEGRATORS).
        output, optional: Whether or not to output data to file (default False).
        atol, optional: The absolute error tolerance of the methods (see SciPy documentation). Defaults to 1.0e-06.
        rtol, optional: The relative error tolerance of the methods (see SciPy documentation). Defaults to 1.0e-03."""

    results = test_ivp.test_integrators(
        end_t, ivp, integrators=integrators, atol=atol, rtol=rtol
    )  # Apply the methods to the IVP
    find_integration_errors(
        results, ivp, output=output
    )  # Find the errors for each method and plot them for each dimension of the IVP


if __name__ == "__main__":
    test_integrator_errors(1.5, stiff_functions.STIFF_IVP, output=True)
