# Tries out SciPy's builtin stiff integrators on a 2D Initial Value Problem
import matplotlib.pyplot as plt  # For plotting graphs
import numpy as np  # For maths and arrays
import stiff_functions  # For the STIFF_IVP2 problem
import test_ivp  # For test_integrators and INTEGRATORS


def main():
    end_t = 100.0  # End point for integration

    print(
        "Testing 2D initial value problem for integrators: {0}...".format(
            ", ".join(test_ivp.INTEGRATORS)
        )
    )

    results = test_ivp.test_integrators(
        end_t, stiff_functions.STIFF_IVP2, integrators=test_ivp.INTEGRATORS
    )  # Apply the integration methods to the IVP... (test_ivp.INTEGRATORS is passed explicity to guard against future code changes where it may no longer be the default)

    figure, ax = plt.subplots(
        2, 1, sharex=True
    )  # Create two separate plots, one above the other, that share an x-axis but have different y-axes

    # Label the axes
    ax[1].set_xlabel("t")
    ax[0].set_ylabel("y1")
    ax[1].set_ylabel("y2")

    for method, solution in results.items():
        if solution.success:  # If this integration method succeeded...
            print("Method '{0}' completed successfully.".format(method))
            ax[0].plot(solution.t, solution.y[0], label=method)  # Plot y1
            ax[1].plot(solution.t, solution.y[1])  # Plot y2

        else:  # If this method failed...
            print("Method '{0}' failed.".format(method))

    # Plot the exact solution
    t = np.linspace(stiff_functions.STIFF_IVP2.t0, end_t, 10000)
    y = stiff_functions.STIFF_IVP2.SolutionFunction(t)
    ax[0].plot(t, y[0], label="Exact")
    ax[1].plot(t, y[1])

    figure.legend(loc="upper right")  # Add a legend to the graph

    plt.show()  # Display the graph


if __name__ == "__main__":
    main()
