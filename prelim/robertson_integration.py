"""Tests integration methods on the IVP given in robertson_ivp.py"""
import argparse
import robertson_ivp
import calc_error
import test_ivp
import matplotlib.pyplot as plt
import numpy as np


def parse_args(args=None):
    """Parses the specified command line arguments or uses sys.argv if none are specified.

    Returns:
        Instance of stiff_problems.IVPTuple for Robertson IVP."""

    parser = argparse.ArgumentParser(
        description="Integrates the Robertson chemical kinetics stiff IVP with the stiff integration methods provided by SciPy."
    )

    parser.add_argument(
        "-us",
        "--use-static",
        dest="use_static",
        help="Use the static initial values in robertson_ivp.TEST_IVP instead of randomly generated values.",
    )

    if args is None:
        # Use sys.argv arguments
        parsed = parser.parse_args()

    else:
        parsed = parser.parse_args(args=args)

    if parsed.use_static:
        return robertson_ivp.TEST_IVP

    else:
        # Use randomly generated values
        return robertson_ivp.get_robertson_ivp()


def get_constraint(y):
    """Takes the y-values of the solution and returns y1 + y2 + y3 (which should equal 1.0)."""
    return y[0] + y[1] + y[2]


def add_to_graph(ax, t, lin_err):
    """Plots the linear error of the solution to the graphs, one for each dimension"""

    # For each dimension of the IVP, get the subplot and linear error
    for axes, err in zip(ax, lin_err[:]):
        axes.plot(t, err)


def save_data(method, t, y, lin_err):
    """Saves t, y, and lin_err to files named by method (transposing y and lin_err into column format)."""

    files = [f.format(method) for f in ("t_{0}.tsv", "y_{0}.tsv", "lin_err_{0}.tsv")]

    # Transpose y and lin_err so the dimensions are the columns, rather than the rows
    data = [t, y.T, lin_err.T]

    for f, d in zip(files, data):
        np.savetxt(f, d, delimiter="\t")


def main():
    ivp = parse_args()

    print("y0 = {0}".format(str(ivp.y0)))

    # The end_t value of 1.0E04 is taken from the graph in figure 104(i) in Butcher, J. C. (2016). Numerical methods for ordinary differential equations / J.C. Butcher (Third edition. ed.). Wiley.
    results = test_ivp.test_integrators(1.0e04, ivp)

    # Subplot for each dimension of IVP, with shared x axis
    figure, ax = plt.subplots(3, 1, sharex=True)

    # Only set x label once, axes share it
    ax[0].set_xlabel("t")

    # y label for each subplot
    ax[0].set_ylabel("err(y1)")
    ax[1].set_ylabel("err(y2)")
    ax[2].set_ylabel("err(y3)")

    # Put methods in list to fix order of keys
    methods = list(results.keys())

    for method, solution in zip(methods, map(results.get, methods)):
        if solution.success:
            print("Method '{0}' completed successfully.".format(method))

            # Number of evaluations of right hand side
            print("\tnfev: {0}".format(solution.nfev))
            # Number of evaluations of the Jacobian
            print("\tnjev: {0}".format(solution.njev))
            # Number of LU decompositions
            print("\tnlu: {0}".format(solution.nlu))

            constraint = get_constraint(solution.y)
            lin_err = calc_error.linear_error(constraint, 1.0)

            add_to_graph(ax, solution.t, lin_err)
            save_data(method, solution.t, solution.y, lin_err)

        else:
            print(
                "Method '{0}' failed with message: '{1}'.".format(
                    method, solution.message
                )
            )

            # Empty graph plot to keep legend ordering correct
            add_to_graph(ax, np.array([]), np.array([[], [], []]))

    plt.legend(methods, loc="upper right")
    plt.show()


if __name__ == "__main__":
    main()
