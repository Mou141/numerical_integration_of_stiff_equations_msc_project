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
        action="store_true",
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


def save_data(method, t, y, lin_err):
    """Saves the t, y, and linear error data of the specified solution to file (transposing y so that the dimensions are the columns rather than rows)."""

    files = [f.format(method) for f in ("t_{0}.tsv", "y_{0}.tsv", "lin_err{0}.tsv")]
    datasets = [t, y.T, lin_err]

    for f, data in zip(files, datasets):
        np.savetxt(f, data, delimiter="\t")


def main():
    ivp = parse_args()

    print("y0 = {0}".format(str(ivp.y0)))

    # The end_t value of 1.0E04 is taken from the graph in figure 104(i) in Butcher, J. C. (2016). Numerical methods for ordinary differential equations / J.C. Butcher (Third edition. ed.). Wiley.
    results = test_ivp.test_integrators(1.0e04, ivp)

    plt.set_xlabel("t")
    plt.set_ylabel("Linear Error")

    for method, solution in results.items():
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

            plt.plot(solution.t, lin_err, label=method)
            save_data(method, solution.t, solution.y, lin_err)

        else:
            print(
                "Method '{0}' failed with message: '{1}'.".format(
                    method, solution.message
                )
            )

    plt.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    main()
