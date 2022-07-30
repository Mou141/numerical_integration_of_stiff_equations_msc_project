"""Integrates the Robertson Chemical Kinetics IVP repeatedly for each integration method and performs statistics on the resulting linear errors."""
import argparse
import sys
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt

import robertson_ivp
import robertson_integration
import test_ivp
import calc_error
import analyse_error


def parse_args(args=None):
    """Parses command line arguments when this subroutine is executed directly and returns the number of integrations to perform for each method."""

    parser = argparse.ArgumentParser(
        description="Integrates the Robertson Chemical Kinetics IVP repeatedly for each method with random initial values and analyses the results."
    )

    parser.add_argument(
        "-n",
        "--number",
        type=int,
        help="The number of times to integrate the IVP for each method.",
        default=10,
        required=False,
        dest="n",
    )

    if args is not None:
        parsed = parser.parse_args(args)

    else:
        parsed = parser.parse_args()

    if parsed.n <= 0:
        print(
            "Number of times to integrate must be greater than 0 (current value: {0}).".format(
                parsed.n
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    return parsed.n


def generate_robertson_ivps(n, gen_func=robertson_ivp.get_random_y0):
    """Returns an iterator that produces n instances of the Robertson IVP with randomly chosen initial values."""

    for _ in range(n):
        yield robertson_ivp.get_robertson_ivp(gen_func=gen_func)


class RobertsonStatsTuple(
    namedtuple("RobertsonStatsTuple", ["l2", "l_inf", "nfev", "njev", "nlu"])
):
    """Named tuple to contain information on executions of a integration method on the Robertson IVP.

    l2: l2 (RMS) of linear error for each execution.
    l_inf: Maximum linear error for each execution.
    nfev: Number of evaluations of right hand side for each execution.
    njev: Number of evaluations of the Jacobian for each execution.
    nlu: Number of LU decompositions for each execution."""

    @classmethod
    def _create_tuple(cls, n):
        """Creates a RobertsonStatsTuple instance to hold the data for n executions of an integration method."""
        # Use np.nan to indicate missing data for floats and 0 for ints
        l2 = np.full(shape=(n,), fill_value=np.nan, dtype=float)
        l_inf = np.full(shape=(n,), fill_value=np.nan, dtype=float)
        nfev = np.zeros(shape=(n,), dtype=int)
        njev = np.zeros(shape=(n,), dtype=int)
        nlu = np.zeros(shape=(n,), dtype=int)

        return cls._make((l2, l_inf, nfev, njev, nlu))


def create_y0_array(n):
    """Creates a numpy array of floats of shape (n, 3) to hold the initial values for each integration of the IVP for each method."""
    # np.nan indicates missing data
    return np.full(shape=(n, 3), fill_value=np.nan, dtype=float)


def create_robertson_method_dict(n, methods):
    """Creates a dictionary which contains a RobertsonStatsTuple instance for each method specified."""

    out = {}

    for m in methods:
        out[m] = RobertsonStatsTuple._create_tuple(n)

    return out


def perform_integration(ivp, methods):
    """Integrates the given Robertson IVP with each of the specified methods, and returns the l2, l_inf, nfev, njev, and nlu values for each method where successful and np.nan values where not."""

    out = {}

    # Perform integration for every method on this ivp
    results = test_ivp.test_integrators(1.0e02, ivp, integrators=methods)

    for method, solution in results.items():
        if solution.success:
            constraint = robertson_integration.get_constraint(solution.y)
            lin_err = calc_error.linear_error(constraint, 1.0)

            l2 = analyse_error.l_2_norm(lin_err)
            l_inf = analyse_error.l_infinity_norm(lin_err)

            out[method] = l2, l_inf, solution.nfev, solution.njev, solution.nlu

        else:
            print(
                "Method '{0}' failed for initial values {1}.".format(
                    method, str(ivp.y0)
                )
            )
            # nan to indicate error for floats and 0 for ints
            out[method] = np.nan, np.nan, 0, 0, 0

    return out


def find_stats(n, methods=test_ivp.INTEGRATORS):
    """Generates n random sets of initial values, performs integration of each set of initial values with each integration method,"""

    random_ivps = generate_robertson_ivps(n)
    y0 = create_y0_array(n)
    stats_dict = create_robertson_method_dict(n, methods)

    for i, ivp in zip(range(0, n), random_ivps):
        y0[i] = ivp.y0

        results = perform_integration(ivp, methods)

        for m, stats in stats_dict.items():
            l2, l_inf, nfev, njev, nlu = results[m]

            stats.l2[i] = l2
            stats.l_inf[i] = l_inf
            stats.nfev[i] = nfev
            stats.njev[i] = njev
            stats.nlu[i] = nlu

    return y0, stats_dict


def save_data(y0, stats):
    """Saves one file, "y0.tsv", which contains all the randomly generated initial values and files for each method which contain the l2, l_inf, nfev, njev, and nlu data."""

    np.savetxt("y0.tsv", y0, delimiter="\t")

    for method, stats_tuple in stats.items():
        # Use generator expression here rather than list comprehension to reduce number of loops
        files = (
            f.format(method)
            for f in [
                "l2_{0}.tsv",
                "l_inf_{0}.tsv",
                "nfev_{0}.tsv",
                "njev_{0}.tsv",
                "nlu_{0}.tsv",
            ]
        )

        for f, d in zip(files, stats_tuple):
            np.savetxt(f, d, delimiter="\t")


def make_l2_histograms(stats_dict):
    """Plots a histogram of the l^2 values for each method using the doane bin edge calculation method."""

    # One plot for each integration method, arranged horizontally, sharing a y-axis (where the counts for each bin are)
    figure, ax = plt.subplots(1, ncols=len(stats_dict), sharey=True)

    if len(stats_dict) == 1:
        # Put ax in a list so it can be subscripted
        ax = [ax]

    # Only set y-label once since y-axis is shared
    ax[0].set_ylabel("N")

    # Log axis for y, linear for x
    ax[0].set_yscale("log")

    methods = list(stats_dict.keys())

    # Gets the stats for the specified method from the dictionary and returns the l2 values specifically
    getter = lambda m: stats_dict[m].l2

    for method, l2, graph in zip(methods, map(getter, methods), ax):
        # Remove nan values
        l2_filtered = l2[np.logical_not(np.isnan(l2))]

        graph.set_title("{0} (N = {1})".format(method, len(l2_filtered)))
        graph.set_xlabel("l^2")
        graph.hist(l2_filtered, bins="doane")

    plt.show()


def make_l_inf_histograms(stats_dict):
    """Plots a histogram of the l^inf values for each method using the doane bin edge calculation method."""

    # One plot for each integration method, arranged horizontally, sharing a y-axis (where the counts for each bin are)
    figure, ax = plt.subplots(1, ncols=len(stats_dict), sharey=True)

    if len(stats_dict) == 1:
        # Put ax in a list so it can be subscripted
        ax = [ax]

    # Only set y-label once since y-axis is shared
    ax[0].set_ylabel("N")

    # log axis for y, linear for x
    ax[0].set_yscale("log")

    methods = list(stats_dict.keys())

    # Gets the stats for the specified method from the dictionary and returns the l_inf values specifically
    getter = lambda m: stats_dict[m].l_inf

    for method, l_inf, graph in zip(methods, map(getter, methods), ax):
        # Filter out nan values
        l_inf_filtered = l_inf[np.logical_not(np.isnan(l_inf))]

        graph.set_title("{0} (N = {1})".format(method, len(l_inf_filtered)))
        graph.set_xlabel("l^∞")
        graph.hist(l_inf_filtered, bins="doane")

    plt.show()


def main():
    n = parse_args()
    y0, stats = find_stats(n)
    save_data(y0, stats)
    make_l2_histograms(stats)
    make_l_inf_histograms(stats)


if __name__ == "__main__":
    main()
