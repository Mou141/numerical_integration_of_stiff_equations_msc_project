"""Measures the execution time for different integration methods at different relative tolerances."""

from cProfile import label
import sys
from pathlib import Path
from turtle import rt

# Path of this file's folder's folder
PARENT_PATH = Path(__file__).resolve().parent.parent

# Path of prelim folder
PRELIM_PATH = PARENT_PATH / "prelim"

sys.path.extend([str(PARENT_PATH), str(PRELIM_PATH)])

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import time

from integrators import exp4
from stiff_functions import STIFF_IVP

# The name and ODESolver instance of each method to test
METHODS = [
    ("EXP4", exp4.EXP4),
    ("Radau", scipy.integrate.Radau),
    ("BDF", scipy.integrate.BDF),
]

# Integration range
T_SPAN = (0.0, 2.0)

# Range of relative tolerances to try
RTOL_RANGE = np.array([1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1])

# Number of times to perform each integration
N = 5


def measure_method(method_name, method_class, rtol, fun, y0, t_span, atol=1.0e-6):
    """Measures the execution time of the integration of the specified function with the specified method."""

    print("Integrating with {0} (rtol = {1})...".format(method_name, rtol))

    # Time of monotonic (i.e. guaranteed not to decrease) clock in nanoseconds
    start_time = time.monotonic_ns()

    results = scipy.integrate.solve_ivp(
        fun, t_span, y0, method=method_class, atol=atol, rtol=rtol
    )

    duration = time.monotonic_ns() - start_time

    if results.success:
        print("Integration successful ({0} ns).".format(duration))
        return duration

    else:
        print("Integration failed.")
        # Using np.inf to indicate failure helps with sorting.
        # This result is guaranteed not to be the minimum unless all values are np.inf.
        return np.inf


def measure_all_methods(methods, rtol_range, fun, y0, t_span, n=5, atol=1.0e-6):
    """For each method and relative tolerance, execute n times and take the lowest execution time."""

    out = {}

    for method_name, method_class in methods:
        times = np.empty(shape=rtol_range.shape, dtype=int)

        for i, rtol in enumerate(rtol_range):
            # Measure execution time n times and take minimum
            times[i] = min(
                [
                    measure_method(
                        method_name, method_class, rtol, fun, y0, t_span, atol=atol
                    )
                    for j in range(n)
                ]
            )

        out[method_name] = times

    return out


def save_data(rtol_range, results_dict):
    """Saves the range of rtol values and the times for each method to file."""

    np.savetxt("rtol.tsv", rtol_range)

    for method_name, times in results_dict.items():
        np.savetxt("{0}.tsv".format(method_name), times)


def main():
    results = measure_all_methods(
        METHODS, RTOL_RANGE, STIFF_IVP.ODEFunction, STIFF_IVP.y0, T_SPAN, N
    )

    save_data(RTOL_RANGE, results)

    plt.xlabel("Relative Tolerance")
    plt.ylabel("Execution Time")

    for method_name, times in results.items():
        plt.plot(RTOL_RANGE, times, label=method_name)

    plt.legend(loc="best")

    plt.show()


if __name__ == "__main__":
    main()
