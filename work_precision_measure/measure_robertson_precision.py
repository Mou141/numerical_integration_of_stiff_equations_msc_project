"""Measures the execution time for different integration methods at different relative tolerances for the Robertson IVP."""

from work_precision_measure import METHODS, PRELIM_PATH, measure_all_methods, save_data

import sys

sys.path.append(str(PRELIM_PATH))

from robertson_ivp import TEST_IVP

import matplotlib.pyplot as plt
import numpy as np

N = 5
T_SPAN = (0.0, 400.0)

RTOL_RANGE = np.array(
    [1.0e-12, 1.0e-11, 1.0e-10, 1.0e-9, 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3]
)


def main():
    results = measure_all_methods(
        METHODS, RTOL_RANGE, TEST_IVP.ODEFunction, TEST_IVP.y0, T_SPAN, n=N
    )

    save_data(RTOL_RANGE, results)

    plt.xlabel("Relative Tolerance")
    plt.ylabel("Minimum Execution Time (ns)")

    plt.xscale("log")

    for method_name, times in results.items():
        plt.plot(RTOL_RANGE, times, marker="o", label=method_name)

    plt.legend(loc="best")

    plt.show()


if __name__ == "__main__":
    main()
